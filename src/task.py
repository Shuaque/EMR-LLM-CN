import os
import json
import torch
import numpy as np
import torch.distributed as dist
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fairseq import models
from fairseq.tasks import FairseqTask, register_task
from fairseq.dataclass import FairseqDataclass

logger = logging.getLogger(__name__)

@dataclass
class MedicalHMRTaskConfig(FairseqDataclass):
    data: str = field(default="", metadata={"help": "Path to data directory"})
    ontology: str = field(default="", metadata={"help": "Path to label bank JSON"})
    matrix_path: str = field(default="", metadata={"help": "Path to correlation matrix .pt"})
    llm_path: str = field(default="", metadata={"help": "Path to pretrained LLM"})
    max_source_len: int = field(default=512)
    
    # Loss component weights
    macro_weight: float = field(default=0.3)
    asl_weight: float = field(default=0.5)
    topo_weight: float = field(default=0.2)
    
    # Threshold for topic balancing
    topic_tau: float = field(default=25.0, metadata={"help": "Capping for topic imbalance ratio"})
    
    save_snapshots: bool = field(default=True)
    snapshot_interval: int = field(default=1)
    topic_k_quotas: List[int] = field(default_factory=lambda: [80, 50, 20, 30])

@register_task("medical_hmr_task", dataclass=MedicalHMRTaskConfig)
class MedicalHMRTask(FairseqTask):
    def __init__(self, cfg: MedicalHMRTaskConfig, **kwargs):
        super().__init__(cfg)
        self.cfg = cfg
        self._ontology_data = None
        self._tokenizer = None
        self._bare_label_queries = None
        
        # Topic weights for dynamic application in criterion
        self.topic_pos_weights = torch.ones(4)
        
        self.is_master = (not dist.is_initialized()) or (dist.get_rank() == 0)
        self.run_id = datetime.now().strftime("%m%d_%H%M")
        
        # Use relative path for snapshots
        self.snapshot_dir = os.path.join(self.cfg.data, "snapshots", self.run_id)
        
        if self.is_master:
            os.makedirs(self.snapshot_dir, exist_ok=True)
            print(f"| [Task] Medical HMR Task Initialized. Snapshots: {self.snapshot_dir}")

    @property
    def ontology_data(self):
        if self._ontology_data is None:
            with open(self.cfg.ontology, "r", encoding="utf-8") as f:
                self._ontology_data = json.load(f)
        return self._ontology_data

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.llm_path, trust_remote_code=True)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def build_model(self, model_cfg):
        # 1. Build model instance
        model = models.build_model(model_cfg, self)
        
        # 2. Load correlation matrix
        if os.path.exists(self.cfg.matrix_path):
            model.load_correlation_matrix(self.cfg.matrix_path)
        
        # 3. Initialize ontology structures and hierarchy
        self._initialize_ontology_structures(model)
        
        # 4. Pre-calculate topic weights for balancing
        self._calculate_topic_reweighting()
        
        return model

    def _calculate_topic_reweighting(self):
        """
        Apply Class-Balanced Reweighting Strategy.
        Formula: gamma = N_neg / (N_pos + eps), w_pos = min(gamma, tau)
        """
        # Update n_pos based on actual stats from the training set
        # Order: Symptom, Test, Surgery, General
        n_pos = torch.tensor([4200, 450, 120, 230]).float()
        n_total = 5000.0
        n_neg = n_total - n_pos
        
        eps = 1e-8
        tau = self.cfg.topic_tau
        
        gamma = n_neg / (n_pos + eps)
        self.topic_pos_weights = torch.clamp(gamma, max=tau)
        
        if self.is_master:
            print(f"| [Task] Topic CB-Weights: {self.topic_pos_weights.tolist()}")

    def _initialize_ontology_structures(self, model):
        """Initialize both label_queries and hierarchical_mask."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 
        
        m = model.module if hasattr(model, 'module') else model
        num_sub = 206
        h_size = m.llm.config.hidden_size
        
        anchors = torch.zeros(num_sub, h_size, device=device, dtype=dtype)
        bare_v = torch.zeros(num_sub, h_size, device=device, dtype=dtype)
        h_mask = torch.zeros(4, num_sub, device=device, dtype=dtype)

        if self.is_master:
            embed_layer = m.llm.get_input_embeddings().to(device=device, dtype=dtype)
            id2label = {int(k): v for k, v in self.ontology_data["id2label"].items()}
            alias_data = self.ontology_data.get("label_alias_data", {})
            
            # Map Chinese topics to indices (Ensure these match your data)
            topic_to_idx = {"症状": 0, "检查": 1, "手术": 2, "一般信息": 3}

            with torch.no_grad():
                for i in range(num_sub):
                    label_path = id2label[i]
                    topic, concept = label_path.split(":", 1)
                    
                    if topic in topic_to_idx:
                        h_mask[topic_to_idx[topic], i] = 1.0
                    
                    b_ids = self.tokenizer(concept, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
                    bare_v[i] = embed_layer(b_ids).mean(dim=1).squeeze(0)
                    
                    aliases = alias_data.get(topic, {}).get(concept, [concept])
                    a_embs = []
                    for a in aliases:
                        # Construct full concept string
                        full_concept = f"{topic}：{a}"
                        ids = self.tokenizer(full_concept, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
                        a_embs.append(embed_layer(ids).mean(dim=1).squeeze(0))
                    anchors[i] = torch.stack(a_embs).mean(dim=0)

        if dist.is_initialized():
            dist.barrier()
            dist.broadcast(h_mask, src=0)
            dist.broadcast(anchors, src=0)
            dist.broadcast(bare_v, src=0)
        
        m.hierarchical_mask.copy_(h_mask)
        m.label_queries.data.copy_(anchors)
        self._bare_label_queries = bare_v.cpu().float().numpy()

    def load_dataset(self, split, **kwargs):
        from .dataset import MedicalHierarchicalDataset
        self.datasets[split] = MedicalHierarchicalDataset(
            json_path=os.path.join(self.cfg.data, f"{split}.json"),
            label_bank_path=self.cfg.ontology,
            tokenizer=self.tokenizer,
            max_source_len=self.cfg.max_source_len
        )

    def begin_epoch(self, epoch, model):
        if self.is_master and self.cfg.save_snapshots and epoch % self.cfg.snapshot_interval == 0:
            m = model.module if hasattr(model, 'module') else model
            self._save_and_log_semantic(f"epoch_{epoch}", m)

    def _save_and_log_semantic(self, stage, m):
        try:
            with torch.no_grad():
                refined = m.get_queries().detach().float().cpu().numpy()
            npz_fn = os.path.join(self.snapshot_dir, f"data_{stage}.npz")
            np.savez(npz_fn, bare=self._bare_label_queries, refined=refined)
            
            from sklearn.manifold import TSNE
            all_v = np.concatenate([self._bare_label_queries, refined], axis=0)
            reduced = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(all_v)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(reduced[:206, 0], reduced[:206, 1], c='gray', alpha=0.1, label='Raw Prior')
            colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
            topics = ["Symptom", "Test", "Surgery", "General"]
            ranges = m.topic_ranges 
            for i, (s, e) in enumerate(ranges):
                plt.scatter(reduced[206+s:206+e, 0], reduced[206+s:206+e, 1], 
                            c=colors[i], label=topics[i], s=40, edgecolors='white', linewidth=0.5)
            plt.title(f"Semantic Topology Evolution: {stage}")
            plt.legend(loc='upper right')
            plt.grid(True, linestyle=':', alpha=0.3)
            plt.savefig(os.path.join(self.snapshot_dir, f"viz_{stage}.png"), dpi=300)
            plt.close()
        except Exception as e:
            logger.error(f"Snapshot Error: {e}")