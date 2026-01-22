import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

from fairseq.models import BaseFairseqModel, register_model
from fairseq.dataclass import FairseqDataclass
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

@dataclass
class MieTaskLlmModelConfig(FairseqDataclass):
    # LLM Backbone Settings
    llm_path: str = field(default="", metadata={"help": "Path to pretrained LLM"})
    max_source_len: int = field(default=512)
    use_lora: bool = field(default=True)
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)
    gradient_checkpointing: bool = field(default=True)
    
    # Architecture Hyperparameters
    dropout: float = field(default=0.1) 
    num_attention_heads: int = field(default=8)
    logic_dropout: float = field(default=0.4) 
    
    # HLWMP Config: Local focus coefficient
    window_size: int = field(default=64, metadata={"help": "Window size for Local-Window Max Pooling"})
    local_focus_alpha: float = field(default=0.8, metadata={"help": "Weight for local peaks (h_local)"})
    
    # Adaptive Resource Quotas (E-IAQ)
    K_max: int = field(default=128)
    K_floor: int = field(default=16)
    k_min: int = field(default=1)
    topic_k_quotas: List[int] = field(default_factory=lambda: [80, 50, 20, 30])
    
    # Scaling and Penalties
    text_logit_scale: float = field(default=8.0) 
    train_soft_penalty: float = field(default=-7.0) 
    logic_salvage_lambda: float = field(default=0.3) 
    token_bias_alpha: float = field(default=0.3) 

@register_model("mie_task_llm_model", dataclass=MieTaskLlmModelConfig)
class MieTaskLlmModel(BaseFairseqModel):
    def __init__(self, cfg: MieTaskLlmModelConfig):
        super().__init__()
        self.cfg = cfg
        
        # 1. LLM Backbone Initialization
        self.llm = AutoModelForCausalLM.from_pretrained(
            cfg.llm_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        )

        if cfg.gradient_checkpointing:
            self.llm.config.use_cache = False
            self.llm.gradient_checkpointing_enable()
            
        if cfg.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], lora_dropout=cfg.lora_dropout,
            )
            self.llm = get_peft_model(self.llm, peft_config)
        
        hidden_size = self.llm.config.hidden_size
        self.num_subtopic = 206 

        # 2. Stage I: Topic Head
        self.label_queries = nn.Parameter(torch.randn(self.num_subtopic, hidden_size) * 0.02)
        self.topic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_size // 2, 4) 
        )

        # 3. Stage III: Sparse Cross-Attention
        self.label_cross_attn = nn.MultiheadAttention(
            hidden_size, cfg.num_attention_heads, batch_first=True, dropout=cfg.dropout
        )
        self.subtopic_proj = nn.Linear(hidden_size, 1)
        self.norm_text, self.norm_label = nn.LayerNorm(hidden_size), nn.LayerNorm(hidden_size)

        # 4. Buffers
        self.register_buffer("correlation_matrix", torch.zeros(self.num_subtopic, self.num_subtopic))
        self.register_buffer("hierarchical_mask", torch.zeros(4, self.num_subtopic))
        self.register_buffer("activity_priors", torch.tensor([0.92, 0.45, 0.08, 0.15])) 
        self.register_buffer("topic_counts", torch.tensor([180, 16, 4, 6])) 
        self.topic_ranges = [(0, 180), (180, 196), (196, 200), (200, 206)]

    @classmethod
    def build_model(cls, cfg: MieTaskLlmModelConfig, task):
        return cls(cfg)

    def load_correlation_matrix(self, path: str):
        if os.path.exists(path): 
            self.correlation_matrix.copy_(torch.load(path))

    def get_queries(self) -> torch.Tensor:
        return self.label_queries

    def _get_quotas(self, probs_topic: torch.Tensor, ratio: float) -> torch.Tensor:
        device = probs_topic.device
        K_total = max(self.cfg.K_floor, int(round(self.cfg.K_max * ratio)))
        w = torch.pow(probs_topic, 2.0) * \
            torch.pow(self.topic_counts.to(device), 0.4) * \
            torch.sqrt(self.activity_priors.to(device))
        w_sum = w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        ks = torch.floor((w / w_sum) * K_total).long().clamp(min=self.cfg.k_min)
        return ks

    def forward(self, input_ids, attention_mask, **kwargs):
        batch_size, dtype = input_ids.shape[0], self.label_queries.dtype
        
        # --- Stage I: Context Encoding (HLWMP Optimized) ---
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        H = self.norm_text(outputs.hidden_states[-1].to(dtype))
        mask_f = attention_mask.unsqueeze(-1).to(dtype)

        # Global Path
        h_global = (H * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-8)

        # Local Window Path (HLWMP)
        window_size = self.cfg.window_size
        T = H.size(1)
        pad_len = (window_size - T % window_size) % window_size
        h_padded = F.pad(H * mask_f, (0, 0, 0, pad_len), value=-1e4)
        
        num_windows = h_padded.size(1) // window_size
        h_local_peaks = h_padded.view(batch_size, num_windows, window_size, -1).max(dim=2)[0]
        h_local = h_local_peaks.mean(dim=1)

        # HLWMP Strategy: Enhance local significance over global
        alpha = self.cfg.local_focus_alpha
        h_ctx = (1 - alpha) * h_global + alpha * h_local

        logits_topic = self.topic_head(h_ctx)
        
        # Distributed Eval: Return only topic logits
        if kwargs.get("get_probs_only", False):
            return {"logits_topic": logits_topic}

        probs_topic = torch.sigmoid(logits_topic)

        # --- Stage II: Retrieval ---
        # Prioritize custom quotas if provided (for strategy testing)
        custom_ks = kwargs.get("custom_quotas", None)
        if custom_ks is not None:
            ks = torch.tensor(custom_ks, device=h_ctx.device).view(batch_size, 4)
        else:
            curr_ratio = torch.rand(1).item() * 0.9 + 0.1 if self.training else kwargs.get("ratio", 1.0)
            ks = self._get_quotas(probs_topic, curr_ratio)

        s_global = torch.matmul(h_ctx, self.label_queries.t()) 
        s_token_max, _ = torch.matmul(H, self.label_queries.t()).max(dim=1) 
        s_heuristic = s_global + self.cfg.token_bias_alpha * s_token_max
        
        logic_boost = torch.matmul(torch.sigmoid(s_heuristic), self.correlation_matrix.to(dtype))
        s_total = s_heuristic + self.cfg.logic_salvage_lambda * logic_boost

        topk_indices = []
        for b in range(batch_size):
            sample_indices = []
            for t_idx, (start, end) in enumerate(self.topic_ranges):
                k = min(int(ks[b, t_idx]), end - start)
                if k > 0:
                    _, local_idx = torch.topk(s_total[b, start:end], k)
                    sample_indices.append(local_idx + start)
            
            if self.training and "target_subtopic" in kwargs:
                target = kwargs["target_subtopic"]
                gt_indices = torch.where(target[b] > 0.5)[0]
                sample_indices.append(gt_indices)
            
            # Avoid empty indices
            if len(sample_indices) == 0:
                topk_indices.append(torch.tensor([], device=h_ctx.device, dtype=torch.long))
            else:
                topk_indices.append(torch.unique(torch.cat(sample_indices)))

        # --- Stage III: Sparse Verification ---
        init_val = self.cfg.train_soft_penalty if self.training else -1e4
        logits_sub = torch.full((batch_size, self.num_subtopic), init_val, device=h_ctx.device, dtype=dtype)
        
        for b in range(batch_size):
            idx = topk_indices[b]
            if idx.numel() == 0: continue
            
            q_sparse = self.label_queries[idx].unsqueeze(0)
            verified_feat, _ = self.label_cross_attn(
                query=self.norm_label(q_sparse), key=H[b:b+1], value=H[b:b+1],
                key_padding_mask=(attention_mask[b:b+1] == 0)
            )
            res = self.subtopic_proj(verified_feat).squeeze(-1) 
            logits_sub[b].scatter_(0, idx, res.squeeze(0).to(dtype) * self.cfg.text_logit_scale)

        # Hierarchical Log-masking Shield
        with torch.no_grad():
            soft_mask = torch.matmul(probs_topic, self.hierarchical_mask)
        # Physical masking: add eps to avoid log(0)
        logits_sub = logits_sub + 12.0 * torch.log(soft_mask + 1e-8)

        return {
            "logits_topic": logits_topic,
            "logits_subtopic": logits_sub,
            "logits_retrieval": s_total,
            "selected_indices": topk_indices,
            "avg_logic_gamma": torch.tensor([0.0], device=h_ctx.device) 
        }