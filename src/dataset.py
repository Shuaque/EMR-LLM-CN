import json
import os
import torch
import numpy as np
from typing import List, Dict, Any
from fairseq.data import FairseqDataset, data_utils

# Standard English comments. Fixed parent_idx_map robust parsing.

class MedicalHierarchicalDataset(FairseqDataset):
    def __init__(
        self,
        json_path: str,
        label_bank_path: str,
        tokenizer: Any,
        max_source_len: int = 512,
    ):
        """
        Loads hierarchical medical data. Supports dialogue_id for cross-window evaluation.
        Fixed: Robust parsing for parent_idx_map (supports both List and Dict).
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len

        # 1. Load Ontology and handle parent_idx_map
        if not os.path.exists(label_bank_path):
            raise FileNotFoundError(f"Ontology file missing at {label_bank_path}")
            
        with open(label_bank_path, "r", encoding="utf-8") as f:
            ontology = json.load(f)
        
        # Robust parsing for List or Dict format
        raw_map = ontology.get("parent_idx_map", [])
        if isinstance(raw_map, list):
            # If List: Index is sub_idx, Value is parent_idx
            self.parent_idx_map = {i: int(v) for i, v in enumerate(raw_map)}
        elif isinstance(raw_map, dict):
            # If Dict: Key is sub_idx, Value is parent_idx
            self.parent_idx_map = {int(k): int(v) for k, v in raw_map.items()}
        else:
            raise TypeError("parent_idx_map in ontology.json must be a List or Dict.")

        self.num_subtopics = 206
        self.num_topics = 4

        # 2. Load Samples
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Data file not found at {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Filter valid samples
        self.samples = [s for s in raw_data if isinstance(s, dict) and 'utterances' in s]
        
        # Fairseq uses 'sizes' for batching efficiency
        self.sizes = np.array([len(" ".join(s["utterances"])) for s in self.samples], dtype=np.int64)
        print(f"| [Dataset] Loaded {len(self.samples)} samples. parent_idx_map size: {len(self.parent_idx_map)}")

    def _build_prompt(self, utterances: List[str]) -> str:
        """
        Constructs a medical extraction prompt in ChatML format.
        """
        context = "\n".join([u.strip() for u in utterances if u.strip()])
        # Note: The prompt text is kept in Chinese to match the pre-trained model's expected input distribution.
        return (
            "<|im_start|>system\n"
            "你是一个医疗专家助手。请仔细分析以下医生与患者的对话，"
            "重点识别：1.症状表现, 2.临床检查, 3.手术治疗, 4.一般信息。\n"
            "请确保提取的特征能够覆盖全文所有的医疗关键信息。<|im_end|>\n"
            "<|im_start|>user\n"
            f"对话内容：\n{context}\n\n"
            "请基于对话提取相关的医疗实体标签。<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def __getitem__(self, index: int):
        s = self.samples[index]
        
        # Preserve dialogue_id to support final dialogue-level F1 calculation
        dial_id = s.get("dialogue_id", f"idx_{index}")
        
        prompt = self._build_prompt(s["utterances"])

        # Tokenize (padding=False as it is handled by the collater)
        tokenized = self.tokenizer(
            prompt, 
            max_length=self.max_source_len, 
            truncation=True, 
            padding=False, 
            add_special_tokens=True
        )
        
        input_ids = torch.LongTensor(tokenized["input_ids"])
        target_subtopic = torch.zeros(self.num_subtopics, dtype=torch.float)
        target_topic = torch.zeros(self.num_topics, dtype=torch.float)
        
        # Label Mapping: Subtopic -> Parent Topic
        label_ids = s.get("label_ids", [])
        for lid in label_ids:
            l_idx = int(lid)
            if l_idx < self.num_subtopics:
                target_subtopic[l_idx] = 1.0
                # Find parent topic via the robust map
                t_idx = self.parent_idx_map.get(l_idx)
                if t_idx is not None and t_idx < self.num_topics:
                    target_topic[t_idx] = 1.0

        return {
            "id": index,
            "dialogue_id": dial_id,
            "input_ids": input_ids,
            "target_topic": target_topic,
            "target_subtopic": target_subtopic,
            "ntokens": len(input_ids),
        }

    def __len__(self): 
        return len(self.samples)

    def num_tokens(self, index): 
        return self.sizes[index]

    def size(self, index): 
        return self.sizes[index]

    def collater(self, samples: List[Dict]):
        """
        Batches data and adds attention masks for pooling.
        """
        if not samples: 
            return {}
        
        # Dynamic padding to the max length in this batch
        input_ids = data_utils.collate_tokens(
            [s["input_ids"] for s in samples], 
            pad_idx=self.tokenizer.pad_token_id, 
            left_pad=False
        )
        
        # Attention Mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "dialogue_id": [s["dialogue_id"] for s in samples], # Essential for aggregation
            "ntokens": sum(s["ntokens"] for s in samples),
            "net_input": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            "target_topic": torch.stack([s["target_topic"] for s in samples]),
            "target_subtopic": torch.stack([s["target_subtopic"] for s in samples]),
        }