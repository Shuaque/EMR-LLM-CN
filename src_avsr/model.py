# -*- coding: utf-8 -*-
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from fairseq import checkpoint_utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.data import Dictionary
from fairseq.models import BaseFairseqModel, register_model

# AV-HuBERT & Whisper
from av_hubert.avhubert.hubert_asr import AVHubertAsrConfig, HubertEncoderWrapper
from transformers import (
    WhisperForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Custom Modules
from .module import WhisperEncoderWrapper, Projector, build_fusion_module, build_adapter

logger = logging.getLogger(__name__)


@dataclass
class MobAVEMR_Config(AVHubertAsrConfig):
    # Paths - Set generic defaults for public release
    w2v_path: str = "/path/to/pretrained/avhubert/base_vox_iter5.pt"
    llm_path: str = "/path/to/pretrained/LLM/Qwen2.5-3B-Instruct"
    whisper_path: str = "/path/to/pretrained/whisper/whisper-large"
    ctc_vocab_path: str = field(default="", metadata={"help": "Path to global subword ctc vocab map"})

    # Dims
    whisper_embed_dim: int = 1280
    avhubert_embed_dim: int = 768
    llama_embed_dim: int = 2560

    # Fusion
    modality_fuse: str = field(
        default="concat",
        metadata={"help": "concat|add|cross-att|align-v2a|align-a2v|glu|mlp"},
    )
    fuse_heads: int = 8
    fuse_layers: int = 1
    align_dropout: float = 0.1

    # CTC
    ctc_vocab_size: int = 0

    # LoRA
    target_modules: str = "q_proj.k_proj.v_proj.o_proj"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_gradient_checkpointing: bool = True

    # Internal
    w2v_args: Optional[Any] = None


@register_model("mob-av-emr", dataclass=MobAVEMR_Config)
class MobAVEMR(BaseFairseqModel):
    def __init__(self, avhubert, whisper, llm_core, tokenizer, embeddings, cfg: MobAVEMR_Config, ctc_map=None):
        """
        Initialize the MobAVEMR model with audio/video encoders, LLM backbone, and fusion modules.
        """
        super().__init__()
        self.cfg = cfg
        self.avhubert = avhubert
        self.whisper = whisper
        self.llama = llm_core
        self.tokenizer = tokenizer
        self.llama_embeddings = embeddings

        actual_llm_dim = int(self.llama_embeddings.weight.size(1))
        self.cfg.llama_embed_dim = actual_llm_dim
        self.llm_dim = actual_llm_dim

        # Freeze Encoders
        for p in self.avhubert.parameters(): p.requires_grad = False
        for p in self.whisper.parameters(): p.requires_grad = False

        self.audio_adapter = build_adapter(cfg, "audio")
        self.video_adapter = build_adapter(cfg, "video")
        self.fusion_module = build_fusion_module(cfg)
        fused_dim = int(self.fusion_module.out_dim)

        self.avfeat_to_llm = Projector(
            input_dim=fused_dim,
            hidden_dim=(fused_dim + self.llm_dim) // 2,
            output_dim=self.llm_dim,
        )

        self.ctc_map = ctc_map
        self.ctc_head = None
        if self.ctc_map is not None:
            ctc_vocab_size = len(self.ctc_map)
            self.ctc_head = nn.Linear(self.llm_dim, ctc_vocab_size)
            with torch.no_grad():
                valid_llm_ids = self.ctc_map[1:]
                if len(valid_llm_ids) > 0:
                    self.ctc_head.weight[1:] = self.llama_embeddings.weight[valid_llm_ids].clone()
                    self.ctc_head.bias.zero_()
            
            llm_to_ctc = torch.full((self.llama_embeddings.weight.size(0),), -1, dtype=torch.long)
            llm_to_ctc[torch.tensor(valid_llm_ids, dtype=torch.long)] = torch.arange(1, len(valid_llm_ids) + 1, dtype=torch.long)
            self.register_buffer("llm_to_ctc_map", llm_to_ctc)

        self._last_av_feat = None
        self._last_prefix_mask = None

    def load_state_dict(self, state_dict, strict=True, model_cfg=None, **kwargs):
        """
        Custom load_state_dict to robustly handle DDP prefixes and dimension mismatches (e.g. adapters).
        """
        curr_state = self.state_dict()
        new_state_dict = {}
        
        # 1. Preprocess keys (remove module. prefix)
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            clean_state_dict[k] = v
            
        # 2. Match and filter keys
        loaded_modules = set()
        for k, v in clean_state_dict.items():
            if k in curr_state:
                # Skip bitsandbytes quantization metadata
                if any(q in k for q in ["absmax", "quant_map", "nested", "quant_state"]):
                    continue
                
                # Dimension check
                if v.shape != curr_state[k].shape:
                    if "ctc_head" in k:
                        logger.warning(f"[Vocab Fix] Discarding mismatched CTC Head: {v.shape} vs {curr_state[k].shape}")
                    else:
                        logger.warning(f"[Shape Warning] Skipping {k}: {v.shape} != {curr_state[k].shape}")
                    continue
                
                new_state_dict[k] = v
                
                # Track loaded modules
                if "audio_adapter" in k: loaded_modules.add("Audio Adapter")
                if "video_adapter" in k: loaded_modules.add("Video Adapter")
                if "avfeat_to_llm" in k: loaded_modules.add("Projector")
        
        # 3. Print diagnostic info
        logger.info(f"[Checkpoint] Loaded {len(new_state_dict)} keys. Key modules loaded: {list(loaded_modules)}")
        if "Projector" not in loaded_modules or "Audio Adapter" not in loaded_modules:
            logger.error("CRITICAL: ADAPTERS OR PROJECTOR NOT LOADED! MODEL WILL FAIL!")

        return super().load_state_dict(new_state_dict, strict=False)

    @classmethod
    def build_model(cls, cfg: MobAVEMR_Config, task):
        """
        Factory method to build the model instance from configuration.
        """
        state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path)
        w2v_args = state.get("cfg", None) or convert_namespace_to_omegaconf(state["args"])
        cfg.w2v_args = w2v_args

        # Config compatibility fix
        task_cfg_dict = OmegaConf.to_container(w2v_args.task, resolve=True)
        for k in ["data", "label_dir", "labels"]:
            if k in task_cfg_dict: task_cfg_dict[k] = "" if k != "labels" else []
        if "input_modality" in task_cfg_dict:
            val = task_cfg_dict["input_modality"]
            if isinstance(val, str): task_cfg_dict["input_modality"] = [val]
        w2v_args.task = OmegaConf.create(task_cfg_dict)
        
        task_pretrain = tasks.setup_task(w2v_args.task)
        
        # Inject dummy dictionary
        target_dict = Dictionary()
        target_dict.add_symbol("<mask>")
        try:
            if hasattr(task_pretrain, "dictionaries") and len(task_pretrain.dictionaries) > 0:
                target_dict = task_pretrain.dictionaries[0]
        except:
            pass

        try:
            encoder = task_pretrain.build_model(w2v_args.model)
        except Exception:
            w2v_args.model.label_rate = -1
            encoder = task_pretrain.build_model(w2v_args.model)

        avhubert = HubertEncoderWrapper(encoder, target_dict)
        
        if "model" in state:
            for k in list(state["model"].keys()):
                if any(bk in k for bk in ["label_embs_concat", "final_proj", "mask_emb"]):
                    state["model"].pop(k)
        avhubert.w2v_model.load_state_dict(state["model"], strict=False)
        avhubert.w2v_model.remove_pretraining_modules()

        whisper = WhisperEncoderWrapper(WhisperForConditionalGeneration.from_pretrained(cfg.whisper_path).model.encoder)

        try:
            device_id = torch.cuda.current_device()
        except:
            device_id = 0
            
        logger.info(f"Loading LLM to GPU {device_id}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        llm = AutoModelForCausalLM.from_pretrained(
            cfg.llm_path, quantization_config=bnb_config, device_map={"": device_id}
        )

        if cfg.use_gradient_checkpointing:
            llm = prepare_model_for_kbit_training(llm, use_gradient_checkpointing=True)

        llm = get_peft_model(llm, LoraConfig(
            r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, 
            target_modules=[m for m in cfg.target_modules.split(".") if m],
            lora_dropout=cfg.lora_dropout, bias="none", task_type="CAUSAL_LM",
        ))
        tokenizer = AutoTokenizer.from_pretrained(cfg.llm_path)

        llm_base = llm.base_model.model if hasattr(llm, "base_model") else llm
        embeddings = llm_base.get_input_embeddings() if hasattr(llm_base, "get_input_embeddings") else None
        if embeddings is None:
            for m in llm_base.modules():
                if isinstance(m, nn.Embedding):
                    embeddings = m
                    break
        if embeddings is None: raise AttributeError("Could not locate embed_tokens.")

        ctc_map = torch.load(cfg.ctc_vocab_path) if cfg.ctc_vocab_path and os.path.exists(cfg.ctc_vocab_path) else None
        return cls(avhubert, whisper, llm_base, tokenizer, embeddings, cfg, ctc_map)

    def forward(self, **kwargs):
        """
        Forward pass handling feature extraction, fusion, and LLM forward.
        """
        src, labels = kwargs["source"], kwargs.get("target_list")
        av_feat, prefix_mask = self.encode_for_llm(src)
        self._last_av_feat, self._last_prefix_mask = av_feat, prefix_mask

        llm_inputs, attn_mask, llm_labels = self.prepare_inputs_labels_for_queries(
            instructions=src["instruction"], queries=av_feat, 
            len_queries=prefix_mask.sum(dim=1).long().tolist(), labels=labels
        )
        out = self.llama(inputs_embeds=llm_inputs, attention_mask=attn_mask, labels=llm_labels, use_cache=False)
        return out.loss, out.logits, llm_labels

    def prepare_inputs_labels_for_queries(self, instructions, queries, len_queries, labels=None):
        """
        Prepares the input sequence for the LLM by concatenating instructions, queries (features), and labels.
        """
        llm_input_list, llm_labels_list, lengths = [], [], []
        B, device = queries.size(0), queries.device

        for i in range(len(instructions)):
            inst_emb = self.llama_embeddings(instructions[i].unsqueeze(0).to(device)).squeeze(0)
            q = queries[i][:int(len_queries[i]), :]
            if labels is not None:
                lab_ids = labels[i].to(device)
                combined = torch.cat([inst_emb, q, self.llama_embeddings(lab_ids.unsqueeze(0)).squeeze(0)], dim=0)
                mask = torch.full((combined.size(0),), -100, dtype=torch.long, device=device)
                mask[inst_emb.size(0) + q.size(0) : ] = lab_ids
                llm_labels_list.append(mask)
            else:
                combined = torch.cat([inst_emb, q], dim=0)
            llm_input_list.append(combined)
            lengths.append(combined.size(0))

        max_len = max(lengths)
        pad_emb = self.llama_embeddings(torch.tensor([self.tokenizer.pad_token_id or self.tokenizer.eos_token_id], device=device)).squeeze(0)
        res_inputs = pad_emb.unsqueeze(0).unsqueeze(0).expand(B, max_len, -1).clone()
        res_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        res_labels = torch.full((B, max_len), -100, dtype=torch.long, device=device) if labels is not None else None

        for i, seq in enumerate(llm_input_list):
            res_inputs[i, :seq.size(0)] = seq
            res_mask[i, :seq.size(0)] = 1
            if labels is not None: res_labels[i, :seq.size(0)] = llm_labels_list[i]
        return res_inputs, res_mask, res_labels

    def encode_for_llm(self, source: dict):
        """
        Encodes audio and video inputs, applies adapters, and fuses them for LLM consumption.
        """
        device, dtype = next(self.llama.parameters()).device, next(self.llama.parameters()).dtype
        padding_mask = source.get("padding_mask") 
        w = self.audio_adapter(self.whisper(source)) if source.get("audio") is not None else None
        
        # --- Video Processing ---
        v = None
        if source.get("video") is not None:
            # 1. Extract raw features
            av_out = self.avhubert(source={"audio":None, "video":source["video"]}, padding_mask=padding_mask)
            v = av_out["encoder_out"].transpose(0, 1) # [B, T, D]
            
            # 2. Safety padding for short videos
            MIN_LEN = 10 
            if v.size(1) < MIN_LEN:
                pad_len = MIN_LEN - v.size(1)
                padding = torch.zeros((v.size(0), pad_len, v.size(2)), device=device, dtype=v.dtype)
                v = torch.cat([v, padding], dim=1)
                
                # Update padding mask
                if padding_mask is not None:
                    pm_pad = torch.ones((v.size(0), pad_len), device=device, dtype=padding_mask.dtype)
                    padding_mask = torch.cat([padding_mask, pm_pad], dim=1)
            
            # 3. Adapter projection
            v = self.video_adapter(v)

        if w is not None and v is not None:
            T = min(w.size(1), v.size(1))
            w, v = w[:, :T, :], v[:, :T, :]
        
        T_out = w.size(1) if w is not None else v.size(1)
        av_feat = self.avfeat_to_llm(self.fusion_module(w, v)).to(device=device, dtype=dtype)
        
        # Prefix mask logic for variable length sequences
        if padding_mask is not None and source.get("video") is not None:
            valid_len = (~padding_mask).long().sum(dim=1) 
            # Account for stride=2 in adapter
            valid_len = torch.clamp(valid_len // 2, min=1, max=T_out)
            
            prefix_mask = torch.zeros(av_feat.size(0), T_out, device=device, dtype=torch.long)
            for i in range(av_feat.size(0)): prefix_mask[i, :valid_len[i]] = 1
        else:
            prefix_mask = torch.ones(av_feat.size(0), T_out, device=device, dtype=torch.long)
            
        return av_feat, prefix_mask

    @torch.no_grad()
    def generate(self, num_beams=5, temperature=0.0, max_new_tokens=100, **net_input):
        """
        Generates text output with parameters passed from eval.yaml via net_input kwargs.
        """
        source = net_input["source"]
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        inst = source["instruction"]
        
        if not isinstance(inst, torch.Tensor):
            lens = [x.numel() for x in inst]
            tmp = inst[0].new_full((len(inst), max(lens)), pad_id)
            for i, t in enumerate(inst): tmp[i, :t.numel()] = t
            inst = tmp
        
        inst = inst.to(next(self.llama.parameters()).device)
        prefix_embeds, prefix_mask = self.encode_for_llm(source)
        inputs_embeds = torch.cat([self.llama.get_input_embeddings()(inst), prefix_embeds.to(dtype=self.llama.dtype)], dim=1)
        attn_mask = torch.cat([(inst != pad_id).long(), prefix_mask.to(inst.device).long()], dim=1)

        # [CRITICAL FIX] Extract generation args from net_input (passed via eval.py)
        # We set defaults but allow overrides
        gen_kwargs = {
            "inputs_embeds": inputs_embeds, 
            "attention_mask": attn_mask,
            "num_beams": num_beams, 
            "do_sample": False, 
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_id, 
            "eos_token_id": self.tokenizer.eos_token_id,
            "early_stopping": True
        }
        
        # Manually map specific generation keys if they exist in net_input
        # These keys come from cfg.generation in eval.py
        if "repetition_penalty" in net_input:
            gen_kwargs["repetition_penalty"] = net_input["repetition_penalty"]
        if "no_repeat_ngram_size" in net_input:
            gen_kwargs["no_repeat_ngram_size"] = net_input["no_repeat_ngram_size"]
        if "length_penalty" in net_input: # map lenpen -> length_penalty
            gen_kwargs["length_penalty"] = net_input["length_penalty"]

        return self.llama.generate(**gen_kwargs)