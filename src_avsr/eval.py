# -*- coding: utf-8 -*-
import os
import sys
import logging
import ast
import torch
import hydra
import editdistance
import numpy as np
from datetime import datetime
from tqdm import tqdm
from omegaconf import DictConfig, open_dict

from fairseq import checkpoint_utils, tasks, utils

# Set up logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("fairseq_cli.eval")

def calculate_metrics(hyps, refs):
    """
    Calculate Character-level Precision, Recall, and F1 scores.
    """
    tp, fp, fn = 0, 0, 0
    for h, r in zip(hyps, refs):
        h_set, r_set = set(list(h)), set(list(r))
        tp += len(h_set & r_set)
        fp += len(h_set - r_set)
        fn += len(r_set - h_set)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1

@hydra.main(config_path="conf", config_name="eval")
def main(cfg: DictConfig):
    """
    Main inference entry point using Hydra for configuration management.
    """
    # 1. Initialize environment
    utils.import_user_module(cfg.common)
    
    # 2. Setup Task
    task = tasks.setup_task(cfg.task)

    # [CRITICAL FIX] Inject overrides into task configuration
    # Must explicitly copy whisper_path from model config to task config
    with open_dict(task.cfg):
        # --- Fix for MissingMandatoryValue: whisper_path ---
        if hasattr(cfg.model, 'whisper_path') and cfg.model.whisper_path:
            task.cfg.whisper_path = cfg.model.whisper_path
            
        # Override data paths
        if cfg.override.data:
            task.cfg.data = cfg.override.data
        if cfg.override.label_dir:
            task.cfg.label_dir = cfg.override.label_dir
            
        # Override model/noise settings
        if cfg.override.llm_path:
            task.cfg.llm_path = cfg.override.llm_path
        if cfg.override.noise_snr is not None:
            task.cfg.noise_snr = cfg.override.noise_snr
        if cfg.override.noise_prob is not None:
            task.cfg.noise_prob = cfg.override.noise_prob
        if cfg.override.noise_wav:
            task.cfg.noise_wav = cfg.override.noise_wav
        
        # Parse modalities
        if cfg.override.modalities:
            try:
                task.cfg.modalities = ast.literal_eval(str(cfg.override.modalities))
            except:
                task.cfg.modalities = cfg.override.modalities

    # 3. Load Model
    logger.info(f"Loading model from {cfg.common_eval.path}")
    
    arg_overrides = {
        "w2v_path": cfg.model.w2v_path,
        "whisper_path": cfg.model.whisper_path,
        "llm_path": cfg.model.llm_path,
        "ctc_vocab_path": cfg.model.ctc_vocab_path
    }
    
    models, _model_args = checkpoint_utils.load_model_ensemble(
        [cfg.common_eval.path], 
        arg_overrides=arg_overrides,
        task=task
    )
    model = models[0].cuda().eval()

    # 4. Prepare Prompt
    tokenizer = model.tokenizer
    messages = [
            {"role": "user", "content": "请将这段语音转写为中文："},
        ]
    prompt_ids = tokenizer.apply_chat_template(messages, add_special_tokens=True, return_tensors="pt")[0].cuda()

    # 5. Load Dataset
    subset = cfg.dataset.gen_subset
    task.load_dataset(subset)
    dataset = task.dataset(subset)
    
    batch_size = cfg.dataset.get('batch_size', 1)
    
    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=batch_size,
        num_workers=cfg.dataset.num_workers,
    ).next_epoch_itr(shuffle=False)

    # 6. Setup Logging
    res_base_dir = cfg.common_eval.results_path or "./results"
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    save_dir = os.path.join(res_base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    log_file_path = os.path.join(save_dir, "eval_log.txt")
    
    logger.info(f"Starting inference on {subset}. Results will be saved to {log_file_path}")

    # Statistics
    hit_count, total_samples = 0, 0
    total_dist, total_char_len = 0, 0
    all_hyps, all_refs = [], []

    # 7. Inference Loop
    with open(log_file_path, "w", encoding="utf-8", buffering=1) as log_f:
        pbar = tqdm(itr, desc="Decoding")
        for sample in pbar:
            sample = utils.move_to_cuda(sample)
            bsz = len(sample["id"])
            
            sample["net_input"]["source"]["instruction"] = [prompt_ids.clone() for _ in range(bsz)]

            with torch.no_grad():
                gen_args = cfg.generation
                output_ids = model.generate(
                    num_beams=gen_args.beam, 
                    temperature=gen_args.temperature,
                    # Pass extra args for module.py to pick up
                    repetition_penalty=getattr(gen_args, 'repetition_penalty', 1.0),
                    no_repeat_ngram_size=getattr(gen_args, 'no_repeat_ngram_size', 0),
                    length_penalty=getattr(gen_args, 'lenpen', 1.0),
                    **sample["net_input"]
                )
            
            hypos = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            refs = [tokenizer.decode(t[(t != tokenizer.pad_token_id) & (t != tokenizer.eos_token_id)], skip_special_tokens=True) 
                    for t in sample["target_list"]]

            for hid, h, r in zip(sample["utt_id"], hypos, refs):
                h_c = h.replace(" ", "").replace("，","").replace("。","")
                r_c = r.replace(" ", "").replace("，","").replace("。","")
                
                # Metrics
                h_set = set(list(h_c))
                r_set = set(list(r_c))
                
                recall_rate = len(h_set & r_set) / len(r_set) if len(r_set) > 0 else 0
                is_hit = recall_rate > 0.7
                if is_hit: hit_count += 1
                
                dist = editdistance.eval(list(h_c), list(r_c))
                raw_cer = (dist / len(r_c)) * 100 if len(r_c) > 0 else 100.0
                cer_val = min(raw_cer, 100.0)
                
                total_dist += dist
                total_char_len += len(r_c)
                total_samples += 1
                all_hyps.append(h_c); all_refs.append(r_c)
                
                # Log
                tqdm.write(f"ID: {hid} | CER: {cer_val:.1f}% | Recall: {recall_rate:.1%} | HIT: {is_hit}")
                tqdm.write(f"REF: {r_c}")
                tqdm.write(f"HYP: {h_c}")
                tqdm.write("-" * 30)
                
                hit_tag = "[PASS]" if is_hit else "[FAIL]"
                log_f.write(f"ID: {hid} {hit_tag} (Recall: {recall_rate:.1%})\nREF: {r_c}\nHYP: {h_c}\nCER: {cer_val:.1f}%\n{'-'*30}\n")
                os.fsync(log_f.fileno())

    # 8. Summary
    final_cer = min(100.0, (total_dist / total_char_len) * 100 if total_char_len > 0 else 100.0)
    final_kwr = (hit_count / total_samples) * 100 if total_samples > 0 else 0
    p, r, f1 = calculate_metrics(all_hyps, all_refs)

    summary = f"\n| Metrics | CER: {final_cer:.2f}% | KWR: {final_kwr:.2f}% | F1: {f1:.4f}\n"
    logger.info("=" * 60)
    logger.info(summary)
    logger.info("=" * 60)
    
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write("\n" + "="*60 + summary + "="*60 + "\n")

if __name__ == "__main__":
    main()