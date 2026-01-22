import os, time, torch, argparse, numpy as np
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from fairseq import checkpoint_utils, tasks, utils

# C3-MRAF Performance Sweep: Real-time Logging Enabled

class Logger:
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
    
    def log(self, msg=""):
        # Use tqdm.write to prevent interfering with the progress bar display
        tqdm.write(msg)
        self.file.write(msg + "\n")
        self.file.flush()
        
    def close(self):
        self.file.close()

def compute_prf(tp, fp, fn):
    """Calculates precision, recall, and f1 score."""
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1

def get_stats(preds, targets):
    """Computes hit statistics and empty match indicators."""
    tp = (preds & targets).sum().item()
    fp = (preds & ~targets).sum().item()
    fn = (~preds & targets).sum().item()
    is_empty_match = (preds.sum() == 0 and targets.sum() == 0)
    return tp, fp, fn, is_empty_match

def log_result_block(logger, title, data, prefix=""):
    """Formatted output block matching the training log structure (Macro-Average)."""
    tp_t, fp_t, fn_t = data["topic"]["tp"], data["topic"]["fp"], data["topic"]["fn"]
    t_p, t_r, t_f = compute_prf(tp_t, fp_t, fn_t)
    
    tp_s, fp_s, fn_s = data["sub"]["tp"], data["sub"]["fp"], data["sub"]["fn"]
    s_p, s_r, s_f = compute_prf(tp_s, fp_s, fn_s)
    
    o_p = (t_p + s_p) / 2
    o_r = (t_r + s_r) / 2
    o_f = (t_f + s_f) / 2

    logger.log("=" * 85)
    logger.log(f"| {prefix}{title} Evaluation Results")
    logger.log("| " + "-" * 81)
    logger.log(f"| Topic   - Prec: {t_p:.4f} | Recall: {t_r:.4f} | F1: {t_f:.4f}")
    logger.log(f"| Sub     - Prec: {s_p:.4f} | Recall: {s_r:.4f} | F1: {s_f:.4f}")
    logger.log(f"| Overall - Prec: {o_p:.4f} | Recall: {o_r:.4f} | F1: {o_f:.4f}")

def run_eval(model, dataset, task, ratio, device, logger):
    """Runs a full test pass with real-time text logging."""
    win_stats = {"topic": {"tp": 0, "fp": 0, "fn": 0, "empty": 0}, "sub": {"tp": 0, "fp": 0, "fn": 0, "empty": 0}}
    dial_store = defaultdict(lambda: {"gt_topic": torch.zeros(4), "pred_topic": torch.zeros(4), "gt_sub": torch.zeros(206), "pred_sub": torch.zeros(206)})
    total_latency, total_tokens = 0, 0
    torch.cuda.reset_peak_memory_stats()
    
    # Retrieve label mapping (ID -> Name)
    try:
        id2label = task.ontology_data.get("id2label", {})
    except Exception as e:
        logger.log(f"| Warning: Could not load ontology labels: {e}")
        id2label = {}

    pbar = tqdm(range(len(dataset)), desc=f"Eval Ratio {ratio:.1f}", leave=False)
    
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for i in pbar:
            s = dataset[i]
            input_ids = s['input_ids'].unsqueeze(0).to(device)
            attn_mask = (input_ids != task.tokenizer.pad_token_id).long().to(device)
            start_t = time.perf_counter()
            
            # Inference with HLWMP
            out = model(input_ids=input_ids, attention_mask=attn_mask, ratio=ratio)
            latency = (time.perf_counter() - start_t) * 1000
            total_latency += (latency / 1000)
            
            # Numerical Alignment
            p_topic = (torch.sigmoid(out['logits_topic'].squeeze(0).float()) > 0.5).long().cpu()
            p_sub = (torch.sigmoid(out['logits_subtopic'].squeeze(0).float()) > 0.5).long().cpu()
            t_topic, t_sub = s['target_topic'].long(), s['target_subtopic'].long()

            # ---------------- Real-time Visualization ----------------
            # Retrieve raw text
            raw_dialogue_list = dataset.samples[i].get("utterances", [])
            dialogue_str = "\n".join(raw_dialogue_list)
            
            # Parse prediction results
            pred_idx = torch.where(p_sub > 0)[0].tolist()
            gt_idx = torch.where(t_sub > 0)[0].tolist()
            
            # Map to label names
            pred_names = [id2label.get(str(idx), str(idx)) for idx in pred_idx]
            gt_names = [id2label.get(str(idx), str(idx)) for idx in gt_idx]
            
            # Construct log message
            log_msg = (
                f"\n{'='*20} Sample {i} [ID: {dataset.samples[i].get('dialogue_id', i)}] {'='*20}\n"
                f"[Dialogue]:\n{dialogue_str}\n"
                f"{'-'*10}\n"
                f"[Prediction]: {pred_names}\n"
                f"[GroundTruth]: {gt_names}\n"
                f"[Latency]: {latency:.2f}ms\n"
            )
            logger.log(log_msg)
            # ---------------------------------------------------------

            for k, p, t in [("topic", p_topic, t_topic), ("sub", p_sub, t_sub)]:
                tp, fp, fn, emp = get_stats(p, t)
                win_stats[k]["tp"] += tp; win_stats[k]["fp"] += fp; win_stats[k]["fn"] += fn
                if emp: win_stats[k]["empty"] += 1

            d = dial_store[s["dialogue_id"]]
            d["gt_topic"] = torch.max(d["gt_topic"], t_topic)
            d["pred_topic"] = torch.max(d["pred_topic"], p_topic)
            d["gt_sub"] = torch.max(d["gt_sub"], t_sub)
            d["pred_sub"] = torch.max(d["pred_sub"], p_sub)
            total_tokens += s['ntokens']

    dial_stats = {"topic": {"tp": 0, "fp": 0, "fn": 0}, "sub": {"tp": 0, "fp": 0, "fn": 0}}
    for d_id in dial_store:
        ds = dial_store[d_id]
        for k, pk, gk in [("topic", "pred_topic", "gt_topic"), ("sub", "pred_sub", "gt_sub")]:
            tp, fp, fn, _ = get_stats(ds[pk].long(), ds[gk].long())
            dial_stats[k]["tp"] += tp; dial_stats[k]["fp"] += fp; dial_stats[k]["fn"] += fn

    avg_lat = (total_latency / len(dataset)) * 1000
    tps = total_tokens / total_latency
    max_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return win_stats, dial_stats, avg_lat, tps, max_vram

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--common-user-dir', type=str, required=True)
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--ratios', type=float, nargs='+', default=[0.1, 0.2, 0.5, 0.8, 1.0])
    
    parser.add_argument('--output-dir', type=str, default='.', help='Base output directory for results')
    
    args = parser.parse_args()

    # Setup Logging with new path structure
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    
    save_dir = os.path.join(args.output_dir, "results", timestamp, "emr")
    
    os.makedirs(save_dir, exist_ok=True)
    log_file_path = os.path.join(save_dir, "eval_sweep.log")
    logger = Logger(log_file_path)

    logger.log(f"| Evaluation started. Saving logs to {log_file_path}")
    logger.log(f"| Args: {args}")

    device = torch.device(args.device)
    logger.log(f"| Loading model from {args.checkpoint_path} on {device}...")
    
    utils.import_user_module(argparse.Namespace(user_dir=args.common_user_dir))
    state = checkpoint_utils.load_checkpoint_to_cpu(args.checkpoint_path)
    task = tasks.setup_task(state["cfg"]["task"])
    task.load_dataset(args.split)
    dataset = task.datasets[args.split]
    models, _ = checkpoint_utils.load_model_ensemble([args.checkpoint_path], task=task)
    model = models[0].to(device).eval()

    for r in args.ratios:
        logger.log(f"\n[PROFILING COMPUTATION QUOTA: {int(r*100)}%]")
        # Pass logger
        w_stats, d_stats, lat, tps, vram = run_eval(model, dataset, task, r, device, logger)
        
        log_result_block(logger, "Window-level", w_stats, prefix=f"Ratio {r} | ")
        logger.log("=" * 85)
        logger.log(f"| Window Empty Match Rate (System Conservative): {w_stats['sub']['empty']/len(dataset):.4f}")
        logger.log("=" * 85)
        log_result_block(logger, "Dialogue-level", d_stats, prefix=f"Ratio {r} | ")
        logger.log("=" * 85)
        logger.log(f"| Efficiency: Latency: {lat:.2f}ms/sample, Throughput: {tps:.1f} tokens/sec, VRAM: {vram:.1f} MB")
        logger.log("=" * 85)
    
    logger.close()

if __name__ == '__main__':
    main()