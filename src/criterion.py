import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

# Integrated CB-Reweight & Full Metric Logging

@dataclass
class MedicalHmrCriterionConfig(FairseqDataclass):
    # Core weights for the three-part loss
    macro_weight: float = field(default=0.3, metadata={"help": "Weight for Topic + Retrieval discovery loss"})
    asl_weight: float = field(default=0.5, metadata={"help": "Weight for ASL subtopic verification"})
    topo_weight: float = field(default=0.2, metadata={"help": "Weight for topological logic"})
    
    # Topic balancing threshold to suppress dominant classes
    topic_tau: float = field(default=25.0, metadata={"help": "Positive weight capping for topics"})
    
    # Topology and discrimination parameters
    intra_sim_alpha: float = field(default=0.2)
    gamma_neg: float = field(default=2.0)
    gamma_pos: float = field(default=1.0)
    asl_clip: float = field(default=0.05)
    sub_threshold: float = field(default=0.5)

@register_criterion("medical_hmr_criterion", dataclass=MedicalHmrCriterionConfig)
class MedicalHmrCriterion(FairseqCriterion):
    def __init__(self, task, cfg: MedicalHmrCriterionConfig):
        super().__init__(task)
        self.cfg = cfg
        # Retrieve pre-calculated topic weights from task
        # Note: task.py must implement topic_pos_weights initialization
        self.topic_pos_weights = getattr(task, "topic_pos_weights", torch.ones(4))

    def forward(self, model, sample, reduce=True):
        start_t = time.time()
        
        # 1. Model Inference (Integrated HLWMP)
        net_output = model(**sample["net_input"], target_subtopic=sample["target_subtopic"])
        
        l_topic = net_output["logits_topic"]      # [B, 4]
        l_ret = net_output["logits_retrieval"]    # [B, 206]
        l_sub = net_output["logits_subtopic"]     # [B, 206]
        sel_indices = net_output["selected_indices"]
        
        t_topic = sample["target_topic"].to(l_topic.dtype)
        t_sub = sample["target_subtopic"].to(l_sub.dtype)
        batch_size = t_topic.size(0)

        # ---------------------------------------------------------------------
        # LOSS 1: Macro Loss (with CB-Reweight Topic Loss)
        # ---------------------------------------------------------------------
        # Apply class-balanced reweighting to handle imbalance
        w_topic = self.topic_pos_weights.to(l_topic.device)
        loss_topic = F.binary_cross_entropy_with_logits(
            l_topic, t_topic, 
            pos_weight=w_topic, 
            reduction="sum" if reduce else "none"
        )
        
        loss_ret = F.binary_cross_entropy_with_logits(l_ret, t_sub, reduction="sum")
        total_macro_loss = (loss_topic + loss_ret) / 2.0

        # ---------------------------------------------------------------------
        # LOSS 2: ASL Verify Loss
        # ---------------------------------------------------------------------
        probs_s = torch.sigmoid(l_sub)
        p_neg = (probs_s + self.cfg.asl_clip).clamp(max=1.0)
        eps = 1e-8
        l_pos = t_sub * torch.pow(1 - probs_s, self.cfg.gamma_pos) * torch.log(probs_s + eps)
        l_neg = (1 - t_sub) * torch.pow(p_neg, self.cfg.gamma_neg) * torch.log(1 - p_neg + eps)
        total_asl_loss = -(l_pos + l_neg).sum()

        # ---------------------------------------------------------------------
        # LOSS 3: Topo Loss
        # ---------------------------------------------------------------------
        curr_m = model
        while hasattr(curr_m, 'module'): curr_m = curr_m.module
        queries, h_mask = curr_m.label_queries, curr_m.hierarchical_mask 
        q_norm = F.normalize(queries, p=2, dim=1)
        sim_matrix = torch.matmul(q_norm, q_norm.t()) 
        with torch.no_grad():
            intra_mask = torch.matmul(h_mask.t(), h_mask) 
            inter_mask = 1.0 - intra_mask
            identity = torch.eye(206, device=sim_matrix.device)
        loss_inter = torch.mean((sim_matrix * inter_mask) ** 2)
        loss_intra = torch.mean((sim_matrix * intra_mask - self.cfg.intra_sim_alpha * intra_mask - (1-self.cfg.intra_sim_alpha)*identity) ** 2)
        total_topo_loss = loss_inter * 2.0 + loss_intra * 0.5

        # Final loss aggregation
        total_loss = (self.cfg.macro_weight * total_macro_loss + 
                      self.cfg.asl_weight * total_asl_loss + 
                      self.cfg.topo_weight * total_topo_loss * batch_size)

        # ---------------------------------------------------------------------
        # 4. Metric Collection (Prepare PRF stats)
        # ---------------------------------------------------------------------
        logging_output = {
            "loss": total_loss.data,
            "loss_macro": total_macro_loss.data,
            "loss_asl": total_asl_loss.data,
            "loss_topo": total_topo_loss.data,
            "sample_size": batch_size,
            "nsentences": batch_size,
            "ntokens": sample["ntokens"],
            "latency": time.time() - start_t,
        }

        with torch.no_grad():
            # Calculate Stage II retrieval coverage
            total_gt = 0
            captured_gt = 0
            for b in range(batch_size):
                gt_set = torch.where(t_sub[b] > 0.5)[0]
                total_gt += len(gt_set)
                if len(gt_set) > 0:
                    captured_gt += torch.isin(gt_set, sel_indices[b]).sum().item()
            logging_output["ret_total_gt"] = total_gt
            logging_output["ret_captured_gt"] = captured_gt

            # Calculate basic PRF components
            for p, lg, tg in [("topic", l_topic, t_topic), ("sub", l_sub, t_sub)]:
                preds = (torch.sigmoid(lg.float()) > self.cfg.sub_threshold).long()
                target_long = tg.long()
                logging_output[f"{p}_tp"] = (preds & target_long).sum().item()
                logging_output[f"{p}_fp"] = (preds & ~target_long).sum().item()
                logging_output[f"{p}_fn"] = (~preds & target_long).sum().item()

        return total_loss, batch_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate and log P, R, F1 (Topic, Sub, Overall)."""
        ss = sum(log.get("sample_size", 0) for log in logging_outputs)
        if ss == 0: return

        # 1. Log Losses
        metrics.log_scalar("loss", sum(log.get("loss", 0) for log in logging_outputs) / ss, ss, round=3)
        metrics.log_scalar("l_macro", sum(log.get("loss_macro", 0) for log in logging_outputs) / ss, ss, round=3)
        metrics.log_scalar("l_topo", sum(log.get("loss_topo", 0) for log in logging_outputs) / ss, ss, round=5)

        # 2. Log Retrieval Quality
        tot_gt = sum(log.get("ret_total_gt", 0) for log in logging_outputs)
        cap_gt = sum(log.get("ret_captured_gt", 0) for log in logging_outputs)
        metrics.log_scalar("ret_coverage", cap_gt / (tot_gt + 1e-8), ss, round=4)

        # 3. PRF Metrics Dashboard (Topic, Sub, Overall)
        def log_prf(prefix):
            tp = sum(log.get(f"{prefix}_tp", 0) for log in logging_outputs)
            fp = sum(log.get(f"{prefix}_fp", 0) for log in logging_outputs)
            fn = sum(log.get(f"{prefix}_fn", 0) for log in logging_outputs)
            p = tp / (tp + fp + 1e-8)
            r = tp / (tp + fn + 1e-8)
            f = 2 * p * r / (p + r + 1e-8)
            # Log to Tensorboard and Console
            metrics.log_scalar(f"{prefix}_p", p, ss, round=4)
            metrics.log_scalar(f"{prefix}_r", r, ss, round=4)
            metrics.log_scalar(f"{prefix}_f1", f, ss, round=4)
            return p, r, f

        tp_p, tp_r, tp_f = log_prf("topic") # Topic P, R, F1
        sb_p, sb_r, sb_f = log_prf("sub")   # Subtopic P, R, F1

        # 4. Overall Metrics (Macro-Average)
        metrics.log_scalar("overall_p", (tp_p + sb_p) / 2, ss, round=4)
        metrics.log_scalar("overall_r", (tp_r + sb_r) / 2, ss, round=4)
        metrics.log_scalar("overall_f1", (tp_f + sb_f) / 2, ss, round=4)

        # 5. Performance Dashboard
        lats = [log.get("latency", 0) for log in logging_outputs]
        metrics.log_scalar("lat_ms", (sum(lats) / len(lats)) * 1000, ss, round=1)