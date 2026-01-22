import math
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

@dataclass
class DecoderOnlyLMLossConfig(FairseqDataclass):
    ctc_weight: float = field(default=0.0, metadata={"help": "Weight for aux CTC loss"})
    ctc_zero_infinity: bool = field(default=True, metadata={"help": "Zero infinity for CTC"})

@register_criterion("decoder_only_language_modeling_loss", dataclass=DecoderOnlyLMLossConfig)
class decoder_only_language_modeling_loss(FairseqCriterion):
    def __init__(self, task, ctc_weight=0.0, ctc_zero_infinity=True):
        super().__init__(task)
        self.ctc_weight = ctc_weight
        self.ctc_zero_infinity = ctc_zero_infinity
        
        try:
            import editdistance
            self._has_editdistance = True
        except ImportError:
            self._has_editdistance = False

    def _levenshtein(self, a: str, b: str) -> int:
        la, lb = len(a), len(b)
        if la == 0: return lb
        if lb == 0: return la
        dp = list(range(lb + 1))
        for i in range(1, la + 1):
            prev = dp[0]
            dp[0] = i
            ca = a[i - 1]
            for j in range(1, lb + 1):
                tmp = dp[j]
                cost = 0 if ca == b[j - 1] else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = tmp
        return dp[lb]

    def _cer_pair(self, hyp: str, ref: str) -> int:
        if self._has_editdistance:
            import editdistance
            return editdistance.eval(list(hyp), list(ref))
        return self._levenshtein(hyp, ref)

    def forward(self, model, sample, reduce=True):
        # 1. Main Forward (CE)
        loss, logits, llm_labels = model(**sample["net_input"])
        sample_size = len(sample["target"])

        # Token Accuracy
        n_correct, total = self.compute_accuracy(logits, llm_labels)

        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "ntokens": utils.item(total.data),
            "nsentences": sample_size,
            "sample_size": sample_size,
            "n_correct": utils.item(n_correct.data),
            "total": utils.item(total.data),
        }

        # 2. Aux CTC Loss
        if self.ctc_weight > 0 and hasattr(model, "_last_av_feat") and getattr(model, "ctc_head", None) is not None:
            av_feat = model._last_av_feat
            prefix_mask = model._last_prefix_mask
            
            if av_feat is not None:
                ctc_logits = model.ctc_head(av_feat)
                
                # [CRITICAL FIX] Convert to float32 for PyTorch CTC loss
                lprobs = F.log_softmax(ctc_logits, dim=-1).to(dtype=torch.float32).transpose(0, 1)

                if prefix_mask is not None:
                    input_lengths = prefix_mask.sum(dim=1).long()
                else:
                    input_lengths = torch.full((av_feat.size(0),), av_feat.size(1), device=av_feat.device, dtype=torch.long)

                if hasattr(model, "llm_to_ctc_map"):
                    mapping = model.llm_to_ctc_map
                    ctc_target_list = []
                    target_lengths_list = []
                    
                    for i in range(llm_labels.size(0)):
                        row = llm_labels[i]
                        row = row[row != -100]
                        if row.numel() > 0:
                            row_mapped = mapping[row]
                            row_mapped = row_mapped[row_mapped != -1]
                            if row_mapped.numel() > 0:
                                ctc_target_list.append(row_mapped)
                                target_lengths_list.append(len(row_mapped))
                            else:
                                ctc_target_list.append(row.new_empty(0))
                                target_lengths_list.append(0)
                        else:
                            ctc_target_list.append(row.new_empty(0))
                            target_lengths_list.append(0)
                    
                    if len(ctc_target_list) > 0:
                        ctc_targets = torch.cat(ctc_target_list)
                        target_lengths = torch.tensor(target_lengths_list, device=av_feat.device, dtype=torch.long)
                        
                        # Note: cudnn.flags disable is sometimes needed for older torch versions with weird ctc bugs
                        # but usually casting to float32 is enough. Keeping it safe.
                        with torch.backends.cudnn.flags(enabled=False):
                            aux_loss = F.ctc_loss(
                                lprobs, ctc_targets, input_lengths, target_lengths, 
                                blank=0, reduction='mean', zero_infinity=self.ctc_zero_infinity
                            )
                        
                        if not torch.isnan(aux_loss) and not torch.isinf(aux_loss):
                            loss += self.ctc_weight * aux_loss
                            logging_output["ctc_loss"] = utils.item(aux_loss.data)

        # 3. Calculate CER (Eval only)
        if not model.training:
            with torch.no_grad():
                shifted_logits = logits[:, :-1, :]
                shifted_labels = llm_labels[:, 1:]
                pred_ids = torch.argmax(shifted_logits, dim=-1)
                
                batch_hyps = []
                batch_refs = []
                for i in range(shifted_labels.size(0)):
                    mask = (shifted_labels[i] != -100)
                    if mask.any():
                        batch_hyps.append(pred_ids[i][mask])
                        batch_refs.append(shifted_labels[i][mask])
                    else:
                        batch_hyps.append(torch.tensor([], dtype=torch.long))
                        batch_refs.append(torch.tensor([], dtype=torch.long))
                
                if hasattr(model, "tokenizer"):
                    hypos = model.tokenizer.batch_decode(batch_hyps, skip_special_tokens=True)
                    refs = model.tokenizer.batch_decode(batch_refs, skip_special_tokens=True)
                    n_err, n_total = 0, 0
                    for h, r in zip(hypos, refs):
                        h_clean = h.strip().replace(" ", "")
                        r_clean = r.strip().replace(" ", "")
                        n_err += self._cer_pair(h_clean, r_clean)
                        n_total += len(r_clean)
                    logging_output["n_err"] = n_err
                    logging_output["n_total"] = n_total

        return loss, sample_size, logging_output

    def compute_accuracy(self, logits, labels):
        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:]
        predictions = torch.argmax(shifted_logits, dim=-1)
        mask = shifted_labels != -100
        correct_predictions = (predictions == shifted_labels) & mask
        n_correct = correct_predictions.sum().float()
        total = mask.sum().float()
        return n_correct, total

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        
        if sample_size > 0:
            metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
            metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["loss"].avg))
        else:
             metrics.log_scalar("loss", 0, 0, round=3)

        ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        ctc_count = sum(1 for log in logging_outputs if log.get("ctc_loss", 0) > 0)
        if ctc_count > 0:
             metrics.log_scalar("ctc_loss", ctc_loss_sum / len(logging_outputs), round=4)

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            n_correct = utils.item(sum(log.get("n_correct", 0) for log in logging_outputs))
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_scalar("total", total)
            metrics.log_derived("accuracy", lambda meters: round(meters["n_correct"].sum * 100.0 / meters["total"].sum, 3))

        n_err = sum(log.get("n_err", 0) for log in logging_outputs)
        n_total = sum(log.get("n_total", 0) for log in logging_outputs)
        metrics.log_scalar("_n_err", n_err)
        metrics.log_scalar("_n_total", n_total)
        if n_total > 0:
            metrics.log_derived("cer", lambda meters: round(meters["_n_err"].sum * 100.0 / meters["_n_total"].sum, 2))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False