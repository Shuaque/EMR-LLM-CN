# dialogue_sample = [
#     "医生：最近咳嗽厉害吗？痰多不多？",
#     "患者：咳嗽挺明显的，尤其是晚上，痰是白色的。",
#     "医生：有没有发烧或者胸痛？",
#     "患者：没有发烧，但是深呼吸的时候肋骨这边有点疼。",
#     "医生：行，那先去拍个CT检查一下，再开点止咳药。"
# ]

# dialogue_sample = [
#     "医生：你平时抽不抽",
#     "患者：我还好，也就偶尔抽抽烟",
#     "医生：有没有发烧或者胸痛？",
#     "患者：没有发烧，但是深呼吸的时候肋骨这边有点疼。",
#     "医生：你平时要多注意身体，保护肺部，要不然有癌症风险。"
# ]
# # =================================================

import os
import torch
import argparse
import logging
import json
from fairseq import checkpoint_utils, utils

# Optimized for C3-MRAF | Symptom-Dominance Debugging | Logic Alignment [cite: 2025-12-28]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("C3_MRAF_Inference")

# ================= Configuration =================
# 🌟 建议更新为您的 3B 最优 Checkpoint
CHECKPOINT_PATH = "/workspace/shuaque/Classification_Semantic_att_LLM/exp/202512/run/28185426_A800_Optimized_Exp2_1_5B_3loss/checkpoints/checkpoint_best.pt"
USER_DIR = "/workspace/shuaque/Classification_Semantic_att_LLM/src_for_optim_subtopic"
ONTOLOGY_PATH = "/workspace/shuaque/Classification_Semantic_att_LLM/data/ontology.json"
dialogue_sample = [
    "医生：最近咳嗽厉害吗？痰多不多？",
    "患者：咳嗽挺明显的，尤其是晚上，痰是白色的。",
    "医生：有没有发烧或者胸痛？",
    "患者：没有发烧，但是深呼吸的时候肋骨这边有点疼。",
    "医生：行，那先去拍个CT检查一下，再开点止咳药。"
]
# dialogue_sample = [
#     "医生：你平时抽不抽",
#     "患者：我还好，也就偶尔抽抽烟",
#     "医生：有没有发烧或者胸痛？",
#     "患者：没有发烧，但是深呼吸的时候肋骨这边有点疼。",
#     "医生：你平时要多注意身体，保护肺部，要不然有癌症风险。"
# ]
# 包含明显“检查(Test)”指令的样本
# dialogue_sample = [
#     "医生：最近咳嗽厉害吗？痰多不多？",
#     "患者：咳嗽挺明显的，尤其是晚上，痰是白色的。",
#     "医生：有没有发烧或者胸痛？",
#     "患者：没有发烧，但是深呼吸的时候肋骨这边有点疼。",
#     "医生：行，那先去拍个拍个CT检查一下，再开点止咳药。"
# ]

# 🌟 设置计算配额：0.1-0.2 是观察 E-IAQ 效率的最佳区间
TEST_RATIO = 0.2
# =================================================

def build_prompt(utterances):
    context = "\n".join([u.strip() for u in utterances if u.strip()])
    return (
        "<|im_start|>system\n"
        "你是一个医疗专家助手。请分析对话并提取：1.症状, 2.检查, 3.手术, 4.一般信息。<|im_end|>\n"
        "<|im_start|>user\n"
        f"对话内容：\n{context}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def main():
    if USER_DIR:
        utils.import_user_module(argparse.Namespace(user_dir=USER_DIR))
    
    logger.info(f"Loading C3-MRAF Model from {CHECKPOINT_PATH}...")
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [CHECKPOINT_PATH],
        arg_overrides={'distributed_training': {'distributed_world_size': 1}}
    )
    model = models[0].cuda().eval().to(dtype=torch.bfloat16)
    
    # 1. 准备映射与层级掩码
    with open(ONTOLOGY_PATH, "r", encoding="utf-8") as f:
        ontology = json.load(f)
    id2label = {int(v): k for k, v in ontology["label2id"].items()}
    topic_names = ["Symptom", "Test", "Surgery", "GeneralInfo"]
    
    # 获取模型的层级掩码用于诊断屏蔽逻辑
    m_inner = model.module if hasattr(model, 'module') else model
    h_mask = m_inner.hierarchical_mask.float().cpu() # [4, 206]

    # 2. 输入处理
    full_prompt = build_prompt(dialogue_sample)
    tokenized = task.tokenizer(full_prompt, max_length=512, truncation=True, return_tensors="pt")
    input_ids, attention_mask = tokenized["input_ids"].cuda(), tokenized["attention_mask"].cuda()

    # 3. 推理
    logger.info(f"Inference Running (Ratio: {TEST_RATIO})...")
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        # 传递 ratio 触发 E-IAQ 动态名额分配 [cite: 2025-12-28]
        output = model(input_ids=input_ids, attention_mask=attention_mask, ratio=TEST_RATIO)
    
    probs_topic = torch.sigmoid(output["logits_topic"].squeeze(0).float())
    # 注意：logits_subtopic 已经过 model.py 中的 Log-masking Shield 处理 [cite: 2025-12-28]
    probs_sub = torch.sigmoid(output["logits_subtopic"].squeeze(0).float())

    # 4. 格式化输出
    print("\n" + "="*75)
    print(f" 🏥 Medical Entity Recognition - C3-MRAF (Diagnostic Mode)")
    print("="*75)
    
    # --- 主类话题结果 ---
    print(f"{'Prob':<8} | {'Topic Category':<15} | {'Status'}")
    print("-" * 45)
    for i, p in enumerate(probs_topic.tolist()):
        status = "✅ ACTIVE" if p > 0.5 else "❌ MASKED"
        print(f"{p:.4f}   | {topic_names[i]:<15} | {status}")

    # --- 子类实体结果 ---
    print("\n" + f"{'Prob':<8} | {'Sub-category (Entities)':<30} | {'Logic Gate'}")
    print("-" * 75)
    
    # 提取 Top-10 概率
    top_v, top_i = torch.topk(probs_sub, k=10)
    for p, idx in zip(top_v.tolist(), top_i.tolist()):
        label = id2label.get(idx, "Unknown")
        
        # 诊断：寻找该子类所属的主类概率
        parent_topic_idx = torch.where(h_mask[:, idx] > 0.5)[0][0].item()
        parent_prob = probs_topic[parent_topic_idx].item()
        
        gate_info = "🟢 Pass" if parent_prob > 0.5 else f"🔴 Blocked by {topic_names[parent_topic_idx]}({parent_prob:.2f})"
        
        marker = "✅" if p > 0.5 else "  "
        print(f"{p:.4f}   | {marker} {label:<30} | {gate_info}")

    # --- 效率指标 ---
    selected_num = len(output["selected_indices"][0])
    print("\n" + "-" * 75)
    print(f"Efficiency: E-IAQ verified {selected_num}/206 entities ({selected_num/206*100:.1f}% budget used)")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()