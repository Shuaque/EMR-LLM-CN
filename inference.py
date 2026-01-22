import torch
import json
import os
import argparse
import sys
import re
import ast
import numpy as np
from fairseq import checkpoint_utils, tasks, utils

#  CASE SELECTION (Uncomment the case you want to test)

# --- CASE 1: Text Only ---
"""
modalities = ["text_only"]

raw_dialogue_inputs = [

    [
        "患者:没有呕吐,就是吐了点奶,",
        "医生:宝宝今天吐奶几次",
        "患者:一次",
        "医生:血丝多吗",
        "医生:呕吐物中有粘液吗?"
        ],

    [
        "患者:心脏的血管堵塞有什么最新治疗方法吗",
        "医生:您好,要看在什么部位,一般可以下支架解决堵塞问题!CT未看到堵塞,有肌桥,若症状较重考虑搭桥手术!"
        ],

    [
        "医生:在看化验单",
        "医生:就这个大便结果来看,是病毒合并细菌感染,这两个药可以继续吃,孩子大便还是不见好吗?",
        "患者:是的",
        "患者:还要给他添点别的药吗?昨天好一点,今天又开始了",
        "医生:大便是什么样子?能拍个图片吗?"
        ],

    [
        "医生:你好,有具体彩超单子吗?",
        "患者:有,刚满月拍的彩超。现在孩子三个月了",
        "医生:拍一个完整的",
        "患者:",
        "医生:孩子现在有什么症状吗?"
        ],
]
"""

# --- CASE 2: Audio/Video Input ---
"""
    ground truth for AVSR:

    mie_dia_1_win_0	pat	体检检查出窦性心动过速
    mie_dia_1_win_1	doc	你好当时做心电图时紧张吗正常人可以出现窦性心动过速
    mie_dia_1_win_2	pat	没有啊以前查出过三尖瓣少量逆流
    mie_dia_1_win_3	doc	三尖瓣少量返流正常人也可以出现平时心率多少

    ground truth for labels:

    "labels": [
      "检查:体检",
      "检查:心电图",
      "症状:心律不齐"
    ]
"""

modalities = ["video", "audio"]

raw_dialogue_input = [
    "患者:/workspace/shuaque/EMR-LLM-CN/data/examples/audio16k/mie_dia_1_win_0.wav",
    "医生:/workspace/shuaque/EMR-LLM-CN/data/examples/video96/mie_dia_1_win_1.mp4, /workspace/shuaque/EMR-LLM-CN/data/examples/audio16k/mie_dia_1_win_1.wav",
    "患者:/workspace/shuaque/EMR-LLM-CN/data/examples/audio16k/mie_dia_1_win_2.wav",
    "医生:/workspace/shuaque/EMR-LLM-CN/data/examples/video96/mie_dia_1_win_3.mp4, /workspace/shuaque/EMR-LLM-CN/data/examples/audio16k/mie_dia_1_win_3.wav",
]

# ==============================================================================

class AVSRPredictor:
    """Wrapper to load the AVSR model (src_avsr) and run transcription."""
    
    def __init__(self, checkpoint_path, user_dir, data_dir, device="cuda:0", overrides=None):
        self.device = device
        print(f"| [AVSR] Loading model from {checkpoint_path}...")
        
        # 1. Import src_avsr module
        utils.import_user_module(argparse.Namespace(user_dir=user_dir))
        
        # 2. Load model and task configuration
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path], 
            arg_overrides=overrides
        )
        self.model = models[0].to(self.device).eval()
        self.task = task
        self.cfg = cfg
        
        # 3. Initialize Generator
        self.generator = self.task.build_generator(models, cfg.generation)
        print("| [AVSR] Model loaded successfully.")

    def transcribe(self, file_path):
        """Run inference on a single audio/video file."""
        if not os.path.exists(file_path):
            print(f"| [AVSR Error] File not found: {file_path}")
            return "[Error: File Missing]"

        try:
            # 1. Load data using Task preprocessing logic
            # This relies on the specific implementation of process_raw_input in your src_avsr task
            sample = self._load_sample_from_file(file_path)
            sample = utils.move_to_cuda(sample, device=self.device)
            
            # 2. Generate
            with torch.no_grad():
                hypos = self.task.inference_step(self.generator, [self.model], sample)
            
            # 3. Decode tokens to text
            top_hypo = hypos[0][0] # Top beam result
            tokens = top_hypo['tokens']
            text = self.task.tokenizer.decode(tokens)
            return text
            
        except Exception as e:
            print(f"| [AVSR Error] Inference failed for {file_path}: {e}")
            return "[Transcription Failed]"

    def _load_sample_from_file(self, file_path):
        """
        Convert file path to model input tensor.
        NOTE: The AVSR task must implement `process_raw_input`.
        """
        if hasattr(self.task, 'process_raw_input'):
             return self.task.process_raw_input(file_path)
        
        raise NotImplementedError("src_avsr Task must implement `process_raw_input` or a similar interface.")

class EMRPredictor:
    """Wrapper to load the EMR model (src) and run entity extraction."""

    def __init__(self, checkpoint_path, user_dir, ontology_path, device="cuda:0"):
        self.device = device
        print(f"| [EMR] Loading model from {checkpoint_path}...")
        
        # Load Ontology
        with open(ontology_path, 'r', encoding='utf-8') as f:
            self.ontology = json.load(f)
        self.id2label = {int(k): v for k, v in self.ontology['id2label'].items()}
        
        # Import src module
        utils.import_user_module(argparse.Namespace(user_dir=user_dir))
        
        # Load Model
        state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path)
        cfg = state["cfg"]
        self.task = tasks.setup_task(cfg.task)
        models, _ = checkpoint_utils.load_model_ensemble([checkpoint_path], task=self.task)
        self.model = models[0].to(self.device).eval()
        self.tokenizer = self.task.tokenizer
        print("| [EMR] Model loaded successfully.")

    def build_prompt(self, utterances):
        context = "\n".join([u.strip() for u in utterances if u.strip()])
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

    def predict(self, dialogue_text_list):
        prompt = self.build_prompt(dialogue_text_list)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, ratio=1.0)
            
        logits = output['logits_subtopic'][0]
        probs = torch.sigmoid(logits).cpu().float()
        return probs

def parse_and_process_dialogue(raw_input, avsr_predictor=None):
    """
    Parses input list. If file path, call AVSR model; if text, keep as is.
    """
    final_dialogue = []
    
    print("\n| [Pipeline] Processing Dialogue Inputs...")
    
    for line in raw_input:
        # 1. Separate Speaker and Content
        if ":" in line:
            speaker, content = line.split(":", 1)
            speaker = speaker.strip()
            content = content.strip()
        else:
            speaker = "Unknown"
            content = line.strip()

        # 2. Check if content is file path or text
        is_file = False
        if avsr_predictor:
            # Handle multiple files "a.wav, b.mp4"
            files = [f.strip() for f in content.split(',')]
            transcribed_segments = []
            
            for f in files:
                # Basic check for media extensions
                if any(f.lower().endswith(ext) for ext in ['.wav', '.mp4', '.avi', '.mp3', '.flac']):
                    is_file = True
                    print(f"|  > Transcribing file: {f}")
                    text = avsr_predictor.transcribe(f)
                    transcribed_segments.append(text)
                else:
                    # Mixed text case
                    transcribed_segments.append(f)
            
            if is_file:
                full_text = "，".join(transcribed_segments)
                final_dialogue.append(f"{speaker}:{full_text}")
                print(f"|  > Result: {speaker}:{full_text}")
                continue

        # 3. Text pass-through
        final_dialogue.append(line)
        if avsr_predictor:
            print(f"|  > Text Passthrough: {line}")

    return final_dialogue

def main():
    parser = argparse.ArgumentParser()
    # EMR Args
    parser.add_argument('--emr-checkpoint', type=str, required=True)
    parser.add_argument('--emr-user-dir', type=str, required=True)
    parser.add_argument('--ontology-path', type=str, required=True)
    
    # AVSR Args
    parser.add_argument('--avsr-checkpoint', type=str, default=None)
    parser.add_argument('--avsr-user-dir', type=str, default=None)
    parser.add_argument('--avsr-data-dir', type=str, default=None) # Dict/Vocab path for AVSR
    
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()


    # 1. Initialize AVSR Model
    avsr_model = None
    if "text_only" not in modalities:
        if not args.avsr_checkpoint:
            print("| [Error] Modalities include audio/video but --avsr-checkpoint is missing.")
            sys.exit(1)
            
        # AVSR specific overrides
        avsr_overrides = {
            "data": args.avsr_data_dir,
            "label_dir": args.avsr_data_dir,
            "task": {"data": args.avsr_data_dir} 
        }
        
        try:
            avsr_model = AVSRPredictor(
                checkpoint_path=args.avsr_checkpoint,
                user_dir=args.avsr_user_dir,
                data_dir=args.avsr_data_dir,
                device=args.device,
                overrides=avsr_overrides
            )
        except Exception as e:
            print(f"| [Critical Error] Failed to load AVSR model: {e}")
            print("| Ensure src_avsr is in PYTHONPATH and implements Fairseq interfaces.")
            sys.exit(1)

    # 2. Initialize EMR Model
    emr_model = EMRPredictor(
        checkpoint_path=args.emr_checkpoint,
        user_dir=args.emr_user_dir,
        ontology_path=args.ontology_path,
        device=args.device
    )

    # 3. Process Input (Transcribe -> Text)
    for input in raw_dialogue_inputs:
        final_text_dialogue = parse_and_process_dialogue(input, avsr_model)

        print("\n" + "="*60)
        print("| Final Context for EMR Model:")
        print("="*60)
        for line in final_text_dialogue:
            print(line)
        print("="*60 + "\n")


        # 4. Run EMR Prediction
        print("| [EMR] Running entity extraction...")
        probs = emr_model.predict(final_text_dialogue)
        
        # 5. Output Results
        # Top-10
        topk_vals, topk_inds = torch.topk(probs, k=10)
        print("\n[Top-10 Probabilities]")
        for rank, (score, idx) in enumerate(zip(topk_vals, topk_inds)):
            idx = idx.item()
            label_name = emr_model.id2label.get(idx, f"Unknown_ID_{idx}")
            print(f"Rank {rank+1:2d} | Prob: {score:.4f} | {label_name}")

        # Final Summary
        threshold = 0.5
        final_indices = torch.where(probs > threshold)[0]
        final_labels = [emr_model.id2label.get(idx.item(), str(idx.item())) for idx in final_indices]
        
        print("\n" + "-"*60)
        print("[Summary: Final Predicted Labels (Threshold > 0.5)]")
        if not final_labels:
            print("No labels detected.")
        else:
            for lbl in final_labels:
                print(f" - {lbl}")
        print("="*60 + "\n")

if __name__ == "__main__":
    main()