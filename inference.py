# -*- coding: utf-8 -*-
import torch
import json
import os
import argparse
import sys
import numpy as np
import cv2
import librosa
import tempfile
import shutil
import re
from tqdm import tqdm
from fairseq import checkpoint_utils, tasks, utils
from omegaconf import open_dict
from transformers import WhisperFeatureExtractor

# Import Dictionary
try:
    from fairseq.data import Dictionary
except ImportError:
    from fairseq.data.dictionary import Dictionary

# ==============================================================================
#  FACE DETECTION IMPORTS (With Safety Fallback)
# ==============================================================================
try:
    from ibug.face_alignment import FANPredictor
    from ibug.face_detection import RetinaFacePredictor
    HAS_FACE_DETECTOR = True
    print("| [Info] Face detector loaded successfully.")
except ImportError:
    HAS_FACE_DETECTOR = False
    print("| [Warning] 'ibug' not found. Will fallback to Center Crop for non-96x96 videos.")


# ==============================================================================
#  INPUT CONFIGURATION
# ==============================================================================

# ------------------------------------------------------------------------------
# GROUP 1: Text-Only Batch (Direct EMR Prediction)
# Define your text dialogues here as a list of lists.
# ------------------------------------------------------------------------------
text_dialogues = [
    # Dialogue 1: Vomiting
    [
        "患者:没有呕吐,就是吐了点奶,",
        "医生:宝宝今天吐奶几次",
        "患者:一次",
        "医生:血丝多吗",
        "医生:呕吐物中有粘液吗?"
    ],
    # Dialogue 2: Heart/Vessel
    [
        "患者:心脏的血管堵塞有什么最新治疗方法吗",
        "医生:您好,要看在什么部位,一般可以下支架解决堵塞问题!CT未看到堵塞,有肌桥,若症状较重考虑搭桥手术!"
    ],
    # Dialogue 3: Stool Analysis
    [
        "医生:在看化验单",
        "医生:就这个大便结果来看,是病毒合并细菌感染,这两个药可以继续吃,孩子大便还是不见好吗?",
        "患者:是的",
        "患者:还要给他添点别的药吗?昨天好一点,今天又开始了",
        "医生:大便是什么样子?能拍个图片吗?"
    ],
    # Dialogue 4: Ultrasound
    [
        "医生:你好,有具体彩超单子吗?",
        "患者:有,刚满月拍的彩超。现在孩子三个月了",
        "医生:拍一个完整的",
        "医生:孩子现在有什么症状吗?"
    ]
]

# ------------------------------------------------------------------------------
# GROUP 2: Multimedia Batch (Requires AVSR Transcription)
# Define specific files with "modality": "audio", "video", or "audio_video"
    # "utterances": [
    #   "患者:体检检查出窦性心动过速",
    #   "医生:你好!当时做心电图时紧张吗?正常人可以出现窦性心动过速!",
    #   "患者:没有啊      以前查出过三尖瓣少量逆流",
    #   "医生:三尖瓣少量返流正常人也可以出现!平时心率多少?"
    # ],
# ------------------------------------------------------------------------------

multimedia_dialogues = [
    {
        "modality": "audio_video",
        "dialogue": [
            "患者:/../../EMR-LLM-CN/data/examples/audio16k/mie_dia_1_win_0.wav",
            "医生:/../../EMR-LLM-CN/data/examples/video96/mie_dia_1_win_1.mp4, /../../EMR-LLM-CN/data/examples/audio16k/mie_dia_1_win_1.wav",
            "患者:/../../EMR-LLM-CN/data/examples/audio16k/mie_dia_1_win_2.wav",
            "医生:/../../EMR-LLM-CN/data/examples/video96/mie_dia_1_win_3.mp4, /../../EMR-LLM-CN/data/examples/audio16k/mie_dia_1_win_3.wav",
        ]
    },
    {
        "modality": "audio_video",
        "dialogue": [
            "患者:/../../EMR-LLM-CN/data/examples/video96/mie_dia_6_win_2.mp4, /../../EMR-LLM-CN/data/examples/audio16k/mie_dia_6_win_2.wav",
            "医生:/../../EMR-LLM-CN/data/examples/video96/mie_dia_6_win_3.mp4, /../../EMR-LLM-CN/data/examples/audio16k/mie_dia_6_win_3.wav",
            "患者:/../../EMR-LLM-CN/data/examples/audio16k/mie_dia_6_win_4.wav",
            "医生:/../../EMR-LLM-CN/data/examples/video96/mie_dia_6_win_5.mp4, /../../EMR-LLM-CN/data/examples/audio16k/mie_dia_6_win_5.wav",
            "患者:/../../EMR-LLM-CN/data/examples/video96/mie_dia_6_win_6.mp4, /../../EMR-LLM-CN/data/examples/audio16k/mie_dia_6_win_6.wav",
        ]
    },

        {
        "modality": "audio_video",
        "dialogue": [
            "患者:/../../EMR-LLM-CN/data/examples/video96/mie_dia_11_win_4.mp4, /../../EMR-LLM-CN/data/examples/audio16k/mie_dia_11_win_4.wav",
            "医生:/../../EMR-LLM-CN/data/examples/video96/mie_dia_11_win_5.mp4, /../../EMR-LLM-CN/data/examples/audio16k/mie_dia_11_win_5.wav",
            "患者:/../../EMR-LLM-CN/data/examples/audio16k/mie_dia_11_win_6.wav",
            "医生:/../../EMR-LLM-CN/data/examples/video96/mie_dia_11_win_7.mp4, /../../EMR-LLM-CN/data/examples/audio16k/mie_dia_11_win_7.wav",
        ]
    }
]

raw_dialogue_inputs = []
for d in text_dialogues:
    raw_dialogue_inputs.append({"modality": "text", "dialogue": d})
raw_dialogue_inputs.extend(multimedia_dialogues)


# ==============================================================================
#  VIDEO PREPROCESSOR
# ==============================================================================
class VideoPreprocessor:
    def __init__(self, device="cuda:0", mean_face_path=None):
        self.device = device
        self.std_size = (256, 256)
        self.crop_height = 96
        self.crop_width = 96
        self.window_margin = 12
        self.start_idx = 48
        self.stop_idx = 68
        self.stable_points = (28, 33, 36, 39, 42, 45, 48, 54)
        
        self.detector_ready = False
        if HAS_FACE_DETECTOR:
            try:
                self.face_detector = RetinaFacePredictor(
                    device=device, threshold=0.8, model=RetinaFacePredictor.get_model("resnet50")
                )
                self.landmark_detector = FANPredictor(device=device, model=None)
                self.detector_ready = True
            except Exception as e:
                print(f"| [Warning] Failed to init detectors: {e}")

        self.mean_face_landmarks = None
        if mean_face_path and os.path.exists(mean_face_path):
            self.mean_face_landmarks = np.load(mean_face_path)

    def load_video(self, filename):
        frames = []
        cap = cv2.VideoCapture(filename)
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return np.array(frames)

    def process_video_frames(self, video_path):
        # 1. Fast Check: Resolution
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            if w == self.crop_width and h == self.crop_height:
                print(f"| [Info] Video {os.path.basename(video_path)} is already 96x96. Skipping detection.")
                return self.load_video(video_path)

        # Load all frames
        vid = self.load_video(video_path)
        if len(vid) == 0: return None

        print(f"| [Info] Processing {os.path.basename(video_path)} (Size != 96x96)...")

        # 2. Check dependencies
        if not self.detector_ready or self.mean_face_landmarks is None:
            return self._center_crop_fallback(vid)

        try:
            # 3. Detection
            landmarks = []
            for frame in vid:
                detected_faces = self.face_detector(frame, rgb=True)
                if len(detected_faces) == 0:
                    landmarks.append(None)
                    continue
                
                max_id = np.argmax([(b[2]-b[0]) + (b[3]-b[1]) for b in detected_faces])
                target_face = detected_faces[max_id]

                # [FIXED] Wrap in numpy array
                target_face_np = np.array([target_face]) 
                face_points, _ = self.landmark_detector(frame, target_face_np, rgb=True)
                landmarks.append(face_points[0])
            
            # Interpolate
            landmarks = self._landmarks_interpolate(landmarks)
            if landmarks is None: return self._center_crop_fallback(vid)

            # Align & Crop
            stable_reference = self._get_stable_reference(self.std_size)
            sequence = []
            
            for frame_idx, frame in enumerate(vid):
                margin = min(self.window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
                smoothed = np.mean([landmarks[x] for x in range(frame_idx - margin, frame_idx + margin + 1)], axis=0)
                smoothed += landmarks[frame_idx].mean(axis=0) - smoothed.mean(axis=0)

                transform = cv2.estimateAffinePartial2D(np.vstack([smoothed[x] for x in self.stable_points]), stable_reference, method=cv2.LMEDS)[0]
                trans_frame = cv2.warpAffine(
                    frame, transform, dsize=self.std_size,
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
                trans_lms = (np.matmul(smoothed, transform[:, :2].transpose()) + transform[:, 2].transpose())

                center_x, center_y = np.mean(trans_lms[self.start_idx : self.stop_idx], axis=0)
                y_min = int(round(np.clip(center_y - self.crop_height//2, 0, trans_frame.shape[0])))
                y_max = int(round(np.clip(center_y + self.crop_height//2, 0, trans_frame.shape[0])))
                x_min = int(round(np.clip(center_x - self.crop_width//2, 0, trans_frame.shape[1])))
                x_max = int(round(np.clip(center_x + self.crop_width//2, 0, trans_frame.shape[1])))
                
                patch = trans_frame[y_min:y_max, x_min:x_max]
                patch = cv2.resize(patch, (self.crop_width, self.crop_height))
                sequence.append(patch)

            return np.array(sequence)

        except Exception as e:
            print(f"| [Warning] Detection failed ({e}). Fallback to Center Crop.")
            return self._center_crop_fallback(vid)

    def _center_crop_fallback(self, frames):
        out = []
        for frame in frames:
            h, w, _ = frame.shape
            min_dim = min(h, w)
            sy, sx = (h - min_dim)//2, (w - min_dim)//2
            crop = frame[sy:sy+min_dim, sx:sx+min_dim]
            crop = cv2.resize(crop, (self.crop_width, self.crop_height))
            out.append(crop)
        return np.array(out)

    def _get_stable_reference(self, target_size):
        stable_reference = np.vstack([self.mean_face_landmarks[x] for x in self.stable_points])
        stable_reference[:, 0] -= (self.std_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (self.std_size[1] - target_size[1]) / 2.0
        return stable_reference

    def _linear_interpolate(self, landmarks, start_idx, stop_idx):
        start_landmarks = landmarks[start_idx]
        stop_landmarks = landmarks[stop_idx]
        delta = stop_landmarks - start_landmarks
        for idx in range(1, stop_idx-start_idx):
            landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
        return landmarks

    def _landmarks_interpolate(self, landmarks):
        valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
        if not valid_frames_idx: return None
        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx-1] > 1:
                landmarks = self._linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
        if valid_frames_idx:
            landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
        return landmarks

# ==============================================================================
#  DATA LOADING HELPERS (With RMS check)
# ==============================================================================

def load_audio_raw(path, target_sample_rate=16000, threshold=0.001):
    try:
        wav, _ = librosa.load(path, sr=target_sample_rate, mono=True)
    except Exception as e:
        print(f"| [Error] Librosa failed to load {path}: {e}")
        return torch.zeros(480000), False

    # RMS Check for silence
    rms = np.sqrt(np.mean(wav**2))
    
    if rms < threshold:
        # Return Silent=True
        return torch.zeros(480000), True
    
    wav = torch.from_numpy(wav).float()
    if wav.std() > 0:
        wav = (wav - wav.mean()) / (wav.std() + 1e-7)
    
    target_length = 480000
    current_length = wav.shape[0]
    if current_length < target_length:
        padding = torch.zeros(target_length - current_length)
        wav = torch.cat([wav, padding], dim=0)
    elif current_length > target_length:
        wav = wav[:target_length]
        
    return wav, False

# ==============================================================================
#  MODEL WRAPPERS (Revised for Multimodal Input)
# ==============================================================================

class AVSRPredictor:
    def __init__(self, checkpoint_path, user_dir, data_dir, device="cuda:0", overrides=None):
        self.device = device
        print(f"| [AVSR] Loading model from {checkpoint_path}...")
        
        utils.import_user_module(argparse.Namespace(user_dir=user_dir))
        
        # 1. Video Preprocessor
        mean_face_path = os.path.join(os.environ.get("ROOT", "/../../EMR-LLM-CN"), 
                                      "data/preparation/detectors/retinaface/20words_mean_face.npy")
        self.video_preprocessor = VideoPreprocessor(device=device, mean_face_path=mean_face_path)

        # 2. Whisper Extractor
        whisper_path = "openai/whisper-large-v2"
        if overrides and "model" in overrides and "whisper_path" in overrides["model"]:
            whisper_path = overrides["model"]["whisper_path"]
        try:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        except:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")

        # 3. Load Model
        try:
            state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path)
            cfg = state["cfg"]
            if overrides:
                if 'task' in overrides:
                    with open_dict(cfg.task):
                        for k, v in overrides['task'].items(): setattr(cfg.task, k, v)
                if 'model' in overrides:
                    with open_dict(cfg.model):
                        for k, v in overrides['model'].items(): setattr(cfg.model, k, v)
            
            task = tasks.setup_task(cfg.task)
            
            dummy_dict = Dictionary()
            dummy_dict.add_symbol("<s>")
            dummy_dict.add_symbol("</s>")
            TaskClass = task.__class__
            setattr(TaskClass, 'target_dictionary', property(lambda self: dummy_dict))
            setattr(TaskClass, 'dictionaries', property(lambda self: [dummy_dict]))

            model = task.build_model(cfg.model)
            model.load_state_dict(state['model'], strict=False)
            model.eval()
            self.model = model.to(self.device)
            self.task = task
            
            self.tokenizer = self.model.tokenizer
            messages = [{"role": "user", "content": "请将这段语音转写为中文："}]
            self.prompt_ids = self.tokenizer.apply_chat_template(
                messages, add_special_tokens=True, return_tensors="pt"
            )[0].to(self.device)
            
            print("| [AVSR] Model loaded successfully.")

        except Exception as e:
            print(f"| [AVSR Critical] Failed to load model: {e}")
            sys.exit(1)

    def transcribe(self, video_path=None, audio_path=None):
        """
        Multimodal transcription. Accepts matched video and audio paths.
        """
        if not video_path and not audio_path:
            return ""

        try:
            # Load Combined Sample
            sample = self._load_multimodal_sample(video_path, audio_path)
            
            if sample is None: 
                return ""

            net_input = sample["net_input"]
            
            for k, v in net_input["source"].items():
                if isinstance(v, torch.Tensor):
                    net_input["source"][k] = v.to(self.device)
            if "padding_mask" in net_input:
                net_input["padding_mask"] = net_input["padding_mask"].to(self.device)

            net_input["source"]["instruction"] = [self.prompt_ids.clone()]

            with torch.no_grad():
                output_ids = self.model.generate(
                    do_sample=False,          
                    num_beams=5,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    max_new_tokens=200,
                    **net_input
                )
                text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            return text.strip()
            
        except Exception as e:
            print(f"| [AVSR Error] Inference failed: {e}")
            return "[Transcription Failed]"

    def _load_multimodal_sample(self, video_path, audio_path):
        source_content = {}
        length = 0
        
        # --- 1. Process Audio ---
        if audio_path and os.path.exists(audio_path):
            raw_audio, is_silent = load_audio_raw(audio_path)
            # If silent and no video, skip
            if is_silent and not video_path:
                return None
            
            features = self.feature_extractor(
                raw_audio.numpy(), sampling_rate=16000, return_tensors="pt"
            ).input_features
            source_content["audio"] = features
            length = 3000
        else:
            # Dummy Audio
            dummy_raw = torch.zeros(480000)
            features = self.feature_extractor(
                dummy_raw.numpy(), sampling_rate=16000, return_tensors="pt"
            ).input_features
            source_content["audio"] = features

        # --- 2. Process Video ---
        if video_path and os.path.exists(video_path):
            video_np = self.video_preprocessor.process_video_frames(video_path)
            if video_np is None: 
                if audio_path:
                    source_content["video"] = None
                else:
                    return None
            else:
                video_tensor = torch.tensor(video_np, dtype=torch.float32)
                video_tensor = video_tensor / 255.0
                video_tensor = (video_tensor - 0.421) / 0.165
                video_gray = 0.299 * video_tensor[:, :, :, 0] + 0.587 * video_tensor[:, :, :, 1] + 0.114 * video_tensor[:, :, :, 2]
                source_content["video"] = video_gray.unsqueeze(0).unsqueeze(0)
                length = video_gray.size(0)
        else:
            source_content["video"] = None

        if source_content.get("audio") is None and source_content.get("video") is None:
            return None

        return {
            "net_input": {
                "source": source_content,
                "padding_mask": torch.zeros((1, length), dtype=torch.bool)
            },
            "id": torch.tensor([0]),
        }

class EMRPredictor:
    def __init__(self, checkpoint_path, user_dir, ontology_path, device="cuda:0"):
        self.device = device
        print(f"| [EMR] Loading model from {checkpoint_path}...")
        
        with open(ontology_path, 'r', encoding='utf-8') as f:
            self.ontology = json.load(f)
        self.id2label = {int(k): v for k, v in self.ontology['id2label'].items()}
        
        utils.import_user_module(argparse.Namespace(user_dir=user_dir))
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

# ==============================================================================
#  PIPELINE LOGIC
# ==============================================================================

def process_dialogue_lines(raw_lines, modality, avsr_predictor=None):
    final_dialogue = []
    if modality == "text": return raw_lines
    if not avsr_predictor: return []

    print(f"| [Pipeline] Modality is {modality.upper()}.")
    for line in raw_lines:
        if ":" in line:
            speaker, content = line.split(":", 1)
            speaker, content = speaker.strip(), content.strip()
        else:
            speaker, content = "Unknown", line.strip()

        # Split multiple files
        files_str = [f.strip() for f in content.split(',')]
        
        video_file = None
        audio_file = None
        
        # Identify pairs
        for f in files_str:
            ext = os.path.splitext(f)[1].lower()
            if ext in ['.mp4', '.avi', '.mov']:
                video_file = f
            elif ext in ['.wav', '.mp3', '.flac']:
                audio_file = f
        
        text_result = ""
        
        # Combined Inference Logic
        if video_file or audio_file:
            log_str = "|  > Transcribing pair: "
            if video_file: log_str += f"[V: {os.path.basename(video_file)}] "
            if audio_file: log_str += f"[A: {os.path.basename(audio_file)}]"
            print(log_str)
            
            # Pass BOTH to the model
            text_result = avsr_predictor.transcribe(video_path=video_file, audio_path=audio_file)
        else:
            # Pass through existing text
            text_result = content

        if text_result:
            final_dialogue.append(f"{speaker}:{text_result}")
            print(f"|  > Result: {speaker}:{text_result}")
        else:
            print(f"|  > Result: {speaker}: [Silent/Empty]")

    return final_dialogue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emr-checkpoint', type=str, required=True)
    parser.add_argument('--emr-user-dir', type=str, required=True)
    parser.add_argument('--ontology-path', type=str, required=True)
    
    parser.add_argument('--avsr-checkpoint', type=str, default=None)
    parser.add_argument('--avsr-user-dir', type=str, default=None)
    parser.add_argument('--avsr-data-dir', type=str, default=None)
    parser.add_argument('--avsr-ctc-vocab', type=str, default=None)
    parser.add_argument('--avsr-w2v-path', type=str, default=None)
    parser.add_argument('--avsr-whisper-path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    avsr_model = None
    if any(item['modality'] != 'text' for item in raw_dialogue_inputs):
        if not args.avsr_checkpoint:
            print("| [Error] Missing AVSR checkpoint.")
            sys.exit(1)
        avsr_overrides = {
            "task": {"data": args.avsr_data_dir, "label_dir": args.avsr_data_dir},
            "model": {}
        }
        if args.avsr_ctc_vocab: avsr_overrides["model"]["ctc_vocab_path"] = args.avsr_ctc_vocab
        if args.avsr_w2v_path: avsr_overrides["model"]["w2v_path"] = args.avsr_w2v_path
        if args.avsr_whisper_path: avsr_overrides["model"]["whisper_path"] = args.avsr_whisper_path
        
        avsr_model = AVSRPredictor(args.avsr_checkpoint, args.avsr_user_dir, args.avsr_data_dir, args.device, avsr_overrides)

    emr_model = EMRPredictor(args.emr_checkpoint, args.emr_user_dir, args.ontology_path, args.device)

    print(f"\n| [Batch Start] Processing {len(raw_dialogue_inputs)} dialogues...")
    for i, item in enumerate(raw_dialogue_inputs):
        modality = item.get("modality", "text")
        lines = item.get("dialogue", [])
        print(f"\n# Processing Dialogue {i+1}/{len(raw_dialogue_inputs)} | Mode: {modality.upper()} #")

        final_text = process_dialogue_lines(lines, modality, avsr_model)

        print("\n| [Context] EMR Input:")
        for line in final_text: print(f"|  {line}")
        
        if not final_text:
            print("| [Warning] No valid text extracted. Skipping.")
            continue

        probs = emr_model.predict(final_text)
        
        topk_vals, topk_inds = torch.topk(probs, k=10)
        print("\n[Top-10 Probabilities]")
        for rank, (score, idx) in enumerate(zip(topk_vals, topk_inds)):
            label_name = emr_model.id2label.get(idx.item(), f"ID_{idx.item()}")
            print(f"Rank {rank+1:2d} | Prob: {score:.4f} | {label_name}")

        final_labels = [emr_model.id2label.get(idx.item(), str(idx.item())) for idx in torch.where(probs > 0.5)[0]]
        print(f"\n[Summary: Final Labels] {final_labels if final_labels else 'None detected'}")

if __name__ == "__main__":
    main()