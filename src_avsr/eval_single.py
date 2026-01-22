# -*- coding: utf-8 -*-
import os
import sys
import logging
import ast
import torch
import hydra
import editdistance
import numpy as np
import subprocess
import tempfile
import cv2
import pickle
import math
import shutil
from datetime import datetime
from tqdm import tqdm
from omegaconf import DictConfig, open_dict

# Fairseq imports
from fairseq import checkpoint_utils, tasks, utils

# Face Detection imports
try:
    from ibug.face_alignment import FANPredictor
    from ibug.face_detection import RetinaFacePredictor
    HAS_FACE_DETECTOR = True
except ImportError:
    HAS_FACE_DETECTOR = False

# Set up logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("fairseq_cli.eval")

# ==========================================
# Video Preprocessing Helpers
# ==========================================

class VideoPreprocessor:
    def __init__(self, device="cuda:0", mean_face_path=None):
        self.device = device
        self.std_size = (256, 256)
        self.crop_height = 96
        self.crop_width = 96
        self.window_margin = 12
        self.start_idx = 48
        self.stop_idx = 68
        # Stable points from the provided snippet
        self.stable_points = (28, 33, 36, 39, 42, 45, 48, 54)
        
        if HAS_FACE_DETECTOR:
            self.face_detector = RetinaFacePredictor(
                device=device,
                threshold=0.8,
                model=RetinaFacePredictor.get_model("resnet50")
            )
            self.landmark_detector = FANPredictor(device=device, model=None)
        else:
            logger.warning("ibug.face_detection not found. Auto-cropping will be disabled.")

        # Load mean face
        if mean_face_path and os.path.exists(mean_face_path):
            self.mean_face_landmarks = np.load(mean_face_path)
        else:
            logger.warning(f"Mean face not found at {mean_face_path}. Alignment might fail.")
            self.mean_face_landmarks = None

    def load_video(self, filename):
        frames = []
        cap = cv2.VideoCapture(filename)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return np.array(frames)

    def is_already_cropped(self, video_path):
        """Check if video resolution is already 96x96"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return False
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        # Allow slight tolerance or exact match
        return (w == self.crop_width and h == self.crop_height)

    def detect_landmarks(self, video_frames):
        landmarks = []
        for frame in video_frames:
            detected_faces = self.face_detector(frame, rgb=True)
            if len(detected_faces) == 0:
                landmarks.append(None)
            else:
                # Pick the largest face
                max_id = np.argmax([(b[2]-b[0]) + (b[3]-b[1]) for b in detected_faces])
                # Detect landmarks
                face_points, _ = self.landmark_detector(frame, [detected_faces[max_id]], rgb=True)
                landmarks.append(face_points[0])
        return landmarks

    def linear_interpolate(self, landmarks, start_idx, stop_idx):
        start_landmarks = landmarks[start_idx]
        stop_landmarks = landmarks[stop_idx]
        delta = stop_landmarks - start_landmarks
        for idx in range(1, stop_idx-start_idx):
            landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
        return landmarks

    def landmarks_interpolate(self, landmarks):
        valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
        if not valid_frames_idx: return None
        
        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx-1] > 1:
                landmarks = self.linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
        
        # Padding
        if valid_frames_idx:
            landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
        return landmarks

    def get_stable_reference(self, target_size):
        stable_reference = np.vstack([self.mean_face_landmarks[x] for x in self.stable_points])
        stable_reference[:, 0] -= (self.std_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (self.std_size[1] - target_size[1]) / 2.0
        return stable_reference

    def process_video(self, video_path, output_path):
        vid = self.load_video(video_path)
        if len(vid) == 0: return False

        # Detect
        landmarks = self.detect_landmarks(vid)
        landmarks = self.landmarks_interpolate(landmarks)
        if not landmarks or self.mean_face_landmarks is None: return False

        # Crop & Transform
        stable_reference = self.get_stable_reference(self.std_size)
        sequence = []
        
        for frame_idx, frame in enumerate(tqdm(vid, desc="Align & Crop")):
            # Smoothing
            margin = min(self.window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
            smoothed = np.mean([landmarks[x] for x in range(frame_idx - margin, frame_idx + margin + 1)], axis=0)
            smoothed += landmarks[frame_idx].mean(axis=0) - smoothed.mean(axis=0)

            # Affine Transform
            transform = cv2.estimateAffinePartial2D(np.vstack([smoothed[x] for x in self.stable_points]), stable_reference, method=cv2.LMEDS)[0]
            trans_frame = cv2.warpAffine(
                frame, transform, dsize=self.std_size,
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            
            # Transformed Landmarks
            trans_lms = (np.matmul(smoothed, transform[:, :2].transpose()) + transform[:, 2].transpose())

            # Cut Patch
            center_x, center_y = np.mean(trans_lms[self.start_idx : self.stop_idx], axis=0)
            y_min = int(round(np.clip(center_y - self.crop_height//2, 0, trans_frame.shape[0])))
            y_max = int(round(np.clip(center_y + self.crop_height//2, 0, trans_frame.shape[0])))
            x_min = int(round(np.clip(center_x - self.crop_width//2, 0, trans_frame.shape[1])))
            x_max = int(round(np.clip(center_x + self.crop_width//2, 0, trans_frame.shape[1])))
            
            patch = trans_frame[y_min:y_max, x_min:x_max]
            # Resize guarantees 96x96 if boundary clipping happened
            patch = cv2.resize(patch, (self.crop_width, self.crop_height))
            sequence.append(patch)

        self.write_video(sequence, output_path)
        return True

    def write_video(self, rois, target_path):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        tmp_dir = tempfile.mkdtemp()
        try:
            for i, roi in enumerate(rois):
                # Save as BGR for cv2
                cv2.imwrite(os.path.join(tmp_dir, f"{i:06d}.png"), cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
            
            cmd = [
                "ffmpeg", "-y", "-f", "image2", "-r", "25",
                "-i", os.path.join(tmp_dir, "%06d.png"),
                "-vcodec", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                target_path, "-loglevel", "error"
            ]
            subprocess.run(cmd, check=True)
        finally:
            shutil.rmtree(tmp_dir)

# ==========================================
# Main Logic
# ==========================================

def calculate_metrics(hyps, refs):
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

def setup_save_directory(cfg):
    base_dir = getattr(cfg.common_eval, 'results_path', './results')
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def normalize_audio(in_path, out_path):
    """Force audio to 16kHz mono using FFmpeg"""
    cmd = f"ffmpeg -y -i {in_path} -acodec pcm_s16le -ac 1 -ar 16000 {out_path} -loglevel error"
    subprocess.call(cmd, shell=True)
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0

def extract_audio_from_video(video_path, out_path):
    """Extract audio track, convert to 16k mono"""
    cmd = f"ffmpeg -y -i {video_path} -vn -acodec pcm_s16le -ac 1 -ar 16000 {out_path} -loglevel error"
    subprocess.call(cmd, shell=True)
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0

def preprocess_single_input(cfg, save_dir):
    in_video = getattr(cfg.override, 'input_video', None)
    in_audio = getattr(cfg.override, 'input_audio', None)

    if not in_video and not in_audio:
        return None

    logger.info("Detected single file input. Processing...")
    
    final_video = "null"
    final_audio = "null"
    modalities = []

    # 1. Process Audio
    if in_audio:
        if os.path.exists(in_audio):
            # Normalization
            target_audio = os.path.join(save_dir, "input_audio_16k.wav")
            if normalize_audio(in_audio, target_audio):
                final_audio = target_audio
                modalities.append("audio")
                logger.info(f"Audio normalized: {target_audio}")
        else:
             logger.error(f"Input audio not found: {in_audio}")

    # 2. Process Video (and potentially extract audio if missing)
    if in_video:
        if os.path.exists(in_video):
            # A. Try extract audio if not provided
            if final_audio == "null":
                extracted_audio = os.path.join(save_dir, "extracted_audio.wav")
                if extract_audio_from_video(in_video, extracted_audio):
                    final_audio = extracted_audio
                    if "audio" not in modalities: modalities.append("audio")
                    logger.info("Audio extracted from video.")
                else:
                    logger.warning("Video appears silent. No audio extracted.")

            # B. Process Video Visuals
            root_dir = os.environ.get("ROOT", "/workspace/shuaque/EMR-LLM-CN")
            mean_face = os.path.join(root_dir, "data/preparation/detectors/retinaface/20words_mean_face.npy")
            preprocessor = VideoPreprocessor(mean_face_path=mean_face)
            
            # Check if already cropped
            if preprocessor.is_already_cropped(in_video):
                logger.info("Video already 96x96. Skipping crop.")
                final_video = in_video
                modalities.append("video")
            else:
                logger.info("Cropping mouth ROI...")
                crop_out = os.path.join(save_dir, "crop_mouth.mp4")
                if preprocessor.process_video(in_video, crop_out):
                    final_video = crop_out
                    modalities.append("video")
                else:
                    logger.error("Video processing failed.")
        else:
            logger.error(f"Input video not found: {in_video}")

    if not modalities:
        raise ValueError("No valid modalities available after preprocessing.")

    # 3. Create Manifest
    tsv_path = os.path.join(save_dir, "test.tsv")
    with open(tsv_path, 'w', encoding='utf-8') as f:
        f.write("/\n")
        # Use safe dummy frame count (e.g. 100) to pass min/max size filters
        f.write(f"single_infer\tNA\t{final_video}\t{final_audio}\t100\t16000\n")

    ltr_path = os.path.join(save_dir, "test.ltr")
    with open(ltr_path, 'w', encoding='utf-8') as f:
        f.write("single_infer\tunknown\tno_label\n")

    logger.info(f"Manifest created: {tsv_path}")
    return save_dir, str(modalities)

@hydra.main(config_path="conf", config_name="eval")
def main(cfg: DictConfig):
    utils.import_user_module(cfg.common)
    save_dir = setup_save_directory(cfg)
    
    # 1. Preprocess
    with open_dict(cfg):
        if 'input_video' not in cfg.override: cfg.override.input_video = None
        if 'input_audio' not in cfg.override: cfg.override.input_audio = None
    
    single_mode_info = preprocess_single_input(cfg, save_dir)
    
    # 2. Setup Task
    task = tasks.setup_task(cfg.task)

    # 3. Apply Overrides
    with open_dict(task.cfg):
        if hasattr(cfg.model, 'whisper_path') and cfg.model.whisper_path:
            task.cfg.whisper_path = cfg.model.whisper_path

        if single_mode_info:
            data_dir, derived_modalities = single_mode_info
            task.cfg.data = data_dir
            task.cfg.label_dir = data_dir
            task.cfg.modalities = ast.literal_eval(derived_modalities)
            cfg.dataset.gen_subset = "test"
            
            # Disable size filtering for single inference
            task.cfg.max_sample_size = None
            task.cfg.max_trim_sample_size = None
            task.cfg.min_sample_size = 0 # Allow short clips
        else:
            # Batch mode
            if cfg.override.data: task.cfg.data = cfg.override.data
            if cfg.override.label_dir: task.cfg.label_dir = cfg.override.label_dir
            if cfg.override.modalities:
                try:
                    task.cfg.modalities = ast.literal_eval(str(cfg.override.modalities))
                except:
                    task.cfg.modalities = cfg.override.modalities

        if cfg.override.llm_path: task.cfg.llm_path = cfg.override.llm_path
        if cfg.override.noise_snr is not None: task.cfg.noise_snr = cfg.override.noise_snr
        if cfg.override.noise_prob is not None: task.cfg.noise_prob = cfg.override.noise_prob
        if cfg.override.noise_wav: task.cfg.noise_wav = cfg.override.noise_wav

    # 4. Load Model
    logger.info(f"Loading model from {cfg.common_eval.path}")
    arg_overrides = {
        "w2v_path": cfg.model.w2v_path,
        "whisper_path": cfg.model.whisper_path,
        "llm_path": cfg.model.llm_path,
        "ctc_vocab_path": cfg.model.ctc_vocab_path
    }
    models, _ = checkpoint_utils.load_model_ensemble([cfg.common_eval.path], arg_overrides=arg_overrides, task=task)
    model = models[0].cuda().eval()

    # 5. Load Data
    tokenizer = model.tokenizer
    messages = [{"role": "user", "content": "请将这段语音转写为中文："}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_special_tokens=True, return_tensors="pt")[0].cuda()

    task.load_dataset(cfg.dataset.gen_subset)
    dataset = task.dataset(cfg.dataset.gen_subset)
    
    if len(dataset) == 0:
        logger.error("Dataset is empty.")
        return

    itr = task.get_batch_iterator(
        dataset=dataset, max_tokens=cfg.dataset.max_tokens, max_sentences=1, num_workers=0
    ).next_epoch_itr(shuffle=False)

    # 6. Inference
    result_log = os.path.join(save_dir, "result.txt")
    with open(result_log, "w", encoding="utf-8") as f:
        for sample in tqdm(itr, desc="Decoding"):
            sample = utils.move_to_cuda(sample)
            bsz = len(sample["id"])
            sample["net_input"]["source"]["instruction"] = [prompt_ids.clone() for _ in range(bsz)]

            with torch.no_grad():
                gen_args = cfg.generation
                output_ids = model.generate(
                    num_beams=gen_args.beam, 
                    temperature=gen_args.temperature,
                    repetition_penalty=getattr(gen_args, 'repetition_penalty', 1.0),
                    no_repeat_ngram_size=getattr(gen_args, 'no_repeat_ngram_size', 0),
                    length_penalty=getattr(gen_args, 'lenpen', 1.0),
                    **sample["net_input"]
                )
            
            hypos = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            refs = [tokenizer.decode(t[(t != tokenizer.pad_token_id) & (t != tokenizer.eos_token_id)], skip_special_tokens=True) 
                    for t in sample["target_list"]]

            for hid, h, r in zip(sample["utt_id"], hypos, refs):
                tqdm.write(f"\nID: {hid}\nHYP: {h}")
                tqdm.write("-" * 30)
                f.write(f"ID: {hid}\nHYP: {h}\n{'-'*30}\n")
                if not single_mode_info: f.write(f"REF: {r}\n")
                f.flush()
                os.fsync(f.fileno())

    logger.info(f"Finished. Saved to {save_dir}")

if __name__ == "__main__":
    main()