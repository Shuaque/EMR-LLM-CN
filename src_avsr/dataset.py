# -*- coding: utf-8 -*-
import itertools
import logging
import os
import sys
import numpy as np
import random
import torch
from scipy.io import wavfile
from fairseq.data.fairseq_dataset import FairseqDataset
from transformers import AutoTokenizer, WhisperProcessor

# Automatic path fix for project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src_avsr.utils as custom_utils

logger = logging.getLogger(__name__)


def load_audio_visual(manifest_path, max_keep, min_keep, frame_rate, label_paths, label_rates, tol=0.1):
    """
    Parses the manifest file to load video/audio paths and sizes.
    Filters samples based on min/max size constraints.
    """
    names, inds, sizes = [], [], []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            if len(items) < 5:
                continue 

            audio_id = items[0]
            video_path = items[2]
            audio_path = items[3]
            num_frames = int(items[4])
            sz = num_frames

            if min_keep is not None and sz < min_keep:
                continue
            elif max_keep is not None and sz > max_keep:
                continue
            else:
                names.append((video_path, audio_path + ":" + audio_id))
                inds.append(ind)
                sizes.append(sz)

    logger.info(f"Loaded {len(names)} samples. Root: {root}")
    return root, names, inds, ind + 1, sizes

def load_label(label_path, inds, tot):
    """
    Loads text labels associated with the selected indices.
    """
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = [line.strip().split('\t')[-1] for line in f if line.strip()]
    return [labels[i] for i in inds]


class mob_av_emr_dataset(FairseqDataset):
    def __init__(self, manifest_path, sample_rate, llm_path, whisper_path, label_paths, label_rates, **kwargs):
        """
        Dataset class for multimodal (Audio-Visual) EMR tasks.
        Handles data loading, augmentation, noise mixing, and tokenization.
        """
        self.label_rates = label_rates
        self.modalities = set(kwargs.get("modalities", ["audio", "video"]))
        self.max_sample_size = kwargs.get("max_sample_size", sys.maxsize)
        self.crop_size = kwargs.get("image_crop_size", 88)

        self.audio_root, self.names, inds, tot, self.sizes = load_audio_visual(
            manifest_path, kwargs.get("max_keep_sample_size"), kwargs.get("min_keep_sample_size"), 
            sample_rate, label_paths, self.label_rates
        )

        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_path)
        if self.llm_tokenizer.pad_token_id is None:
            self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eos_token_id

        self.label_list = [load_label(p, inds, tot) for p in label_paths]
        
        self.transform = custom_utils.Compose([
            custom_utils.Normalize(0.0, 255.0),
            custom_utils.CenterCrop((self.crop_size, self.crop_size)),
            custom_utils.Normalize(kwargs.get("image_mean", 0), kwargs.get("image_std", 1)),
        ])
        
        self.noise = None
        self.noise_prob = kwargs.get("noise_prob", 0.0)
        self.snr_target = kwargs.get("noise_snr", 0)
        noise_fn = kwargs.get("noise_fn")
        
        if noise_fn and str(noise_fn).strip():
            try:
                sr, noise = wavfile.read(noise_fn)
                if noise.ndim == 2: noise = noise.mean(axis=1)
                
                # Smart normalization logic for noise file
                if np.issubdtype(noise.dtype, np.integer):
                    noise = noise.astype(np.float32) / 32768.0
                else:
                    noise = noise.astype(np.float32)
                noise = noise - float(np.mean(noise))
                self.noise = noise
                logger.info(f"Loaded noise: {noise_fn}")
            except:
                pass

    def _load_wav_16k_mono(self, path: str):
        """
        Robustly loads 16kHz mono audio.
        [Critical Fix] Handles Float32 vs Int16 scaling to prevent silence issues.
        """
        sr, data = wavfile.read(path)
        assert sr == 16000, f"Expected 16kHz, got {sr}"
        
        if data.ndim == 2:
            data = data.mean(axis=1)
            
        # [CRITICAL FIX] Only divide by 32768 if data is integer type.
        # If float, assume it is already normalized [-1, 1].
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / 32768.0
        else:
            data = data.astype(np.float32)
            
        data = data - float(np.mean(data))
        return data, sr

    def _mix_noise(self, clean):
        """
        Mixes background noise into the clean audio based on SNR target.
        """
        if self.noise is None or self.noise_prob <= 0: return clean
        if random.random() > self.noise_prob: return clean
        
        noise_len = len(self.noise)
        wav_len = len(clean)
        if noise_len >= wav_len:
            start = random.randint(0, noise_len - wav_len)
            noise_seg = self.noise[start:start+wav_len]
        else:
            repeats = int(np.ceil(wav_len / noise_len))
            noise_seg = np.tile(self.noise, repeats)[:wav_len]
            
        # SNR mixing
        clean_p = np.sum(clean ** 2) + 1e-10
        noise_p = np.sum(noise_seg ** 2) + 1e-10
        target_noise_p = clean_p / (10 ** (self.snr_target / 10))
        scale = np.sqrt(target_noise_p / noise_p)
        
        return clean + noise_seg * scale

    def load_feature(self, name_pair):
        """
        Loads raw audio and video features.
        """
        video_fn, audio_fn = name_pair
        video_feats, audio_feats = None, None

        if "video" in self.modalities:
            is_valid = video_fn and str(video_fn).lower() not in ["null", "none", ""]
            if is_valid:
                try:
                    path = os.path.join(self.audio_root, video_fn) if not os.path.isabs(video_fn) else video_fn
                    raw_v = custom_utils.load_video(path)
                    v_np = self.transform(raw_v)
                    video_feats = np.expand_dims(v_np, axis=-1)
                except:
                    is_valid = False
            
            if not is_valid:
                # Dummy black frames
                video_feats = np.zeros((1, self.crop_size, self.crop_size, 1), dtype=np.float32)

        if "audio" in self.modalities:
            path = audio_fn.split(":")[0]
            try:
                wav, sr = self._load_wav_16k_mono(path)
                wav = self._mix_noise(wav)
                audio_feats = self.whisper_processor(wav, sampling_rate=16000, return_tensors="pt").input_features
            except Exception as e:
                logger.warning(f"Audio error {path}: {e}")
                # Dummy silence
                audio_feats = self.whisper_processor(np.zeros(16000), sampling_rate=16000, return_tensors="pt").input_features

        return video_feats, audio_feats

    def __getitem__(self, index):
        v_feats, a_feats = self.load_feature(self.names[index])
        if v_feats is not None: v_feats = torch.from_numpy(v_feats.astype(np.float32))
        
        label_text = self.label_list[0][index]
        label_ids = self.llm_tokenizer(label_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        label_ids = torch.cat([label_ids, torch.tensor([self.llm_tokenizer.eos_token_id])])
        
        # Minimal Prompt to reduce hallucination
        messages = [{"role": "user", "content": "请将这段语音转写为中文："}]
        inst_ids = self.llm_tokenizer.apply_chat_template(messages, add_special_tokens=True, return_tensors="pt")[0]

        return {
            "id": index,
            "fid": self.names[index][1].split(":")[1],
            "video_source": v_feats,
            "audio_source": a_feats,
            "label_list": label_ids,
            "text_source": [inst_ids]
        }

    def __len__(self): return len(self.sizes)
    def num_tokens(self, index): return self.sizes[index]
    def size(self, index): return self.sizes[index]
    def ordered_indices(self): return np.arange(len(self))

    def collater(self, samples):
        """
        Collates samples into a batch. Handles padding for audio, video, and text.
        """
        samples = [s for s in samples if s["id"] is not None]
        if not samples: return {}

        a_src = [s["audio_source"] for s in samples]
        v_src = [s["video_source"] for s in samples]
        
        coll_audio = torch.cat(a_src, dim=0) if a_src[0] is not None else None
        
        coll_video = None
        padding_mask = None
        if v_src[0] is not None:
            # Video collation with padding
            max_len = max(len(v) for v in v_src)
            # In eval mode, max_len might be large, no clipping applied here.
            feat_shape = list(v_src[0].shape[1:])
            coll_video = v_src[0].new_zeros([len(v_src), max_len] + feat_shape)
            padding_mask = torch.BoolTensor(len(v_src), max_len).fill_(False)
            
            for i, v in enumerate(v_src):
                coll_video[i, :len(v)] = v
                if len(v) < max_len: padding_mask[i, len(v):] = True
            coll_video = coll_video.permute(0, 4, 1, 2, 3)

        targets = [s["label_list"] for s in samples]
        pad_id = self.llm_tokenizer.pad_token_id
        max_target = max(len(t) for t in targets)
        coll_targets = torch.full((len(targets), max_target), pad_id, dtype=torch.long)
        for i, t in enumerate(targets):
            coll_targets[i, :len(t)] = t

        return {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {
                "source": {
                    "audio": coll_audio,
                    "video": coll_video,
                    "instruction": [s["text_source"][0] for s in samples],
                    "padding_mask": padding_mask
                },
                "target_list": targets
            },
            "utt_id": [s["fid"] for s in samples],
            "target": coll_targets,
            "target_list": targets
        }