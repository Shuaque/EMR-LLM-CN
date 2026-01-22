import logging
import sys
from typing import List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from fairseq.data import Dictionary
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING

DBG = True if len(sys.argv) == 1 else False
if DBG:
    from src.dataset import mob_av_emr_dataset
else:
    from .dataset import mob_av_emr_dataset

logger = logging.getLogger(__name__)

# task.py

@dataclass
class MobAVEMR_TrainingConfig(FairseqDataclass):
    # Data configuration
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    labels: List[str] = field(
        default=("ltr",), 
        metadata={"help": "label file extensions"}
    )
    label_dir: Optional[str] = field(default=None, metadata={"help": "separate label directory"})
    label_rate: int = field(default=-1, metadata={"help": "label frame rate, -1 for sequence"})

    # Audio configuration
    sample_rate: int = field(default=16_000, metadata={"help": "target sample rate"})
    max_sample_size: Optional[int] = field(default=700, metadata={"help": "max audio length"})
    min_sample_size: Optional[int] = field(default=None, metadata={"help": "min audio length"})
    
    # [Modified] Removed interpolation, default to None
    max_trim_sample_size: Optional[int] = field(default=None, metadata={"help": "trim size for batching"})
    
    normalize: bool = field(default=False, metadata={"help": "normalize waveform"})
    pad_audio: bool = field(default=False, metadata={"help": "pad audio to batch max length"})
    random_crop: bool = field(default=True, metadata={"help": "always crop from start if False"})
    stack_order_audio: int = field(default=1, metadata={"help": "stack n frames as input"})
    skip_verify: bool = field(default=False, metadata={"help": "skip alignment check"})

    # Whisper configuration
    whisper_path: str = field(default="/path/to/pretrained/whisper/whisper-large", metadata={"help": "path to whisper model"})

    # Visual configuration
    image_crop_size: int = field(default=88, metadata={"help": "lip ROI crop size"})
    image_mean: float = field(default=0.421, metadata={"help": "image mean"})
    image_std: float = field(default=0.165, metadata={"help": "image std"})
    image_aug: bool = field(default=False, metadata={"help": "use augmentation in training"})

    # Modalities
    modalities: Optional[List[str]] = field(
            default=("audio", "video"), 
            metadata={"help": "modalities to load"}
        )
    is_s2s: bool = field(default=False, metadata={"help": "seq2seq training"})

    # Tokenizer params
    tokenizer_bpe_name: Optional[str] = field(default=None, metadata={"help": "bpe tokenizer name"})
    tokenizer_bpe_model: Optional[str] = field(default=None, metadata={"help": "bpe tokenizer model path"})

    # Noise Augmentation
    noise_wav: Optional[str] = field(default=None, metadata={"help": "noise wav file path"})
    noise_prob: float = field(default=0.0, metadata={"help": "noise mix probability"})
    noise_snr: float = field(default=0.0, metadata={"help": "noise SNR (dB), e.g. -20"})
    noise_num: int = field(default=1, metadata={"help": "number of noise wavs to mix"})

    # Fine-tuning
    fine_tuning: bool = field(default=False, metadata={"help": "fine-tune AV-HuBERT"})
    llm_path: str = field(default=MISSING, metadata={"help": "path to LLM checkpoint"})
    pdb: bool = field(default=False, metadata={"help": "enable pdb debug"})
    log_noise_debug: bool = field(default=True, metadata={"help": "log noise apply info in dataset"})

@register_task("mob-av-emr_training", dataclass=MobAVEMR_TrainingConfig)
class MobAVEMR_TrainingTask(FairseqTask):
    cfg: MobAVEMR_TrainingConfig

    def __init__(self, cfg: MobAVEMR_TrainingConfig) -> None:
        super().__init__(cfg)
        self.fine_tuning = cfg.fine_tuning
        
        # [Modified] Manual default value logic
        if self.cfg.max_trim_sample_size is None:
            self.cfg.max_trim_sample_size = self.cfg.max_sample_size

    @property
    def source_dictionary(self) -> Optional[Dictionary]: return None
    @property
    def target_dictionary(self) -> Optional[Dictionary]: return None
    @property
    def dictionaries(self) -> List[Dictionary]: return None

    @classmethod
    def setup_task(cls, cfg: MobAVEMR_TrainingConfig, **kwargs) -> "MobAVEMR_TrainingTask":
        if cfg.pdb:
            import pdb; pdb.set_trace()
        return cls(cfg)

    def get_label_dir(self) -> str:
        return self.cfg.label_dir if self.cfg.label_dir is not None else self.cfg.data

    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"
        paths = [f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels]
        image_aug = self.cfg.image_aug if split == "train" else False

        self.datasets[split] = mob_av_emr_dataset(
            manifest_path=manifest,
            sample_rate=self.cfg.sample_rate,
            llm_path=self.cfg.llm_path,
            whisper_path=self.cfg.whisper_path,
            label_paths=paths,
            label_rates=self.cfg.label_rate,
            max_keep_sample_size=self.cfg.max_sample_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            max_sample_size=self.cfg.max_trim_sample_size,
            pad_audio=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
            store_labels=True,
            random_crop=self.cfg.random_crop,
            single_target=False,
            stack_order_audio=self.cfg.stack_order_audio,
            skip_verify=self.cfg.skip_verify,
            image_mean=self.cfg.image_mean,
            image_std=self.cfg.image_std,
            image_crop_size=self.cfg.image_crop_size,
            image_aug=image_aug,
            modalities=self.cfg.modalities,
            is_s2s=self.cfg.is_s2s,
            noise_fn=self.cfg.noise_wav,
            noise_prob=float(self.cfg.noise_prob),
            snr_target=float(self.cfg.noise_snr),
            noise_num=int(self.cfg.noise_num),
            log_noise_debug=bool(self.cfg.log_noise_debug),
        )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(self, indices: np.array, *args, **kwargs) -> np.array:
        return indices