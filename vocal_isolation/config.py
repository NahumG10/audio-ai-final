from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AudioConfig:
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    freq_min: float = 80.0
    freq_max: float = 16000.0

    @property
    def n_freq_bins(self) -> int:
        return self.n_fft // 2 + 1


@dataclass
class DataConfig:
    musdb_root: str = "./data/musdb18"
    segment_duration: float = 5.0
    is_wav: bool = True
    num_workers: int = 4

    @property
    def segment_samples(self) -> int:
        return int(self.segment_duration * 44100)


@dataclass
class AugmentConfig:
    enable: bool = True
    pitch_shift_range: float = 1.0
    time_stretch_range: float = 0.05
    gain_db_range: float = 3.0
    remix_prob: float = 0.5


@dataclass
class ModelConfig:
    in_channels: int = 1
    encoder_channels: list = field(default_factory=lambda: [32, 64, 128, 256, 512])
    decoder_channels: list = field(default_factory=lambda: [256, 128, 64, 32, 16])
    kernel_size: int = 5
    dropout: float = 0.3


@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    grad_clip: float = 5.0
    checkpoint_dir: str = "./vocal_isolation/checkpoints"
    log_dir: str = "./vocal_isolation/logs"
    device: str = "auto"

    def resolve_device(self) -> str:
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return self.device


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
