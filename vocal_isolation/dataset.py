import torch
import numpy as np
import librosa
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from .config import Config


class STFT:
    """Handles forward and inverse STFT transforms."""

    def __init__(self, config: Config):
        self.n_fft = config.audio.n_fft
        self.hop_length = config.audio.hop_length
        self.window = torch.hann_window(self.n_fft)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(waveform.device),
            return_complex=True,
        )
        magnitude = stft.abs()
        phase = stft.angle()
        return magnitude, phase

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        complex_spec = torch.polar(magnitude, phase)
        waveform = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(magnitude.device),
        )
        return waveform


class MUSDBDataset(Dataset):
    """
    Dataset for MUSDB18-HQ (WAV format).

    Expected folder structure:
        musdb18/
          train/
            TrackName/
              mixture.wav
              vocals.wav
              (drums.wav, bass.wav, other.wav — optional)
          test/
            TrackName/
              mixture.wav
              vocals.wav
    """

    def __init__(self, config: Config, split: str = "train", augment: bool = False):
        self.config = config
        self.sr = config.audio.sample_rate
        self.segment_samples = config.data.segment_samples
        self.n_fft = config.audio.n_fft
        self.hop_length = config.audio.hop_length
        self.augment = augment and config.augment.enable
        self.aug_cfg = config.augment

        root = Path(config.data.musdb_root)

        if split in ("train", "valid"):
            track_dirs = sorted((root / "train").iterdir())
            if split == "valid":
                self.track_dirs = track_dirs[-10:]
            else:
                self.track_dirs = track_dirs[:-10]
        else:
            self.track_dirs = sorted((root / "test").iterdir())

        self._segments = self._build_segment_index()

    def _get_track_duration(self, track_dir: Path) -> int:
        """Get total samples in a track without loading full audio."""
        import soundfile as sf
        info = sf.info(str(track_dir / "mixture.wav"))
        return int(info.frames)

    def _build_segment_index(self) -> list[tuple[int, int]]:
        """Pre-compute (track_idx, start_sample) pairs for all segments."""
        segments = []
        for i, track_dir in enumerate(self.track_dirs):
            total_samples = self._get_track_duration(track_dir)
            n_segments = max(1, total_samples // self.segment_samples)
            for s in range(n_segments):
                start = s * self.segment_samples
                if start + self.segment_samples <= total_samples:
                    segments.append((i, start))
        return segments

    def __len__(self) -> int:
        return len(self._segments)

    def _load_segment(self, track_idx: int, start_sample: int):
        track_dir = self.track_dirs[track_idx]
        dur_samples = self.segment_samples

        mix, _ = librosa.load(
            str(track_dir / "mixture.wav"),
            sr=self.sr, mono=True,
            offset=start_sample / self.sr,
            duration=dur_samples / self.sr,
        )
        vocals, _ = librosa.load(
            str(track_dir / "vocals.wav"),
            sr=self.sr, mono=True,
            offset=start_sample / self.sr,
            duration=dur_samples / self.sr,
        )

        # Pad if shorter than expected
        if len(mix) < dur_samples:
            mix = np.pad(mix, (0, dur_samples - len(mix)))
        if len(vocals) < dur_samples:
            vocals = np.pad(vocals, (0, dur_samples - len(vocals)))

        return torch.from_numpy(mix).float(), torch.from_numpy(vocals).float()

    def _augment_audio(self, mix: torch.Tensor, vocals: torch.Tensor):
        if not self.augment:
            return mix, vocals

        # Random gain
        gain_db = np.random.uniform(-self.aug_cfg.gain_db_range, self.aug_cfg.gain_db_range)
        gain = 10 ** (gain_db / 20)
        mix = mix * gain
        vocals = vocals * gain

        # Remix: randomly adjust vocal/accompaniment ratio
        if np.random.rand() < self.aug_cfg.remix_prob:
            accompaniment = mix - vocals
            vocal_gain = np.random.uniform(0.7, 1.3)
            vocals = vocals * vocal_gain
            mix = accompaniment + vocals

        return mix, vocals

    def __getitem__(self, idx: int):
        track_idx, start_sample = self._segments[idx]
        mix, vocals = self._load_segment(track_idx, start_sample)
        mix, vocals = self._augment_audio(mix, vocals)

        mix_stft = torch.stft(
            mix, n_fft=self.n_fft, hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft), return_complex=True,
        )
        vocals_stft = torch.stft(
            vocals, n_fft=self.n_fft, hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft), return_complex=True,
        )

        mix_mag = mix_stft.abs()
        vocals_mag = vocals_stft.abs()

        mix_mag_log = torch.log1p(mix_mag).unsqueeze(0)       # (1, F, T)
        vocals_mag_log = torch.log1p(vocals_mag).unsqueeze(0) # (1, F, T)

        return {
            "mix_mag": mix_mag_log,
            "vocals_mag": vocals_mag_log,
            "mix_phase": mix_stft.angle(),
            "mix_mag_raw": mix_mag,
        }


def create_dataloaders(config: Config) -> dict[str, DataLoader]:
    train_ds = MUSDBDataset(config, split="train", augment=True)
    valid_ds = MUSDBDataset(config, split="valid", augment=False)
    test_ds = MUSDBDataset(config, split="test", augment=False)

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=config.train.batch_size,
            shuffle=True, num_workers=config.data.num_workers,
            pin_memory=True, drop_last=True,
        ),
        "valid": DataLoader(
            valid_ds, batch_size=config.train.batch_size,
            shuffle=False, num_workers=config.data.num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds, batch_size=1,
            shuffle=False, num_workers=0,
        ),
    }
    return loaders
