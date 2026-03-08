import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from .config import Config
from .model import UNet, build_model
from .dataset import STFT


class VocalIsolator:
    """Inference wrapper for separating vocals from a music file."""

    def __init__(self, checkpoint_path: str, config: Config | None = None):
        self.config = config or Config()
        self.device = self.config.train.resolve_device()
        self.stft = STFT(self.config)

        self.model = build_model(self.config.model)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def separate(self, audio_path: str, output_dir: str | None = None) -> dict[str, np.ndarray]:
        """
        Separate vocals from accompaniment.

        Args:
            audio_path: Path to input audio file.
            output_dir: If provided, saves separated audio files.

        Returns:
            Dict with 'vocals' and 'accompaniment' numpy arrays.
        """
        waveform, sr = librosa.load(audio_path, sr=self.config.audio.sample_rate, mono=True)
        waveform_tensor = torch.from_numpy(waveform).float()

        # Process in chunks to manage memory
        segment_samples = self.config.data.segment_samples
        total_samples = len(waveform_tensor)
        vocals_full = torch.zeros(total_samples)

        for start in range(0, total_samples, segment_samples):
            end = min(start + segment_samples, total_samples)
            chunk = waveform_tensor[start:end]

            # Pad if needed
            if len(chunk) < segment_samples:
                pad_len = segment_samples - len(chunk)
                chunk = torch.nn.functional.pad(chunk, (0, pad_len))

            mix_mag, mix_phase = self.stft.forward(chunk)
            mix_mag_log = torch.log1p(mix_mag).unsqueeze(0).unsqueeze(0).to(self.device)

            mask = self.model(mix_mag_log).squeeze(0).squeeze(0).cpu()
            vocals_mag = mask * mix_mag
            vocals_chunk = self.stft.inverse(vocals_mag, mix_phase)

            chunk_len = min(segment_samples, end - start)
            actual_len = min(chunk_len, len(vocals_chunk))
            vocals_full[start:start + actual_len] = vocals_chunk[:actual_len]

        vocals_np = vocals_full.numpy()
        accompaniment_np = waveform - vocals_np

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            stem = Path(audio_path).stem

            sf.write(str(out / f"{stem}_vocals.wav"), vocals_np, sr)
            sf.write(str(out / f"{stem}_accompaniment.wav"), accompaniment_np, sr)
            sf.write(str(out / f"{stem}_original.wav"), waveform, sr)
            print(f"Saved outputs to {out}")

        return {"vocals": vocals_np, "accompaniment": accompaniment_np}


def run_inference(checkpoint_path: str, audio_path: str, output_dir: str):
    isolator = VocalIsolator(checkpoint_path)
    result = isolator.separate(audio_path, output_dir)
    print(f"Vocal isolation complete. Vocals shape: {result['vocals'].shape}")
    return result
