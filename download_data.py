"""
Dataset Setup for Vocal Isolation Project
==========================================

OPTION 1: Generate a synthetic mini-dataset for testing the pipeline.
    python download_data.py --synthetic

OPTION 2: Download MUSDB18-HQ (real dataset, requires Zenodo access).
    1. Go to: https://zenodo.org/records/3338372
    2. Click "Request access" (academic use, approved within a day)
    3. Download and extract to ./data/musdb18/
    4. The folder structure should be:
        data/musdb18/
          train/
            A Classic Education - NightOwl/
              mixture.wav
              vocals.wav
              drums.wav
              bass.wav
              other.wav
            ...
          test/
            ...

OPTION 3: Use the smaller MUSDB18 (stems, 4.4GB, needs ffmpeg).
    pip install musdb stempeg
    python -c "import musdb; musdb.DB(download=True, root='./data/musdb18')"
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path


def generate_synthetic_dataset(root: str = "./data/musdb18", n_train: int = 15, n_test: int = 5):
    """
    Generate synthetic audio tracks to test the pipeline.

    Creates simple sine-wave based tracks where:
    - Vocals = sine waves at typical vocal frequencies (200-800 Hz)
    - Accompaniment = sine waves at other frequencies + noise
    - Mixture = vocals + accompaniment
    """
    sr = 44100
    duration = 30.0  # seconds per track
    root = Path(root)

    for split, n_tracks in [("train", n_train), ("test", n_test)]:
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_tracks):
            track_name = f"synthetic_track_{i:03d}"
            track_dir = split_dir / track_name
            track_dir.mkdir(parents=True, exist_ok=True)

            t = np.linspace(0, duration, int(sr * duration), endpoint=False)

            # Vocals: combination of sine waves at vocal frequencies
            vocal_freq = np.random.uniform(150, 500)
            vocal_freq2 = vocal_freq * np.random.uniform(1.5, 3.0)
            vocals = (
                0.3 * np.sin(2 * np.pi * vocal_freq * t)
                + 0.15 * np.sin(2 * np.pi * vocal_freq2 * t)
            )
            # Add slight vibrato
            vibrato = 0.02 * np.sin(2 * np.pi * 5 * t)
            vocals = vocals * (1 + vibrato)

            # Accompaniment: lower + higher frequency content
            bass_freq = np.random.uniform(60, 120)
            high_freq = np.random.uniform(1000, 4000)
            accompaniment = (
                0.2 * np.sin(2 * np.pi * bass_freq * t)
                + 0.1 * np.sin(2 * np.pi * high_freq * t)
                + 0.05 * np.random.randn(len(t))
            )

            # Drums: periodic impulses
            beat_period = int(sr * 60 / np.random.uniform(90, 140))
            drums = np.zeros_like(t)
            for j in range(0, len(t), beat_period):
                decay = np.exp(-np.arange(min(4410, len(t) - j)) / 1000)
                drums[j:j + len(decay)] += 0.15 * decay * np.random.randn(len(decay))

            mixture = vocals + accompaniment + drums

            # Normalize
            peak = max(np.abs(mixture).max(), 1e-6)
            vocals = (vocals / peak).astype(np.float32)
            accompaniment = (accompaniment / peak).astype(np.float32)
            drums = (drums / peak).astype(np.float32)
            mixture = (mixture / peak).astype(np.float32)

            # Stereo (duplicate mono to both channels)
            vocals_stereo = np.stack([vocals, vocals], axis=-1)
            mixture_stereo = np.stack([mixture, mixture], axis=-1)
            accomp_stereo = np.stack([accompaniment, accompaniment], axis=-1)
            drums_stereo = np.stack([drums, drums], axis=-1)
            other_stereo = np.zeros_like(vocals_stereo)

            sf.write(str(track_dir / "mixture.wav"), mixture_stereo, sr)
            sf.write(str(track_dir / "vocals.wav"), vocals_stereo, sr)
            sf.write(str(track_dir / "bass.wav"), accomp_stereo, sr)
            sf.write(str(track_dir / "drums.wav"), drums_stereo, sr)
            sf.write(str(track_dir / "other.wav"), other_stereo, sr)

            print(f"  [{split}] Created: {track_name} ({duration:.0f}s)")

    print(f"\nSynthetic dataset created at: {root}")
    print(f"  Train tracks: {n_train}")
    print(f"  Test tracks:  {n_test}")
    print(f"\nYou can now run: python main.py train --musdb_root {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up dataset for vocal isolation")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic test dataset")
    parser.add_argument("--root", default="./data/musdb18",
                        help="Output directory")
    parser.add_argument("--n_train", type=int, default=15,
                        help="Number of synthetic training tracks")
    parser.add_argument("--n_test", type=int, default=5,
                        help="Number of synthetic test tracks")
    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_dataset(args.root, args.n_train, args.n_test)
    else:
        print(__doc__)
