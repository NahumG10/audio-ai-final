"""Download MUSDB18 and convert stems to WAV using ffmpeg directly."""

import ssl
import certifi
import subprocess
import sys
from pathlib import Path

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

FFMPEG = "C:\\Users\\nahum\\AppData\\Local\\Programs\\Python\\Python313\\Scripts\\ffmpeg.exe"

STEM_MAP = {
    0: "mixture",
    1: "drums",
    2: "bass",
    3: "other",
    4: "vocals",
}


def convert_stem_to_wavs(stem_path: Path, output_dir: Path):
    """Extract all 5 stems from an mp4 stem file to individual WAVs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for stream_idx, name in STEM_MAP.items():
        wav_path = output_dir / f"{name}.wav"
        if wav_path.exists():
            continue

        cmd = [
            FFMPEG, "-y", "-loglevel", "error",
            "-i", str(stem_path),
            "-map", f"0:{stream_idx}",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            str(wav_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    Warning: Failed to extract {name} from {stem_path.name}: {result.stderr.strip()}")


def download_and_convert():
    stems_root = Path("./data/musdb18_stems")
    wav_root = Path("./data/musdb18")

    # Step 1: Download if not already present
    stem_files = list(stems_root.rglob("*.stem.mp4"))
    if not stem_files:
        print("Step 1: Downloading MUSDB18 sample dataset...")
        import musdb
        musdb.DB(root=str(stems_root), download=True, setup_file=None)
        stem_files = list(stems_root.rglob("*.stem.mp4"))
        print(f"Downloaded {len(stem_files)} stem files.\n")
    else:
        print(f"Found {len(stem_files)} existing stem files.\n")

    # Step 2: Convert stems to WAV
    print("Step 2: Converting stems to WAV...")

    for subset_dir in sorted(stems_root.iterdir()):
        if not subset_dir.is_dir():
            continue
        subset_name = subset_dir.name
        stems_in_dir = sorted(subset_dir.glob("*.stem.mp4"))

        if not stems_in_dir:
            continue

        print(f"\n  {subset_name} ({len(stems_in_dir)} tracks):")

        for i, stem_file in enumerate(stems_in_dir):
            track_name = stem_file.name.replace(".stem.mp4", "")
            out_dir = wav_root / subset_name / track_name
            convert_stem_to_wavs(stem_file, out_dir)
            print(f"    [{i+1}/{len(stems_in_dir)}] {track_name}")

    # Count results
    train_tracks = len(list((wav_root / "train").iterdir())) if (wav_root / "train").exists() else 0
    test_tracks = len(list((wav_root / "test").iterdir())) if (wav_root / "test").exists() else 0

    print(f"\nDone! WAV files saved to: {wav_root}")
    print(f"  Train tracks: {train_tracks}")
    print(f"  Test tracks:  {test_tracks}")
    print(f"\nRun training: python main.py train --musdb_root {wav_root} --epochs 100")


if __name__ == "__main__":
    download_and_convert()
