# Vocal Isolation from Stereo Music

U-Net-based deep learning system for isolating vocals from mixed music recordings. Operates in the STFT magnitude domain using soft masking.

## Results

Evaluated on MUSDB18 test set (80 segments, 55 tracks):

| Metric | Mean | Median |
|--------|------|--------|
| SDR | 12.61 dB | 7.89 dB |
| SIR | 12.11 dB | 7.20 dB |
| SAR | 12.45 dB | 6.99 dB |

## Setup

```bash
pip install -r requirements.txt
```

## Download Dataset

```bash
python download_musdb.py
```

This downloads MUSDB18 and converts stems to WAV format. Requires `ffmpeg` in PATH.

## Usage

**Train:**
```bash
python main.py train
```

**Evaluate:**
```bash
python main.py evaluate
```

**Separate vocals from a song:**
```bash
python main.py separate path/to/song.wav -o output_dir
```

## Architecture

- 5-layer U-Net encoder-decoder with skip connections
- Input: log-magnitude STFT spectrogram (1025 × 431)
- Output: soft mask [0, 1] applied to mixture magnitude
- ~35.6M parameters
- Loss: L1 + spectral convergence

## Project Structure

```
├── main.py                  # CLI entry point
├── download_musdb.py        # Dataset download & conversion
├── download_data.py         # Synthetic data generator (for testing)
├── requirements.txt
└── vocal_isolation/
    ├── config.py            # Hyperparameters & settings
    ├── dataset.py           # Data loading, STFT, augmentation
    ├── model.py             # U-Net architecture
    ├── train.py             # Training loop
    ├── evaluate.py          # SDR/SIR/SAR evaluation
    ├── inference.py         # Vocal separation pipeline
    └── visualize.py         # Plotting utilities
```
