import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path


def plot_spectrogram(mag: np.ndarray, sr: int, hop_length: int, title: str, ax=None):
    """Plot a magnitude spectrogram."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 4))
    librosa.display.specshow(
        librosa.amplitude_to_db(mag, ref=np.max),
        sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz", ax=ax,
    )
    ax.set_title(title)
    ax.set_ylim(0, 16000)


def plot_separation_result(
    mix_wav: np.ndarray,
    vocals_wav: np.ndarray,
    accomp_wav: np.ndarray,
    sr: int,
    hop_length: int,
    n_fft: int,
    save_path: str | None = None,
):
    """Visualize separation results: mix, predicted vocals, predicted accompaniment."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    for wav, title, ax in zip(
        [mix_wav, vocals_wav, accomp_wav],
        ["Original Mix", "Predicted Vocals", "Predicted Accompaniment"],
        axes,
    ):
        spec = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))
        plot_spectrogram(spec, sr, hop_length, title, ax)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_mask(mask: np.ndarray, sr: int, hop_length: int, save_path: str | None = None):
    """Visualize the predicted vocal mask."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    librosa.display.specshow(
        mask, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz", ax=ax,
    )
    ax.set_title("Predicted Vocal Mask")
    ax.set_ylim(0, 16000)
    plt.colorbar(ax.images[0], ax=ax, label="Mask Value")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_training_history(history: dict, save_path: str | None = None):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(epochs, history["valid_loss"], label="Valid Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
