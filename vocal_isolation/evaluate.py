import torch
import numpy as np
from tqdm import tqdm
from .config import Config
from .model import UNet
from .dataset import STFT


def _compute_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute Signal-to-Distortion Ratio (dB) between reference and estimate."""
    reference = reference.flatten()
    estimate = estimate.flatten()
    min_len = min(len(reference), len(estimate))
    reference = reference[:min_len]
    estimate = estimate[:min_len]

    noise = estimate - reference
    ref_energy = np.sum(reference ** 2) + 1e-10
    noise_energy = np.sum(noise ** 2) + 1e-10
    return float(10 * np.log10(ref_energy / noise_energy))


def _compute_sir(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute Signal-to-Interference Ratio (dB)."""
    reference = reference.flatten()
    estimate = estimate.flatten()
    min_len = min(len(reference), len(estimate))
    reference = reference[:min_len]
    estimate = estimate[:min_len]

    # Project estimate onto reference
    dot = np.sum(reference * estimate)
    ref_energy = np.sum(reference ** 2) + 1e-10
    s_target = (dot / ref_energy) * reference
    e_interf = estimate - s_target
    target_energy = np.sum(s_target ** 2) + 1e-10
    interf_energy = np.sum(e_interf ** 2) + 1e-10
    return float(10 * np.log10(target_energy / interf_energy))


def _compute_sar(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute Signal-to-Artifacts Ratio (dB)."""
    reference = reference.flatten()
    estimate = estimate.flatten()
    min_len = min(len(reference), len(estimate))
    reference = reference[:min_len]
    estimate = estimate[:min_len]

    noise = estimate - reference
    est_energy = np.sum(estimate ** 2) + 1e-10
    noise_energy = np.sum(noise ** 2) + 1e-10
    return float(10 * np.log10(est_energy / noise_energy))


@torch.no_grad()
def evaluate_track(
    model: UNet,
    mix_audio: np.ndarray,
    vocals_audio: np.ndarray,
    config: Config,
    device: str,
) -> dict[str, float]:
    """
    Evaluate a single track using SDR, SIR, SAR metrics.

    Args:
        mix_audio: mono mix waveform (num_samples,)
        vocals_audio: ground truth mono vocals (num_samples,)
    """
    model.eval()
    stft_transform = STFT(config)

    mix_tensor = torch.from_numpy(mix_audio).float()
    mix_mag, mix_phase = stft_transform.forward(mix_tensor)

    mix_mag_log = torch.log1p(mix_mag).unsqueeze(0).unsqueeze(0).to(device)

    mask = model(mix_mag_log).squeeze(0).squeeze(0).cpu()
    vocals_mag_pred = mask * mix_mag
    vocals_pred = stft_transform.inverse(vocals_mag_pred, mix_phase)

    min_len = min(len(vocals_pred), len(vocals_audio))
    pred_np = vocals_pred[:min_len].numpy()
    ref_np = vocals_audio[:min_len]

    return {
        "SDR": _compute_sdr(ref_np, pred_np),
        "SIR": _compute_sir(ref_np, pred_np),
        "SAR": _compute_sar(ref_np, pred_np),
    }


@torch.no_grad()
def evaluate_dataset(model: UNet, test_loader, config: Config, device: str) -> dict:
    """Run evaluation across the entire test set."""
    model.eval()
    all_results = {"SDR": [], "SIR": [], "SAR": []}
    stft_transform = STFT(config)

    pbar = tqdm(test_loader, desc="Evaluating")
    for batch in pbar:
        mix_mag = batch["mix_mag"].to(device)
        mix_phase = batch["mix_phase"]
        mix_mag_raw = batch["mix_mag_raw"]
        vocals_mag_target = batch["vocals_mag"]

        mask = model(mix_mag).cpu()

        for i in range(mask.shape[0]):
            pred_mag = mask[i, 0] * mix_mag_raw[i]
            pred_wav = stft_transform.inverse(pred_mag, mix_phase[i])

            target_mag = torch.expm1(vocals_mag_target[i, 0])
            target_wav = stft_transform.inverse(target_mag, mix_phase[i])

            pred_np = pred_wav.numpy()
            target_np = target_wav.numpy()

            try:
                all_results["SDR"].append(_compute_sdr(target_np, pred_np))
                all_results["SIR"].append(_compute_sir(target_np, pred_np))
                all_results["SAR"].append(_compute_sar(target_np, pred_np))
            except Exception:
                continue

        if all_results["SDR"]:
            pbar.set_postfix(SDR=f"{np.mean(all_results['SDR']):.2f}")

    summary = {
        metric: {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
        }
        for metric, vals in all_results.items()
        if vals
    }
    return summary
