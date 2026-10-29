"""
Data / domain-shift diagnostics (port of models/diagnostics.py).
Every function saves a PNG (and returns its path); pass show=True for interactive use.
Hard-coded paths and the 250 Hz assumption are gone: pass fs and checkpoint paths explicitly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis

from scorer.data.recordings import load_array, to_windows_channels_time
from scorer.data.signal import robust_normalize


def _plt():
    """Import pyplot with the Agg backend unless a backend is already set."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def _finish(fig, path, show):
    """Lay out ``fig``, save it to ``path`` (if given), show it if asked, close it, and return ``path``."""
    plt = _plt()
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_embeddings(x_good: np.ndarray, x_bad: np.ndarray, encoder_path, path=None, device="cpu", show=False):
    """Plot a 2-D PCA of encoder embeddings for a reference and an out-of-domain window set.

    Args:
        x_good: Normalised reference windows [N, C, T].
        x_bad: Normalised out-of-domain windows [N, C, T].
        encoder_path: Encoder checkpoint.
        path: PNG to write, or None.
        device: Device for the encoder.
        show: Also open an interactive window.
    """
    import torch
    from sklearn.decomposition import PCA

    from scorer.models.checkpoints import load_encoder

    enc, _ = load_encoder(encoder_path, device)
    with torch.inference_mode():
        g = enc(torch.as_tensor(x_good, dtype=torch.float32, device=device)).cpu().numpy()
        b = enc(torch.as_tensor(x_bad, dtype=torch.float32, device=device)).cpu().numpy()
    xy = PCA(n_components=2).fit_transform(np.vstack([g, b]))
    fig, ax = _plt().subplots(figsize=(7, 7))
    ax.scatter(xy[: len(g), 0], xy[: len(g), 1], s=8, alpha=0.5, label="reference")
    ax.scatter(xy[len(g):, 0], xy[len(g):, 1], s=8, alpha=0.5, label="OOD")
    ax.legend()
    ax.set_title(f"latent space: {Path(encoder_path).name}")
    return _finish(fig, path, show)


def plot_domain_diagnostics(clean: np.ndarray, ood: np.ndarray, fs: float, path=None, show=False):
    """Plot amplitude histograms and Welch power spectra of a reference and an OOD signal (1-D)."""
    plt = _plt()
    clean, ood = np.ravel(clean), np.ravel(ood)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(-8, 8, 200)
    axes[0].hist(clean, bins=bins, density=True, histtype="step", label="reference")
    axes[0].hist(ood, bins=bins, density=True, histtype="step", label="OOD")
    axes[0].set_title("amplitude distribution")
    axes[0].legend()
    for sig, name in ((clean, "reference"), (ood, "OOD")):
        f, p = welch(sig, fs=fs, nperseg=int(fs * 2))
        axes[1].semilogy(f, p, label=name)
    axes[1].set_xlim(0, fs / 2)
    axes[1].set_xlabel("Hz")
    axes[1].set_title("power spectral density")
    axes[1].legend()
    return _finish(fig, path, show)


def check_tensors(raw: np.ndarray, processed: np.ndarray, path=None, show=False):
    """Plot per-window SD of raw vs processed windows [N, C, T], plus the loudest and quietest window."""
    plt = _plt()
    raw_sd, proc_sd = raw[:, 0].std(-1), processed[:, 0].std(-1)
    worst, best = int(np.argmax(raw_sd)), int(np.argmin(raw_sd))
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    axes[0].plot(raw_sd, color="gray", alpha=0.6, label="raw window SD")
    axes[0].plot(proc_sd, color="red", label="processed window SD")
    axes[0].set_yscale("log")
    axes[0].legend()
    for ax, i, name in ((axes[1], worst, "loudest"), (axes[2], best, "quietest")):
        ax.plot(raw[i, 0], color="gray", label="raw")
        ax.plot(processed[i, 0], color="red", label="processed")
        ax.set_title(f"{name} window {i}")
        ax.legend()
    return _finish(fig, path, show)


def theta_delta_ratio(x: np.ndarray, fs: float) -> np.ndarray:
    """Return per-window theta (6-9 Hz) / delta (1-4 Hz) power for windows ``x`` [N, T]."""
    f, p = welch(x, fs=fs, nperseg=min(x.shape[-1], int(fs)), axis=-1)
    d = p[..., (f >= 1) & (f <= 4)].sum(-1)
    t = p[..., (f >= 6) & (f <= 9)].sum(-1)
    return t / (d + 1e-12)


def compare_ood(ood_file, reference_file, fs: float, out_dir, encoder_paths=(), lowpass_hz: float | None = 49.0,
                channel: int = 0, device="cpu", show=False) -> dict:
    """Compare an out-of-domain recording with a reference recording.

    Both raw X files are normalised the way ``prepare`` does it. Writes PCA plots (one per
    encoder), window-SD, amplitude/PSD and theta/delta figures to ``out_dir``.

    Returns:
        ``{"stats": clipping %, kurtosis and median theta/delta for both, "files": paths}``.
    """
    out_dir = Path(out_dir)
    load = lambda f: to_windows_channels_time(load_array(Path(f)), f)[:, channel:channel + 1].astype(np.float32)  # noqa
    ref_raw, ood_raw = load(reference_file), load(ood_file)
    ref, _ = robust_normalize(ref_raw, fs, lowpass_hz)
    ood, _ = robust_normalize(ood_raw, fs, lowpass_hz)
    files = {}
    for p in encoder_paths:
        files[f"pca_{Path(p).stem}"] = plot_embeddings(ref[:1000], ood[:1000], p, out_dir / f"pca_{Path(p).stem}.png",
                                                       device, show)
    files["tensors"] = check_tensors(ood_raw, ood, out_dir / "ood_window_sd.png", show)
    files["domain"] = plot_domain_diagnostics(ref[:, 0].ravel(), ood[:, 0].ravel(), fs, out_dir / "domain.png", show)
    zr = (ref_raw.ravel() - ref_raw.mean()) / ref_raw.std()
    zo = (ood_raw.ravel() - ood_raw.mean()) / ood_raw.std()
    stats = {
        "reference_clip_pct": float(np.mean(zr == zr.max()) * 100), "reference_kurtosis": float(kurtosis(zr)),
        "ood_clip_pct": float(np.mean(zo == zo.max()) * 100), "ood_kurtosis": float(kurtosis(zo)),
        "reference_theta_delta_median": float(np.median(theta_delta_ratio(ref_raw[:, 0], fs))),
        "ood_theta_delta_median": float(np.median(theta_delta_ratio(ood_raw[:, 0], fs))),
    }
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(theta_delta_ratio(ref_raw[:, 0], fs), bins=60, alpha=0.5, density=True, label="reference")
    ax.hist(theta_delta_ratio(ood_raw[:, 0], fs), bins=60, alpha=0.5, density=True, label="OOD")
    ax.set_title("theta / delta ratio per window")
    ax.legend()
    files["tdr"] = _finish(fig, out_dir / "theta_delta.png", show)
    return {"stats": stats, "files": {k: str(v) for k, v in files.items()}}


def plot_trace(cache_recording, scheme, index: int | None = None, path=None, show=False, seed: int = 0):
    """Plot one prepared window with its label; a quick check that X and y line up.

    Args:
        cache_recording: A ``CachedRecording``.
        scheme: Label scheme for the title.
        index: Window to show; None picks a random one (seeded).
        path: PNG to write, or None.
        show: Also open an interactive window.
        seed: Seed for the random pick.
    """
    rng = np.random.default_rng(seed)
    x = cache_recording.x()
    cls = cache_recording.classes(scheme)
    i = int(rng.integers(0, len(cls))) if index is None else index
    fig, ax = _plt().subplots(figsize=(8, 2))
    ax.plot(np.asarray(x[i, 0], dtype=np.float32))
    name = scheme.class_names[cls[i]] if cls[i] >= 0 else "unlabeled"
    ax.set_title(f"{cache_recording.dataset}/{cache_recording.name} window {i}: {name}")
    return _finish(fig, path, show)
