"""Plotting library for EEG signals, spectra, topographies, and contrast analysis.

Conforms to Section V.1.1 requirements:
- Time-domain waveforms comparing raw and filtered traces with task cue spans.
- Frequency-domain Welch Power Spectral Density comparing raw and filtered passbands.
- 2D scalp topography head maps across mu (8-12 Hz) and beta (13-30 Hz) bands.
- Sensorimotor contrast spectra on C3 and C4 demonstrating contralateral ERD.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.figure import Figure
from mne.viz import plot_topomap

from tpv.config import H_FREQ, L_FREQ


def plot_time_domain(
    raw: mne.io.BaseRaw,
    raw_filt: mne.io.BaseRaw,
    picks: Sequence[str] = ("C3", "Cz", "C4"),
    t_start: float = 0.0,
    duration: float = 12.0,
    out_path: Path | None = None,
) -> Figure:
    """Plot side-by-side time-domain comparison of raw vs. filtered signals."""
    valid_picks = [ch for ch in picks if ch in raw.ch_names]
    if not valid_picks:
        valid_picks = raw.ch_names[:3]

    t_end = min(t_start + duration, raw.times[-1])
    s_idx = raw.time_as_index(t_start)[0]
    e_idx = raw.time_as_index(t_end)[0]

    raw_data, times_slice = raw[valid_picks, s_idx:e_idx]
    filt_data, _ = raw_filt[valid_picks, s_idx:e_idx]

    n_ch = len(valid_picks)
    fig, axes = plt.subplots(n_ch, 2, figsize=(14, max(3.5, 2.3 * n_ch)), sharex=True)
    if n_ch == 1:
        axes = np.array([axes])

    event_colors = {"T0": "#7f7f7f", "T1": "#2ca02c", "T2": "#ff7f0e"}

    for i, ch in enumerate(valid_picks):
        # Raw trace (µV)
        axes[i, 0].plot(times_slice, raw_data[i] * 1e6, color="#1f77b4", lw=0.9)
        axes[i, 0].set_ylabel(f"{ch} (µV)", fontweight="bold")
        axes[i, 0].grid(True, linestyle=":", alpha=0.6)
        if i == 0:
            axes[i, 0].set_title(
                "Raw Signal (Unfiltered)", fontsize=11, fontweight="bold"
            )

        # Filtered trace (µV)
        axes[i, 1].plot(times_slice, filt_data[i] * 1e6, color="#d62728", lw=0.9)
        axes[i, 1].grid(True, linestyle=":", alpha=0.6)
        if i == 0:
            axes[i, 1].set_title(
                f"Filtered Signal ({L_FREQ:.0f}–{H_FREQ:.0f} Hz)",
                fontsize=11,
                fontweight="bold",
            )

        # Highlight event intervals
        for ax in (axes[i, 0], axes[i, 1]):
            for onset, dur, desc in zip(
                raw.annotations.onset,
                raw.annotations.duration,
                raw.annotations.description,
                strict=True,
            ):
                if (onset + dur >= t_start) and (onset <= t_end):
                    col = event_colors.get(desc, "#9467bd")
                    ax.axvspan(
                        max(float(onset), t_start),
                        min(float(onset + dur), t_end),
                        color=col,
                        alpha=0.15,
                    )
                    ax.axvline(
                        float(onset), color=col, linestyle="--", alpha=0.7, lw=1.0
                    )
                    ylims = ax.get_ylim()
                    ax.text(
                        float(onset) + 0.05,
                        ylims[1] * 0.75,
                        desc,
                        color=col,
                        fontweight="bold",
                        fontsize=8,
                    )

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Time-Domain Waveforms (Raw vs. Zero-Phase FIR Bandpass [{t_start:.1f}s – {t_end:.1f}s])",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_frequency_domain(
    raw: mne.io.BaseRaw,
    raw_filt: mne.io.BaseRaw,
    picks: Sequence[str] = ("C3", "Cz", "C4"),
    out_path: Path | None = None,
) -> Figure:
    """Plot Power Spectral Density (PSD) using MNE's compute_psd (Welch method)."""
    valid_picks = [ch for ch in picks if ch in raw.ch_names]
    if not valid_picks:
        valid_picks = raw.ch_names[:3]

    sfreq = raw.info["sfreq"]
    spec_raw = (
        raw.copy()
        .pick(valid_picks)
        .compute_psd(
            method="welch",
            fmin=0.0,
            fmax=sfreq / 2,
            n_fft=256,
            verbose=False,
        )
    )
    spec_filt = (
        raw_filt.copy()
        .pick(valid_picks)
        .compute_psd(
            method="welch",
            fmin=0.0,
            fmax=sfreq / 2,
            n_fft=256,
            verbose=False,
        )
    )

    psd_raw, freqs = spec_raw.get_data(return_freqs=True)
    psd_filt, _ = spec_filt.get_data(return_freqs=True)

    n_ch = len(valid_picks)
    fig, axes = plt.subplots(1, n_ch, figsize=(max(6.0, 5.0 * n_ch), 4.5), sharey=True)
    if n_ch == 1:
        axes = [axes]

    for i, ch in enumerate(valid_picks):
        ax = axes[i]
        p_raw_db = 10 * np.log10(np.maximum(psd_raw[i] * 1e12, 1e-12))
        p_filt_db = 10 * np.log10(np.maximum(psd_filt[i] * 1e12, 1e-12))

        ax.plot(
            freqs,
            p_raw_db,
            label="Raw (Unfiltered)",
            color="#1f77b4",
            lw=1.5,
            alpha=0.75,
        )
        ax.plot(
            freqs,
            p_filt_db,
            label=f"Filtered ({L_FREQ:.0f}–{H_FREQ:.0f} Hz)",
            color="#d62728",
            lw=1.8,
        )

        ax.axvspan(
            L_FREQ,
            H_FREQ,
            color="gold",
            alpha=0.15,
            label=f"Passband [{L_FREQ:.0f}–{H_FREQ:.0f} Hz]",
        )
        ax.axvspan(8.0, 12.0, color="green", alpha=0.10, label="Mu (8–12 Hz)")
        ax.axvspan(13.0, 30.0, color="orange", alpha=0.08, label="Beta (13–30 Hz)")

        ax.set_title(f"Channel {ch} - Power Spectrum", fontweight="bold", fontsize=11)
        ax.set_xlabel("Frequency (Hz)")
        if i == 0:
            ax.set_ylabel("PSD (dB / µV²/Hz)")
        ax.set_xlim(0, sfreq / 2)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Frequency Domain: Welch Power Spectral Density (Raw vs. Filtered)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_task_topomaps(
    epochs: mne.BaseEpochs,
    out_path: Path | None = None,
) -> Figure:
    """Plot scalp topographies for Task 1 vs. Task 2 across Mu and Beta bands."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    spec_t1 = epochs["T1"].compute_psd(
        method="welch", fmin=0.5, fmax=45.0, n_fft=256, verbose=False
    )
    spec_t2 = epochs["T2"].compute_psd(
        method="welch", fmin=0.5, fmax=45.0, n_fft=256, verbose=False
    )

    psds_t1, freqs = spec_t1.get_data(return_freqs=True)
    psds_t2, _ = spec_t2.get_data(return_freqs=True)

    mean_t1 = np.mean(psds_t1, axis=0)
    mean_t2 = np.mean(psds_t2, axis=0)

    bands = [
        ("Mu (8–12 Hz)", (8.0, 12.0), axes[0]),
        ("Beta (13–30 Hz)", (13.0, 30.0), axes[1]),
    ]

    for band_name, (fmin, fmax), (ax1, ax2) in bands:
        mask = (freqs >= fmin) & (freqs <= fmax)
        pow_t1 = np.mean(mean_t1[:, mask], axis=-1)
        pow_t2 = np.mean(mean_t2[:, mask], axis=-1)

        pow_t1_db = 10 * np.log10(np.maximum(pow_t1 * 1e12, 1e-12))
        pow_t2_db = 10 * np.log10(np.maximum(pow_t2 * 1e12, 1e-12))

        # Shared limits for fair visual comparison
        vmin = min(pow_t1_db.min(), pow_t2_db.min())
        vmax = max(pow_t1_db.max(), pow_t2_db.max())

        im1, _ = plot_topomap(
            pow_t1_db,
            epochs.info,
            axes=ax1,
            cmap="viridis",
            vlim=(vmin, vmax),
            show=False,
        )
        ax1.set_title(f"Task 1 (T1) — {band_name}", fontsize=11, fontweight="bold")
        fig.colorbar(
            im1,
            ax=ax1,
            orientation="horizontal",
            pad=0.08,
            shrink=0.7,
            label="Power (dB / µV²/Hz)",
        )

        im2, _ = plot_topomap(
            pow_t2_db,
            epochs.info,
            axes=ax2,
            cmap="viridis",
            vlim=(vmin, vmax),
            show=False,
        )
        ax2.set_title(f"Task 2 (T2) — {band_name}", fontsize=11, fontweight="bold")
        fig.colorbar(
            im2,
            ax=ax2,
            orientation="horizontal",
            pad=0.08,
            shrink=0.7,
            label="Power (dB / µV²/Hz)",
        )

    fig.suptitle(
        "Task-Specific Scalp Topography on Shared Scales (T1 vs. T2)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_sensorimotor_contrast(
    epochs: mne.BaseEpochs,
    out_path: Path | None = None,
) -> Figure:
    """Plot C3 vs. C4 sensorimotor spectra for T1 vs. T2 trials."""
    fig, (ax_c3, ax_c4) = plt.subplots(1, 2, figsize=(11, 4.5))

    c3_idx = epochs.ch_names.index("C3")
    c4_idx = epochs.ch_names.index("C4")

    spec_t1 = epochs["T1"].compute_psd(
        method="welch", fmin=4.0, fmax=35.0, n_fft=256, verbose=False
    )
    psd_t1, f_erd = spec_t1.get_data(return_freqs=True)
    mean_t1 = np.mean(psd_t1, axis=0)

    spec_t2 = epochs["T2"].compute_psd(
        method="welch", fmin=4.0, fmax=35.0, n_fft=256, verbose=False
    )
    psd_t2, _ = spec_t2.get_data(return_freqs=True)
    mean_t2 = np.mean(psd_t2, axis=0)

    # C3 plot
    ax_c3.plot(
        f_erd, 10 * np.log10(mean_t1[c3_idx] * 1e12), label="T1", color="#2ca02c", lw=2
    )
    ax_c3.plot(
        f_erd, 10 * np.log10(mean_t2[c3_idx] * 1e12), label="T2", color="#ff7f0e", lw=2
    )
    ax_c3.axvspan(8, 12, color="gray", alpha=0.15, label="Mu band (8-12 Hz)")
    ax_c3.axvspan(13, 30, color="gray", alpha=0.08, label="Beta band (13-30 Hz)")
    ax_c3.set_title("Electrode C3 (Left Motor Cortex)", fontweight="bold")
    ax_c3.set_xlabel("Frequency (Hz)")
    ax_c3.set_ylabel("PSD (dB / µV²/Hz)")
    ax_c3.grid(True, linestyle=":", alpha=0.6)
    ax_c3.legend(fontsize=8)

    # C4 plot
    ax_c4.plot(
        f_erd, 10 * np.log10(mean_t1[c4_idx] * 1e12), label="T1", color="#2ca02c", lw=2
    )
    ax_c4.plot(
        f_erd, 10 * np.log10(mean_t2[c4_idx] * 1e12), label="T2", color="#ff7f0e", lw=2
    )
    ax_c4.axvspan(8, 12, color="gray", alpha=0.15, label="Mu band (8-12 Hz)")
    ax_c4.axvspan(13, 30, color="gray", alpha=0.08, label="Beta band (13-30 Hz)")
    ax_c4.set_title("Electrode C4 (Right Motor Cortex)", fontweight="bold")
    ax_c4.set_xlabel("Frequency (Hz)")
    ax_c4.grid(True, linestyle=":", alpha=0.6)
    ax_c4.legend(fontsize=8)

    fig.suptitle(
        "Sensorimotor Contrast Spectra (C3 vs. C4 during T1 and T2)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig
