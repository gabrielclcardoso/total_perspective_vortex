"""CLI script for visualization

Loads raw PhysioNet recordings, applies zero-phase FIR filtering,
extracts active trials, and delegates rendering to `tpv.plots`.

Usage:
    python -m tpv.visualize
    python -m tpv.visualize --subject 1 --run 4 --no-show
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from tpv.config import (
    H_FREQ,
    L_FREQ,
    PLOTS_DIR,
    TASK_DESCRIPTIONS,
)
from tpv.plots import (
    plot_frequency_domain,
    plot_sensorimotor_contrast,
    plot_task_topomaps,
    plot_time_domain,
)
from tpv.preprocessing import (
    extract_epochs,
    filter_raw,
    load_raw,
)


def main(argv=None):
    # Parse arguments
    args = parse_arguments(argv)
    if not args.show:
        matplotlib.use("Agg")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("=== Total Perspective Vortex — EEG Visualization (Section V.1.1) ===")
    print(
        f"Loading Subject S{args.subject:03d}, Run R{args.run:02d}: "
        f"{TASK_DESCRIPTIONS.get(args.run, 'Unknown')}"
    )

    # Load data
    raw = load_raw(args.subject, args.run)
    print(
        f"✓ Raw EEG loaded: {len(raw.ch_names)} channels, "
        f"{raw.times[-1]:.1f}s @ {raw.info['sfreq']:.0f} Hz"
    )

    print(
        f"Applying zero-phase FIR bandpass filter ({L_FREQ:.1f} - {H_FREQ:.1f} Hz)..."
    )
    raw_filt = filter_raw(raw, l_freq=L_FREQ, h_freq=H_FREQ)
    print("✓ Filtering complete")

    print("Extracting 4.0s trial epochs for active cues (T1 vs. T2)...")
    epochs = extract_epochs(raw_filt)
    n_t1 = len(epochs["T1"]) if "T1" in epochs.event_id else 0
    n_t2 = len(epochs["T2"]) if "T2" in epochs.event_id else 0
    print(
        f"✓ Epoch extraction complete: {len(epochs)} total trials (T1: {n_t1}, T2: {n_t2})"
    )

    # Generate plots
    time_path = args.out_dir / "raw_vs_filtered_time.png"
    print(f"Generating Time-Domain Comparison -> {time_path.name}")
    plot_time_domain(raw, raw_filt, picks=args.channels, out_path=time_path)

    psd_path = args.out_dir / "raw_vs_filtered_psd.png"
    print(f"Generating Frequency-Domain Comparison -> {psd_path.name}")
    plot_frequency_domain(raw, raw_filt, picks=args.channels, out_path=psd_path)

    topo_path = args.out_dir / "task_topomaps.png"
    print(f"Generating Task Scalp Topographies -> {topo_path.name}")
    plot_task_topomaps(epochs, out_path=topo_path)

    contrast_path = args.out_dir / "sensorimotor_c3_c4_contrast.png"
    print(f"Generating Sensorimotor Contrast Spectra -> {contrast_path.name}")
    plot_sensorimotor_contrast(epochs, out_path=contrast_path)

    print(f"✓ All figures successfully saved in: {args.out_dir.resolve()}")

    if args.show:
        plt.show()


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        prog="tpv.visualize",
        description="Visualize EEG preprocessing, filtering, spectra, and topographies.",
    )
    parser.add_argument("--subject", type=int, default=1, help="Subject ID (1 - 109)")
    parser.add_argument("--run", type=int, default=4, help="Run number (3 - 14)")
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["C3", "Cz", "C4"],
        help="Channels to compare in time and frequency domains",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PLOTS_DIR,
        help="Output directory to save figures",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display figures in interactive GUI window",
    )
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    main()
