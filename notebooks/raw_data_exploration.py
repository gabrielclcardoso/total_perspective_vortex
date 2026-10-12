import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Total Perspective Vortex - Raw EEG Data Exploration
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Subject Selection
    """)
    return


@app.cell
def _():
    import marimo as mo
    from tpv.config import DATA_DIR, ROOT_DIR

    return DATA_DIR, mo


@app.cell
def _(mo):
    subject_ui = mo.ui.number(start=1, stop=109, step=1, value=1, label="Subject")
    run_ui = mo.ui.dropdown(
        options=[str(r) for r in range(1, 15)],
        value="4",
        label="Run (Task)",
    )
    mo.hstack([subject_ui, run_ui], justify="start")
    return run_ui, subject_ui


@app.cell
def _(DATA_DIR, mo, run_ui, subject_ui):
    mo.md(f"""
    **Selected Configuration:**
    - **Subject:** S{subject_ui.value:03d}
    - **Run:** R{int(run_ui.value):02d}
    - **Data Directory:** `{DATA_DIR}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data visualization
    """)
    return


@app.cell
def _():
    import mne
    import mne.datasets.eegbci as eegbci
    import matplotlib.pyplot as plt
    from tpv.download import download_run

    return download_run, eegbci, mne, plt


@app.cell
def _(DATA_DIR, download_run, eegbci, mne, mo, run_ui, subject_ui):
    sub = int(subject_ui.value)
    run = int(run_ui.value)

    # Check if the file is already downloaded to local cache
    cached_file = (
        DATA_DIR
        / "MNE-eegbci-data"
        / "files"
        / "eegmmidb"
        / "1.0.0"
        / f"S{sub:03d}"
        / f"S{sub:03d}R{run:02d}.edf"
    )
    is_cached = cached_file.exists()

    status_title = (
        f"Loading cached S{sub:03d}R{run:02d}..."
        if is_cached
        else f"Downloading S{sub:03d}R{run:02d} from AWS S3 mirror (~2.5 MB)..."
    )

    with mo.status.spinner(title=status_title):
        download_run(sub, run)
        files = eegbci.load_data(
            subjects=sub,
            runs=[run],
            path=DATA_DIR,
            update_path=False,
        )
        # Parse raw EDF
        raw = mne.io.read_raw_edf(files[0], preload=True)
        eegbci.standardize(raw)
        montage = mne.channels.make_standard_montage('colin27_1005')
        raw.set_montage(montage)

    mo.output.replace(
        mo.md(
            f"✓ **Loaded S{sub:03d}R{run:02d}.edf** "
            f"({'from local cache' if is_cached else 'downloaded from AWS S3 mirror'}) | "
            f"**{raw.times[-1]:.1f}s** recording, **{len(raw.ch_names)} channels** @ **{raw.info['sfreq']:.0f} Hz**"
        )
    )
    return (raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. Electrode Layout (Topomap)
    """)
    return


@app.cell
def _(raw):
    # Plot sensor locations on head model
    fig_sensors = raw.plot_sensors(show_names=True, kind='topomap')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Raw EEG Waveforms (Time Domain)
    Inspect continuous voltage signals over time. Event markers are overlaid:
    - **`T0`** (gray): Rest
    - **`T1`** (green): Left fist / Both fists
    - **`T2`** (orange): Right fist / Both feet
    """)
    return


@app.cell
def _(mo, raw):
    max_time = max(0, int(raw.times[-1] - 5))
    start_time_ui = mo.ui.slider(start=0, stop=max_time, step=1, value=0, label="Start Time (s)")
    duration_ui = mo.ui.slider(start=2, stop=50, step=1, value=10, label="Window (s)")

    default_picks = [ch for ch in ["C3", "Cz", "C4"] if ch in raw.ch_names]
    channel_ui = mo.ui.multiselect(
        options=raw.ch_names,
        value=default_picks,
        label="Select Channels",
    )
    mo.vstack([
        mo.hstack([start_time_ui, duration_ui], justify="start"),
        channel_ui,
    ])
    return channel_ui, duration_ui, start_time_ui


@app.cell
def _(channel_ui, duration_ui, plt, raw, start_time_ui):
    picks = channel_ui.value
    t_start = float(start_time_ui.value)
    t_end = min(t_start + float(duration_ui.value), raw.times[-1])

    start_idx = raw.time_as_index(t_start)[0]
    stop_idx = raw.time_as_index(t_end)[0]
    data_slice, times_slice = raw[picks, start_idx:stop_idx]

    fig_traces, axes = plt.subplots(len(picks), 1, figsize=(10, 1.8 * len(picks)), sharex=True)
    if len(picks) == 1:
        axes = [axes]

    # Plot each selected channel
    for i, ch_name in enumerate(picks):
        axes[i].plot(times_slice, data_slice[i] * 1e6, color="#1f77b4", lw=0.9)
        axes[i].set_ylabel(f"{ch_name}\n(µV)", rotation=0, labelpad=25, va="center")
        axes[i].grid(True, linestyle=":", alpha=0.6)

    # Highlight trial event annotations within visible window
    event_colors = {"T0": "#7f7f7f", "T1": "#2ca02c", "T2": "#ff7f0e"}
    for annot in raw.annotations:
        onset = annot["onset"]
        dur = annot["duration"]
        desc = annot["description"]
        if (onset + dur >= t_start) and (onset <= t_end):
            col = event_colors.get(desc, "#9467bd")
            for ax in axes:
                ax.axvspan(max(onset, t_start), min(onset + dur, t_end), color=col, alpha=0.15)
                ax.axvline(onset, color=col, linestyle="--", alpha=0.8, lw=1.2)
                y_max = ax.get_ylim()[1]
                ax.text(onset + 0.05, y_max * 0.75, desc, color=col, fontweight="bold", fontsize=9)

    axes[-1].set_xlabel("Time (seconds)")
    fig_traces.suptitle(f"Raw Signals: {', '.join(picks)} ({t_start:.1f}s – {t_end:.1f}s)", fontsize=11)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
