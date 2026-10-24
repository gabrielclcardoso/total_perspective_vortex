import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Total Perspective Vortex — EEG Preprocessing & Signal Exploration

    This notebook covers **Phase 1: Preprocessing, Parsing and Formatting (Section V.1.1)** of the 42 BCI project.

    Raw, noisy 64-channel brain recordings from the PhysioNet EEGBCI dataset are taken and prepared them for machine learning:

    1. **Electrode Layout:** Mapping the 64 sensors on the scalp using the international 10–10 system (`colin27_1005` montage).
    2. **Zero-Phase Filtering (8.0 – 30.0 Hz):** Isolating the movement rhythms ($\mu$: 8–12 Hz and $\beta$: 13–30 Hz) while discarding slow eye blinks ($<8\text{ Hz}$) and muscle/powerline noise ($>30\text{ Hz}$).
    3. **Time-Domain Waveforms:** Side-by-side comparison of continuous voltage traces before and after filtering, overlaid with rest (`T0`) and movement cues (`T1`, `T2`).
    4. **Frequency-Domain Spectrum (PSD):** Computing Power Spectral Density using Welch's averaged periodogram to verify filter cutoffs.
    5. **Epoch Extraction (Trial Segmentation):** Slicing continuous 2-minute streams into discrete, labeled 4-second trials around movement cues for classifier training.
    6. **Task-Specific Scalp Topography:** Mapping 2D head maps of $\mu$ and $\beta$ power during Task 1 vs. Task 2 on a shared scale, demonstrating that visual differences are too subtle to detect with the naked eye.
    7. **Left vs. Right Brain Response ($C3$ vs. $C4$ Spectra):** Inspecting the primary motor sensors to examine contralateral power drops, proving why machine learning (Phase 2: CSP) is necessary to decode movement thoughts.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import mne
    import numpy as np
    from mne.datasets import eegbci
    from mne.viz import plot_topomap

    from tpv.config import DATA_DIR
    from tpv.download import download_run

    return DATA_DIR, download_run, eegbci, mne, mo, np, plot_topomap, plt


@app.cell
def _():
    TASK_DESCRIPTIONS = {
        1: "Baseline, eyes open (60s)",
        2: "Baseline, eyes closed (60s)",
        3: "Task 1 (Real): Open/close left or right fist (T1=Left, T2=Right)",
        4: "Task 2 (Imagined): Imagine opening/closing left or right fist (T1=Left, T2=Right)",
        5: "Task 3 (Real): Open/close both fists or both feet (T1=Both Fists, T2=Both Feet)",
        6: "Task 4 (Imagined): Imagine opening/closing both fists or both feet (T1=Both Fists, T2=Both Feet)",
        7: "Task 1 (Real): Open/close left or right fist (T1=Left, T2=Right)",
        8: "Task 2 (Imagined): Imagine opening/closing left or right fist (T1=Left, T2=Right)",
        9: "Task 3 (Real): Open/close both fists or both feet (T1=Both Fists, T2=Both Feet)",
        10: "Task 4 (Imagined): Imagine opening/closing both fists or both feet (T1=Both Fists, T2=Both Feet)",
        11: "Task 1 (Real): Open/close left or right fist (T1=Left, T2=Right)",
        12: "Task 2 (Imagined): Imagine opening/closing left or right fist (T1=Left, T2=Right)",
        13: "Task 3 (Real): Open/close both fists or both feet (T1=Both Fists, T2=Both Feet)",
        14: "Task 4 (Imagined): Imagine opening/closing both fists or both feet (T1=Both Fists, T2=Both Feet)",
    }
    return (TASK_DESCRIPTIONS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Subject & Task Selection
    Select any of the 109 PhysioNet subjects and one of the 14 experimental runs.
    """)
    return


@app.cell
def _(TASK_DESCRIPTIONS, mo):
    _options = {f"Run {r:02d} — {TASK_DESCRIPTIONS[r]}": r for r in range(1, 15)}
    subject_ui = mo.ui.number(start=1, stop=109, step=1, value=1, label="Subject ID")
    run_ui = mo.ui.dropdown(
        options=_options,
        value=list(_options.keys())[
            3
        ],  # Default to Run 4 (Task 2: Imagined Left/Right Fist)
        label="Run / Task",
    )
    mo.hstack([subject_ui, run_ui], justify="start", gap=2)
    return run_ui, subject_ui


@app.cell
def _(DATA_DIR, TASK_DESCRIPTIONS, mo, run_ui, subject_ui):
    sub_num = int(subject_ui.value)
    run_num = int(run_ui.value)
    mo.md(f"""
    **Current Configuration:**
    - **Subject:** `S{sub_num:03d}`
    - **Run:** `R{run_num:02d}`: *{TASK_DESCRIPTIONS[run_num]}*
    - **Storage Path:** `{DATA_DIR}`
    """)
    return run_num, sub_num


@app.cell
def _(DATA_DIR, download_run, eegbci, mne, mo, run_num, sub_num):
    _cached_file = (
        DATA_DIR
        / "MNE-eegbci-data"
        / "files"
        / "eegmmidb"
        / "1.0.0"
        / f"S{sub_num:03d}"
        / f"S{sub_num:03d}R{run_num:02d}.edf"
    )
    _is_cached = _cached_file.exists()

    _status_title = (
        f"Loading cached S{sub_num:03d}R{run_num:02d}..."
        if _is_cached
        else f"Downloading S{sub_num:03d}R{run_num:02d} from AWS S3 mirror (~2.5 MB)..."
    )

    with mo.status.spinner(title=_status_title):
        download_run(sub_num, run_num)
        _files = eegbci.load_data(
            subjects=sub_num,
            runs=[run_num],
            path=DATA_DIR,
            update_path=False,
        )
        raw = mne.io.read_raw_edf(_files[0], preload=True)
        eegbci.standardize(raw)
        _montage = mne.channels.make_standard_montage("colin27_1005")
        raw.set_montage(_montage)

    mo.output.replace(
        mo.md(
            f"✓ **Loaded S{sub_num:03d}R{run_num:02d}.edf** "
            f"({'cached locally' if _is_cached else 'downloaded from S3'}) | "
            f"**{raw.times[-1]:.1f}s** recording, **{len(raw.ch_names)} channels** @ **{raw.info['sfreq']:.0f} Hz**"
        )
    )
    return (raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. Electrode Montage Layout
    The recording uses 64 electrodes under the international 10-10 system.
    Key regions of interest for motor cortex decoding:
    - **C3**: Left sensorimotor cortex (contralateral to right hand)
    - **C4**: Right sensorimotor cortex (contralateral to left hand)
    - **Cz**: Midline sensorimotor cortex (contralateral to feet)
    - Surrounding motor references: **FC3, FC4, CP3, CP4**
    """)
    return


@app.cell
def _(raw):
    raw.plot_sensors(show_names=True, kind="topomap", show=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. Bandpass Filtering (8.0 – 30.0 Hz)

    ### Why Filter the Signal?
    Raw brain signals are extremely faint, and the sensors easily pick up unwanted electrical noise from the body and the room. A **8.0 to 30.0 Hz bandpass filter** is applied to keep only the brain waves related to movement while discarding the rest:

    #### 1. What is Kept (8 – 30 Hz: The "Motor" Brain Waves)
    * **The "Idling" Rhythm:** When the body is resting, the movement centers of the brain produce steady, rhythmic electrical waves between **8 and 30 Hz** (called the *Mu* and *Beta* rhythms).
    * **What Happens During Movement:** When it moves or imagines moving the hand, the opposite side of the brain goes to work. This activity disrupts the steady rhythm, causing the wave's power to drop.

    #### 2. What is Removed (Unwanted Noise)
    * **Below 8 Hz (Eye & Movement Noise):** Eye blinks, eye movements, and sweat create massive, slow electrical waves (1–4 Hz) that easily drown out the faint brain signal. Cutting everything below 8 Hz removes this distortion.
    * **Above 30 Hz (Muscle & Wall Power Noise):** Clenching the jaw or tensing neck muscles produces fast electrical vibrations (30+ Hz). In addition, electrical cords in the room emit a constant 50/60 Hz hum. Cutting everything above 30 Hz eliminates both.

    > **In short:** The 8–30 Hz filter keeps the brain waves that reflect motor thoughts and throws away eye blinks, muscle twitches, and electrical hum.
    """)
    return


@app.cell
def _(mo):
    l_freq_ui = mo.ui.slider(
        start=0.5, stop=20.0, step=0.5, value=8.0, label="Low Cutoff (Hz)"
    )
    h_freq_ui = mo.ui.slider(
        start=15.0, stop=60.0, step=1.0, value=30.0, label="High Cutoff (Hz)"
    )
    fir_window_ui = mo.ui.dropdown(
        options=["hamming", "hann", "blackman"],
        value="hamming",
        label="FIR Window",
    )
    mo.hstack([l_freq_ui, h_freq_ui, fir_window_ui], justify="start", gap=2)
    return fir_window_ui, h_freq_ui, l_freq_ui


@app.cell
def _(fir_window_ui, h_freq_ui, l_freq_ui, raw):
    l_freq = float(l_freq_ui.value)
    h_freq = float(h_freq_ui.value)
    fir_window = str(fir_window_ui.value)

    # Filter a copy of raw to preserve raw for comparison
    raw_filtered = raw.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design="firwin",
        fir_window=fir_window,
        skip_by_annotation="edge",
        verbose=False,
    )
    return h_freq, l_freq, raw_filtered


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. Visual Comparison: Time-Domain Waveforms
    Compare continuous voltage traces **before** and **after** filtering for selected channels.
    Event markers:
    - **`T0`** (gray): Rest interval
    - **`T1`** (green): Task 1 (e.g. Left fist / Both fists)
    - **`T2`** (orange): Task 2 (e.g. Right fist / Both feet)
    """)
    return


@app.cell
def _(mo, raw):
    _max_t = max(0, int(raw.times[-1] - 5))
    start_time_ui = mo.ui.slider(
        start=0, stop=_max_t, step=1, value=0, label="Start Time (s)"
    )
    duration_ui = mo.ui.slider(start=2, stop=30, step=1, value=10, label="Window (s)")
    _motor_ch = [ch for ch in ["C3", "Cz", "C4"] if ch in raw.ch_names]
    channel_ui = mo.ui.multiselect(
        options=raw.ch_names,
        value=_motor_ch if _motor_ch else raw.ch_names[:3],
        label="Channels to Compare",
    )
    mo.vstack(
        [
            mo.hstack([start_time_ui, duration_ui], justify="start", gap=2),
            channel_ui,
        ]
    )
    return channel_ui, duration_ui, start_time_ui


@app.cell
def _(channel_ui, duration_ui, np, plt, raw, raw_filtered, start_time_ui):
    _picks = channel_ui.value
    if not _picks:
        fig_time = plt.figure(figsize=(8, 2))
        plt.text(
            0.5,
            0.5,
            "Select at least one channel above to visualize",
            ha="center",
            va="center",
        )
        plt.axis("off")
    else:
        _t_start = float(start_time_ui.value)
        _t_end = min(_t_start + float(duration_ui.value), raw.times[-1])
        _s_idx = raw.time_as_index(_t_start)[0]
        _e_idx = raw.time_as_index(_t_end)[0]

        _raw_data, _times_slice = raw[_picks, _s_idx:_e_idx]
        _filt_data, _ = raw_filtered[_picks, _s_idx:_e_idx]

        _n_ch = len(_picks)
        fig_time, _axes = plt.subplots(
            _n_ch, 2, figsize=(14, max(3.0, 2.2 * _n_ch)), sharex=True
        )
        if _n_ch == 1:
            _axes = np.array([_axes])

        _event_colors = {"T0": "#7f7f7f", "T1": "#2ca02c", "T2": "#ff7f0e"}

        for _i, _ch in enumerate(_picks):
            # Raw trace
            _axes[_i, 0].plot(
                _times_slice, _raw_data[_i] * 1e6, color="#1f77b4", lw=0.9
            )
            _axes[_i, 0].set_ylabel(f"{_ch} (µV)", fontweight="bold")
            _axes[_i, 0].grid(True, linestyle=":", alpha=0.6)
            if _i == 0:
                _axes[_i, 0].set_title(
                    "Raw Signal (Unfiltered)", fontsize=11, fontweight="bold"
                )

            # Filtered trace
            _axes[_i, 1].plot(
                _times_slice, _filt_data[_i] * 1e6, color="#d62728", lw=0.9
            )
            _axes[_i, 1].grid(True, linestyle=":", alpha=0.6)
            if _i == 0:
                _axes[_i, 1].set_title(
                    "Filtered Signal (8–30 Hz)", fontsize=11, fontweight="bold"
                )

            # Overlap event annotations
            for _ax in (_axes[_i, 0], _axes[_i, 1]):
                for _annot in raw.annotations:
                    _onset = _annot["onset"]
                    _dur = _annot["duration"]
                    _desc = _annot["description"]
                    if (_onset + _dur >= _t_start) and (_onset <= _t_end):
                        _col = _event_colors.get(_desc, "#9467bd")
                        _ax.axvspan(
                            max(_onset, _t_start),
                            min(_onset + _dur, _t_end),
                            color=_col,
                            alpha=0.15,
                        )
                        _ax.axvline(
                            _onset, color=_col, linestyle="--", alpha=0.7, lw=1.0
                        )
                        _ylims = _ax.get_ylim()
                        _ax.text(
                            _onset + 0.05,
                            _ylims[1] * 0.75,
                            _desc,
                            color=_col,
                            fontweight="bold",
                            fontsize=8,
                        )

        _axes[-1, 0].set_xlabel("Time (s)")
        _axes[-1, 1].set_xlabel("Time (s)")
        fig_time.suptitle(
            f"Time-Domain Comparison ({_t_start:.1f}s – {_t_end:.1f}s)",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. Frequency Domain & Fourier Transform (PSD)

    ### Measuring the Strength of Each Frequency
    Instead of looking at the signal over time, **Fourier Transform (FFT)** is used to measure how much power exists at each frequency (from 0 to 80 Hz):

    * **Power Spectral Density (PSD):** Shows how "loud" each frequency is in the EEG.
    * **Welch’s Averaging:** Slices the recording into short, smoothed chunks (using a Hann window) and averages their Fourier spectra together to eliminate random noise.
    * **Objective:**
        1. Visually verify that the filter successfully suppressed unwanted frequencies (< 8 Hz and > 30 Hz).
        2. Inspect the power of the motor rhythms ($\mu$ and $\beta$) across the spectrum.
    """)
    return


@app.cell
def _(channel_ui, h_freq, l_freq, np, plt, raw, raw_filtered):
    _picks = channel_ui.value
    if not _picks:
        fig_psd = plt.figure(figsize=(8, 2))
        plt.text(0.5, 0.5, "Select channels to visualize PSD", ha="center")
        plt.axis("off")
    else:
        _sfreq = raw.info["sfreq"]
        _spec_raw = raw.compute_psd(
            method="welch",
            picks=_picks,
            fmin=0.0,
            fmax=_sfreq / 2,
            n_fft=256,
            verbose=False,
        )
        _spec_filt = raw_filtered.compute_psd(
            method="welch",
            picks=_picks,
            fmin=0.0,
            fmax=_sfreq / 2,
            n_fft=256,
            verbose=False,
        )

        _psd_raw, _freqs = _spec_raw.get_data(return_freqs=True)
        _psd_filt, _ = _spec_filt.get_data(return_freqs=True)

        _n_ch = len(_picks)
        fig_psd, _axes = plt.subplots(
            1, _n_ch, figsize=(max(6.0, 5.0 * _n_ch), 4.5), sharey=True
        )
        if _n_ch == 1:
            _axes = [_axes]

        for _i, _ch in enumerate(_picks):
            _ax = _axes[_i]
            # Convert to dB scale: 10 * log10(PSD in µV^2/Hz)
            _p_raw_db = 10 * np.log10(np.maximum(_psd_raw[_i] * 1e12, 1e-12))
            _p_filt_db = 10 * np.log10(np.maximum(_psd_filt[_i] * 1e12, 1e-12))

            _ax.plot(
                _freqs,
                _p_raw_db,
                label="Raw (Unfiltered)",
                color="#1f77b4",
                lw=1.5,
                alpha=0.75,
            )
            _ax.plot(
                _freqs,
                _p_filt_db,
                label=f"Filtered ({l_freq:.0f}–{h_freq:.0f} Hz)",
                color="#d62728",
                lw=1.8,
            )

            # Shaded passband and sub-bands
            _ax.axvspan(
                l_freq,
                h_freq,
                color="gold",
                alpha=0.15,
                label=f"Passband [{l_freq:.0f}–{h_freq:.0f} Hz]",
            )
            _ax.axvspan(8.0, 12.0, color="green", alpha=0.10, label="Mu (8–12 Hz)")
            _ax.axvspan(13.0, 30.0, color="orange", alpha=0.08, label="Beta (13–30 Hz)")

            _ax.set_title(
                f"Channel {_ch} - Power Spectrum", fontweight="bold", fontsize=11
            )
            _ax.set_xlabel("Frequency (Hz)")
            if _i == 0:
                _ax.set_ylabel("PSD (dB / µV²/Hz)")
            _ax.set_xlim(0, _sfreq / 2)
            _ax.grid(True, linestyle=":", alpha=0.6)
            _ax.legend(fontsize=8, loc="upper right")

        fig_psd.suptitle(
            "Frequency Domain Comparison (Welch PSD)", fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5. Epoch Extraction (Slicing the Recording into Trials)

    ### Turning a Continuous Stream into Machine Learning Examples

    **Epoching** cuts out the specific 4-second clip every time the person was instructed to perform a movement task:

    * **What is Extracted (The Tasks):**
      * **`T1` Trials:** Left Fist movement (Tasks 1 & 2) or Both Fists (Tasks 3 & 4).
      * **`T2` Trials:** Right Fist movement (Tasks 1 & 2) or Both Feet (Tasks 3 & 4).
    * **The Time Window ($t \in [0.0\text{s}, 4.0\text{s}]$):**
      * Captures the full 4 seconds the target was visible on the screen.

    > **Result:** Turns the continuous 2-minute recording into ~15 labeled trial blocks ready to be fed into a classification pipeline.
    """)
    return


@app.cell
def _(mo):
    tmin_ui = mo.ui.slider(
        start=-1.0, stop=0.5, step=0.1, value=0.0, label="t_min (s relative to cue)"
    )
    tmax_ui = mo.ui.slider(
        start=1.0, stop=4.5, step=0.1, value=4.0, label="t_max (s relative to cue)"
    )
    mo.hstack([tmin_ui, tmax_ui], justify="start", gap=2)
    return tmax_ui, tmin_ui


@app.cell
def _(mne, mo, raw_filtered, tmax_ui, tmin_ui):
    _events, _event_id = mne.events_from_annotations(raw_filtered, verbose=False)

    # Filter to only keep task cues T1 and T2
    _target_event_id = {k: v for k, v in _event_id.items() if k in ("T1", "T2")}

    epochs = mne.Epochs(
        raw_filtered,
        events=_events,
        event_id=_target_event_id,
        tmin=float(tmin_ui.value),
        tmax=float(tmax_ui.value),
        proj=False,
        baseline=None,
        preload=True,
        verbose=False,
    )

    _n_t1 = len(epochs["T1"]) if "T1" in epochs.event_id else 0
    _n_t2 = len(epochs["T2"]) if "T2" in epochs.event_id else 0
    _data_shape = epochs.get_data().shape

    mo.md(f"""
    **Extracted Epochs Summary:**
    - **Total Trials:** {len(epochs)} (T1: {_n_t1}, T2: {_n_t2})
    - **Data Matrix Shape:** `{_data_shape}` $\\to$ *(trials, channels, timepoints)*
    - **Sampling Rate:** `{epochs.info["sfreq"]:.0f} Hz` ({_data_shape[2]} samples per trial)
    """)
    return (epochs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 6. Task-Specific Scalp Topography (T1 vs. T2)

    Scalp power during **Task 1 ($T1$)** vs. **Task 2 ($T2$)** is plotted on the same color scale to see if the two motor thoughts produce visibly different brain maps.

    **Observation:**
    The two heat maps look almost indistinguishable. Background rhythms dominate, and the task difference is too small to spot with the naked eye. This visual similarity provides the direct justification for using machine learning algorithms (like CSP) rather than visual inspection to decode movement thoughts.
    """)
    return


@app.cell
def _(mo):
    band_choice_ui = mo.ui.radio(
        options=["Mu Band (8–12 Hz)", "Beta Band (13–30 Hz)"],
        value="Mu Band (8–12 Hz)",
        label="Select Frequency Band",
    )
    mo.hstack([band_choice_ui])
    return (band_choice_ui,)


@app.cell
def _(band_choice_ui, epochs, np, plot_topomap, plt):
    fig_topo = plt.figure(figsize=(12, 4.8))
    _has_t1 = "T1" in epochs.event_id and len(epochs["T1"]) > 0
    _has_t2 = "T2" in epochs.event_id and len(epochs["T2"]) > 0

    if _has_t1 and _has_t2:
        _spec_t1 = epochs["T1"].compute_psd(
            method="welch", fmin=0.5, fmax=45.0, n_fft=256, verbose=False
        )
        _spec_t2 = epochs["T2"].compute_psd(
            method="welch", fmin=0.5, fmax=45.0, n_fft=256, verbose=False
        )

        _psds_t1, _freqs = _spec_t1.get_data(return_freqs=True)
        _psds_t2, _ = _spec_t2.get_data(return_freqs=True)

        _mean_t1 = np.mean(_psds_t1, axis=0)
        _mean_t2 = np.mean(_psds_t2, axis=0)

        _is_mu = "Mu" in band_choice_ui.value
        _band_name = "Mu (8–12 Hz)" if _is_mu else "Beta (13–30 Hz)"
        _mask = (
            (_freqs >= 8.0) & (_freqs <= 12.0)
            if _is_mu
            else (_freqs >= 13.0) & (_freqs <= 30.0)
        )

        _power_t1 = np.mean(_mean_t1[:, _mask], axis=-1)
        _power_t2 = np.mean(_mean_t2[:, _mask], axis=-1)

        _power_t1_db = 10 * np.log10(np.maximum(_power_t1 * 1e12, 1e-12))
        _power_t2_db = 10 * np.log10(np.maximum(_power_t2 * 1e12, 1e-12))

        # Shared color limits for a direct, fair visual comparison
        _vmin = min(_power_t1_db.min(), _power_t2_db.min())
        _vmax = max(_power_t1_db.max(), _power_t2_db.max())

        _ax1 = plt.subplot(1, 2, 1)
        _im1, _ = plot_topomap(
            _power_t1_db,
            epochs.info,
            axes=_ax1,
            cmap="viridis",
            vlim=(_vmin, _vmax),
            show=False,
        )
        _ax1.set_title(
            f"Task 1: Left Fist (T1) — {_band_name}", fontsize=11, fontweight="bold"
        )
        plt.colorbar(
            _im1,
            ax=_ax1,
            orientation="horizontal",
            pad=0.08,
            shrink=0.7,
            label="Power (dB / µV²/Hz)",
        )

        _ax2 = plt.subplot(1, 2, 2)
        _im2, _ = plot_topomap(
            _power_t2_db,
            epochs.info,
            axes=_ax2,
            cmap="viridis",
            vlim=(_vmin, _vmax),
            show=False,
        )
        _ax2.set_title(
            f"Task 2: Right Fist (T2) — {_band_name}", fontsize=11, fontweight="bold"
        )
        plt.colorbar(
            _im2,
            ax=_ax2,
            orientation="horizontal",
            pad=0.08,
            shrink=0.7,
            label="Power (dB / µV²/Hz)",
        )

        fig_topo.suptitle(
            f"Task-Specific Scalp Topography ({_band_name})",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()
    else:
        plt.text(
            0.5,
            0.5,
            "Both T1 and T2 trials required for task comparison topomap",
            ha="center",
        )
        plt.axis("off")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 7. Left vs. Right Brain Response ($C3$ vs. $C4$)

    Because the left brain controls the right hand and vice versa:
    * **Left Fist ($T1$):** Activates the **right motor cortex ($C4$)**, causing wave power to drop.
    * **Right Fist ($T2$):** Activates the **left motor cortex ($C3$)**, causing wave power to drop.

    The curves below compare the power across frequencies for both electrodes during both tasks. The difference between the lines is very subtle and why we need machine learning rather than visual inspection to decode movement thoughts.
    """)
    return


@app.cell
def _(epochs, np, plt):
    plt.figure(figsize=(10, 4))
    _has_t1 = "T1" in epochs.event_id and len(epochs["T1"]) > 0
    _has_t2 = "T2" in epochs.event_id and len(epochs["T2"]) > 0
    _has_ch = ("C3" in epochs.ch_names) and ("C4" in epochs.ch_names)

    if _has_t1 and _has_t2 and _has_ch:
        _c3_idx = epochs.ch_names.index("C3")
        _c4_idx = epochs.ch_names.index("C4")

        # Average PSD for T1
        _spec_t1 = epochs["T1"].compute_psd(
            method="welch", fmin=4.0, fmax=35.0, n_fft=256, verbose=False
        )
        _psd_t1, _f_erd = _spec_t1.get_data(return_freqs=True)
        _mean_t1 = np.mean(_psd_t1, axis=0)

        # Average PSD for T2
        _spec_t2 = epochs["T2"].compute_psd(
            method="welch", fmin=4.0, fmax=35.0, n_fft=256, verbose=False
        )
        _psd_t2, _ = _spec_t2.get_data(return_freqs=True)
        _mean_t2 = np.mean(_psd_t2, axis=0)

        _ax_c3 = plt.subplot(1, 2, 1)
        _ax_c3.plot(
            _f_erd,
            10 * np.log10(_mean_t1[_c3_idx] * 1e12),
            label="T1 (Left)",
            color="#2ca02c",
            lw=2,
        )
        _ax_c3.plot(
            _f_erd,
            10 * np.log10(_mean_t2[_c3_idx] * 1e12),
            label="T2 (Right)",
            color="#ff7f0e",
            lw=2,
        )
        _ax_c3.axvspan(8, 12, color="gray", alpha=0.15, label="Mu band")
        _ax_c3.axvspan(13, 30, color="gray", alpha=0.08, label="Beta band")
        _ax_c3.set_title("Electrode C3 (Left Motor Cortex)", fontweight="bold")
        _ax_c3.set_xlabel("Frequency (Hz)")
        _ax_c3.set_ylabel("PSD (dB / µV²/Hz)")
        _ax_c3.grid(True, linestyle=":", alpha=0.6)
        _ax_c3.legend(fontsize=8)

        _ax_c4 = plt.subplot(1, 2, 2)
        _ax_c4.plot(
            _f_erd,
            10 * np.log10(_mean_t1[_c4_idx] * 1e12),
            label="T1 (Left)",
            color="#2ca02c",
            lw=2,
        )
        _ax_c4.plot(
            _f_erd,
            10 * np.log10(_mean_t2[_c4_idx] * 1e12),
            label="T2 (Right)",
            color="#ff7f0e",
            lw=2,
        )
        _ax_c4.axvspan(8, 12, color="gray", alpha=0.15, label="Mu band")
        _ax_c4.axvspan(13, 30, color="gray", alpha=0.08, label="Beta band")
        _ax_c4.set_title("Electrode C4 (Right Motor Cortex)", fontweight="bold")
        _ax_c4.set_xlabel("Frequency (Hz)")
        _ax_c4.grid(True, linestyle=":", alpha=0.6)
        _ax_c4.legend(fontsize=8)

        plt.suptitle(
            "Class Power Difference (T1 vs. T2) across Sensorimotor Electrodes",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()
    else:
        plt.text(
            0.5,
            0.5,
            "Both T1 and T2 trials with C3/C4 needed for contrast plot",
            ha="center",
        )
        plt.axis("off")
    return


if __name__ == "__main__":
    app.run()
