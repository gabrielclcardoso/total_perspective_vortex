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

    return eegbci, mne, plt


@app.cell
def _(DATA_DIR, eegbci, mne, mo, run_ui, subject_ui):
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
        else f"Downloading S{sub:03d}R{run:02d} from PhysioNet (~2.5 MB, may take a minute)..."
    )

    with mo.status.spinner(title=status_title):
        files = eegbci.load_data(
            subjects=sub,
            runs=[run],
            path=DATA_DIR,
            update_path=False,
        )
        # Parse raw EDF
        raw = mne.io.read_raw_edf(files[0], preload=True)

    mo.output.replace(
        mo.md(
            f"✓ **Loaded S{sub:03d}R{run:02d}.edf** "
            f"({'from local cache' if is_cached else 'downloaded from PhysioNet'}) | "
            f"**{raw.times[-1]:.1f}s** recording, **{len(raw.ch_names)} channels** @ **{raw.info['sfreq']:.0f} Hz**"
        )
    )
    return (raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualize Electrodes Location
    """)
    return


@app.cell
def _(eegbci, mne, plt, raw):
    # 1. Clean channel names ('C3..' -> 'C3') and set standard 10-05 montage
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage('colin27_1005')
    raw.set_montage(montage)

    # 2. Plot sensor locations
    fig = raw.plot_sensors(show_names=True, kind='topomap')
    return (fig,)


if __name__ == "__main__":
    app.run()
