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
def _(DATA_DIR, eegbci, mne, run_ui, subject_ui):
    files = eegbci.load_data(
        subjects=subject_ui.value,
        runs=[int(run_ui.value)],
        path=DATA_DIR,
        update_path=False,
    )

    # Parse raw EDF
    raw = mne.io.read_raw_edf(files[0], preload=True)
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
    plt.show()
    return


if __name__ == "__main__":
    app.run()
