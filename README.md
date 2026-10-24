# Total Perspective Vortex — EEG Brain-Computer Interface

A Brain-Computer Interface (BCI) machine learning pipeline developed for the 42 School project. The system classifies motor imagery and execution tasks (e.g., Left vs. Right fist) from 64-channel EEG recordings in the PhysioNet EEGBCI dataset.

---

## Development Milestones & Key Decisions

### 1. Data Acquisition & Montage Mapping

- **Parallel S3 Ingestion:** Created a multi-threaded downloader (`tpv.download`) fetching directly from the AWS PhysioNet mirror into MNE's standard cache directory.
- **10–10 Electrode Layout:** Mapped the 64 electrode channels onto the international 10–10 system (`colin27_1005` montage).

### 2. Zero-Phase Bandpass Filtering (8.0 – 30.0 Hz)

- **Target Rhythms:** Isolated sensorimotor $\mu$ (8–12 Hz) and $\beta$ (13–30 Hz) rhythms responsible for Event-Related Desynchronization (ERD).
- **Noise Rejection:** Suppressed low-frequency ocular artifacts and DC drift (< 8 Hz) as well as muscle tension EMG and 50/60 Hz powerline hum (> 30 Hz).
- **Decision (Filter before Epoching):** Filtering is applied to continuous recordings _before_ trial slicing to prevent FIR filter boundary/edge ringing artifacts.

### 3. Trial Extraction & Labeling

- **Time Window:** Sliced 4.0-second epochs ($t \in [0.0\text{s}, 4.0\text{s}]$) locked to visual task cue onsets.
- **Decision (Discarding Rest):** Rest intervals (`T0`) are discarded; active cues (`T1` and `T2`) are remapped to binary target labels `{1, 2}` to match the subject's required classification format.

### 4. Visualization & Signal Exploration (Section V.1.1)

- Implemented `tpv.plots` and `tpv.visualize` (`make visualize`) to generate 4 figures in `plots/`:
  1. **Time-Domain Traces:** Raw vs. zero-phase filtered waveforms with shaded trial cues.
  2. **Frequency Spectra (PSD):** Welch periodogram confirming steep attenuation outside the 8–30 Hz passband.
  3. **Task Topographies:** 2D head maps across $\mu$ and $\beta$ bands on shared color scales, visually illustrating why raw scalp differences are too subtle for human inspection.
  4. **Sensorimotor Contrast ($C3$ vs. $C4$):** Power curves demonstrating contralateral motor cortex power drops.

---

## Code Architecture

```
src/tpv/
├── config.py         # Global paths, hardware parameters, and filter constants
├── download.py       # High-speed AWS S3 parallel dataset downloader
├── preprocessing.py  # MNE loading, continuous filtering, and trial epoching
├── plots.py          # Modular Matplotlib/MNE plotting library
├── visualize.py      # Standalone CLI driver for Section V.1.1 ('make visualize')
└── cli.py            # CLI entrypoint for train and prediction
notebooks/
└── preprocessing_exploration.py  # Interactive Marimo exploration notebook
```

---

## Quickstart

```bash
# 1. Setup virtual environment and dependencies
make setup

# 2. Download sample data (Subject 1, Run 4) or all subjects
make download SUBJECT=1

# 3. Generate Section V.1.1 diagnostic figures into plots/
make visualize

# 4. (Optional) Run interactive Marimo exploration notebook
make notebook
```
