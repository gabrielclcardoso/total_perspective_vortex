"""Phase 1: Preprocessing, Parsing, and Epoching.

Handles:
- Loading PhysioNet EDF files with MNE and setting standard 10-10 montages.
- Zero-phase FIR bandpass filtering (8.0 - 30.0 Hz) to isolate sensorimotor rhythms (mu and beta).
- Epoch extraction around task cues (T1 vs. T2) with label mapping to integers {1, 2}.
- Converting MNE Epochs into clean NumPy feature tensors X and target labels y.
"""

from collections.abc import Sequence

import mne
import numpy as np

from tpv.config import (
    EVENT_IDS,
    FIR_DESIGN,
    FIR_WINDOW,
    H_FREQ,
    L_FREQ,
    MONTAGE_NAME,
    T_MAX,
    T_MIN,
)
from tpv.download import download_run, get_edf_path


def load_and_preprocess_runs(
    subject: int,
    runs: Sequence[int],
    tmin: float = T_MIN,
    tmax: float = T_MAX,
    l_freq: float = L_FREQ,
    h_freq: float = H_FREQ,
) -> tuple[np.ndarray, np.ndarray, mne.BaseEpochs]:
    """End-to-end preprocessing for one or more runs concatenated together."""
    epoch_list = []
    for r in runs:
        raw = load_raw(subject, r)
        raw_filt = filter_raw(raw, l_freq=l_freq, h_freq=h_freq)
        ep = extract_epochs(raw_filt, tmin=tmin, tmax=tmax)
        epoch_list.append(ep)

    all_epochs = (
        epoch_list[0]
        if len(epoch_list) == 1
        else mne.concatenate_epochs(epoch_list, verbose=False)
    )
    X, y = get_epoch_data_and_labels(all_epochs)
    return X, y, all_epochs


def load_raw(subject: int, run: int, preload: bool = True) -> mne.io.BaseRaw:
    """Load a single PhysioNet EEGBCI EDF recording with MNE."""
    edf_path = get_edf_path(subject, run)
    if not edf_path.exists():
        download_run(subject, run)

    raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose="ERROR")
    mne.datasets.eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage(MONTAGE_NAME)
    raw.set_montage(montage, on_missing="ignore")
    return raw


def filter_raw(
    raw: mne.io.BaseRaw,
    l_freq: float = L_FREQ,
    h_freq: float = H_FREQ,
    copy: bool = True,
) -> mne.io.BaseRaw:
    """Apply zero-phase FIR bandpass filter to continuous EEG data.

    Continuous signals are filtered before epoching to eliminate boundary
    artifacts at trial edges.
    """
    target = raw.copy() if copy else raw
    target.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design=FIR_DESIGN,
        fir_window=FIR_WINDOW,
        skip_by_annotation="edge",
        verbose=False,
    )
    return target


def extract_epochs(
    raw: mne.io.BaseRaw,
    tmin: float = T_MIN,
    tmax: float = T_MAX,
    event_ids: dict[str, int] | None = None,
) -> mne.Epochs:
    """Slice continuous EEG into trial epochs around active movement cues.

    Filters out rest periods (T0) and extracts active cues (T1 and T2),
    remapping their labels to integer target codes {1, 2}.
    """
    if event_ids is None:
        event_ids = EVENT_IDS

    raw_events, annot_desc_to_id = mne.events_from_annotations(raw, verbose=False)
    annot_map: dict[str, int] = dict(annot_desc_to_id)

    # Find the integer ID assigned by MNE to each target annotation
    raw_id_to_target = {
        annot_map[desc]: target_id
        for desc, target_id in event_ids.items()
        if desc in annot_map
    }

    # Filter events to only keep target cues (e.g. T1, T2) and remap their label
    filtered_events = []
    for ev in raw_events:
        if ev[2] in raw_id_to_target:
            filtered_events.append([ev[0], 0, raw_id_to_target[ev[2]]])

    if not filtered_events:
        raise ValueError(
            f"No matching target events ({list(event_ids.keys())}) found in recording."
        )

    filtered_events = np.array(filtered_events, dtype=int)

    epochs = mne.Epochs(
        raw,
        events=filtered_events,
        event_id=event_ids,
        tmin=tmin,
        tmax=tmax,
        proj=False,
        baseline=None,
        preload=True,
        verbose=False,
    )
    return epochs


def get_epoch_data_and_labels(
    epochs: mne.BaseEpochs,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract data matrix X and ground truth label vector y from Epochs."""
    X = epochs.get_data(copy=True)
    y = epochs.events[:, 2].astype(int)
    return X, y
