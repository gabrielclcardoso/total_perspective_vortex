"""Project-wide configuration, paths, and signal-processing constants."""

from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
PLOTS_DIR = ROOT_DIR / "plots"

# Dataset dimensions and hardware specifications
N_SUBJECTS = 109
N_CHANNELS = 64
SFREQ = 160.0  # Hz
MONTAGE_NAME = "colin27_1005"  # Standard 10-10 electrode montage for PhysioNet EEGBCI

# 8.0 - 30.0 Hz isolates sensorimotor mu (8-12 Hz) and beta (13-30 Hz) rhythms
# while eliminating ocular / slow drift artifacts (< 8 Hz) and muscle EMG / line noise (> 30 Hz)
L_FREQ = 8.0  # Hz
H_FREQ = 30.0  # Hz
FIR_DESIGN = "firwin"
FIR_WINDOW = "hamming"

# Epoching parameters
# 4.0-second task execution window starting at target cue onset (t = 0.0s)
T_MIN = 0.0  # seconds relative to cue onset
T_MAX = 4.0  # seconds relative to cue onset

# Event codes and target labels
# In PhysioNet EEGBCI:
# T0: Rest interval
# T1: Left fist (tasks 1, 2) or Both fists (tasks 3, 4) -> mapped to label 1
# T2: Right fist (tasks 1, 2) or Both feet (tasks 3, 4) -> mapped to label 2
REST_EVENT = "T0"
EVENT_IDS = {"T1": 1, "T2": 2}
TARGET_LABELS = (1, 2)

# Experimental task descriptions and run groupings
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

# Grouping runs by motor condition (12 task runs: 3 runs per condition)
TASK_RUNS = {
    1: (3, 7, 11),  # Task 1: Real left vs. right fist
    2: (4, 8, 12),  # Task 2: Imagined left vs. right fist
    3: (5, 9, 13),  # Task 3: Real both fists vs. both feet
    4: (6, 10, 14),  # Task 4: Imagined both fists vs. both feet
}

RUN_TO_TASK = {
    3: 1,
    7: 1,
    11: 1,
    4: 2,
    8: 2,
    12: 2,
    5: 3,
    9: 3,
    13: 3,
    6: 4,
    10: 4,
    14: 4,
}

# Real-time simulation and grading constraints
STREAM_DELAY_S = 2.0
ACCURACY_THRESHOLD = 0.60
