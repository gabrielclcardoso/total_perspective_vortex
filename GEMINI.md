---
name: total-perspective-vortex-rules
description: Project constraints for the 42 Total Perspective Vortex BCI project.
trigger: always_on
---

# Total Perspective Vortex - Project Constraints

This directory contains the "Total Perspective Vortex" project for 42 School. It is a Brain-Computer Interface (BCI) project using EEG data.

When helping me with code or architecture in this folder, you MUST strictly adhere to the following constraints:

1. **Allowed Libraries:** 
   - `MNE` (for EEG data parsing/filtering).
   - `scikit-learn` (for classification, pipelines, validation).
   - `numpy` / `scipy` (allowed for eigenvalues, singular values, and covariance matrix estimation).
   - *DO NOT suggest deep learning libraries like TensorFlow or PyTorch.*

2. **Forbidden Tools:**
   - `mne-realtime` is explicitly **FORBIDDEN**. We must manually implement the 2-second delay playback simulation.

3. **Architecture Requirements:**
   - **Scikit-learn Pipeline:** Our custom dimensionality reduction algorithm (e.g., PCA, CSP, ICA) must be integrated seamlessly into a scikit-learn `Pipeline`. It must inherit from `BaseEstimator` and `TransformerMixin`.
   - **Validation:** We must use `cross_val_score` on the *entire* processing pipeline, not just the classifier.

4. **Functional Requirements:**
   - **Stream Playback:** The prediction script must read the file to simulate a real-time stream. It must predict output after a strict delay of 2 seconds after the data chunk is sent to the pipeline.
   - **Accuracy Threshold:** The final test data (which includes never-learned data and all six types of experiments) must achieve a mean accuracy of >= 60%.
