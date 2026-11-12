# Numeric 1–12 LSTM v4/v5 Experiment Log

## 1. Background

The v3 LSTM achieved 79.17% accuracy (19/24) on the AI Hub validation set, but showed poor generalization on directly recorded webcam videos.

Analysis indicated several differences between AI Hub data and webcam inputs:

- Missing-hand coordinates were represented as raw zeros and became extreme values after normalization.
- Raw MediaPipe landmark coordinates depended strongly on camera position and hand scale.
- Webcam clips contained unnecessary frames before and after the actual sign.
- Full-sequence temporal resampling could dilute useful temporal patterns.

## 2. v4 Preprocessing

v4 introduced:

- Mask-aware missing-hand handling
- Wrist-centered landmark coordinates
- Hand-scale normalization using the wrist-to-middle-MCP distance
- Nearest-neighbor resampling for presence masks
- Active Left-hand temporal cropping

Validation results:

| Version | Validation Accuracy |
|---|---:|
| v3 | 79.17% (19/24) |
| v4a | 66.67% (16/24) |
| v4b | 75.00% (18/24) |

Temporal cropping improved repeated webcam samples of class 5.

## 3. Sliding Window

Training landmark sequence length:

- Training samples: 204
- Median sequence length: 52 frames
- Mean sequence length: 55.73 frames

Based on this distribution, the first sliding-window configuration was:

- Window size: 48 frames
- Stride: 8 frames
- Each window resampled to 64 timesteps before entering the LSTM

To prevent long videos from dominating training, sample weights were adjusted so that the total contribution of each original video was equal.

Training set:

- Original videos: 204
- Sliding windows: 709

## 4. v5 Sliding-Window LSTM

The LSTM architecture itself was kept unchanged from v4b.

Validation:

| Evaluation Method | Accuracy |
|---|---:|
| v5 Full Sequence | 75.00% (18/24) |
| v5 Sliding-Window Aggregation | 91.67% (22/24) |

Window predictions were aggregated using the mean softmax probability for each original video.

## 5. Webcam Test

Preliminary webcam samples:

| Video | Target | v4b Sliding | v5 Sliding |
|---|---:|---|---|
| test_4 | 4 | 10 (incorrect) | 10 (incorrect) |
| test_5 | 5 | 5 (correct) | 5 (correct) |
| test_5_re1 | 5 | 5 (correct) | 5 (correct) |
| test_5_re2 | 5 | 5 (correct) | 5 (correct) |
| test_6 | 6 | 9 (incorrect) | 6 (correct) |

Preliminary external accuracy improved from 3/5 to 4/5.

The webcam set is too small and has been used during development, so it should not be treated as an independent final test set.

## 6. Current Inference Pipeline

MediaPipe landmark sequence

→ Active Left-hand crop

→ 48-frame sliding windows, stride 8

→ Wrist-centered normalization

→ Hand-scale normalization

→ 64-timestep resampling

→ Mask-aware normalization

→ LSTM

→ Window softmax probabilities

→ Mean probability aggregation

→ Final class prediction

## 7. Next Step

Collect a new external webcam test set after the next team meeting.

Recommended initial evaluation set:

- Classes 1–12
- At least 3 repetitions per class
- 36 videos per participant

The current v5 model should be preserved without modification until the new external set is evaluated.
