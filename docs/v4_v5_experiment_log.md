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

## 8. v6 Stacked LSTM

A stacked LSTM model was evaluated while keeping the v5 sliding-window preprocessing and dataset unchanged.

Architecture:

- LSTM(64, return_sequences=True, L2=0.005)
- Batch Normalization
- Dropout(0.3)
- LSTM(32, return_sequences=False, L2=0.005)
- Batch Normalization
- Dropout(0.3)
- Dense(12, Softmax, L2=0.005)

Validation results:

| Model | Full Sequence | Sliding-Window Aggregation |
|---|---:|---:|
| v5 | 75.00% (18/24) | 91.67% (22/24) |
| v6 | 83.33% (20/24) | 91.67% (22/24) |

The stacked LSTM improved full-sequence validation performance while maintaining the same sliding-window aggregated accuracy.

## 9. External Test Set v1

A new external webcam test set was collected before further model modification.

Dataset:

- 12 classes (1–12)
- 3 repetitions per class
- Total: 36 videos
- File naming: test_0101, test_0102, ..., test_1203
- Repetitions 01/02: MP4
- Repetition 03: MOV

All 36 videos were successfully processed with MediaPipe and converted to landmark sequences.

### External Test Results

| Model | Accuracy |
|---|---:|
| v5 | 38.89% (14/36) |
| v6 | 44.44% (16/36) |

v6 showed a 5.55 percentage-point improvement over v5 on the new external set.

### v6 Per-Class Accuracy

| Class | Accuracy |
|---|---:|
| 1 | 0/3 |
| 2 | 2/3 |
| 3 | 2/3 |
| 4 | 1/3 |
| 5 | 3/3 |
| 6 | 2/3 |
| 7 | 0/3 |
| 8 | 2/3 |
| 9 | 0/3 |
| 10 | 2/3 |
| 11 | 1/3 |
| 12 | 1/3 |

A large performance gap was observed between the AI Hub validation set and newly recorded external videos.

For the v6 model:

- MP4 repetitions 01/02: 14/24 = 58.33%
- MOV repetition 03: 2/12 = 16.67%

This does not prove that the MOV format itself is the cause, because repetition number, recording condition, device, orientation, and file format are confounded.

Classes 1, 7, and 9 failed on all three external repetitions, indicating that the current limitation is not explained only by the MOV samples.

## 10. Current Status

Current primary candidate:

- v6 stacked LSTM
- Sliding window: 48 frames
- Stride: 8 frames
- Resampled sequence length: 64
- Mean softmax probability aggregation

Development validation:

- 91.67% (22/24)

New external test:

- 44.44% (16/36)

The next step is to analyze the domain gap between AI Hub videos and externally recorded videos before further model tuning.

In particular:

- video resolution
- frame rate
- orientation
- MediaPipe hand detection rate
- recording device / camera condition
- class-specific confusion

should be investigated.
