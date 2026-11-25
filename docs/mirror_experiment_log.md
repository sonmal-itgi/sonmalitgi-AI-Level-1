# Mirror Preprocessing Experiment

Date: 2026-11-26

## 1. Background

The original AI Hub training videos do not have a mirror effect,
whereas the intended webcam input in the learning service is mirrored.

When users follow the original learning videos while viewing
a mirrored webcam preview, they may perform signs using
the opposite physical hand.

This also introduces an input orientation mismatch between
the training data and the intended inference data.

The objective of this experiment is to align the learning
content, model training data, and actual webcam input orientation.

## 2. Mirror Landmark Extraction

Horizontal flipping was applied before MediaPipe processing:

    frame = cv2.flip(frame, 1)

The original videos were preserved.

The mirrored landmarks were extracted into a separate directory:

    numeric_1to12/landmarks_mirror/

Dataset:

- Training: 204 videos
- Validation: 24 videos
- Total: 228 videos
- Feature dimension: 128
- Average sequence length: 57.2 frames
- Average hand detection rate: 97.9%

The original MediaPipe configuration and annotated temporal
segments were preserved.

## 3. Landmark Verification

All 228 mirrored landmark files matched their original files.

No errors were found in:

- File matching
- Feature dimensions
- Frame counts
- Empty sequences
- NaN or Inf values
- Presence masks
- Missing-hand coordinates

### Hand Detection Rate

| Split | Original | Mirror |
|---|---:|---:|
| Training | 97.84% | 97.77% |
| Validation | 99.15% | 99.06% |

### Presence Mask Agreement

| Split | Left to Right | Right to Left |
|---|---:|---:|
| Training | 99.38% | 99.20% |
| Validation | 99.20% | 99.90% |

### Coordinate Reflection MAE

| Split | Direction | X | Y | Z |
|---|---|---:|---:|---:|
| Training | Left to Right | 0.003447 | 0.003056 | 0.005565 |
| Training | Right to Left | 0.003655 | 0.003294 | 0.008115 |
| Validation | Left to Right | 0.002727 | 0.001721 | 0.005853 |
| Validation | Right to Left | 0.002435 | 0.002142 | 0.006738 |

The extracted coordinates closely matched the expected
horizontal reflection.

Hand detection performance was largely preserved.

## 4. Mirror Sliding-Window Dataset

The active crop reference was changed from Left to Right.

The remaining preprocessing conditions were preserved:

- Window size: 48 frames
- Stride: 8 frames
- Target sequence length: 64
- Feature dimension: 128
- Wrist-centered normalization
- Hand-scale normalization
- Linear interpolation for coordinates
- Nearest-neighbor interpolation for presence masks
- Mask-aware missing-hand handling
- Source-balanced sample weighting

| Dataset | Original | Mirror |
|---|---:|---:|
| Training videos | 204 | 204 |
| Training windows | 709 | 705 |
| Validation videos | 24 | 24 |
| Validation windows | 94 | 96 |
| Mean sample weight | 1.0 | 1.0 |
| Weight sum per class | 59.08 | 58.75 |

No invalid hands were reported during mirror training
dataset preprocessing.

The small differences in window counts are consistent with
minor differences in hand detection and active crop boundaries.

## 5. Model Training

Both mirror models were trained from scratch using the
mirrored dataset and newly calculated normalization statistics.

The original models and their weights were preserved.

### v5_mirror

- Input: (64, 128)
- LSTM(32)
- Dropout(0.4)
- Dense(24, ReLU)
- Dropout(0.3)
- Dense(12, Softmax)

### v6_mirror

- Input: (64, 128)
- LSTM(64, return_sequences=True)
- Batch Normalization
- Dropout(0.3)
- LSTM(32, return_sequences=False)
- Batch Normalization
- Dropout(0.3)
- Dense(12, Softmax)

L2 regularization of 0.005 was preserved from v6.

Both architectures use unidirectional LSTM layers.

## 6. AI Hub Validation Results

| Model | Full Sequence | Sliding Aggregation |
|---|---:|---:|
| v5 | 75.00% (18/24) | 91.67% (22/24) |
| v5_mirror | 83.33% (20/24) | 91.67% (22/24) |
| v6 | 83.33% (20/24) | 91.67% (22/24) |
| v6_mirror | 79.17% (19/24) | 87.50% (21/24) |

Sliding-window predictions were aggregated using
the mean softmax probability.

### Remaining Sliding Aggregation Errors

v5_mirror:

- Class 1, CROWD19 -> Class 11
- Class 8, CROWD18 -> Class 9

v6_mirror:

- Class 1, CROWD19 -> Class 11
- Class 4, CROWD19 -> Class 3
- Class 8, CROWD18 -> Class 9

## 7. External Test Dataset v2

The original External Test v1 contained:

- Repetitions 01/02: 24 incorrectly oriented recordings
- Repetition 03: 12 correctly oriented recordings

The 24 incorrectly oriented recordings were replaced with
new recordings using the intended physical hand and orientation.

The existing 12 correctly oriented recordings were preserved.

External Test v2:

- Classes: 1-12
- Repetitions per class: 3
- Total videos: 36
- New recordings: 24
- Existing recordings: 12
- Successful landmark extraction: 36/36

No additional horizontal flip was applied during extraction
of the external videos.

## 8. External Test Results

All four models were evaluated on the same External Test v2.

| Model | Overall (36) | New (24) | Existing (12) |
|---|---:|---:|---:|
| v5 | 11.11% (4/36) | 12.50% (3/24) | 8.33% (1/12) |
| v5_mirror | 50.00% (18/36) | 45.83% (11/24) | 58.33% (7/12) |
| v6 | 13.89% (5/36) | 12.50% (3/24) | 16.67% (2/12) |
| v6_mirror | 52.78% (19/36) | 54.17% (13/24) | 50.00% (6/12) |

Both mirror models improved overall external accuracy
by 38.89 percentage points.

### Per-Class Results

| Class | v5 | v5_mirror | v6 | v6_mirror |
|---|---:|---:|---:|---:|
| 1 | 2/3 | 0/3 | 0/3 | 0/3 |
| 2 | 0/3 | 0/3 | 0/3 | 3/3 |
| 3 | 0/3 | 3/3 | 1/3 | 3/3 |
| 4 | 0/3 | 0/3 | 1/3 | 0/3 |
| 5 | 0/3 | 3/3 | 3/3 | 3/3 |
| 6 | 1/3 | 3/3 | 0/3 | 3/3 |
| 7 | 0/3 | 3/3 | 0/3 | 1/3 |
| 8 | 0/3 | 1/3 | 0/3 | 0/3 |
| 9 | 0/3 | 0/3 | 0/3 | 0/3 |
| 10 | 0/3 | 3/3 | 0/3 | 2/3 |
| 11 | 1/3 | 0/3 | 0/3 | 3/3 |
| 12 | 0/3 | 2/3 | 0/3 | 1/3 |

## 9. Interpretation

The results strongly support the hypothesis that the
orientation mismatch contributed to poor external performance.

The measured improvement represents the combined effect of:

- Mirrored training videos
- Right-hand active cropping
- Recalculated normalization statistics
- Model retraining

It should not be interpreted as an isolated measurement
of horizontal flipping alone.

Classes 1, 4, and 9 failed across all external recordings
for both mirror models.

Major remaining confusion patterns include:

- Class 1 -> Class 11
- Class 4 -> Class 10
- Classes 7, 8, and 9 -> Classes 6 or 7

The stacked LSTM achieved the highest external accuracy,
but its advantage over the single LSTM was only one video.

The current results do not establish the general superiority
of either architecture.

## 10. Limitations

- AI Hub validation contains only 24 original videos.
- Validation was used during model development.
- External Test v2 includes only 36 recordings.
- Twelve external recordings were reused from External Test v1.
- Models were trained using a single seed.
- Remaining domain differences have not been fully analyzed.
- Correct class prediction does not guarantee correct hand usage.
- Real-time streaming performance has not yet been established.

## 11. Current Status

Primary candidate:

    models_v6_stacked_mirror/
        lstm_1to12_v6_stacked_mirror.keras
        normalization.npz

Alternative candidate:

    models_v5_sw_mirror/
        lstm_1to12_v5_sw_mirror.keras
        normalization.npz

Current inference pipeline:

Mirrored webcam input
-> MediaPipe Hands
-> 128-dimensional landmark sequence
-> Active Right-hand Crop
-> Sliding Window (48 frames, stride 8)
-> Geometry Normalization
-> Temporal Resampling (64 timesteps)
-> Mask-aware Normalization
-> LSTM
-> Mean Probability Aggregation
-> Class Prediction

## 12. Next Steps

1. Analyze external landmark detection and handedness.
2. Compare AI Hub and external landmark distributions.
3. Investigate class-specific confusion patterns.
4. Improve preprocessing based on the observed failure causes.
5. Evaluate alternative architectures, including BiLSTM.
6. Collect a new independent external test set.
7. Verify correct hand usage and real-time service performance.

All original and mirror models should be preserved for
subsequent controlled comparisons.
