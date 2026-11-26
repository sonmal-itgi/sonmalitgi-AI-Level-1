# LSTM Architecture Comparison and Cross-signer Evaluation

- Experiment period: 2026-11-26 ~ 2026-11-27
- Project: 손말잇기 (Sonmalitgi)
- Task: Korean Sign Language Number Recognition (1–12)

---

## 1. Experiment Objective

기존 Mirror Preprocessing 및 Sliding Window 기반 LSTM 모델을
기준으로 Architecture에 따른 분류 성능을 비교한다.

주요 실험 목적은 다음과 같다.

- LSTM Unit 수 증가에 따른 성능 변화 확인
- Single LSTM과 Stacked LSTM의 성능 비교
- Bidirectional LSTM 및 Temporal Pooling 적용 효과 확인
- AI Hub Validation과 External Test 간 성능 차이 분석
- 새로운 Signer에 대한 Generalization 성능 검증
- 향후 Feature 개선 실험에 사용할 Baseline 선정

---

## 2. Dataset

### 2.1 AI Hub Dataset

| Dataset | Signers | Classes | Videos |
|---|---:|---:|---:|
| Training | 17 | 12 | 204 |
| Validation | 2 | 12 | 24 |

- Training: CROWD01–CROWD17
- Validation: CROWD18–CROWD19
- Target classes: Numbers 1–12
- Signer-independent training/validation split

### 2.2 Sliding Window Dataset

| Dataset | Original videos | Windows |
|---|---:|---:|
| Training | 204 | 705 |
| Validation | 24 | 96 |

Dataset file:

    numeric_1to12/lstm_dataset_v5_sw_mirror.npz

모든 Architecture 비교 실험에서 동일한 Mirror 기반
Sliding Window Dataset을 사용한다.

### 2.3 External Test v2

- Signers: 1
- Classes: 12
- Videos per class: 3
- Total videos: 36
- Filename: test_XX01 ~ test_XX03

### 2.4 External Test v3

External Test v2에 새로운 Signer의 영상 12개를 추가하였다.

| Dataset | Signer 01 | Signer 02 | Total |
|---|---:|---:|---:|
| External Test v2 | 36 | 0 | 36 |
| External Test v3 | 36 | 12 | 48 |

- Signer 01: repetition 01, 02, 03
- Signer 02: repetition 04
- Total classes: 12
- Videos per class: 4

Filename examples:

    test_0101.mp4
    test_0102.mp4
    test_0103.MOV
    test_0104.mp4

External Test v3 dataset verification:

- Total videos: 48
- Missing videos: 0
- Unexpected filenames: 0
- Duplicate filenames: 0
- Landmark extraction success: 48
- Landmark extraction failure: 0

---

## 3. Common Preprocessing

Architecture 비교 시 입력 데이터 및 전처리 조건을 유지하였다.

### 3.1 Landmark Extraction

- MediaPipe Hands
- Maximum hands: 2
- Landmarks per hand: 21
- Coordinates per landmark: x, y, z
- Hand presence masks: 2

Feature dimension:

    Left hand:  63
    Right hand: 63
    Hand masks:  2
    Total:     128

### 3.2 Mirror Preprocessing

AI Hub 영상에 Horizontal Flip을 적용하여
실제 서비스의 Mirror 영상 방향과 일치시킨다.

External Test 영상은 Mirror 방향으로 저장된 영상을
사용하므로 추가적인 Horizontal Flip을 적용하지 않는다.

### 3.3 Geometry and Temporal Preprocessing

- Right-hand-based Active Crop
- Wrist-centered coordinates
- Hand-scale normalization
- Sliding Window size: 48 frames
- Sliding Window stride: 8 frames
- Temporal Resampling: 64 frames
- Mask-aware Standardization
- Source-balanced sample weighting

Final model input:

    (64, 128)

### 3.4 Prediction Aggregation

각 Sliding Window의 Softmax Probability를 평균하여
영상 단위 최종 Prediction을 산출한다.

---

## 4. Model Architectures

### 4.1 v5_mirror

    Input (64, 128)
    LSTM (32)
    Dropout (0.4)
    Dense (24, ReLU)
    Dropout (0.3)
    Dense (12, Softmax)

### 4.2 v6_mirror

    Input (64, 128)
    LSTM (64, return_sequences=True)
    Batch Normalization
    Dropout (0.3)
    LSTM (32)
    Batch Normalization
    Dropout (0.3)
    Dense (12, Softmax)

L2 regularization is applied to LSTM kernel/recurrent
weights and the output Dense kernel.

### 4.3 v7_expanded_mirror

    Input (64, 128)
    LSTM (256, return_sequences=True)
    Batch Normalization
    Dropout (0.3)
    LSTM (128)
    Batch Normalization
    Dropout (0.3)
    Dense (12, Softmax)

v6와 동일한 Architecture에서 LSTM Units만 증가시켰다.

### 4.4 v8_bilstm_pooling_mirror

    Input (64, 128)
    Layer Normalization
    Dense (64)
    GELU
    Dropout (0.0)
    Bidirectional LSTM (64 units per direction)
    Global Average Pooling + Global Max Pooling
    Concatenate (256)
    Dropout (0.0)
    Dense (12, Softmax)

기존 입력 데이터와의 비교를 위해 입력 크기를
(64, 128)로 유지하였다.

v8에는 L2 regularization을 적용하지 않았다.
따라서 v6/v7/v9와의 비교는 Architecture 및
Regularization 구성이 함께 달라진 비교이다.

### 4.5 v9_medium_mirror

    Input (64, 128)
    LSTM (128, return_sequences=True)
    Batch Normalization
    Dropout (0.3)
    LSTM (64)
    Batch Normalization
    Dropout (0.3)
    Dense (12, Softmax)

v6와 v7의 중간 규모 Architecture를 구성하였다.

### 4.6 Model Size Comparison

| Model | Architecture | Total Parameters |
|---|---|---:|
| v5 | Single LSTM (32) | 21,700 |
| v6 | Stacked LSTM (64, 32) | 62,604 |
| v7 | Stacked LSTM (256, 128) | 594,444 |
| v8 | BiLSTM + Temporal Pooling | 77,644 |
| v9 | Stacked LSTM (128, 64) | 182,540 |

Parameter 수는 Architecture 기반 이론적 계산값이며,
Batch Normalization의 non-trainable parameters를 포함한다.
Optimizer state는 포함하지 않는다.

---

## 5. Training Configuration

| Configuration | Value |
|---|---|
| Framework | TensorFlow / Keras |
| GPU | NVIDIA GeForce RTX 3060 |
| Random Seed | 42 |
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| Batch Size | 8 |
| Maximum Epochs | 150 |
| Early Stopping Monitor | val_loss |
| Early Stopping Patience | 20 |
| Input Shape | (64, 128) |
| Output Classes | 12 |

v6, v7, v9의 L2 regularization coefficient는 0.005이다.

학습 로그 기반 수렴 과정 및 Overfitting 분석은
향후 별도로 수행할 예정이다.

---

## 6. AI Hub Validation Results

| Model | Full Accuracy | Sliding Accuracy |
|---|---:|---:|
| v5 | 20/24 (83.33%) | 22/24 (91.67%) |
| v6 | 19/24 (79.17%) | 21/24 (87.50%) |
| v7 | 22/24 (91.67%) | 22/24 (91.67%) |
| v8 | 21/24 (87.50%) | 21/24 (87.50%) |
| v9 | 17/24 (70.83%) | 19/24 (79.17%) |

v7은 Full Accuracy에서 가장 높은 성능을 보였다.

v5와 v7은 Sliding Window 평가에서
동일하게 91.67%의 정확도를 달성하였다.

v9는 비교 모델 중 가장 낮은 Validation Accuracy를 보였다.

Validation Dataset은 학습 중 모델 선택에도 사용되었으므로
독립적인 최종 Test Dataset으로 간주하지 않는다.

---

## 7. External Test v2 Results

| Model | Correct / Total | Accuracy |
|---|---:|---:|
| v5 | 18/36 | 50.00% |
| v6 | 19/36 | 52.78% |
| v7 | 19/36 | 52.78% |
| v8 | 17/36 | 47.22% |
| v9 | 24/36 | 66.67% |

v9는 기존 Signer에 대해 가장 높은 정확도를 달성하였다.

그러나 단일 Signer의 영상만을 사용하였기 때문에
특정 Signer에 대한 모델 편향 가능성을 고려해야 한다.

---

## 8. External Test v3 Results

### 8.1 Evaluation Metrics

Overall Accuracy:

    (Correct_signer01 + Correct_signer02) / 48

Signer-balanced Accuracy:

    (Accuracy_signer01 + Accuracy_signer02) / 2

Signer Accuracy Gap:

    abs(Accuracy_signer01 - Accuracy_signer02)

Signer-balanced Accuracy는 각 Signer에게 동일한
가중치를 부여하여 영상 수 불균형의 영향을 줄인다.

### 8.2 Overall and Signer-level Results

| Model | Signer 01 | Signer 02 | Overall | Balanced | Gap |
|---|---:|---:|---:|---:|---:|
| v5 | 18/36 (50.00%) | 5/12 (41.67%) | 23/48 (47.92%) | 45.83% | 8.33%p |
| v6 | 19/36 (52.78%) | 6/12 (50.00%) | 25/48 (52.08%) | 51.39% | 2.78%p |
| v7 | 19/36 (52.78%) | 8/12 (66.67%) | 27/48 (56.25%) | 59.72% | 13.89%p |
| v8 | 17/36 (47.22%) | 6/12 (50.00%) | 23/48 (47.92%) | 48.61% | 2.78%p |
| v9 | 24/36 (66.67%) | 9/12 (75.00%) | 33/48 (68.75%) | 70.83% | 8.33%p |

Evaluation integrity check:

- Models: 5
- Videos per model: 48
- Total predictions: 240
- Integrity check: PASS

다섯 모델 모두 기존 Signer에 대한 Accuracy가
External Test v2의 결과와 동일하게 재현되었다.

### 8.3 Signer 01 Recording Groups

| Model | Repetition 01/02 | Repetition 03 |
|---|---:|---:|
| v5 | 11/24 (45.83%) | 7/12 (58.33%) |
| v6 | 13/24 (54.17%) | 6/12 (50.00%) |
| v7 | 13/24 (54.17%) | 6/12 (50.00%) |
| v8 | 10/24 (41.67%) | 7/12 (58.33%) |
| v9 | 16/24 (66.67%) | 8/12 (66.67%) |

### 8.4 Per-class Accuracy

각 셀은 클래스별 정분류 영상 수를 나타낸다.
클래스당 전체 영상 수는 4개이다.

| Class | v5 | v6 | v7 | v8 | v9 |
|---|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 2 | 0 | 4 |
| 2 | 0 | 4 | 4 | 4 | 4 |
| 3 | 3 | 3 | 1 | 4 | 2 |
| 4 | 0 | 0 | 0 | 0 | 4 |
| 5 | 4 | 4 | 4 | 1 | 4 |
| 6 | 4 | 4 | 3 | 4 | 3 |
| 7 | 4 | 1 | 4 | 2 | 4 |
| 8 | 1 | 0 | 1 | 1 | 1 |
| 9 | 0 | 0 | 0 | 0 | 0 |
| 10 | 4 | 3 | 3 | 3 | 3 |
| 11 | 0 | 4 | 1 | 0 | 0 |
| 12 | 3 | 2 | 4 | 4 | 4 |
| Total | 23 | 25 | 27 | 23 | 33 |

### 8.5 Signer 02 Predictions

각 셀은 신규 Signer 영상에 대한 Prediction을 나타낸다.

| Target | v5 | v6 | v7 | v8 | v9 |
|---|---:|---:|---:|---:|---:|
| 1 | 2 | 11 | 1 | 2 | 1 |
| 2 | 12 | 2 | 2 | 2 | 2 |
| 3 | 12 | 2 | 3 | 3 | 3 |
| 4 | 10 | 10 | 2 | 3 | 4 |
| 5 | 5 | 5 | 5 | 10 | 5 |
| 6 | 6 | 6 | 6 | 6 | 6 |
| 7 | 7 | 6 | 7 | 7 | 7 |
| 8 | 6 | 6 | 7 | 6 | 6 |
| 9 | 7 | 6 | 8 | 6 | 8 |
| 10 | 10 | 10 | 10 | 10 | 10 |
| 11 | 2 | 11 | 1 | 2 | 1 |
| 12 | 12 | 12 | 12 | 12 | 12 |

---

## 9. Error Analysis

### 9.1 Common Classification Errors

숫자 8과 9는 신규 Signer에 대해서도
다섯 모델 모두 오분류하였다.

특히 숫자 9는 두 Signer의 영상 4개에 대한
20건의 Prediction에서 모두 오분류되었다.

전체 240건의 Prediction에서도
숫자 9가 출력된 사례는 존재하지 않았다.

이는 특정 Signer만의 문제보다는 Feature 표현,
Landmark 추출 및 전처리 과정에서 공통적인 문제가
존재할 가능성을 시사한다.

단, 현재 결과만으로 원인을 확정할 수는 없다.

### 9.2 v9 Confusion Matrix

Rows: True classes
Columns: Predicted classes

      1  2  3  4  5  6  7  8  9 10 11 12
   1  4  0  0  0  0  0  0  0  0  0  0  0
   2  0  4  0  0  0  0  0  0  0  0  0  0
   3  0  2  2  0  0  0  0  0  0  0  0  0
   4  0  0  0  4  0  0  0  0  0  0  0  0
   5  0  0  0  0  4  0  0  0  0  0  0  0
   6  0  0  0  0  0  3  1  0  0  0  0  0
   7  0  0  0  0  0  0  4  0  0  0  0  0
   8  0  0  0  0  0  1  2  1  0  0  0  0
   9  0  0  0  0  0  1  0  3  0  0  0  0
  10  0  0  0  0  0  0  0  0  0  3  0  1
  11  4  0  0  0  0  0  0  0  0  0  0  0
  12  0  0  0  0  0  0  0  0  0  0  0  4

### 9.3 v9 Major Classification Errors

| True Class | Predicted Class | Count |
|---|---|---:|
| 11 | 1 | 4 |
| 9 | 8 | 3 |
| 8 | 7 | 2 |
| 3 | 2 | 2 |
| 6 | 7 | 1 |
| 8 | 6 | 1 |
| 9 | 6 | 1 |
| 10 | 12 | 1 |

전체 오분류 15건 중 숫자 8, 9, 11에 대한
오분류가 11건으로 73.33%를 차지하였다.

### 9.4 Class 4 Recognition

v9는 숫자 4의 전체 영상 4개를 모두 올바르게 분류하였다.

| Filename | Prediction | Confidence |
|---|---:|---:|
| test_0401 | 4 | 95.5% |
| test_0402 | 4 | 94.6% |
| test_0403 | 4 | 86.3% |
| test_0404 | 4 | 91.8% |

나머지 네 모델은 숫자 4의 모든 영상을 오분류하였다.

---

## 10. Results and Discussion

### 10.1 Architecture Capacity

동일한 Stacked LSTM 계열에서 Unit 수를 변경한 결과:

| Model | LSTM Units | Parameters | External v3 |
|---|---|---:|---:|
| v6 | 64, 32 | 62,604 | 52.08% |
| v9 | 128, 64 | 182,540 | 68.75% |
| v7 | 256, 128 | 594,444 | 56.25% |

LSTM Unit 수의 증가가 외부 영상 분류 성능의
단조로운 증가로 이어지지는 않았다.

v9는 v7보다 약 69.3% 적은 Parameter를 사용하면서
External Test v3에서 12.50%p 높은 정확도를 달성하였다.

이는 현재 실험 조건에서 관찰된 결과이며,
최적 Model Capacity가 확정되었음을 의미하지 않는다.

### 10.2 Cross-signer Generalization

v9는 기존 Signer와 신규 Signer 모두에서
비교 모델 중 가장 높은 Accuracy를 달성하였다.

- Signer 01: 66.67%
- Signer 02: 75.00%
- Overall Accuracy: 68.75%
- Signer-balanced Accuracy: 70.83%

신규 Signer에 대해서도 v9의 상대적 우수성이
관찰되었다.

다만 신규 Signer의 영상 수는 12개이며,
v7과 v9의 신규 Signer 정답 수 차이는 1개이다.
일반화 성능의 통계적 우수성을 확정하기에는
추가적인 평가 데이터가 필요하다.

### 10.3 Validation and External Performance

v9는 AI Hub Validation에서 가장 낮은 정확도를
보였지만 External Test에서는 가장 높은 정확도를 보였다.

따라서 AI Hub Validation Accuracy만으로 실제
서비스 환경에서의 성능을 판단하기는 어렵다.

AI Hub와 External Dataset 간 성능 차이가
Overfitting 때문인지, Domain Shift 때문인지,
또는 다른 원인 때문인지는 추가 분석이 필요하다.

---

## 11. Current Model Selection

향후 Feature Engineering 및 전처리 개선 실험에서는
v9_medium_mirror를 우선 Baseline으로 사용한다.

선정 근거:

- 두 External Signer에서 가장 높은 Accuracy
- 가장 높은 Overall Accuracy
- 가장 높은 Signer-balanced Accuracy
- v7 대비 작은 Model Size

이는 현재까지의 실험 결과에 기반한 잠정적인 선정이며,
최종 서비스 모델 확정을 의미하지 않는다.

---

## 12. Limitations

- Training 원본 영상 수가 204개로 제한적이다.
- Sliding Window는 원본 영상 및 Signer의 수를 증가시키지 않는다.
- AI Hub Validation Signer는 2명이다.
- External Test Signer는 2명이다.
- 신규 Signer의 클래스당 영상 수는 1개이다.
- Architecture 비교는 단일 Random Seed에 기반한다.
- 기존 External Test 영상은 모델 선정에 반복 사용되었다.
- External Test v3 역시 이후 모델 개선에 사용될 경우
  독립적인 최종 평가 데이터로 사용할 수 없다.
- Signer 특성과 촬영 환경의 영향을 분리하지 못하였다.
- 학습 로그 기반 Overfitting 분석은 아직 수행하지 않았다.
- 실제 Inference Latency 및 메모리 사용량은 측정하지 않았다.
- 연속적인 실시간 사용자 입력에 대한 평가는 수행하지 않았다.
- Softmax Confidence는 수어 동작의 정확성이나
  실제 정답 확률을 직접 보장하지 않는다.

---

## 13. Future Work

1. v7, v8, v9 학습 로그 및 수렴 과정 분석
2. AI Hub와 External Dataset의 원본 영상 비교
3. Landmark 추출 품질 및 검출 실패 원인 분석
4. 숫자 1/11 및 6/7/8/9의 Feature 분포 분석
5. Joint Angle, Bone Vector, Finger Distance 검토
6. Palm Orientation 및 Hand Rotation 분석
7. v9 Architecture를 고정한 Feature Ablation Study
8. Multiple Random Seed 실험
9. 신규 Signer를 활용한 독립적인 Test Dataset 구축
10. 실제 서비스 환경에서의 Inference Latency 평가

---

## 14. Related Scripts

    scripts/train_lstm_1to12_v7_expanded_mirror.py
    scripts/train_lstm_1to12_v8_bilstm_pooling_mirror.py
    scripts/train_lstm_1to12_v9_medium_mirror.py
    scripts/evaluate_external_test_v2_all_mirror.py
    scripts/extract_external_test_v3_landmarks.py
    scripts/evaluate_external_test_v3_all_mirror.py

Experiment output:

    local_inference/work/external_test_v3/
    external_test_v3_all_mirror_results.csv

Training logs:

    logs/v7_expanded_mirror.log
    logs/v8_bilstm_pooling_mirror.log
    logs/v9_medium_mirror.log

원본 영상, Landmark, 학습 Dataset, 모델 파일 및
원본 로그는 Git Repository에 포함하지 않는다.

스크립트의 실행 기준 경로는 기존 실험 작업 디렉터리인
~/handmalitgi/aihub이다.

Repository의 scripts/ 폴더에서 직접 실행할 경우
BASE_DIR 및 Dataset 상대경로 설정을 확인해야 한다.
