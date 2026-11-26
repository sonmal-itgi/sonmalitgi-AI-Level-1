# v9 Medium Mirror - Deployment Model

손말잇기(Sonmalitgi) 웹서비스 연동용
한국수어 지숫자 1~12 분류 모델.

- Model version: v9_medium_mirror
- Architecture: Stacked LSTM
- Release date: 2026-11-27
- Status: Development Baseline

## 1. Package Contents

| File | Description |
|---|---|
| lstm_1to12_v9_medium_mirror.keras | 학습된 모델 및 가중치 |
| normalization.npz | Training 기반 Mean / Std |
| inference.py | 전처리 및 추론 모듈 |
| requirements.txt | Python 라이브러리 의존성 |
| verify_inference.py | 기존 평가 결과 재현성 검증 |
| README.md | 모델 연동 가이드 |

별도의 모델 학습 없이 추론할 수 있다.

## 2. Model Architecture

    Input (64, 128)
        |
    LSTM (128, return_sequences=True)
        |
    Batch Normalization
        |
    Dropout (0.3)
        |
    LSTM (64)
        |
    Batch Normalization
        |
    Dropout (0.3)
        |
    Dense (12, Softmax)

- Total parameters: 182,540
- Trainable parameters: 182,156
- Non-trainable parameters: 384
- Model file size: approximately 2.2 MB
- Input shape: (None, 64, 128)
- Output shape: (None, 12)
- Classes: Numbers 1–12

## 3. Installation

### 3.1 Tested Environment

| Component | Version |
|---|---|
| Python | 3.12.3 |
| TensorFlow | 2.21.0 |
| Keras | 3.15.1 |
| NumPy | 2.5.4 |
| Operating System | WSL Ubuntu |
| GPU | NVIDIA GeForce RTX 3060 |

GPU 환경에서 기존 실험 결과와의 재현성을 검증하였다.
CPU에서도 추론할 수 있으나 성능 및 수치 재현성은
별도로 검증해야 한다.

### 3.2 Download

    git clone https://github.com/sonmal-itgi/sonmalitgi-AI.git

    cd sonmalitgi-AI/models/v9_medium_mirror

이미 Repository를 내려받은 경우:

    git pull origin main

### 3.3 Install Dependencies

Python 3.12 기반 별도 가상환경 사용을 권장한다.

    python -m pip install -r requirements.txt

MediaPipe는 inference.py 내부에서 직접 실행하지 않는다.
Frontend 또는 별도의 Landmark Extraction 모듈에서
MediaPipe Landmark를 추출하여 전달해야 한다.

## 4. Quick Start

    import numpy as np
    from inference import SignLanguageRecognizer

    recognizer = SignLanguageRecognizer()

    landmarks = np.load("sample_landmarks.npy")

    result = recognizer.predict(landmarks)

    print(result)

sample_landmarks.npy는 사용자가 준비한 Raw Landmark
파일을 의미하며 본 Repository에는 포함되어 있지 않다.

SignLanguageRecognizer는 서버 시작 시 한 번만 생성하고
이후 사용자 입력에 대해 재사용하는 것을 권장한다.

## 5. Input Specification

### 5.1 Raw Landmark Sequence

입력 형상:

    (T, 128)

- T: 시간 순서대로 수집된 Frame 수
- 128: 각 Frame의 Feature Dimension
- dtype: float32 권장

| Index | Feature | Dimension |
|---|---|---:|
| 0–62 | Left Hand Landmarks | 63 |
| 63–125 | Right Hand Landmarks | 63 |
| 126 | Left Hand Presence Mask | 1 |
| 127 | Right Hand Presence Mask | 1 |

각 손에는 MediaPipe Hands의 Landmark 21개가 포함된다.

좌표 순서:

    x0, y0, z0, x1, y1, z1, ..., x20, y20, z20

MediaPipe의 normalized image landmarks를 사용한다.
World landmarks나 Pixel 좌표를 대신 사용하지 않는다.

### 5.2 Missing Hands

손이 검출된 경우:

    Presence Mask = 1

손이 검출되지 않은 경우:

    Presence Mask = 0
    해당 손의 63개 좌표 = 0

모든 Frame에서 양손이 검출되지 않으면
inference.py는 ValueError를 발생시킨다.

### 5.3 Mirror Orientation

중요: 모델은 Mirror 방향으로 학습되었다.

사용자 입력 영상은 기존 External Test와 동일한
Selfie Mirror 방향이어야 한다.

- 실제 Landmark 추출 대상 영상의 Mirror 방향을 확인한다.
- 이미 Mirror 처리된 영상에는 추가 Flip을 적용하지 않는다.
- CSS를 이용한 화면 반전만으로는 Landmark 추출 대상
  영상 자체가 반전되었다고 볼 수 없다.
- MediaPipe Handedness의 Left / Right 구분을 유지한다.
- 화면상 손의 위치만으로 Left / Right를 결정하지 않는다.
- 손의 Landmark 순서를 임의로 변경하지 않는다.

정확한 연동을 위해 기존 Landmark 추출 코드도 참고한다.

    scripts/extract_external_test_v3_landmarks.py
    scripts/extract_landmarks_1to12_mirror.py

### 5.4 Do Not Normalize Before Inference

inference.py는 Raw Landmark Sequence를 입력받는다.

다음 전처리를 Frontend에서 중복 수행하지 않는다.

- Wrist-centered normalization
- Hand-scale normalization
- Sliding Window
- Temporal Resampling
- Mean / Std Standardization

## 6. Inference Pipeline

    Raw Landmark Sequence (T, 128)
        |
    Input Validation
        |
    Right-hand Active Crop
        |
    Sliding Window
        |
    Geometry Normalization
        |
    Temporal Resampling
        |
    Mask-aware Standardization
        |
    Model Input (N, 64, 128)
        |
    v9 Stacked LSTM
        |
    Mean Probability Aggregation
        |
    Prediction Result

### Preprocessing Configuration

| Configuration | Value |
|---|---|
| Window size | 48 frames |
| Stride | 8 frames |
| Target frames | 64 |
| Active crop | Right Hand |
| Geometry normalization | Wrist-centered / Hand-scale |
| Standardization | Weighted Masked |
| Coordinate dimensions | 126 |
| Mask dimensions | 2 |

Training에서 저장한 normalization.npz를 사용한다.
사용자 입력마다 Mean 및 Std를 재계산하지 않는다.

## 7. Output Specification

반환 형식: Python dictionary

    {
        "prediction": 4,
        "confidence": 0.955,
        "probabilities": [...],
        "raw_frames": 88,
        "active_frames": 88,
        "num_windows": 6
    }

위 값은 출력 형식 설명을 위한 예시이다.

| Key | Description |
|---|---|
| prediction | 예측한 숫자 클래스 (1–12) |
| confidence | 예측 클래스의 평균 Softmax 값 (0–1) |
| probabilities | 클래스 1–12의 평균 Softmax 값 |
| raw_frames | 입력 Frame 수 |
| active_frames | Active Crop 이후 Frame 수 |
| num_windows | 추론에 사용된 Sliding Window 수 |

probabilities[0]은 숫자 1의 출력값이며,
probabilities[11]은 숫자 12의 출력값이다.

confidence는 실제 정답 확률이나 자세 정확도를
직접적으로 보장하지 않는다.

## 8. Web Service Integration

권장 구조:

    Frontend (Webcam)
        |
    MediaPipe Landmark Extraction
        |
    Raw Landmark Sequence
        |
    Python Backend API
        |
    SignLanguageRecognizer.predict()
        |
    JSON Response
        |
    Frontend Result Display

inference.py는 Python Backend에서 실행한다.

Frontend에서 JSON 형태의 Landmark를 전송하는 경우
predict()에 리스트 또는 NumPy 배열로 전달할 수 있다.

사용자별 입력 Sequence를 독립적으로 관리해야 한다.

모델은 한 번의 수어 수행이 포함된 Sequence를
입력받도록 설계되었다.

동작 시작 및 종료 구간 판단, 실시간 Buffer 관리,
API Endpoint 및 사용자별 요청 처리는 별도 구현이 필요하다.

TensorFlow.js 변환 및 브라우저 직접 추론은
현재 배포 범위에 포함되지 않는다.

## 9. Evaluation Results

| Dataset | Accuracy |
|---|---:|
| AI Hub Validation Full | 17/24 (70.83%) |
| AI Hub Validation Sliding | 19/24 (79.17%) |
| External Test v2 | 24/36 (66.67%) |
| External Test v3 Signer 01 | 24/36 (66.67%) |
| External Test v3 Signer 02 | 9/12 (75.00%) |
| External Test v3 Overall | 33/48 (68.75%) |
| Signer-balanced Accuracy | 70.83% |

External Test v3는 두 명의 Signer로 구성되었다.

현재 모델은 숫자 8, 9, 11에 대해
상대적으로 낮은 분류 성능을 보인다.

## 10. Deployment Verification

기존 External Test v3의 평가 결과와 비교하였다.

| Metric | Result |
|---|---:|
| Prediction matches | 48/48 |
| Confidence matches | 48/48 |
| Raw / Active Frame matches | 48/48 |
| Window count matches | 48/48 |
| Max confidence difference | 0.0 |
| Accuracy | 33/48 (68.75%) |

Deployment inference verification: PASS

검증 실행 방법:

    python verify_inference.py \
        --landmark-dir /path/to/external_test_v3 \
        --reference-csv /path/to/external_test_v3_all_mirror_results.csv

검증을 위해서는 기존 External Test v3 Landmark와
평가 CSV가 별도로 필요하다.

해당 데이터는 Repository에 포함하지 않는다.

## 11. Limitations

- 숫자 1–12만 지원한다.
- 수어 단어 및 문장 인식은 지원하지 않는다.
- 손가락 자세의 세부적인 교정 기능은 포함하지 않는다.
- 숫자 8, 9, 11 인식 성능에 개선이 필요하다.
- 학습에 사용되지 않은 외부 Signer 수가 제한적이다.
- 연속 Webcam 입력에서의 성능은 아직 검증하지 않았다.
- 실제 서비스의 Inference Latency는 측정하지 않았다.
- 정해진 클래스 이외의 동작을 구분하는
  Unknown / Rejection 기능은 아직 구현되지 않았다.
- 현재 모델은 최종 서비스 모델이 아닌 개발용 Baseline이다.

## 12. Data Source

본 모델의 학습에는 AI Hub에서 제공하는
'수어 영상' 데이터셋을 활용하였다.

- Dataset: 수어 영상
- Provider: AI Hub
- Website: https://www.aihub.or.kr/
- Dataset ID: 103
- Dataset URL:
  https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&dataSetSn=103&topMenu=100

Training: 17 Signers, 204 Videos
Validation: 2 Signers, 24 Videos

원본 영상 및 개별 Landmark 데이터는
본 배포 패키지에 포함하지 않는다.

## 13. Related Documentation

    docs/mirror_experiment_log.md
    docs/architecture_experiment_log.md

학습 및 평가 과정에 대한 상세한 내용은
위 실험 기록 문서를 참고한다.
