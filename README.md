# 손말잇기 AI

손말잇기 서비스의 한국수어 인식 AI 개발 저장소

## 현재 목표

AI Hub 수어 영상 데이터를 활용하여 지숫자 인식 모델 구축

초기 AI 파이프라인:
MP4 -> MediaPipe -> Landmark Sequence -> LSTM -> 지숫자 분류

## 현재 진행 상황

- AI Hub 수어 영상 데이터 다운로드 및 구조 분석
- 지문자 / 지숫자 데이터 구분
- 지숫자 FS0801 ~ FS1000 확인
- Validation 지숫자 영상 400개 추출
- 데이터 파일명 불일치 처리
- Training 데이터 준비 예정

## 프로젝트 구조

```text
scripts/          데이터 분석 및 추출용 스크립트
src/
  preprocessing/ MediaPipe 전처리
  training/      모델 학습
  inference/     실시간 추론
docs/             개발 및 데이터 관련 문서


## 현재 AI 파이프라인

현재 첫 번째 수어 인식 모델은 AI Hub 지숫자 데이터 중 다음 12개 클래스를 사용하고 있습니다.

| Class ID | Label |
|---|---:|
| FS0801 | 1 |
| FS0802 | 2 |
| FS0803 | 3 |
| FS0804 | 4 |
| FS0805 | 5 |
| FS0806 | 6 |
| FS0807 | 7 |
| FS0808 | 8 |
| FS0809 | 9 |
| FS0810 | 10 |
| FS0811 | 11 |
| FS0812 | 12 |

현재 파이프라인:

```text
AI Hub MP4
    ↓
morpheme annotation의 start ~ end 구간 추출
    ↓
MediaPipe Hands
    ↓
양손 landmark (128 features / frame)
    ↓
64-frame temporal resampling
    ↓
LSTM
    ↓
1 ~ 12 classification

### Baseline v3

- Training: CROWD01 ~ CROWD17
- Training signers: 17
- Training videos: 204
- Validation: CROWD18, CROWD19
- Validation videos: 24
- Validation loss: **0.6629**
- Validation accuracy: **79.17% (19/24)**

Training signer 수를 17명까지 확장한 결과, 동일한 전처리 및 LSTM 구조에서 validation accuracy가 79.17%까지 증가했습니다.

### Baseline Comparison

| Version | Training Signers | Training Videos | Validation Accuracy |
|---|---:|---:|---:|
| v1 | 2 | 24 | 8.33% |
| v2 | 5 | 60 | 37.50% |
| v3 | 17 | 204 | 79.17% |

현재 결과는 Training signer 다양성이 새로운 signer에 대한 일반화 성능에 큰 영향을 준다는 것을 보여줍니다.

다음 단계에서는 baseline v3를 기준으로 landmark 정규화, 짧은 annotation 구간 처리, presence mask 처리 방식 등의 전처리 개선을 실험할 예정입니다.