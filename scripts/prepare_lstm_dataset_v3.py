import csv
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent

MANIFEST_PATH = (
    BASE_DIR
    / "numeric_1to12"
    / "landmark_manifest_v3.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "numeric_1to12"
    / "lstm_dataset_v3.npz"
)

SEQUENCE_LENGTH = 64


def resample_sequence(sequence, target_length=64):
    """
    (원본 프레임 수, feature_dim)
    형태의 시퀀스를

    (target_length, feature_dim)

    형태로 시간축 선형 보간한다.
    """

    original_length = len(sequence)

    if original_length == 0:
        raise ValueError("빈 landmark sequence입니다.")

    # 이미 원하는 길이면 그대로 반환
    if original_length == target_length:
        return sequence.astype(np.float32)

    # 원래 시간 위치
    old_positions = np.linspace(
        0.0,
        1.0,
        original_length
    )

    # 새 시간 위치
    new_positions = np.linspace(
        0.0,
        1.0,
        target_length
    )

    result = np.zeros(
        (
            target_length,
            sequence.shape[1]
        ),
        dtype=np.float32
    )

    # feature 하나씩 보간
    for feature_index in range(
        sequence.shape[1]
    ):
        result[:, feature_index] = np.interp(
            new_positions,
            old_positions,
            sequence[:, feature_index]
        )

    return result


# -----------------------------------------
# 데이터 저장용
# -----------------------------------------

X_train = []
y_train = []

X_val = []
y_val = []

train_info = []
val_info = []


# -----------------------------------------
# Manifest 읽기
# -----------------------------------------

with open(
    MANIFEST_PATH,
    "r",
    encoding="utf-8-sig"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        landmark_path = Path(
            row["landmark_path"]
        )

        sequence = np.load(
            landmark_path
        )

        # -----------------------------
        # 64 frame으로 변환
        # -----------------------------

        sequence = resample_sequence(
            sequence,
            SEQUENCE_LENGTH
        )

        # -----------------------------
        # label
        #
        # 실제 숫자:
        # 1 ~ 12
        #
        # 신경망 class index:
        # 0 ~ 11
        # -----------------------------

        number_label = int(
            row["label"]
        )

        class_index = (
            number_label - 1
        )

        info = {
            "class_id": row["class_id"],
            "label": number_label,
            "signer": row["signer"],
            "video_name": row["video_name"],
        }

        # -----------------------------
        # Training / Validation 분리
        # -----------------------------

        if row["split"] == "Training":

            X_train.append(sequence)
            y_train.append(class_index)
            train_info.append(info)

        elif row["split"] == "Validation":

            X_val.append(sequence)
            y_val.append(class_index)
            val_info.append(info)


# -----------------------------------------
# numpy 배열 변환
# -----------------------------------------

X_train = np.array(
    X_train,
    dtype=np.float32
)

y_train = np.array(
    y_train,
    dtype=np.int64
)

X_val = np.array(
    X_val,
    dtype=np.float32
)

y_val = np.array(
    y_val,
    dtype=np.int64
)


# -----------------------------------------
# 저장
# -----------------------------------------

np.savez_compressed(
    OUTPUT_PATH,

    X_train=X_train,
    y_train=y_train,

    X_val=X_val,
    y_val=y_val,
)


# -----------------------------------------
# 결과 출력
# -----------------------------------------

print("=" * 60)
print("LSTM Dataset 생성 완료")
print("=" * 60)

print()
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print()
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print()
print(
    "Training labels:",
    sorted(set(y_train.tolist()))
)

print(
    "Validation labels:",
    sorted(set(y_val.tolist()))
)

print()
print("저장 위치:")
print(OUTPUT_PATH)