from pathlib import Path
import argparse
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ==================================================
# 기본 설정
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

EPOCHS = 150
BATCH_SIZE = 4


# ==================================================
# argument
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--version",
    choices=[
        "v4a",
        "v4b",
    ],
    required=True,
)

args = parser.parse_args()

VERSION = args.version


# ==================================================
# 경로
# ==================================================

DATA_PATH = (
    BASE_DIR
    / "numeric_1to12"
    / f"lstm_dataset_{VERSION}.npz"
)

MODEL_DIR = (
    BASE_DIR
    / "numeric_1to12"
    / f"models_{VERSION}"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = (
    MODEL_DIR
    / f"lstm_1to12_{VERSION}.keras"
)

NORMALIZATION_PATH = (
    MODEL_DIR
    / "normalization.npz"
)


# ==================================================
# 데이터 로드
# ==================================================

data = np.load(
    DATA_PATH,
    allow_pickle=True,
)

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

train_names = data["train_names"]
val_names = data["val_names"]


print()
print("=" * 60)
print("Dataset")
print("=" * 60)

print("version:", VERSION)

print(
    "X_train:",
    X_train.shape,
)

print(
    "y_train:",
    y_train.shape,
)

print(
    "X_val:",
    X_val.shape,
)

print(
    "y_val:",
    y_val.shape,
)


# ==================================================
# Mask-aware normalization
#
# 중요:
#
# v3:
# missing hand의 raw 0도
# (0 - mean) / std 처리됨
#
# v4:
# 실제 손이 존재하는 좌표만
# mean/std 계산 및 normalization
#
# missing hand는 계속 0 유지
# ==================================================

def calculate_masked_normalization(X):

    # ----------------------------------------------
    # presence mask
    # ----------------------------------------------

    left_present = (
        X[:, :, 126] > 0.5
    )

    right_present = (
        X[:, :, 127] > 0.5
    )


    # ----------------------------------------------
    # 실제 존재하는 손 좌표만 가져오기
    # ----------------------------------------------

    left_values = (
        X[:, :, 0:63][
            left_present
        ]
    )

    right_values = (
        X[:, :, 63:126][
            right_present
        ]
    )


    # ----------------------------------------------
    # feature별 mean/std
    # ----------------------------------------------

    left_mean = left_values.mean(
        axis=0
    )

    left_std = left_values.std(
        axis=0
    )

    right_mean = right_values.mean(
        axis=0
    )

    right_std = right_values.std(
        axis=0
    )


    # 너무 작은 std 방지
    left_std[
        left_std < 1e-6
    ] = 1.0

    right_std[
        right_std < 1e-6
    ] = 1.0


    mean = np.concatenate([
        left_mean,
        right_mean,
    ]).astype(np.float32)

    std = np.concatenate([
        left_std,
        right_std,
    ]).astype(np.float32)


    return mean, std


def apply_masked_normalization(
    X,
    mean,
    std,
):

    output = X.copy().astype(
        np.float32
    )


    # ----------------------------------------------
    # presence
    # ----------------------------------------------

    left_present = (
        output[:, :, 126] > 0.5
    )

    right_present = (
        output[:, :, 127] > 0.5
    )


    # ----------------------------------------------
    # Left
    # ----------------------------------------------

    left_coords = (
        output[:, :, 0:63]
    )

    left_coords[
        left_present
    ] = (
        left_coords[
            left_present
        ]
        - mean[0:63]
    ) / std[0:63]


    # ----------------------------------------------
    # Right
    # ----------------------------------------------

    right_coords = (
        output[:, :, 63:126]
    )

    right_coords[
        right_present
    ] = (
        right_coords[
            right_present
        ]
        - mean[63:126]
    ) / std[63:126]


    # ----------------------------------------------
    # missing hand는 반드시 0
    # ----------------------------------------------

    left_coords[
        ~left_present
    ] = 0.0

    right_coords[
        ~right_present
    ] = 0.0


    # presence mask는 그대로 0/1
    output[:, :, 126:128] = (
        output[:, :, 126:128] > 0.5
    ).astype(np.float32)


    return output


# ==================================================
# Training 데이터에서만 mean/std 계산
# ==================================================

mean, std = (
    calculate_masked_normalization(
        X_train
    )
)

X_train_normalized = (
    apply_masked_normalization(
        X_train,
        mean,
        std,
    )
)

X_val_normalized = (
    apply_masked_normalization(
        X_val,
        mean,
        std,
    )
)


# ==================================================
# Normalization 저장
# ==================================================

np.savez(
    NORMALIZATION_PATH,

    mean=mean.reshape(
        1,
        1,
        126,
    ),

    std=std.reshape(
        1,
        1,
        126,
    ),

    normalization_type=np.array(
        "masked"
    ),
)


print()
print("=" * 60)
print("Normalization")
print("=" * 60)

print(
    "mean range:",
    float(mean.min()),
    "~",
    float(mean.max()),
)

print(
    "std range:",
    float(std.min()),
    "~",
    float(std.max()),
)


# ==================================================
# Missing hand 검증
# ==================================================

left_missing = (
    X_train_normalized[:, :, 126]
    < 0.5
)

right_missing = (
    X_train_normalized[:, :, 127]
    < 0.5
)

left_missing_values = (
    X_train_normalized[
        :, :, 0:63
    ][left_missing]
)

right_missing_values = (
    X_train_normalized[
        :, :, 63:126
    ][right_missing]
)


print()
print("Missing-hand check")

print(
    "Left missing max abs:",
    np.abs(
        left_missing_values
    ).max()
    if left_missing_values.size
    else 0.0
)

print(
    "Right missing max abs:",
    np.abs(
        right_missing_values
    ).max()
    if right_missing_values.size
    else 0.0
)


# ==================================================
# Model
#
# v3와 동일한 구조 유지
# ==================================================

model = keras.Sequential([
    layers.Input(
        shape=(64, 128)
    ),

    layers.LSTM(
        32
    ),

    layers.Dropout(
        0.4
    ),

    layers.Dense(
        24,
        activation="relu",
    ),

    layers.Dropout(
        0.3
    ),

    layers.Dense(
        12,
        activation="softmax",
    ),
])


model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss=(
        "sparse_categorical_crossentropy"
    ),

    metrics=[
        "accuracy"
    ],
)


print()
print("=" * 60)
print("Model")
print("=" * 60)

model.summary()


# ==================================================
# Early stopping
# ==================================================

early_stopping = (
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1,
    )
)


# ==================================================
# 학습
# ==================================================

print()
print("=" * 60)
print(
    f"Training {VERSION}"
)
print("=" * 60)

history = model.fit(
    X_train_normalized,
    y_train,

    validation_data=(
        X_val_normalized,
        y_val,
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping
    ],

    verbose=1,
)


# ==================================================
# Validation 평가
# ==================================================

val_loss, val_accuracy = (
    model.evaluate(
        X_val_normalized,
        y_val,
        verbose=0,
    )
)


print()
print("=" * 60)
print("Validation")
print("=" * 60)

print(
    "loss:",
    f"{val_loss:.4f}"
)

print(
    "accuracy:",
    f"{val_accuracy * 100:.2f}%"
)


# ==================================================
# Validation prediction
# ==================================================

predictions = model.predict(
    X_val_normalized,
    verbose=0,
)

predicted_classes = np.argmax(
    predictions,
    axis=1,
)


print()
print("=" * 60)
print("Validation predictions")
print("=" * 60)


correct = 0

for i in range(
    len(y_val)
):

    target = (
        int(y_val[i]) + 1
    )

    prediction = (
        int(
            predicted_classes[i]
        )
        + 1
    )

    confidence = float(
        predictions[
            i,
            predicted_classes[i]
        ]
    )

    is_correct = (
        target == prediction
    )

    if is_correct:
        correct += 1

    print(
        f"{val_names[i]} "
        f"target={target:2d} "
        f"pred={prediction:2d} "
        f"conf={confidence * 100:5.1f}% "
        f"{'O' if is_correct else 'X'}"
    )


print()

print(
    "Correct:",
    correct,
    "/",
    len(y_val),
)

print(
    "Accuracy:",
    f"{correct / len(y_val) * 100:.2f}%"
)


# ==================================================
# 모델 저장
# ==================================================

model.save(
    MODEL_PATH
)


print()
print("=" * 60)
print("저장 완료")
print("=" * 60)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Normalization:",
    NORMALIZATION_PATH
)