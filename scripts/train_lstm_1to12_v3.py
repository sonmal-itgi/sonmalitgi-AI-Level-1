from pathlib import Path
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ==================================================
# 기본 설정
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "numeric_1to12"
    / "lstm_dataset_v3.npz"
)

MODEL_DIR = (
    BASE_DIR
    / "numeric_1to12"
    / "models_v3"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR
    / "lstm_1to12.keras"
)


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==================================================
# 데이터 로드
# ==================================================

data = np.load(DATA_PATH)

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]


print("===== Dataset =====")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)


# ==================================================
# Feature 정규화
#
# landmark 좌표 126개만 정규화
# 마지막 2개는 hand presence mask이므로 그대로 유지
# ==================================================

coordinate_train = X_train[:, :, :126]

mean = coordinate_train.mean(
    axis=(0, 1),
    keepdims=True
)

std = coordinate_train.std(
    axis=(0, 1),
    keepdims=True
)

std[std < 1e-6] = 1.0


X_train_normalized = X_train.copy()
X_val_normalized = X_val.copy()


X_train_normalized[:, :, :126] = (
    X_train[:, :, :126] - mean
) / std


X_val_normalized[:, :, :126] = (
    X_val[:, :, :126] - mean
) / std


# ==================================================
# LSTM 모델
# ==================================================

model = keras.Sequential([
    layers.Input(
        shape=(64, 128)
    ),

    layers.LSTM(
        32,
        return_sequences=False
    ),

    layers.Dropout(
        0.4
    ),

    layers.Dense(
        24,
        activation="relu"
    ),

    layers.Dropout(
        0.3
    ),

    layers.Dense(
        12,
        activation="softmax"
    ),
])


model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


print()
print("===== Model =====")

model.summary()


# ==================================================
# Callback
# ==================================================

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
)


# ==================================================
# 학습
# ==================================================

print()
print("===== Training =====")

history = model.fit(
    X_train_normalized,
    y_train,

    validation_data=(
        X_val_normalized,
        y_val
    ),

    epochs=150,
    batch_size=4,

    callbacks=[
        early_stopping
    ],

    verbose=1,
)


# ==================================================
# Validation 평가
# ==================================================

print()
print("===== Validation Evaluation =====")

val_loss, val_accuracy = model.evaluate(
    X_val_normalized,
    y_val,
    verbose=0
)

print(
    f"Validation loss: {val_loss:.4f}"
)

print(
    f"Validation accuracy: "
    f"{val_accuracy * 100:.2f}%"
)


# ==================================================
# 각 Validation 영상 예측
# ==================================================

probabilities = model.predict(
    X_val_normalized,
    verbose=0
)

predictions = np.argmax(
    probabilities,
    axis=1
)


print()
print("===== Validation Predictions =====")

correct = 0

for index, (
    true_class,
    pred_class
) in enumerate(
    zip(y_val, predictions),
    start=1
):

    # class index 0~11
    # 실제 숫자 1~12
    true_number = int(
        true_class + 1
    )

    predicted_number = int(
        pred_class + 1
    )

    is_correct = (
        true_number
        == predicted_number
    )

    if is_correct:
        correct += 1

    mark = (
        "O"
        if is_correct
        else "X"
    )

    confidence = (
        probabilities[
            index - 1,
            pred_class
        ]
        * 100
    )

    print(
        f"{index:02d} | "
        f"정답={true_number:2d} | "
        f"예측={predicted_number:2d} | "
        f"confidence={confidence:5.1f}% | "
        f"{mark}"
    )


print()
print(
    f"정답 개수: "
    f"{correct}/{len(y_val)}"
)


# ==================================================
# 모델 저장
# ==================================================

model.save(
    MODEL_PATH
)


# 정규화 값도 저장
NORMALIZATION_PATH = (
    MODEL_DIR
    / "normalization.npz"
)

np.savez(
    NORMALIZATION_PATH,
    mean=mean,
    std=std,
)


print()
print("===== 저장 완료 =====")
print("모델:")
print(MODEL_PATH)

print()
print("정규화 정보:")
print(NORMALIZATION_PATH)