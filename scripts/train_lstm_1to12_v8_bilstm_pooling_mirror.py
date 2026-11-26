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
    / "lstm_dataset_v5_sw_mirror.npz"
)

MODEL_DIR = (
    BASE_DIR
    / "numeric_1to12"
    / "models_v8_bilstm_pooling_mirror"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = (
    MODEL_DIR
    / "lstm_1to12_v8_bilstm_pooling_mirror.keras"
)

NORMALIZATION_PATH = (
    MODEL_DIR
    / "normalization.npz"
)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


EPOCHS = 150
BATCH_SIZE = 8

LEFT_SLICE = slice(0, 63)
RIGHT_SLICE = slice(63, 126)

LEFT_MASK = 126
RIGHT_MASK = 127


# ==================================================
# 데이터 로드
# ==================================================

data = np.load(
    DATA_PATH,
    allow_pickle=True,
)

X_train = data[
    "X_train"
].astype(np.float32)

y_train = data[
    "y_train"
].astype(np.int64)

train_source_names = data[
    "train_source_names"
]

train_sample_weights = data[
    "train_sample_weights"
].astype(np.float32)


X_val_full = data[
    "X_val_full"
].astype(np.float32)

y_val_full = data[
    "y_val_full"
].astype(np.int64)

val_full_names = data[
    "val_full_names"
]


X_val_windows = data[
    "X_val_windows"
].astype(np.float32)

y_val_windows = data[
    "y_val_windows"
].astype(np.int64)

val_window_names = data[
    "val_window_names"
]


print()
print("=" * 70)
print("DATASET")
print("=" * 70)

print(
    "X_train:",
    X_train.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "X_val_full:",
    X_val_full.shape
)

print(
    "X_val_windows:",
    X_val_windows.shape
)

print(
    "Training sources:",
    len(
        np.unique(
            train_source_names
        )
    )
)

print(
    "Validation sources:",
    len(
        np.unique(
            val_window_names
        )
    )
)


# ==================================================
# Validation window sample weights
#
# Validation에서도 긴 영상이 window가 많다는 이유만으로
# val_loss에 더 큰 영향을 주지 않도록
# 영상별 총 weight를 동일하게 맞춤
# ==================================================

unique_val_sources, val_counts = (
    np.unique(
        val_window_names,
        return_counts=True,
    )
)

val_count_map = {
    name: count
    for name, count
    in zip(
        unique_val_sources,
        val_counts,
    )
}

val_weight_scale = (
    len(val_window_names)
    / len(unique_val_sources)
)

val_sample_weights = np.array(
    [
        val_weight_scale
        / val_count_map[name]

        for name in val_window_names
    ],
    dtype=np.float32,
)


print()
print(
    "Train weight mean:",
    train_sample_weights.mean()
)

print(
    "Validation weight mean:",
    val_sample_weights.mean()
)


# ==================================================
# Weighted mask-aware normalization
#
# 중요:
#
# sliding windows가 많은 원본 영상이
# normalization 통계까지 지배하지 않도록
# sample weight를 mean/std 계산에도 적용
# ==================================================

def calculate_weighted_stats(
    X,
    sample_weights,
):

    # ----------------------------------------------
    # Left
    # ----------------------------------------------

    left_present = (
        X[:, :, LEFT_MASK] > 0.5
    )

    left_coords = (
        X[:, :, LEFT_SLICE]
    )


    left_frame_weights = (
        sample_weights[:, None]
        * left_present.astype(
            np.float32
        )
    )


    left_total_weight = (
        left_frame_weights.sum()
    )


    left_mean = (
        (
            left_coords
            * left_frame_weights[
                :,
                :,
                None
            ]
        ).sum(
            axis=(0, 1)
        )
        / left_total_weight
    )


    left_var = (
        (
            (
                left_coords
                - left_mean
            ) ** 2
            * left_frame_weights[
                :,
                :,
                None
            ]
        ).sum(
            axis=(0, 1)
        )
        / left_total_weight
    )


    left_std = np.sqrt(
        left_var
    )


    # ----------------------------------------------
    # Right
    # ----------------------------------------------

    right_present = (
        X[:, :, RIGHT_MASK] > 0.5
    )

    right_coords = (
        X[:, :, RIGHT_SLICE]
    )


    right_frame_weights = (
        sample_weights[:, None]
        * right_present.astype(
            np.float32
        )
    )


    right_total_weight = (
        right_frame_weights.sum()
    )


    right_mean = (
        (
            right_coords
            * right_frame_weights[
                :,
                :,
                None
            ]
        ).sum(
            axis=(0, 1)
        )
        / right_total_weight
    )


    right_var = (
        (
            (
                right_coords
                - right_mean
            ) ** 2
            * right_frame_weights[
                :,
                :,
                None
            ]
        ).sum(
            axis=(0, 1)
        )
        / right_total_weight
    )


    right_std = np.sqrt(
        right_var
    )


    # ----------------------------------------------
    # 너무 작은 std 방지
    # ----------------------------------------------

    left_std[
        left_std < 1e-6
    ] = 1.0

    right_std[
        right_std < 1e-6
    ] = 1.0


    mean = np.concatenate(
        [
            left_mean,
            right_mean,
        ]
    ).astype(np.float32)


    std = np.concatenate(
        [
            left_std,
            right_std,
        ]
    ).astype(np.float32)


    return (
        mean,
        std,
    )


# ==================================================
# Normalization 적용
# ==================================================

def apply_normalization(
    X,
    mean,
    std,
):

    output = X.copy().astype(
        np.float32
    )


    left_present = (
        output[:, :, LEFT_MASK] > 0.5
    )

    right_present = (
        output[:, :, RIGHT_MASK] > 0.5
    )


    # ----------------------------------------------
    # Left
    # ----------------------------------------------

    left = output[
        :,
        :,
        LEFT_SLICE
    ]

    left[
        left_present
    ] = (
        left[
            left_present
        ]
        - mean[:63]
    ) / std[:63]

    left[
        ~left_present
    ] = 0.0


    # ----------------------------------------------
    # Right
    # ----------------------------------------------

    right = output[
        :,
        :,
        RIGHT_SLICE
    ]

    right[
        right_present
    ] = (
        right[
            right_present
        ]
        - mean[63:126]
    ) / std[63:126]

    right[
        ~right_present
    ] = 0.0


    # mask는 정확히 0/1
    output[
        :,
        :,
        126:128
    ] = (
        output[
            :,
            :,
            126:128
        ] > 0.5
    ).astype(np.float32)


    return output


# ==================================================
# Training data만 사용해 통계 계산
# ==================================================

mean, std = (
    calculate_weighted_stats(
        X_train,
        train_sample_weights,
    )
)


X_train_norm = apply_normalization(
    X_train,
    mean,
    std,
)

X_val_windows_norm = (
    apply_normalization(
        X_val_windows,
        mean,
        std,
    )
)

X_val_full_norm = (
    apply_normalization(
        X_val_full,
        mean,
        std,
    )
)


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
        "weighted_masked"
    ),
)


print()
print("=" * 70)
print("NORMALIZATION")
print("=" * 70)

print(
    "mean range:",
    float(mean.min()),
    "~",
    float(mean.max())
)

print(
    "std range:",
    float(std.min()),
    "~",
    float(std.max())
)


# ==================================================
# Missing-hand 검증
# ==================================================

left_missing = (
    X_train_norm[
        :,
        :,
        LEFT_MASK
    ] < 0.5
)

right_missing = (
    X_train_norm[
        :,
        :,
        RIGHT_MASK
    ] < 0.5
)


left_missing_values = (
    X_train_norm[
        :,
        :,
        LEFT_SLICE
    ][left_missing]
)

right_missing_values = (
    X_train_norm[
        :,
        :,
        RIGHT_SLICE
    ][right_missing]
)


print()
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
# Stacked LSTM
#
# LSTM(64, return_sequences=True, L2=0.005)
# → Batch Normalization
# → Dropout(0.3)
# → LSTM(32, return_sequences=False, L2=0.005)
# → Batch Normalization
# → Dropout(0.3)
# → Dense(12, Softmax, L2=0.005)
# ==================================================

L2_VALUE = 0.005



# ==================================================
# Bidirectional LSTM + Temporal Pooling
# ==================================================

inputs = keras.Input(
    shape=(64, 128)
)

# Layer Normalization
x = layers.LayerNormalization(
    axis=-1
)(inputs)

# Linear Projection: 128 -> 64
x = layers.Dense(
    64
)(x)

x = layers.Activation(
    "gelu"
)(x)

x = layers.Dropout(
    0.0
)(x)

# Bidirectional LSTM
# Forward 64 + Backward 64 = 128
x = layers.Bidirectional(

    layers.LSTM(
        64,
        return_sequences=True,
    ),

    merge_mode="concat",

)(x)

# Temporal Mean Pooling
mean_pool = layers.GlobalAveragePooling1D()(x)

# Temporal Max Pooling
max_pool = layers.GlobalMaxPooling1D()(x)

# 128 + 128 = 256
x = layers.Concatenate()([
    mean_pool,
    max_pool,
])

x = layers.Dropout(
    0.0
)(x)

# Classification
outputs = layers.Dense(
    12,
    activation="softmax",
)(x)

model = keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="v8_bilstm_pooling_mirror",
)



model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss=(
        "sparse_categorical_crossentropy"
    ),

    metrics=[
        keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )
    ],

    weighted_metrics=[
        keras.metrics.SparseCategoricalAccuracy(
            name="weighted_accuracy"
        )
    ],
)


print()
print("=" * 70)
print("MODEL")
print("=" * 70)

model.summary()


# ==================================================
# Early stopping
#
# sliding-window validation loss 기준
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
# Training
# ==================================================

print()
print("=" * 70)
print("TRAINING V5 SLIDING WINDOW")
print("=" * 70)


history = model.fit(

    X_train_norm,
    y_train,

    sample_weight=(
        train_sample_weights
    ),

    validation_data=(
        X_val_windows_norm,
        y_val_windows,
        val_sample_weights,
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping
    ],

    verbose=1,
)


# ==================================================
# 1. Validation full-sequence 평가
# ==================================================

full_prob = model.predict(
    X_val_full_norm,
    verbose=0,
)

full_pred = np.argmax(
    full_prob,
    axis=1,
)

full_correct = (
    full_pred
    == y_val_full
)


print()
print("=" * 70)
print("VALIDATION - FULL SEQUENCE")
print("=" * 70)

print(
    "Accuracy:",
    f"{full_correct.mean() * 100:.2f}% "
    f"({full_correct.sum()}/{len(y_val_full)})"
)


for i in range(
    len(y_val_full)
):

    confidence = float(
        full_prob[
            i,
            full_pred[i]
        ]
    )

    print(
        f"{val_full_names[i]} "
        f"target={y_val_full[i] + 1:2d} "
        f"pred={full_pred[i] + 1:2d} "
        f"conf={confidence * 100:5.1f}% "
        f"{'O' if full_correct[i] else 'X'}"
    )


# ==================================================
# 2. Sliding-window probability aggregation
#
# 같은 원본 영상의 모든 window probability 평균
# ==================================================

window_prob = model.predict(
    X_val_windows_norm,
    verbose=0,
)


aggregated_correct = 0
aggregated_total = 0


print()
print("=" * 70)
print("VALIDATION - SLIDING WINDOW AGGREGATION")
print("=" * 70)


for source_name in val_full_names:

    indices = np.where(
        val_window_names
        == source_name
    )[0]


    if len(indices) == 0:
        continue


    source_prob = (
        window_prob[
            indices
        ]
    )


    mean_prob = (
        source_prob.mean(
            axis=0
        )
    )


    prediction = int(
        np.argmax(
            mean_prob
        )
    )


    confidence = float(
        mean_prob[
            prediction
        ]
    )


    target = int(
        y_val_windows[
            indices[0]
        ]
    )


    is_correct = (
        prediction
        == target
    )


    aggregated_total += 1

    if is_correct:
        aggregated_correct += 1


    print(
        f"{source_name} "
        f"windows={len(indices):2d} "
        f"target={target + 1:2d} "
        f"pred={prediction + 1:2d} "
        f"conf={confidence * 100:5.1f}% "
        f"{'O' if is_correct else 'X'}"
    )


print()

print(
    "Aggregated accuracy:",
    f"{aggregated_correct / aggregated_total * 100:.2f}% "
    f"({aggregated_correct}/{aggregated_total})"
)


# ==================================================
# 모델 저장
# ==================================================

model.save(
    MODEL_PATH
)


print()
print("=" * 70)
print("저장 완료")
print("=" * 70)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Normalization:",
    NORMALIZATION_PATH
)