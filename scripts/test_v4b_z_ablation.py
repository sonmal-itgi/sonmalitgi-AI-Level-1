from pathlib import Path

import numpy as np
import tensorflow as tf


# ==================================================
# 기본 설정
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "numeric_1to12"
    / "lstm_dataset_v4b.npz"
)

MODEL_PATH = (
    BASE_DIR
    / "numeric_1to12"
    / "models_v4b"
    / "lstm_1to12_v4b.keras"
)

NORMALIZATION_PATH = (
    BASE_DIR
    / "numeric_1to12"
    / "models_v4b"
    / "normalization.npz"
)

USER_ROOT = (
    BASE_DIR
    / "local_inference"
    / "work"
    / "v2"
)

USER_TESTS = {
    "test_4": 4,
    "test_5": 5,
    "test_5_re1": 5,
    "test_5_re2": 5,
    "test_6": 6,
}

TARGET_FRAMES = 64

LEFT_MASK = 126
RIGHT_MASK = 127

EPS = 1e-6


# ==================================================
# z feature index
# ==================================================

LEFT_Z = np.arange(
    2,
    63,
    3,
)

RIGHT_Z = np.arange(
    65,
    126,
    3,
)

Z_INDICES = np.concatenate(
    [
        LEFT_Z,
        RIGHT_Z,
    ]
)


# ==================================================
# v4b geometry normalization
# ==================================================

def normalize_hand(hand):

    wrist = hand[0].copy()

    centered = (
        hand - wrist
    )

    scale = np.linalg.norm(
        centered[9, :2]
    )

    if scale < EPS:

        span_x = np.ptp(
            hand[:, 0]
        )

        span_y = np.ptp(
            hand[:, 1]
        )

        scale = max(
            span_x,
            span_y,
        )

    if scale < EPS:
        return None

    return (
        centered / scale
    ).astype(np.float32)


def geometry_normalize(sequence):

    output = (
        sequence.copy()
        .astype(np.float32)
    )

    for frame in range(
        len(output)
    ):

        # Left
        if output[
            frame,
            LEFT_MASK
        ] > 0.5:

            hand = output[
                frame,
                :63
            ].reshape(21, 3)

            norm = normalize_hand(
                hand
            )

            if norm is None:

                output[
                    frame,
                    :63
                ] = 0

                output[
                    frame,
                    LEFT_MASK
                ] = 0

            else:

                output[
                    frame,
                    :63
                ] = norm.reshape(-1)


        # Right
        if output[
            frame,
            RIGHT_MASK
        ] > 0.5:

            hand = output[
                frame,
                63:126
            ].reshape(21, 3)

            norm = normalize_hand(
                hand
            )

            if norm is None:

                output[
                    frame,
                    63:126
                ] = 0

                output[
                    frame,
                    RIGHT_MASK
                ] = 0

            else:

                output[
                    frame,
                    63:126
                ] = norm.reshape(-1)

    return output


# ==================================================
# 64-frame
# ==================================================

def resample(sequence):

    old_time = np.linspace(
        0,
        1,
        len(sequence),
    )

    new_time = np.linspace(
        0,
        1,
        TARGET_FRAMES,
    )

    coords = np.zeros(
        (
            TARGET_FRAMES,
            126,
        ),
        dtype=np.float32,
    )

    for i in range(126):

        coords[:, i] = np.interp(
            new_time,
            old_time,
            sequence[:, i],
        )


    nearest = np.rint(
        np.linspace(
            0,
            len(sequence) - 1,
            TARGET_FRAMES,
        )
    ).astype(int)

    masks = (
        sequence[
            nearest,
            126:128
        ] > 0.5
    ).astype(np.float32)


    output = np.concatenate(
        [
            coords,
            masks,
        ],
        axis=1,
    )


    left_missing = (
        output[:, LEFT_MASK]
        == 0
    )

    right_missing = (
        output[:, RIGHT_MASK]
        == 0
    )

    output[
        left_missing,
        :63
    ] = 0

    output[
        right_missing,
        63:126
    ] = 0

    return output


# ==================================================
# mean/std normalization
# ==================================================

def apply_normalization(
    sequence,
    mean,
    std,
):

    output = (
        sequence.copy()
        .astype(np.float32)
    )

    left_present = (
        output[:, LEFT_MASK]
        > 0.5
    )

    right_present = (
        output[:, RIGHT_MASK]
        > 0.5
    )

    left = output[:, :63]

    left[left_present] = (
        left[left_present]
        - mean[:63]
    ) / std[:63]

    left[~left_present] = 0


    right = output[:, 63:126]

    right[right_present] = (
        right[right_present]
        - mean[63:126]
    ) / std[63:126]

    right[~right_present] = 0

    return output


# ==================================================
# 예측 함수
# ==================================================

def predict(
    model,
    sequence,
    remove_z=False,
):

    X = sequence.copy()

    if remove_z:

        X[:, Z_INDICES] = 0.0

    prob = model.predict(
        X[np.newaxis, ...],
        verbose=0,
    )[0]

    pred = int(
        np.argmax(prob)
    )

    return (
        pred + 1,
        float(prob[pred]),
        prob,
    )


# ==================================================
# 모델 / normalization
# ==================================================

model = tf.keras.models.load_model(
    MODEL_PATH
)

norm = np.load(
    NORMALIZATION_PATH
)

mean = norm[
    "mean"
].reshape(-1)

std = norm[
    "std"
].reshape(-1)


# ==================================================
# AI Hub Validation
# ==================================================

data = np.load(
    DATA_PATH,
    allow_pickle=True,
)

X_val = data[
    "X_val"
].astype(np.float32)

y_val = data[
    "y_val"
]

names = data[
    "val_names"
]


normal_pred = []
no_z_pred = []


for sequence in X_val:

    sequence = apply_normalization(
        sequence,
        mean,
        std,
    )

    pred1, _, _ = predict(
        model,
        sequence,
        remove_z=False,
    )

    pred2, _, _ = predict(
        model,
        sequence,
        remove_z=True,
    )

    normal_pred.append(
        pred1 - 1
    )

    no_z_pred.append(
        pred2 - 1
    )


normal_pred = np.array(
    normal_pred
)

no_z_pred = np.array(
    no_z_pred
)


print()
print("=" * 70)
print("AI HUB VALIDATION")
print("=" * 70)

print(
    "Original:",
    f"{(normal_pred == y_val).mean() * 100:.2f}%"
)

print(
    "Z removed:",
    f"{(no_z_pred == y_val).mean() * 100:.2f}%"
)


# ==================================================
# USER
# ==================================================

print()
print("=" * 70)
print("USER TEST")
print("=" * 70)

print(
    f"{'Video':<15}"
    f"{'Target':>8}"
    f"{'Original':>15}"
    f"{'No-Z':>15}"
)

for name, target in USER_TESTS.items():

    path = (
        USER_ROOT
        / f"{name}.npy"
    )

    raw = np.load(
        path
    ).astype(np.float32)

    sequence = geometry_normalize(
        raw
    )

    sequence = resample(
        sequence
    )

    sequence = apply_normalization(
        sequence,
        mean,
        std,
    )


    pred_original, conf_original, _ = (
        predict(
            model,
            sequence,
            remove_z=False,
        )
    )

    pred_no_z, conf_no_z, _ = (
        predict(
            model,
            sequence,
            remove_z=True,
        )
    )


    original_text = (
        f"{pred_original}"
        f" ({conf_original*100:.1f}%)"
    )

    no_z_text = (
        f"{pred_no_z}"
        f" ({conf_no_z*100:.1f}%)"
    )


    print(
        f"{name:<15}"
        f"{target:>8}"
        f"{original_text:>15}"
        f"{no_z_text:>15}"
    )