from pathlib import Path
import numpy as np


# ==================================================
# 기본 설정
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "numeric_1to12"
    / "lstm_dataset_v4b.npz"
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

LEFT_SLICE = slice(0, 63)
RIGHT_SLICE = slice(63, 126)

LEFT_MASK = 126
RIGHT_MASK = 127

EPS = 1e-6


# ==================================================
# USER temporal crop
# ==================================================

def crop_active_left_hand(sequence):

    present = sequence[:, LEFT_MASK] > 0.5

    indices = np.where(
        present
    )[0]

    if len(indices) == 0:
        return sequence

    return sequence[
        indices[0]:indices[-1] + 1
    ]


# ==================================================
# v4b geometry normalization
# ==================================================

def normalize_hand(hand):

    wrist = hand[0].copy()

    centered = hand - wrist

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

    output = sequence.copy().astype(
        np.float32
    )

    for frame in range(len(output)):

        if output[
            frame,
            LEFT_MASK
        ] > 0.5:

            hand = output[
                frame,
                LEFT_SLICE
            ].reshape(21, 3)

            normalized = normalize_hand(
                hand
            )

            if normalized is None:

                output[
                    frame,
                    LEFT_SLICE
                ] = 0.0

                output[
                    frame,
                    LEFT_MASK
                ] = 0.0

            else:

                output[
                    frame,
                    LEFT_SLICE
                ] = normalized.reshape(-1)


        if output[
            frame,
            RIGHT_MASK
        ] > 0.5:

            hand = output[
                frame,
                RIGHT_SLICE
            ].reshape(21, 3)

            normalized = normalize_hand(
                hand
            )

            if normalized is None:

                output[
                    frame,
                    RIGHT_SLICE
                ] = 0.0

                output[
                    frame,
                    RIGHT_MASK
                ] = 0.0

            else:

                output[
                    frame,
                    RIGHT_SLICE
                ] = normalized.reshape(-1)

    return output


# ==================================================
# 64-frame resampling
# ==================================================

def resample(sequence):

    if len(sequence) == 1:

        return np.repeat(
            sequence,
            TARGET_FRAMES,
            axis=0,
        ).astype(np.float32)


    old_time = np.linspace(
        0.0,
        1.0,
        len(sequence),
    )

    new_time = np.linspace(
        0.0,
        1.0,
        TARGET_FRAMES,
    )

    coordinates = np.zeros(
        (TARGET_FRAMES, 126),
        dtype=np.float32,
    )

    for i in range(126):

        coordinates[:, i] = np.interp(
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
            coordinates,
            masks,
        ],
        axis=1,
    )


    left_missing = (
        output[:, LEFT_MASK] == 0
    )

    right_missing = (
        output[:, RIGHT_MASK] == 0
    )

    output[
        left_missing,
        LEFT_SLICE
    ] = 0.0

    output[
        right_missing,
        RIGHT_SLICE
    ] = 0.0


    return output.astype(
        np.float32
    )


# ==================================================
# Mask-aware normalization
# ==================================================

def apply_normalization(
    sequence,
    mean,
    std,
):

    output = sequence.copy().astype(
        np.float32
    )

    left_present = (
        output[:, LEFT_MASK] > 0.5
    )

    right_present = (
        output[:, RIGHT_MASK] > 0.5
    )


    left = output[:, LEFT_SLICE]

    left[left_present] = (
        left[left_present]
        - mean[:63]
    ) / std[:63]

    left[~left_present] = 0.0


    right = output[:, RIGHT_SLICE]

    right[right_present] = (
        right[right_present]
        - mean[63:126]
    ) / std[63:126]

    right[~right_present] = 0.0


    return output


def preprocess_user(
    raw,
    mean,
    std,
):

    sequence = crop_active_left_hand(
        raw
    )

    sequence = geometry_normalize(
        sequence
    )

    sequence = resample(
        sequence
    )

    sequence = apply_normalization(
        sequence,
        mean,
        std,
    )

    return sequence


# ==================================================
# Left hand trajectory distance
# ==================================================

def trajectory_distance(
    sequence_a,
    sequence_b,
):

    present_a = (
        sequence_a[:, LEFT_MASK] > 0.5
    )

    present_b = (
        sequence_b[:, LEFT_MASK] > 0.5
    )

    common = (
        present_a & present_b
    )

    if not np.any(common):
        return np.inf


    a = (
        sequence_a[
            common,
            LEFT_SLICE
        ]
        .reshape(-1, 21, 3)
    )

    b = (
        sequence_b[
            common,
            LEFT_SLICE
        ]
        .reshape(-1, 21, 3)
    )


    distances = np.linalg.norm(
        a - b,
        axis=2,
    )

    return float(
        distances.mean()
    )


# ==================================================
# 세 가지 classifier
# ==================================================

def classify(
    query,
    X_train,
    y_train,
):

    distances = np.array([
        trajectory_distance(
            query,
            train_sequence,
        )
        for train_sequence in X_train
    ])


    # ------------------------------------------
    # 1-NN
    # ------------------------------------------

    nearest_index = np.argmin(
        distances
    )

    pred_1nn = int(
        y_train[nearest_index]
    )


    # ------------------------------------------
    # Weighted 5-NN
    # ------------------------------------------

    nearest_5 = np.argsort(
        distances
    )[:5]

    scores = np.zeros(
        12,
        dtype=np.float64,
    )

    for index in nearest_5:

        label = int(
            y_train[index]
        )

        weight = 1.0 / (
            distances[index] + 1e-6
        )

        scores[label] += weight

    pred_5nn = int(
        np.argmax(scores)
    )


    # ------------------------------------------
    # Class mean distance
    # ------------------------------------------

    class_mean_distances = []

    for class_index in range(12):

        class_distances = distances[
            y_train == class_index
        ]

        class_mean_distances.append(
            class_distances.mean()
        )

    class_mean_distances = np.array(
        class_mean_distances
    )

    pred_mean = int(
        np.argmin(
            class_mean_distances
        )
    )


    return (
        pred_1nn,
        pred_5nn,
        pred_mean,
    )


# ==================================================
# Load
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
]

X_val = data[
    "X_val"
].astype(np.float32)

y_val = data[
    "y_val"
]


normalization = np.load(
    NORMALIZATION_PATH
)

mean = normalization[
    "mean"
].reshape(-1)

std = normalization[
    "std"
].reshape(-1)


# Training / Validation은 이미
# geometry normalization + 64-frame 상태
X_train_normalized = np.stack([
    apply_normalization(
        sequence,
        mean,
        std,
    )
    for sequence in X_train
])

X_val_normalized = np.stack([
    apply_normalization(
        sequence,
        mean,
        std,
    )
    for sequence in X_val
])


# ==================================================
# AI Hub Validation
# ==================================================

correct_1nn = 0
correct_5nn = 0
correct_mean = 0


for query, target in zip(
    X_val_normalized,
    y_val,
):

    p1, p5, pm = classify(
        query,
        X_train_normalized,
        y_train,
    )

    correct_1nn += (
        p1 == target
    )

    correct_5nn += (
        p5 == target
    )

    correct_mean += (
        pm == target
    )


print()
print("=" * 70)
print("AI HUB VALIDATION")
print("=" * 70)

print(
    "1-NN:",
    f"{correct_1nn}/{len(y_val)}",
    f"({correct_1nn / len(y_val) * 100:.2f}%)"
)

print(
    "Weighted 5-NN:",
    f"{correct_5nn}/{len(y_val)}",
    f"({correct_5nn / len(y_val) * 100:.2f}%)"
)

print(
    "Class mean distance:",
    f"{correct_mean}/{len(y_val)}",
    f"({correct_mean / len(y_val) * 100:.2f}%)"
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
    f"{'1-NN':>10}"
    f"{'5-NN':>10}"
    f"{'Mean':>10}"
)


for name, target in USER_TESTS.items():

    path = (
        USER_ROOT
        / f"{name}.npy"
    )

    raw = np.load(
        path
    ).astype(np.float32)

    query = preprocess_user(
        raw,
        mean,
        std,
    )

    p1, p5, pm = classify(
        query,
        X_train_normalized,
        y_train,
    )

    print(
        f"{name:<15}"
        f"{target:>8}"
        f"{p1 + 1:>10}"
        f"{p5 + 1:>10}"
        f"{pm + 1:>10}"
    )