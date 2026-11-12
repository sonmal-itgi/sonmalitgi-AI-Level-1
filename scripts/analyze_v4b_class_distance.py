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
# Active Left-hand crop
# ==================================================

def crop_active_left_hand(sequence):

    present = (
        sequence[:, LEFT_MASK] > 0.5
    )

    indices = np.where(
        present
    )[0]

    if len(indices) == 0:
        return sequence

    start = indices[0]
    end = indices[-1] + 1

    return sequence[start:end]


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

        # Left
        if output[frame, LEFT_MASK] > 0.5:

            hand = output[
                frame,
                LEFT_SLICE
            ].reshape(21, 3)

            norm = normalize_hand(
                hand
            )

            if norm is None:

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
                ] = norm.reshape(-1)


        # Right
        if output[frame, RIGHT_MASK] > 0.5:

            hand = output[
                frame,
                RIGHT_SLICE
            ].reshape(21, 3)

            norm = normalize_hand(
                hand
            )

            if norm is None:

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
                ] = norm.reshape(-1)

    return output


# ==================================================
# 64-frame resampling
# ==================================================

def resample(sequence):

    if len(sequence) == 0:
        raise ValueError(
            "빈 sequence입니다."
        )

    if len(sequence) == 1:

        output = np.repeat(
            sequence,
            TARGET_FRAMES,
            axis=0,
        )

        return output.astype(
            np.float32
        )


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


# ==================================================
# USER preprocessing
#
# 현재 실제 inference와 동일:
# crop
# → geometry normalization
# → 64 frame
# → mean/std normalization
# ==================================================

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
# Left-hand trajectory distance
#
# 각 landmark의 3D Euclidean distance를
# 공통으로 Left hand가 존재하는 frame에서 평균
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
        present_a
        & present_b
    )

    if not np.any(common):
        return np.inf


    left_a = (
        sequence_a[
            common,
            LEFT_SLICE
        ]
        .reshape(-1, 21, 3)
    )

    left_b = (
        sequence_b[
            common,
            LEFT_SLICE
        ]
        .reshape(-1, 21, 3)
    )


    # frame × landmark별 3D 거리
    distances = np.linalg.norm(
        left_a - left_b,
        axis=2,
    )

    return float(
        distances.mean()
    )


# ==================================================
# Load Training
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

train_names = data[
    "train_names"
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


# Training dataset은 이미
# geometry-normalized + 64-frame 상태이므로
# mean/std만 적용
X_train_normalized = np.stack(
    [
        apply_normalization(
            sequence,
            mean,
            std,
        )
        for sequence in X_train
    ]
)


# ==================================================
# USER 분석
# ==================================================

for user_name, target in USER_TESTS.items():

    path = (
        USER_ROOT
        / f"{user_name}.npy"
    )

    raw = np.load(
        path
    ).astype(np.float32)

    user = preprocess_user(
        raw,
        mean,
        std,
    )


    # ----------------------------------------------
    # 모든 Training sample과 거리 계산
    # ----------------------------------------------

    distances = []

    for i, train_sequence in enumerate(
        X_train_normalized
    ):

        distance = trajectory_distance(
            user,
            train_sequence,
        )

        distances.append(
            (
                int(y_train[i]) + 1,
                distance,
                str(train_names[i]),
            )
        )


    # ----------------------------------------------
    # class별 통계
    # ----------------------------------------------

    class_results = []

    for class_number in range(
        1,
        13,
    ):

        class_distances = [
            distance
            for label, distance, _
            in distances
            if label == class_number
        ]

        class_results.append(
            {
                "class": class_number,

                "mean": float(
                    np.mean(
                        class_distances
                    )
                ),

                "median": float(
                    np.median(
                        class_distances
                    )
                ),

                "min": float(
                    np.min(
                        class_distances
                    )
                ),
            }
        )


    class_results = sorted(
        class_results,
        key=lambda x: x["mean"],
    )


    nearest_samples = sorted(
        distances,
        key=lambda x: x[1],
    )[:5]


    # ----------------------------------------------
    # 출력
    # ----------------------------------------------

    print()
    print("=" * 90)

    print(
        f"{user_name} "
        f"(TARGET={target})"
    )

    print("=" * 90)

    print()
    print(
        "===== CLASS DISTANCE RANKING "
        "(mean over 17 training signers) ====="
    )

    print(
        f"{'Rank':<6}"
        f"{'Class':>8}"
        f"{'Mean':>14}"
        f"{'Median':>14}"
        f"{'Min':>14}"
    )


    for rank, result in enumerate(
        class_results,
        start=1,
    ):

        marker = (
            " <-- TARGET"
            if result["class"] == target
            else ""
        )

        print(
            f"{rank:<6}"
            f"{result['class']:>8}"
            f"{result['mean']:>14.4f}"
            f"{result['median']:>14.4f}"
            f"{result['min']:>14.4f}"
            f"{marker}"
        )


    print()
    print(
        "===== 5 NEAREST TRAINING SAMPLES ====="
    )

    for rank, (
        label,
        distance,
        filename,
    ) in enumerate(
        nearest_samples,
        start=1,
    ):

        marker = (
            " <-- TARGET"
            if label == target
            else ""
        )

        print(
            f"{rank}. "
            f"class={label:2d} "
            f"distance={distance:.4f} "
            f"{filename}"
            f"{marker}"
        )


print()
print("=" * 90)
print("분석 완료")
print("=" * 90)