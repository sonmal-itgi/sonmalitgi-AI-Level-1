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

TARGET_FRAMES = 64

LEFT_SLICE = slice(0, 63)
RIGHT_SLICE = slice(63, 126)

LEFT_MASK_INDEX = 126
RIGHT_MASK_INDEX = 127

WRIST_INDEX = 0
SCALE_INDEX = 9

EPS = 1e-6


# ==================================================
# USER test 파일과 정답
# ==================================================

USER_TESTS = {
    "test_4": 4,
    "test_5": 5,
    "test_5_re1": 5,
    "test_5_re2": 5,
    "test_6": 6,
}


# ==================================================
# v4b geometry normalization
# ==================================================

def normalize_hand_geometry(hand):

    wrist = hand[WRIST_INDEX].copy()

    centered = hand - wrist

    scale = np.linalg.norm(
        centered[SCALE_INDEX, :2]
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


def normalize_sequence_geometry(sequence):

    output = sequence.copy().astype(
        np.float32
    )

    for frame_idx in range(len(output)):

        # Left
        if (
            output[
                frame_idx,
                LEFT_MASK_INDEX
            ] > 0.5
        ):

            hand = output[
                frame_idx,
                LEFT_SLICE
            ].reshape(21, 3)

            normalized = (
                normalize_hand_geometry(
                    hand
                )
            )

            if normalized is None:

                output[
                    frame_idx,
                    LEFT_SLICE
                ] = 0.0

                output[
                    frame_idx,
                    LEFT_MASK_INDEX
                ] = 0.0

            else:

                output[
                    frame_idx,
                    LEFT_SLICE
                ] = normalized.reshape(-1)


        # Right
        if (
            output[
                frame_idx,
                RIGHT_MASK_INDEX
            ] > 0.5
        ):

            hand = output[
                frame_idx,
                RIGHT_SLICE
            ].reshape(21, 3)

            normalized = (
                normalize_hand_geometry(
                    hand
                )
            )

            if normalized is None:

                output[
                    frame_idx,
                    RIGHT_SLICE
                ] = 0.0

                output[
                    frame_idx,
                    RIGHT_MASK_INDEX
                ] = 0.0

            else:

                output[
                    frame_idx,
                    RIGHT_SLICE
                ] = normalized.reshape(-1)

    return output


# ==================================================
# 64 frame resampling
# ==================================================

def resample_sequence(
    sequence,
    target_frames=64,
):

    if len(sequence) == 0:
        raise ValueError(
            "빈 sequence입니다."
        )

    sequence = sequence.astype(
        np.float32
    )

    if len(sequence) == 1:

        output = np.repeat(
            sequence,
            target_frames,
            axis=0,
        )

        output[:, 126:128] = (
            output[:, 126:128] > 0.5
        ).astype(np.float32)

        return output


    coordinates = sequence[:, :126]

    old_time = np.linspace(
        0,
        1,
        len(sequence),
    )

    new_time = np.linspace(
        0,
        1,
        target_frames,
    )

    new_coordinates = np.zeros(
        (target_frames, 126),
        dtype=np.float32,
    )

    for feature_idx in range(126):

        new_coordinates[
            :,
            feature_idx
        ] = np.interp(
            new_time,
            old_time,
            coordinates[:, feature_idx],
        )


    nearest_indices = np.rint(
        np.linspace(
            0,
            len(sequence) - 1,
            target_frames,
        )
    ).astype(int)

    new_masks = sequence[
        nearest_indices,
        126:128
    ]

    new_masks = (
        new_masks > 0.5
    ).astype(np.float32)


    output = np.concatenate(
        [
            new_coordinates,
            new_masks,
        ],
        axis=1,
    )


    # missing hand = 0
    left_missing = (
        output[:, LEFT_MASK_INDEX] == 0
    )

    right_missing = (
        output[:, RIGHT_MASK_INDEX] == 0
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
# Mask-aware mean/std normalization
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
        output[:, LEFT_MASK_INDEX] > 0.5
    )

    right_present = (
        output[:, RIGHT_MASK_INDEX] > 0.5
    )


    left = output[:, LEFT_SLICE]

    left[
        left_present
    ] = (
        left[left_present]
        - mean[:63]
    ) / std[:63]

    left[
        ~left_present
    ] = 0.0


    right = output[:, RIGHT_SLICE]

    right[
        right_present
    ] = (
        right[right_present]
        - mean[63:126]
    ) / std[63:126]

    right[
        ~right_present
    ] = 0.0


    return output


# ==================================================
# Left hand x/y/z 통계
# ==================================================

def get_axis_stats(sequence):

    left_present = (
        sequence[:, LEFT_MASK_INDEX] > 0.5
    )

    presence_rate = (
        left_present.mean() * 100
    )

    left = (
        sequence[
            left_present,
            LEFT_SLICE
        ]
        .reshape(-1, 21, 3)
    )

    result = {}

    for axis_index, axis_name in enumerate(
        ["x", "y", "z"]
    ):

        values = left[
            :,
            :,
            axis_index
        ].reshape(-1)

        abs_values = np.abs(
            values
        )

        result[axis_name] = {
            "mean": float(
                values.mean()
            ),

            "mean_abs": float(
                abs_values.mean()
            ),

            "p95_abs": float(
                np.percentile(
                    abs_values,
                    95,
                )
            ),

            "p99_abs": float(
                np.percentile(
                    abs_values,
                    99,
                )
            ),

            "over_2": float(
                (
                    abs_values > 2
                ).mean() * 100
            ),

            "over_3": float(
                (
                    abs_values > 3
                ).mean() * 100
            ),

            "max_abs": float(
                abs_values.max()
            ),
        }

    return (
        presence_rate,
        result,
    )


def print_stats(
    title,
    sequence,
):

    presence_rate, stats = (
        get_axis_stats(
            sequence
        )
    )

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)

    print(
        f"Left presence: "
        f"{presence_rate:.1f}%"
    )

    print()
    print(
        f"{'Axis':<6}"
        f"{'mean':>10}"
        f"{'mean|z|':>12}"
        f"{'P95|z|':>12}"
        f"{'P99|z|':>12}"
        f"{'|z|>2':>12}"
        f"{'|z|>3':>12}"
        f"{'max|z|':>12}"
    )

    for axis in [
        "x",
        "y",
        "z",
    ]:

        s = stats[axis]

        print(
            f"{axis:<6}"
            f"{s['mean']:>10.3f}"
            f"{s['mean_abs']:>12.3f}"
            f"{s['p95_abs']:>12.3f}"
            f"{s['p99_abs']:>12.3f}"
            f"{s['over_2']:>11.2f}%"
            f"{s['over_3']:>11.2f}%"
            f"{s['max_abs']:>12.3f}"
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

normalization = np.load(
    NORMALIZATION_PATH
)

mean = normalization[
    "mean"
].reshape(-1)

std = normalization[
    "std"
].reshape(-1)


# ==================================================
# AI Hub Training normalize
# ==================================================

X_train_normalized = []

for sequence in X_train:

    X_train_normalized.append(
        apply_normalization(
            sequence,
            mean,
            std,
        )
    )

X_train_normalized = np.stack(
    X_train_normalized
)


# ==================================================
# 전체 Training baseline
# ==================================================

training_flat = (
    X_train_normalized.reshape(
        -1,
        128,
    )
)

print_stats(
    "AI HUB TRAINING - ALL CLASSES",
    training_flat,
)


# ==================================================
# USER 분석
# ==================================================

for name, target in USER_TESTS.items():

    landmark_path = (
        USER_ROOT
        / f"{name}.npy"
    )

    if not landmark_path.exists():

        print()
        print(
            "[파일 없음]",
            landmark_path
        )

        continue


    # ----------------------------------------------
    # Raw USER
    # → v4b geometry
    # → 64 frames
    # → training mean/std normalization
    # ----------------------------------------------

    user = np.load(
        landmark_path
    ).astype(np.float32)

    user = (
        normalize_sequence_geometry(
            user
        )
    )

    user = resample_sequence(
        user,
        TARGET_FRAMES,
    )

    user = apply_normalization(
        user,
        mean,
        std,
    )


    # ----------------------------------------------
    # 동일 target class의 Training 데이터
    # ----------------------------------------------

    class_index = (
        target - 1
    )

    target_train = (
        X_train_normalized[
            y_train == class_index
        ]
    )

    target_train_flat = (
        target_train.reshape(
            -1,
            128,
        )
    )


    print_stats(
        f"AI HUB TRAINING - TARGET {target}",
        target_train_flat,
    )

    print_stats(
        f"USER {name} - TARGET {target}",
        user,
    )


print()
print("=" * 90)
print("분석 완료")
print("=" * 90)