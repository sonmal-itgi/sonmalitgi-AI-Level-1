from pathlib import Path
import re

import numpy as np


# ==================================================
# 기본 설정
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

LANDMARK_ROOT = (
    BASE_DIR
    / "numeric_1to12"
    / "landmarks_v3"
)

TRAIN_DIR = LANDMARK_ROOT / "training"
VAL_DIR = LANDMARK_ROOT / "validation"

OUTPUT_V4A = (
    BASE_DIR
    / "numeric_1to12"
    / "lstm_dataset_v4a.npz"
)

OUTPUT_V4B = (
    BASE_DIR
    / "numeric_1to12"
    / "lstm_dataset_v4b.npz"
)

TARGET_FRAMES = 64

LEFT_SLICE = slice(0, 63)
RIGHT_SLICE = slice(63, 126)

LEFT_MASK_INDEX = 126
RIGHT_MASK_INDEX = 127

# MediaPipe Hands
# landmark 0 = wrist
# landmark 9 = Middle Finger MCP
WRIST_INDEX = 0
SCALE_INDEX = 9

EPS = 1e-6


# ==================================================
# 파일명 → class index
# FS0801 -> 0
# FS0802 -> 1
# ...
# FS0812 -> 11
# ==================================================

def get_class_index(path: Path):

    match = re.search(
        r"FS(\d{4})",
        path.name,
    )

    if match is None:
        raise ValueError(
            f"class id를 찾을 수 없습니다: {path.name}"
        )

    fs_id = int(match.group(1))

    class_index = fs_id - 801

    if not 0 <= class_index <= 11:
        raise ValueError(
            f"1~12 클래스 범위가 아닙니다: {path.name}"
        )

    return class_index


# ==================================================
# 한 프레임의 한 손 normalize
#
# 1. wrist를 원점으로 이동
# 2. wrist -> middle MCP 거리로 scale normalize
# ==================================================

def normalize_hand_geometry(hand):

    # hand: (21, 3)

    wrist = hand[WRIST_INDEX].copy()

    centered = hand - wrist

    # x, y 좌표 기준 손 크기
    scale = np.linalg.norm(
        centered[SCALE_INDEX, :2]
    )

    # 예외 상황에서는 손 bounding box 사용
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

    normalized = centered / scale

    return normalized.astype(
        np.float32
    )


# ==================================================
# 전체 sequence에 geometry normalization
# ==================================================

def normalize_sequence_geometry(sequence):

    output = sequence.copy().astype(
        np.float32
    )

    invalid_hands = 0

    for frame_idx in range(len(output)):

        # ------------------------------------------
        # Left hand
        # ------------------------------------------

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

                invalid_hands += 1

            else:

                output[
                    frame_idx,
                    LEFT_SLICE
                ] = normalized.reshape(-1)


        # ------------------------------------------
        # Right hand
        # ------------------------------------------

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

                invalid_hands += 1

            else:

                output[
                    frame_idx,
                    RIGHT_SLICE
                ] = normalized.reshape(-1)


    return output, invalid_hands


# ==================================================
# 64-frame resampling
#
# 좌표:
#   linear interpolation
#
# presence mask:
#   nearest-neighbor
#
# 마지막에 missing hand 좌표를 다시 0으로 고정
# ==================================================

def resample_sequence(
    sequence,
    target_frames=64,
):

    if len(sequence) == 0:
        raise ValueError(
            "빈 landmark sequence입니다."
        )

    sequence = sequence.astype(
        np.float32
    )

    # ------------------------------------------
    # 1-frame 예외 처리
    # ------------------------------------------

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


    # ------------------------------------------
    # 좌표
    # ------------------------------------------

    coordinates = sequence[:, :126]

    old_time = np.linspace(
        0.0,
        1.0,
        len(sequence),
    )

    new_time = np.linspace(
        0.0,
        1.0,
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
            coordinates[
                :,
                feature_idx
            ],
        )


    # ------------------------------------------
    # mask는 linear interpolation 금지
    # nearest-neighbor 사용
    # ------------------------------------------

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


    # ------------------------------------------
    # 합치기
    # ------------------------------------------

    output = np.concatenate(
        [
            new_coordinates,
            new_masks,
        ],
        axis=1,
    )


    # ------------------------------------------
    # 매우 중요:
    #
    # interpolation 때문에 missing frame에도
    # 좌표가 남아 있을 수 있으므로
    # mask=0인 손은 다시 정확히 0으로 설정
    # ------------------------------------------

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
# dataset 생성
# ==================================================

def build_dataset(
    directory,
    geometry_normalization=False,
):

    X = []
    y = []
    names = []

    invalid_total = 0

    paths = sorted(
        directory.glob("*.npy")
    )

    for path in paths:

        sequence = np.load(
            path
        ).astype(np.float32)

        if sequence.ndim != 2:

            raise ValueError(
                f"잘못된 shape: "
                f"{path.name} "
                f"{sequence.shape}"
            )

        if sequence.shape[1] != 128:

            raise ValueError(
                f"128차원이 아닙니다: "
                f"{path.name} "
                f"{sequence.shape}"
            )


        # ------------------------------------------
        # v4b에서만 geometry normalization
        # ------------------------------------------

        if geometry_normalization:

            sequence, invalid_count = (
                normalize_sequence_geometry(
                    sequence
                )
            )

            invalid_total += (
                invalid_count
            )


        # ------------------------------------------
        # temporal resampling
        # ------------------------------------------

        sequence = resample_sequence(
            sequence,
            TARGET_FRAMES,
        )


        X.append(
            sequence
        )

        y.append(
            get_class_index(
                path
            )
        )

        names.append(
            path.name
        )


    X = np.stack(
        X
    ).astype(np.float32)

    y = np.array(
        y,
        dtype=np.int64,
    )

    names = np.array(
        names
    )


    return (
        X,
        y,
        names,
        invalid_total,
    )


# ==================================================
# 통계 출력
# ==================================================

def print_dataset_stats(
    name,
    X,
    y,
):

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(
        "X shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    print(
        "classes:",
        np.unique(
            y
        ).tolist()
    )

    left_present = (
        X[:, :, LEFT_MASK_INDEX] > 0.5
    )

    right_present = (
        X[:, :, RIGHT_MASK_INDEX] > 0.5
    )

    print(
        "Left presence:",
        f"{left_present.mean() * 100:.1f}%"
    )

    print(
        "Right presence:",
        f"{right_present.mean() * 100:.1f}%"
    )

    print(
        "Both presence:",
        f"{(left_present & right_present).mean() * 100:.1f}%"
    )


# ==================================================
# MAIN
# ==================================================

print(
    "===== v4a dataset 생성 ====="
)

# --------------------------------------------------
# v4a
#
# geometry normalization 없음
#
# 개선:
# - mask nearest-neighbor
# - missing hand = 0 유지
# --------------------------------------------------

(
    X_train_v4a,
    y_train_v4a,
    train_names,
    invalid_train_v4a,
) = build_dataset(
    TRAIN_DIR,
    geometry_normalization=False,
)

(
    X_val_v4a,
    y_val_v4a,
    val_names,
    invalid_val_v4a,
) = build_dataset(
    VAL_DIR,
    geometry_normalization=False,
)


np.savez_compressed(
    OUTPUT_V4A,

    X_train=X_train_v4a,
    y_train=y_train_v4a,

    X_val=X_val_v4a,
    y_val=y_val_v4a,

    train_names=train_names,
    val_names=val_names,
)


print_dataset_stats(
    "V4A TRAIN",
    X_train_v4a,
    y_train_v4a,
)

print_dataset_stats(
    "V4A VALIDATION",
    X_val_v4a,
    y_val_v4a,
)

print(
    "\n저장:",
    OUTPUT_V4A
)


# --------------------------------------------------
# v4b
#
# v4a +
# wrist-centered
# hand-size normalization
# --------------------------------------------------

print()
print(
    "===== v4b dataset 생성 ====="
)

(
    X_train_v4b,
    y_train_v4b,
    _,
    invalid_train_v4b,
) = build_dataset(
    TRAIN_DIR,
    geometry_normalization=True,
)

(
    X_val_v4b,
    y_val_v4b,
    _,
    invalid_val_v4b,
) = build_dataset(
    VAL_DIR,
    geometry_normalization=True,
)


np.savez_compressed(
    OUTPUT_V4B,

    X_train=X_train_v4b,
    y_train=y_train_v4b,

    X_val=X_val_v4b,
    y_val=y_val_v4b,

    train_names=train_names,
    val_names=val_names,
)


print_dataset_stats(
    "V4B TRAIN",
    X_train_v4b,
    y_train_v4b,
)

print_dataset_stats(
    "V4B VALIDATION",
    X_val_v4b,
    y_val_v4b,
)

print(
    "\ninvalid hands (train):",
    invalid_train_v4b
)

print(
    "invalid hands (validation):",
    invalid_val_v4b
)

print(
    "\n저장:",
    OUTPUT_V4B
)


print()
print("=" * 60)
print("완료")
print("=" * 60)

print(
    "v4a:",
    OUTPUT_V4A
)

print(
    "v4b:",
    OUTPUT_V4B
)