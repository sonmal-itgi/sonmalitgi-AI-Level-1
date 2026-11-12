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

OUTPUT_PATH = (
    BASE_DIR
    / "numeric_1to12"
    / "lstm_dataset_v5_sw.npz"
)


WINDOW_SIZE = 48
STRIDE = 8

TARGET_FRAMES = 64

LEFT_SLICE = slice(0, 63)
RIGHT_SLICE = slice(63, 126)

LEFT_MASK_INDEX = 126
RIGHT_MASK_INDEX = 127

WRIST_INDEX = 0
SCALE_INDEX = 9

EPS = 1e-6


# ==================================================
# 파일명 -> class index
# ==================================================

def get_class_index(path):

    match = re.search(
        r"FS(\d{4})",
        path.name,
    )

    if match is None:
        raise ValueError(
            f"class id를 찾을 수 없습니다: {path.name}"
        )

    fs_id = int(
        match.group(1)
    )

    class_index = (
        fs_id - 801
    )

    if not 0 <= class_index <= 11:
        raise ValueError(
            f"1~12 범위가 아닙니다: {path.name}"
        )

    return class_index


# ==================================================
# Active Left-hand crop
# ==================================================

def crop_active_left_hand(sequence):

    present = (
        sequence[
            :,
            LEFT_MASK_INDEX
        ] > 0.5
    )

    indices = np.where(
        present
    )[0]

    if len(indices) == 0:
        return sequence

    start = int(
        indices[0]
    )

    end = int(
        indices[-1] + 1
    )

    return sequence[
        start:end
    ]


# ==================================================
# Sliding windows
# ==================================================

def make_windows(
    sequence,
    window_size=48,
    stride=8,
):

    n = len(sequence)

    if n == 0:
        return []

    # ----------------------------------------------
    # 짧은 sequence
    # → 전체를 하나의 window로 사용
    # ----------------------------------------------

    if n <= window_size:

        return [
            (
                0,
                n,
                sequence,
            )
        ]


    windows = []

    start = 0

    while (
        start + window_size
        <= n
    ):

        end = (
            start
            + window_size
        )

        windows.append(
            (
                start,
                end,
                sequence[
                    start:end
                ],
            )
        )

        start += stride


    # ----------------------------------------------
    # 마지막 프레임까지 반드시 포함
    # ----------------------------------------------

    last_start = (
        n - window_size
    )

    if (
        windows[-1][0]
        != last_start
    ):

        windows.append(
            (
                last_start,
                n,
                sequence[
                    last_start:n
                ],
            )
        )


    return windows


# ==================================================
# 한 손 geometry normalization
# ==================================================

def normalize_hand_geometry(hand):

    wrist = hand[
        WRIST_INDEX
    ].copy()

    centered = (
        hand - wrist
    )

    scale = np.linalg.norm(
        centered[
            SCALE_INDEX,
            :2
        ]
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


# ==================================================
# sequence geometry normalization
# ==================================================

def normalize_sequence_geometry(sequence):

    output = sequence.copy().astype(
        np.float32
    )

    invalid_count = 0


    for frame_idx in range(
        len(output)
    ):

        # ------------------------------------------
        # Left
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

                invalid_count += 1

            else:

                output[
                    frame_idx,
                    LEFT_SLICE
                ] = normalized.reshape(-1)


        # ------------------------------------------
        # Right
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

                invalid_count += 1

            else:

                output[
                    frame_idx,
                    RIGHT_SLICE
                ] = normalized.reshape(-1)


    return (
        output,
        invalid_count,
    )


# ==================================================
# 64-frame resampling
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


    if len(sequence) == 1:

        output = np.repeat(
            sequence,
            target_frames,
            axis=0,
        )

        output[
            :,
            126:128
        ] = (
            output[
                :,
                126:128
            ] > 0.5
        ).astype(np.float32)

        return output


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


    # ----------------------------------------------
    # 좌표 = linear interpolation
    # ----------------------------------------------

    coordinates = np.zeros(
        (
            target_frames,
            126,
        ),
        dtype=np.float32,
    )

    for feature_idx in range(
        126
    ):

        coordinates[
            :,
            feature_idx
        ] = np.interp(
            new_time,
            old_time,
            sequence[
                :,
                feature_idx
            ],
        )


    # ----------------------------------------------
    # mask = nearest-neighbor
    # ----------------------------------------------

    nearest_indices = np.rint(
        np.linspace(
            0,
            len(sequence) - 1,
            target_frames,
        )
    ).astype(int)


    masks = (
        sequence[
            nearest_indices,
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


    # ----------------------------------------------
    # missing hand = 정확히 0
    # ----------------------------------------------

    left_missing = (
        output[
            :,
            LEFT_MASK_INDEX
        ] == 0
    )

    right_missing = (
        output[
            :,
            RIGHT_MASK_INDEX
        ] == 0
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
# Training sliding-window dataset
# ==================================================

def build_training_dataset():

    X = []
    y = []

    source_names = []

    window_starts = []
    window_ends = []

    windows_per_source = []

    invalid_total = 0


    paths = sorted(
        TRAIN_DIR.glob("*.npy")
    )


    for path in paths:

        sequence = np.load(
            path
        ).astype(np.float32)


        if (
            sequence.ndim != 2
            or sequence.shape[1] != 128
        ):

            raise ValueError(
                f"잘못된 shape: "
                f"{path.name} "
                f"{sequence.shape}"
            )


        # ------------------------------------------
        # 실제 inference와 동일하게
        # active region 먼저 crop
        # ------------------------------------------

        sequence = (
            crop_active_left_hand(
                sequence
            )
        )


        windows = make_windows(
            sequence,
            WINDOW_SIZE,
            STRIDE,
        )


        if len(windows) == 0:
            continue


        windows_per_source.append(
            (
                path.name,
                len(windows),
            )
        )


        label = get_class_index(
            path
        )


        for (
            start,
            end,
            window,
        ) in windows:

            window, invalid_count = (
                normalize_sequence_geometry(
                    window
                )
            )

            invalid_total += (
                invalid_count
            )


            window = resample_sequence(
                window,
                TARGET_FRAMES,
            )


            X.append(
                window
            )

            y.append(
                label
            )

            source_names.append(
                path.name
            )

            window_starts.append(
                start
            )

            window_ends.append(
                end
            )


    X = np.stack(
        X
    ).astype(np.float32)

    y = np.array(
        y,
        dtype=np.int64,
    )

    source_names = np.array(
        source_names
    )

    window_starts = np.array(
        window_starts,
        dtype=np.int32,
    )

    window_ends = np.array(
        window_ends,
        dtype=np.int32,
    )


    # ==================================================
    # Sample weight
    #
    # 한 원본 영상에서 window가 많이 생성되더라도
    # 전체 contribution은 다른 원본 영상과 같게 유지
    # ==================================================

    unique_sources, counts = (
        np.unique(
            source_names,
            return_counts=True,
        )
    )

    source_count_map = {
        source: count
        for source, count
        in zip(
            unique_sources,
            counts,
        )
    }


    num_windows = len(
        source_names
    )

    num_sources = len(
        unique_sources
    )


    # 평균 sample weight가 1이 되도록 scale
    weight_scale = (
        num_windows
        / num_sources
    )


    sample_weights = np.array(
        [
            weight_scale
            / source_count_map[name]

            for name in source_names
        ],
        dtype=np.float32,
    )


    return (
        X,
        y,
        source_names,
        window_starts,
        window_ends,
        sample_weights,
        windows_per_source,
        invalid_total,
    )


# ==================================================
# Validation
#
# 1) full active sequence
# 2) sliding windows
# 둘 다 저장
# ==================================================

def build_validation_dataset():

    X_full = []
    y_full = []
    full_names = []

    X_windows = []
    y_windows = []

    window_names = []
    window_starts = []
    window_ends = []


    paths = sorted(
        VAL_DIR.glob("*.npy")
    )


    for path in paths:

        raw = np.load(
            path
        ).astype(np.float32)


        active = crop_active_left_hand(
            raw
        )


        label = get_class_index(
            path
        )


        # ------------------------------------------
        # Full active sequence
        # ------------------------------------------

        full, _ = (
            normalize_sequence_geometry(
                active
            )
        )

        full = resample_sequence(
            full,
            TARGET_FRAMES,
        )


        X_full.append(
            full
        )

        y_full.append(
            label
        )

        full_names.append(
            path.name
        )


        # ------------------------------------------
        # Sliding windows
        # ------------------------------------------

        windows = make_windows(
            active,
            WINDOW_SIZE,
            STRIDE,
        )


        for (
            start,
            end,
            window,
        ) in windows:

            window, _ = (
                normalize_sequence_geometry(
                    window
                )
            )

            window = resample_sequence(
                window,
                TARGET_FRAMES,
            )


            X_windows.append(
                window
            )

            y_windows.append(
                label
            )

            window_names.append(
                path.name
            )

            window_starts.append(
                start
            )

            window_ends.append(
                end
            )


    return (
        np.stack(
            X_full
        ).astype(np.float32),

        np.array(
            y_full,
            dtype=np.int64,
        ),

        np.array(
            full_names
        ),

        np.stack(
            X_windows
        ).astype(np.float32),

        np.array(
            y_windows,
            dtype=np.int64,
        ),

        np.array(
            window_names
        ),

        np.array(
            window_starts,
            dtype=np.int32,
        ),

        np.array(
            window_ends,
            dtype=np.int32,
        ),
    )


# ==================================================
# MAIN
# ==================================================

print(
    "===== v5 sliding-window dataset 생성 ====="
)


(
    X_train,
    y_train,
    train_source_names,
    train_window_starts,
    train_window_ends,
    train_sample_weights,
    windows_per_source,
    invalid_total,
) = build_training_dataset()


(
    X_val_full,
    y_val_full,
    val_full_names,
    X_val_windows,
    y_val_windows,
    val_window_names,
    val_window_starts,
    val_window_ends,
) = build_validation_dataset()


# ==================================================
# 저장
# ==================================================

np.savez_compressed(

    OUTPUT_PATH,

    X_train=X_train,
    y_train=y_train,

    train_source_names=(
        train_source_names
    ),

    train_window_starts=(
        train_window_starts
    ),

    train_window_ends=(
        train_window_ends
    ),

    train_sample_weights=(
        train_sample_weights
    ),


    X_val_full=X_val_full,
    y_val_full=y_val_full,

    val_full_names=(
        val_full_names
    ),


    X_val_windows=(
        X_val_windows
    ),

    y_val_windows=(
        y_val_windows
    ),

    val_window_names=(
        val_window_names
    ),

    val_window_starts=(
        val_window_starts
    ),

    val_window_ends=(
        val_window_ends
    ),


    window_size=np.array(
        WINDOW_SIZE
    ),

    stride=np.array(
        STRIDE
    ),
)


# ==================================================
# 통계
# ==================================================

print()
print("=" * 70)
print("TRAIN")
print("=" * 70)

print(
    "Original videos:",
    len(
        np.unique(
            train_source_names
        )
    )
)

print(
    "Sliding windows:",
    len(X_train)
)

print(
    "X_train:",
    X_train.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "sample weight mean:",
    train_sample_weights.mean()
)

print(
    "sample weight min:",
    train_sample_weights.min()
)

print(
    "sample weight max:",
    train_sample_weights.max()
)

print(
    "invalid hands:",
    invalid_total
)


window_counts = np.array(
    [
        count
        for _, count
        in windows_per_source
    ]
)


print()
print(
    "windows/source min:",
    window_counts.min()
)

print(
    "windows/source median:",
    np.median(
        window_counts
    )
)

print(
    "windows/source mean:",
    window_counts.mean()
)

print(
    "windows/source max:",
    window_counts.max()
)


print()
print("=" * 70)
print("WINDOWS PER CLASS")
print("=" * 70)

for class_index in range(12):

    count = np.sum(
        y_train
        == class_index
    )

    weight_sum = np.sum(
        train_sample_weights[
            y_train == class_index
        ]
    )

    print(
        f"class {class_index + 1:2d}: "
        f"windows={count:4d} "
        f"weight_sum={weight_sum:.2f}"
    )


print()
print("=" * 70)
print("VALIDATION")
print("=" * 70)

print(
    "Full videos:",
    X_val_full.shape
)

print(
    "Sliding windows:",
    X_val_windows.shape
)

print(
    "Validation sources:",
    len(
        np.unique(
            val_window_names
        )
    )
)


print()
print("=" * 70)
print("저장 완료")
print("=" * 70)

print(
    OUTPUT_PATH
)