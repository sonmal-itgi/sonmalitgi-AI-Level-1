from pathlib import Path
import argparse

import numpy as np
import tensorflow as tf


# ==================================================
# 기본 설정
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

TARGET_FRAMES = 64

LEFT_SLICE = slice(0, 63)
RIGHT_SLICE = slice(63, 126)

LEFT_MASK_INDEX = 126
RIGHT_MASK_INDEX = 127

WRIST_INDEX = 0
SCALE_INDEX = 9

EPS = 1e-6


# ==================================================
# Argument
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--landmarks",
    required=True,
    help="원본 landmark .npy 파일",
)

parser.add_argument(
    "--target",
    type=int,
    default=None,
    help="정답 숫자 (1~12)",
)

parser.add_argument(
    "--version",
    choices=["v4a", "v4b"],
    default="v4b",
)

args = parser.parse_args()

VERSION = args.version


# ==================================================
# 모델 경로
# ==================================================

MODEL_DIR = (
    BASE_DIR
    / "numeric_1to12"
    / f"models_{VERSION}"
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
# 한 손 geometry normalization
#
# v4b에서만 사용
# ==================================================

def normalize_hand_geometry(hand):

    # hand shape = (21, 3)

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

    normalized = centered / scale

    return normalized.astype(
        np.float32
    )


# ==================================================
# 전체 sequence geometry normalization
# ==================================================

def normalize_sequence_geometry(sequence):

    output = sequence.copy().astype(
        np.float32
    )

    for frame_idx in range(len(output)):

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

            else:

                output[
                    frame_idx,
                    RIGHT_SLICE
                ] = normalized.reshape(-1)

    return output


# ==================================================
# 64-frame resampling
#
# coordinates = linear
# masks       = nearest
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

        output[:, 126:128] = (
            output[:, 126:128] > 0.5
        ).astype(np.float32)

        return output


    # ------------------------------------------
    # coordinates
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
    # masks = nearest-neighbor
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


    output = np.concatenate(
        [
            new_coordinates,
            new_masks,
        ],
        axis=1,
    )


    # ------------------------------------------
    # missing hand는 정확히 0
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


def crop_active_left_hand(sequence):

    present = (
        sequence[:, LEFT_MASK_INDEX]
        > 0.5
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
# Mask-aware normalization
# ==================================================

def apply_masked_normalization(
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


    # ------------------------------------------
    # Left
    # ------------------------------------------

    left_coords = output[:, 0:63]

    left_coords[
        left_present
    ] = (
        left_coords[
            left_present
        ]
        - mean[0:63]
    ) / std[0:63]

    left_coords[
        ~left_present
    ] = 0.0


    # ------------------------------------------
    # Right
    # ------------------------------------------

    right_coords = output[:, 63:126]

    right_coords[
        right_present
    ] = (
        right_coords[
            right_present
        ]
        - mean[63:126]
    ) / std[63:126]

    right_coords[
        ~right_present
    ] = 0.0


    # mask는 0/1 유지
    output[:, 126:128] = (
        output[:, 126:128] > 0.5
    ).astype(np.float32)

    return output


# ==================================================
# landmark load
# ==================================================

landmark_path = Path(
    args.landmarks
)

sequence = np.load(
    landmark_path
).astype(np.float32)

raw_frames = len(sequence)

sequence = crop_active_left_hand(
    sequence
)

print(
    "temporal crop:",
    raw_frames,
    "->",
    len(sequence),
)

print()
print("=" * 60)
print("Input")
print("=" * 60)

print(
    "version:",
    VERSION
)

print(
    "landmark:",
    landmark_path
)

print(
    "raw shape:",
    sequence.shape
)


if (
    sequence.ndim != 2
    or sequence.shape[1] != 128
):
    raise ValueError(
        f"잘못된 landmark shape: {sequence.shape}"
    )


# ==================================================
# v4b geometry normalization
# ==================================================

if VERSION == "v4b":

    sequence = (
        normalize_sequence_geometry(
            sequence
        )
    )


# ==================================================
# 64-frame
# ==================================================

sequence = resample_sequence(
    sequence,
    TARGET_FRAMES,
)

print(
    "64-frame shape:",
    sequence.shape
)


# ==================================================
# normalization load
# ==================================================

normalization = np.load(
    NORMALIZATION_PATH
)

mean = normalization[
    "mean"
].reshape(-1)

std = normalization[
    "std"
].reshape(-1)


print(
    "normalization:",
    NORMALIZATION_PATH
)


# ==================================================
# Mask-aware normalization
# ==================================================

sequence = (
    apply_masked_normalization(
        sequence,
        mean,
        std,
    )
)


# ==================================================
# 검증
# ==================================================

left_missing = (
    sequence[:, LEFT_MASK_INDEX]
    < 0.5
)

right_missing = (
    sequence[:, RIGHT_MASK_INDEX]
    < 0.5
)

left_missing_max = (
    np.abs(
        sequence[
            left_missing,
            0:63
        ]
    ).max()
    if np.any(left_missing)
    else 0.0
)

right_missing_max = (
    np.abs(
        sequence[
            right_missing,
            63:126
        ]
    ).max()
    if np.any(right_missing)
    else 0.0
)


print(
    "Left missing max abs:",
    left_missing_max
)

print(
    "Right missing max abs:",
    right_missing_max
)


# ==================================================
# Model
# ==================================================

model = tf.keras.models.load_model(
    MODEL_PATH
)

X = sequence[
    np.newaxis,
    ...
]

probabilities = model.predict(
    X,
    verbose=0,
)[0]


# ==================================================
# 결과
# ==================================================

prediction_index = int(
    np.argmax(
        probabilities
    )
)

prediction = (
    prediction_index + 1
)

confidence = float(
    probabilities[
        prediction_index
    ]
)


top3_indices = np.argsort(
    probabilities
)[::-1][:3]


print()
print("=" * 60)
print("AI 추론 결과")
print("=" * 60)

print(
    "예측 숫자:",
    prediction
)

print(
    "confidence:",
    f"{confidence * 100:.1f}%"
)

print()
print("Top 3")

for rank, index in enumerate(
    top3_indices,
    start=1,
):

    print(
        f"{rank}. "
        f"{index + 1} "
        f"({probabilities[index] * 100:.1f}%)"
    )


if args.target is not None:

    print()

    print(
        "목표 숫자:",
        args.target
    )

    print(
        "판정:",
        "정답"
        if prediction == args.target
        else "오답"
    )