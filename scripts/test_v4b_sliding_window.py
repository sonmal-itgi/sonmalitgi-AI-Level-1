from pathlib import Path

import numpy as np
import tensorflow as tf


# ==================================================
# 기본 설정
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

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


WINDOW_SIZE = 48
STRIDE = 8

TARGET_FRAMES = 64

LEFT_SLICE = slice(0, 63)
RIGHT_SLICE = slice(63, 126)

LEFT_MASK = 126
RIGHT_MASK = 127

EPS = 1e-6


# ==================================================
# Active Left hand 구간 crop
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
# Geometry normalization
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
        if output[
            frame,
            LEFT_MASK
        ] > 0.5:

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
        if output[
            frame,
            RIGHT_MASK
        ] > 0.5:

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
        (
            TARGET_FRAMES,
            126,
        ),
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
# 하나의 window 전처리
# ==================================================

def preprocess_window(
    sequence,
    mean,
    std,
):

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
# Sliding windows 생성
# ==================================================

def make_windows(
    sequence,
    window_size,
    stride,
):

    n = len(sequence)

    # window보다 짧으면 전체 sequence 하나 사용
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


    # 마지막 부분이 정확히 포함되지 않았다면
    # 끝에 맞춘 window 하나 추가
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
# Load model
# ==================================================

model = tf.keras.models.load_model(
    MODEL_PATH
)

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
# Test
# ==================================================

for name, target in USER_TESTS.items():

    print()
    print("=" * 90)
    print(
        f"{name} "
        f"(TARGET={target})"
    )
    print("=" * 90)


    path = (
        USER_ROOT
        / f"{name}.npy"
    )

    raw = np.load(
        path
    ).astype(np.float32)


    # ------------------------------------------
    # 기존에 효과가 확인된 active crop 먼저 적용
    # ------------------------------------------

    active = crop_active_left_hand(
        raw
    )


    windows = make_windows(
        active,
        WINDOW_SIZE,
        STRIDE,
    )


    print(
        "Raw frames   :",
        len(raw)
    )

    print(
        "Active frames:",
        len(active)
    )

    print(
        "Window size  :",
        WINDOW_SIZE
    )

    print(
        "Stride       :",
        STRIDE
    )

    print(
        "Num windows  :",
        len(windows)
    )

    print()


    probabilities = []


    # ------------------------------------------
    # 각 window 추론
    # ------------------------------------------

    for window_index, (
        start,
        end,
        window,
    ) in enumerate(
        windows,
        start=1,
    ):

        processed = preprocess_window(
            window,
            mean,
            std,
        )

        probability = model.predict(
            processed[
                np.newaxis,
                ...
            ],
            verbose=0,
        )[0]


        probabilities.append(
            probability
        )


        pred_index = int(
            np.argmax(
                probability
            )
        )

        confidence = float(
            probability[
                pred_index
            ]
        )


        print(
            f"Window {window_index:2d} "
            f"[{start:3d}:{end:3d}] "
            f"-> "
            f"{pred_index + 1:2d} "
            f"({confidence * 100:5.1f}%)"
        )


    probabilities = np.stack(
        probabilities
    )


    # ==================================================
    # Aggregation 1
    # 모든 window probability 평균
    # ==================================================

    mean_prob = probabilities.mean(
        axis=0
    )

    mean_pred = int(
        np.argmax(
            mean_prob
        )
    )

    mean_conf = float(
        mean_prob[
            mean_pred
        ]
    )


    # ==================================================
    # Aggregation 2
    # 가장 confidence 높은 window
    # ==================================================

    window_confidences = (
        probabilities.max(
            axis=1
        )
    )

    best_window_index = int(
        np.argmax(
            window_confidences
        )
    )

    best_prob = probabilities[
        best_window_index
    ]

    best_pred = int(
        np.argmax(
            best_prob
        )
    )

    best_conf = float(
        best_prob[
            best_pred
        ]
    )


    # ==================================================
    # Aggregation 3
    # confidence 상위 3 window 평균
    # ==================================================

    k = min(
        3,
        len(probabilities),
    )

    top_indices = np.argsort(
        window_confidences
    )[::-1][:k]

    top_prob = probabilities[
        top_indices
    ].mean(axis=0)

    top_pred = int(
        np.argmax(
            top_prob
        )
    )

    top_conf = float(
        top_prob[
            top_pred
        ]
    )


    # ==================================================
    # 결과 출력
    # ==================================================

    print()
    print(
        "----- Aggregation -----"
    )

    print(
        "Mean probability :",
        f"{mean_pred + 1}",
        f"({mean_conf * 100:.1f}%)",
        "O"
        if mean_pred + 1 == target
        else "X",
    )

    print(
        "Best window      :",
        f"{best_pred + 1}",
        f"({best_conf * 100:.1f}%)",
        "O"
        if best_pred + 1 == target
        else "X",
    )

    print(
        f"Top-{k} mean       :",
        f"{top_pred + 1}",
        f"({top_conf * 100:.1f}%)",
        "O"
        if top_pred + 1 == target
        else "X",
    )