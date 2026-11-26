from pathlib import Path
import csv
import re

import numpy as np
import tensorflow as tf


# ==================================================
# 기본 설정
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

LANDMARK_DIR = (
    BASE_DIR
    / "local_inference"
    / "work"
    / "external_test_v2"
)

OUTPUT_CSV = (
    LANDMARK_DIR
    / "external_test_v2_all_mirror_results.csv"
)

WINDOW_SIZE = 48
STRIDE = 8
TARGET_FRAMES = 64

LEFT_SLICE = slice(0, 63)
RIGHT_SLICE = slice(63, 126)

LEFT_MASK = 126
RIGHT_MASK = 127

EPS = 1e-6



MODELS = {

    "v5_mirror": {
        "model": (
            BASE_DIR
            / "numeric_1to12"
            / "models_v5_sw_mirror"
            / "lstm_1to12_v5_sw_mirror.keras"
        ),
        "normalization": (
            BASE_DIR
            / "numeric_1to12"
            / "models_v5_sw_mirror"
            / "normalization.npz"
        ),
    },

    "v6_mirror": {
        "model": (
            BASE_DIR
            / "numeric_1to12"
            / "models_v6_stacked_mirror"
            / "lstm_1to12_v6_stacked_mirror.keras"
        ),
        "normalization": (
            BASE_DIR
            / "numeric_1to12"
            / "models_v6_stacked_mirror"
            / "normalization.npz"
        ),
    },

    "v7_expanded_mirror": {
        "model": (
            BASE_DIR
            / "numeric_1to12"
            / "models_v7_expanded_mirror"
            / "lstm_1to12_v7_expanded_mirror.keras"
        ),
        "normalization": (
            BASE_DIR
            / "numeric_1to12"
            / "models_v7_expanded_mirror"
            / "normalization.npz"
        ),
    },

    "v8_bilstm_pooling_mirror": {
        "model": (
            BASE_DIR
            / "numeric_1to12"
            / "models_v8_bilstm_pooling_mirror"
            / "lstm_1to12_v8_bilstm_pooling_mirror.keras"
        ),
        "normalization": (
            BASE_DIR
            / "numeric_1to12"
            / "models_v8_bilstm_pooling_mirror"
            / "normalization.npz"
        ),
    },

    "v9_medium_mirror": {
        "model": (
            BASE_DIR
            / "numeric_1to12"
            / "models_v9_medium_mirror"
            / "lstm_1to12_v9_medium_mirror.keras"
        ),
        "normalization": (
            BASE_DIR
            / "numeric_1to12"
            / "models_v9_medium_mirror"
            / "normalization.npz"
        ),
    },
}



# ==================================================
# 파일명
#
# test_0101.npy
#      ^^ target = 01
#        ^^ repetition = 01
# ==================================================

def parse_filename(path):

    match = re.fullmatch(
        r"test_(\d{2})(\d{2})",
        path.stem,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    target = int(match.group(1))
    repetition = int(match.group(2))

    return target, repetition


# ==================================================
# Active crop
# ==================================================

def crop_active_hand(sequence, mask_index=LEFT_MASK):

    present = (
        sequence[:, mask_index] > 0.5
    )

    indices = np.where(
        present
    )[0]

    if len(indices) == 0:
        return sequence

    return sequence[
        indices[0]:indices[-1] + 1
    ]


# ==================================================
# Geometry normalization
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


        # Right
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
# Resampling
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


    coords = np.zeros(
        (TARGET_FRAMES, 126),
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
# Sliding windows
# ==================================================

def make_windows(sequence):

    n = len(sequence)

    if n <= WINDOW_SIZE:

        return [
            sequence
        ]


    windows = []

    start = 0

    while (
        start + WINDOW_SIZE
        <= n
    ):

        windows.append(
            sequence[
                start:start + WINDOW_SIZE
            ]
        )

        start += STRIDE


    last_start = (
        n - WINDOW_SIZE
    )

    last_window = sequence[
        last_start:n
    ]

    if last_start % STRIDE != 0:

        windows.append(
            last_window
        )


    return windows


# ==================================================
# Normalization
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
# 영상 하나 전처리
# ==================================================

def preprocess_video(
    raw,
    mean,
    std,
    crop_mask=LEFT_MASK,
):

    active = crop_active_hand(
        raw,
        crop_mask,
    )

    windows = make_windows(
        active
    )

    processed = []

    for window in windows:

        window = geometry_normalize(
            window
        )

        window = resample(
            window
        )

        window = apply_normalization(
            window,
            mean,
            std,
        )

        processed.append(
            window
        )


    return (
        np.stack(processed),
        len(active),
        len(windows),
    )


# ==================================================
# Confusion matrix 출력
# ==================================================

def print_confusion_matrix(
    matrix,
):

    print()

    header = "T\\P " + " ".join(
        f"{i:>3}"
        for i in range(1, 13)
    )

    print(header)

    for target in range(12):

        row = " ".join(
            f"{matrix[target, pred]:>3}"
            for pred in range(12)
        )

        print(
            f"{target + 1:>3} {row}"
        )


# ==================================================
# Test files
# ==================================================

paths = sorted(
    [
        p
        for p in LANDMARK_DIR.glob(
            "test_*.npy"
        )
        if parse_filename(p)
        is not None
    ]
)


print(
    "External test samples:",
    len(paths)
)

if len(paths) != 36:

    print(
        "[WARNING] expected 36 samples"
    )


# ==================================================
# 모델별 평가
# ==================================================

all_rows = []


for version, config in MODELS.items():

    print()
    print("=" * 80)
    print(
        f"MODEL {version.upper()}"
    )
    print("=" * 80)


    model = tf.keras.models.load_model(
        config["model"]
    )


    norm = np.load(
        config["normalization"]
    )

    mean = norm[
        "mean"
    ].reshape(-1)

    std = norm[
        "std"
    ].reshape(-1)


    confusion = np.zeros(
        (12, 12),
        dtype=np.int32,
    )


    class_correct = np.zeros(
        12,
        dtype=np.int32,
    )

    class_total = np.zeros(
        12,
        dtype=np.int32,
    )


    total_correct = 0


    for path in paths:

        target, repetition = (
            parse_filename(path)
        )


        raw = np.load(
            path
        ).astype(np.float32)


        X, active_frames, num_windows = (
            preprocess_video(
                raw,
                mean,
                std,
                crop_mask=(
                    RIGHT_MASK
                    if version.endswith("_mirror")
                    else LEFT_MASK
                ),
            )
        )


        probabilities = model.predict(
            X,
            verbose=0,
        )


        # ------------------------------------------
        # mean probability aggregation
        # ------------------------------------------

        mean_prob = probabilities.mean(
            axis=0
        )


        pred_index = int(
            np.argmax(
                mean_prob
            )
        )

        prediction = (
            pred_index + 1
        )

        confidence = float(
            mean_prob[
                pred_index
            ]
        )


        correct = (
            prediction == target
        )


        total_correct += int(
            correct
        )

        class_total[
            target - 1
        ] += 1

        class_correct[
            target - 1
        ] += int(
            correct
        )

        confusion[
            target - 1,
            prediction - 1
        ] += 1


        print(
            f"{path.stem} "
            f"target={target:2d} "
            f"pred={prediction:2d} "
            f"conf={confidence*100:5.1f}% "
            f"windows={num_windows:2d} "
            f"{'O' if correct else 'X'}"
        )


        all_rows.append({

            "model":
                version,

            "filename":
                path.name,

            "target":
                target,

            "repetition":
                repetition,

            "raw_frames":
                len(raw),

            "active_frames":
                active_frames,

            "num_windows":
                num_windows,

            "prediction":
                prediction,

            "confidence":
                confidence,

            "correct":
                int(correct),
        })


    # ==================================================
    # 전체 결과
    # ==================================================

    print()
    print("-" * 80)

    print(
        "Overall accuracy:",
        f"{total_correct}/{len(paths)} "
        f"({total_correct / len(paths) * 100:.2f}%)"
    )


    print()
    print("Per-class accuracy")

    for class_number in range(
        1,
        13,
    ):

        correct = class_correct[
            class_number - 1
        ]

        total = class_total[
            class_number - 1
        ]

        print(
            f"class {class_number:2d}: "
            f"{correct}/{total} "
            f"({correct / total * 100:.1f}%)"
        )


    print()
    print("Confusion matrix")

    print_confusion_matrix(
        confusion
    )


# ==================================================
# CSV
# ==================================================

fieldnames = [

    "model",

    "filename",

    "target",

    "repetition",

    "raw_frames",

    "active_frames",

    "num_windows",

    "prediction",

    "confidence",

    "correct",
]


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        all_rows
    )


print()
print("=" * 80)
print("평가 완료")
print("=" * 80)

print(
    "결과 CSV:",
    OUTPUT_CSV
)

# ==================================================
# Model comparison
# ==================================================

print()
print("=" * 80)
print("MODEL COMPARISON - EXTERNAL TEST V2")
print("=" * 80)

for version in MODELS:

    rows = [
        row for row in all_rows
        if row["model"] == version
    ]

    groups = {

        "Overall": rows,

        "New recordings (01/02)": [
            row for row in rows
            if row["repetition"] in (1, 2)
        ],

        "Existing recordings (03)": [
            row for row in rows
            if row["repetition"] == 3
        ],
    }

    print()
    print(version.upper())

    for name, group in groups.items():

        correct = sum(
            row["correct"]
            for row in group
        )

        total = len(group)

        if total > 0:

            print(
                f"{name:28s}: "
                f"{correct:2d}/{total:2d} "
                f"({correct / total * 100:.2f}%)"
            )
