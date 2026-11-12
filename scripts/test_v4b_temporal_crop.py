from pathlib import Path

import numpy as np
import tensorflow as tf


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

TARGET_FRAMES = 64

LEFT_MASK = 126
RIGHT_MASK = 127

EPS = 1e-6


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

        span_x = np.ptp(hand[:, 0])
        span_y = np.ptp(hand[:, 1])

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
                :63
            ].reshape(21, 3)

            normalized = normalize_hand(
                hand
            )

            if normalized is None:

                output[
                    frame,
                    :63
                ] = 0.0

                output[
                    frame,
                    LEFT_MASK
                ] = 0.0

            else:

                output[
                    frame,
                    :63
                ] = normalized.reshape(-1)


        # Right
        if output[frame, RIGHT_MASK] > 0.5:

            hand = output[
                frame,
                63:126
            ].reshape(21, 3)

            normalized = normalize_hand(
                hand
            )

            if normalized is None:

                output[
                    frame,
                    63:126
                ] = 0.0

                output[
                    frame,
                    RIGHT_MASK
                ] = 0.0

            else:

                output[
                    frame,
                    63:126
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
        :63
    ] = 0.0

    output[
        right_missing,
        63:126
    ] = 0.0


    return output.astype(
        np.float32
    )


# ==================================================
# Mean/std normalization
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


    left = output[:, :63]

    left[left_present] = (
        left[left_present]
        - mean[:63]
    ) / std[:63]

    left[~left_present] = 0.0


    right = output[:, 63:126]

    right[right_present] = (
        right[right_present]
        - mean[63:126]
    ) / std[63:126]

    right[~right_present] = 0.0

    return output


# ==================================================
# Temporal crop
# ==================================================

def crop_first_last(sequence):

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


def crop_longest(sequence):

    present = (
        sequence[:, LEFT_MASK] > 0.5
    )

    best_start = 0
    best_end = 0

    current_start = None

    for i, value in enumerate(present):

        if value and current_start is None:

            current_start = i

        if (
            not value
            and current_start is not None
        ):

            if (
                i - current_start
                > best_end - best_start
            ):

                best_start = current_start
                best_end = i

            current_start = None


    if current_start is not None:

        if (
            len(present) - current_start
            > best_end - best_start
        ):

            best_start = current_start
            best_end = len(present)


    if best_end <= best_start:
        return sequence

    return sequence[
        best_start:best_end
    ]


# ==================================================
# Prediction
# ==================================================

def preprocess(
    raw,
    mean,
    std,
):

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

    return sequence


def predict(
    model,
    sequence,
):

    probability = model.predict(
        sequence[np.newaxis, ...],
        verbose=0,
    )[0]

    index = int(
        np.argmax(probability)
    )

    return (
        index + 1,
        float(probability[index]),
    )


# ==================================================
# Load
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

print()
print("=" * 100)
print("TEMPORAL CROP TEST")
print("=" * 100)

print(
    f"{'Video':<14}"
    f"{'Target':>8}"
    f"{'Raw frames':>12}"
    f"{'First-Last':>12}"
    f"{'Longest':>10}"
    f"{'Original':>18}"
    f"{'First-Last':>18}"
    f"{'Longest':>18}"
)


for name, target in USER_TESTS.items():

    path = (
        USER_ROOT
        / f"{name}.npy"
    )

    raw = np.load(
        path
    ).astype(np.float32)


    first_last = crop_first_last(
        raw
    )

    longest = crop_longest(
        raw
    )


    original_processed = preprocess(
        raw,
        mean,
        std,
    )

    first_last_processed = preprocess(
        first_last,
        mean,
        std,
    )

    longest_processed = preprocess(
        longest,
        mean,
        std,
    )


    p1, c1 = predict(
        model,
        original_processed,
    )

    p2, c2 = predict(
        model,
        first_last_processed,
    )

    p3, c3 = predict(
        model,
        longest_processed,
    )


    original_text = (
        f"{p1} ({c1*100:.1f}%)"
    )

    first_last_text = (
        f"{p2} ({c2*100:.1f}%)"
    )

    longest_text = (
        f"{p3} ({c3*100:.1f}%)"
    )


    print(
        f"{name:<14}"
        f"{target:>8}"
        f"{len(raw):>12}"
        f"{len(first_last):>12}"
        f"{len(longest):>10}"
        f"{original_text:>18}"
        f"{first_last_text:>18}"
        f"{longest_text:>18}"
    )