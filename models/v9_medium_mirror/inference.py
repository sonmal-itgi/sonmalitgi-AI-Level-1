from pathlib import Path

import numpy as np
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "lstm_1to12_v9_medium_mirror.keras"
)

NORMALIZATION_PATH = (
    BASE_DIR / "normalization.npz"
)


WINDOW_SIZE = 48

STRIDE = 8

TARGET_FRAMES = 64

LEFT_SLICE = slice(0, 63)

RIGHT_SLICE = slice(63, 126)

LEFT_MASK = 126

RIGHT_MASK = 127

EPS = 1e-6

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
# Sign Language Recognizer
# ==================================================

class SignLanguageRecognizer:

    def __init__(
        self,
        model_path=MODEL_PATH,
        normalization_path=NORMALIZATION_PATH,
    ):

        self.model = tf.keras.models.load_model(
            model_path,
            compile=False,
        )

        with np.load(
            normalization_path,
            allow_pickle=False,
        ) as data:

            self.mean = data["mean"].reshape(-1)
            self.std = data["std"].reshape(-1)

            self.normalization_type = str(
                data["normalization_type"].item()
            )

        if self.mean.shape != (126,):
            raise ValueError("Invalid mean shape")

        if self.std.shape != (126,):
            raise ValueError("Invalid std shape")

        if not np.isfinite(self.mean).all():
            raise ValueError("Invalid mean values")

        if not np.isfinite(self.std).all():
            raise ValueError("Invalid std values")

        if not np.all(self.std > 0):
            raise ValueError("Std must be positive")

        if self.normalization_type != "weighted_masked":
            raise ValueError("Invalid normalization type")

        if self.model.input_shape != (None, 64, 128):
            raise ValueError("Invalid model input shape")

        if self.model.output_shape != (None, 12):
            raise ValueError("Invalid model output shape")


    def predict(self, landmark_sequence):

        raw = np.asarray(
            landmark_sequence,
            dtype=np.float32,
        )

        # 입력 데이터 검증

        if raw.ndim != 2 or raw.shape[1] != 128:

            raise ValueError(
                "Expected landmark shape: (T, 128)"
            )

        if len(raw) == 0:
            raise ValueError("Empty landmark sequence")

        if not np.isfinite(raw).all():
            raise ValueError("Invalid landmark values")

        if not np.isin(raw[:, 126:128], [0, 1]).all():
            raise ValueError("Invalid hand presence masks")

        if not np.any(raw[:, 126:128] > 0.5):
            raise ValueError("No hands detected")


        # 기존 v9 전처리

        X, active_frames, num_windows = preprocess_video(
            raw,
            self.mean,
            self.std,
            crop_mask=RIGHT_MASK,
        )

        if not np.any(X[:, :, 126:128] > 0.5):
            raise ValueError("No valid hand landmarks")


        # Sliding Window별 Prediction

        probabilities = self.model.predict(
            X,
            verbose=0,
        )


        # Mean Probability Aggregation

        mean_probability = np.mean(
            probabilities,
            axis=0,
        )

        if not np.isfinite(mean_probability).all():
            raise ValueError("Invalid prediction values")

        prediction = int(
            np.argmax(mean_probability)
        ) + 1

        confidence = float(
            mean_probability[prediction - 1]
        )


        return {

            "prediction":
                prediction,

            "confidence":
                confidence,

            "probabilities":
                mean_probability.tolist(),

            "raw_frames":
                len(raw),

            "active_frames":
                active_frames,

            "num_windows":
                num_windows,
        }

