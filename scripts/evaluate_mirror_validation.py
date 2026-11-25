from pathlib import Path

import numpy as np
import tensorflow as tf


# ==================================================
# 기본 설정
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "numeric_1to12"
    / "lstm_dataset_v5_sw_mirror.npz"
)

LEFT_SLICE = slice(0, 63)
RIGHT_SLICE = slice(63, 126)

LEFT_MASK = 126
RIGHT_MASK = 127


MODELS = {

    "v5_mirror": (
        BASE_DIR
        / "numeric_1to12"
        / "models_v5_sw_mirror"
        / "lstm_1to12_v5_sw_mirror.keras"
    ),

    "v6_mirror": (
        BASE_DIR
        / "numeric_1to12"
        / "models_v6_stacked_mirror"
        / "lstm_1to12_v6_stacked_mirror.keras"
    ),
}


# ==================================================
# Validation Dataset 로드
# ==================================================

with np.load(DATA_PATH) as data:

    X_val_full = data["X_val_full"]
    y_val_full = data["y_val_full"]
    val_full_names = data["val_full_names"]

    X_val_windows = data["X_val_windows"]
    y_val_windows = data["y_val_windows"]
    val_window_names = data["val_window_names"]


assert X_val_full.shape == (24, 64, 128)
assert X_val_windows.shape == (96, 64, 128)

assert len(np.unique(val_full_names)) == 24
assert len(np.unique(val_window_names)) == 24

assert set(val_full_names) == set(val_window_names)


print("===== Validation Dataset =====")
print("Full sequence:", X_val_full.shape)
print("Sliding windows:", X_val_windows.shape)
print("Original videos:", len(val_full_names))


# ==================================================
# Mask-aware Normalization
#
# Geometry Normalization과 Resampling은
# Dataset 생성 단계에서 이미 완료됨.
#
# 학습 당시 저장한 mean/std만 적용.
# ==================================================

def apply_normalization(X, mean, std):

    output = X.copy().astype(np.float32)

    mean = mean.reshape(-1)
    std = std.reshape(-1)

    left_present = (
        output[:, :, LEFT_MASK] > 0.5
    )

    right_present = (
        output[:, :, RIGHT_MASK] > 0.5
    )


    left = output[:, :, LEFT_SLICE]

    left[left_present] = (
        left[left_present] - mean[:63]
    ) / std[:63]

    left[~left_present] = 0.0


    right = output[:, :, RIGHT_SLICE]

    right[right_present] = (
        right[right_present] - mean[63:126]
    ) / std[63:126]

    right[~right_present] = 0.0


    if not np.isfinite(output).all():

        raise ValueError(
            "Normalization 이후 NaN 또는 Inf 발생"
        )

    return output


# ==================================================
# Full Sequence Evaluation
# ==================================================

def evaluate_full(model, X):

    probabilities = model.predict(
        X,
        verbose=0,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    correct_count = int(
        np.sum(predictions == y_val_full)
    )

    total = len(y_val_full)

    print()
    print("=" * 70)
    print("VALIDATION - FULL SEQUENCE")
    print("=" * 70)

    for i, name in enumerate(val_full_names):

        target = int(y_val_full[i]) + 1
        prediction = int(predictions[i]) + 1

        confidence = float(
            probabilities[i, predictions[i]]
        )

        correct = prediction == target

        print(
            f"{name} "
            f"target={target:2d} "
            f"pred={prediction:2d} "
            f"conf={confidence * 100:5.1f}% "
            f"{'O' if correct else 'X'}"
        )

    accuracy = correct_count / total

    print()
    print(
        f"Accuracy: {accuracy * 100:.2f}% "
        f"({correct_count}/{total})"
    )

    return correct_count, total


# ==================================================
# Sliding Window Aggregation
# ==================================================

def evaluate_sliding(model, X):

    probabilities = model.predict(
        X,
        verbose=0,
    )

    correct_count = 0

    print()
    print("=" * 70)
    print("VALIDATION - SLIDING WINDOW AGGREGATION")
    print("=" * 70)


    for name, full_label in zip(
        val_full_names,
        y_val_full,
    ):

        indices = np.where(
            val_window_names == name
        )[0]

        assert len(indices) > 0

        labels = y_val_windows[indices]

        assert np.all(labels == full_label)

        target = int(full_label) + 1


        mean_probability = probabilities[
            indices
        ].mean(axis=0)

        prediction_index = int(
            np.argmax(mean_probability)
        )

        prediction = prediction_index + 1

        confidence = float(
            mean_probability[prediction_index]
        )

        correct = prediction == target

        correct_count += int(correct)


        print(
            f"{name} "
            f"windows={len(indices):2d} "
            f"target={target:2d} "
            f"pred={prediction:2d} "
            f"conf={confidence * 100:5.1f}% "
            f"{'O' if correct else 'X'}"
        )


    total = len(val_full_names)

    accuracy = correct_count / total

    print()
    print(
        f"Aggregated accuracy: "
        f"{accuracy * 100:.2f}% "
        f"({correct_count}/{total})"
    )

    return correct_count, total


# ==================================================
# 모델별 평가
# ==================================================

results = {}


for version, model_path in MODELS.items():

    print()
    print("=" * 80)
    print(f"MODEL {version.upper()}")
    print("=" * 80)


    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )


    normalization_path = (
        model_path.parent
        / "normalization.npz"
    )


    with np.load(normalization_path) as norm:

        mean = norm["mean"]
        std = norm["std"]


    X_full = apply_normalization(
        X_val_full,
        mean,
        std,
    )

    X_windows = apply_normalization(
        X_val_windows,
        mean,
        std,
    )


    full_correct, full_total = evaluate_full(
        model,
        X_full,
    )


    sliding_correct, sliding_total = evaluate_sliding(
        model,
        X_windows,
    )


    results[version] = {

        "full_correct": full_correct,
        "full_total": full_total,

        "sliding_correct": sliding_correct,
        "sliding_total": sliding_total,
    }


# ==================================================
# 최종 결과 비교
# ==================================================

print()
print("=" * 80)
print("MIRROR VALIDATION COMPARISON")
print("=" * 80)


for version, result in results.items():

    fc = result["full_correct"]
    ft = result["full_total"]

    sc = result["sliding_correct"]
    st = result["sliding_total"]

    print()
    print(version.upper())

    print(
        f"Full Sequence : "
        f"{fc}/{ft} "
        f"({fc / ft * 100:.2f}%)"
    )

    print(
        f"Sliding       : "
        f"{sc}/{st} "
        f"({sc / st * 100:.2f}%)"
    )


print()
print("=" * 80)
print("평가 완료")
print("=" * 80)