from pathlib import Path
import argparse
import csv

import numpy as np

from inference import SignLanguageRecognizer


parser = argparse.ArgumentParser()

parser.add_argument(
    "--landmark-dir",
    type=Path,
    required=True,
)

parser.add_argument(
    "--reference-csv",
    type=Path,
    required=True,
)

args = parser.parse_args()


# ==================================================
# 기존 평가 결과 로드
# ==================================================

with args.reference_csv.open(
    encoding="utf-8",
    newline="",
) as f:

    reference_rows = [
        row for row in csv.DictReader(f)
        if row["model"] == "v9_medium_mirror"
    ]

assert len(reference_rows) == 48

expected = {
    f"test_{c:02d}{r:02d}.npy"
    for c in range(1, 13)
    for r in range(1, 5)
}

assert {
    row["filename"]
    for row in reference_rows
} == expected


# ==================================================
# 배포 모델 로드
# ==================================================

recognizer = SignLanguageRecognizer()

prediction_matches = 0
confidence_matches = 0
frame_matches = 0
window_matches = 0
correct = 0

confidence_diffs = []


# ==================================================
# 영상별 추론 결과 비교
# ==================================================

for row in reference_rows:

    filename = row["filename"]

    raw = np.load(
        args.landmark_dir / filename,
        allow_pickle=False,
    )

    result = recognizer.predict(raw)

    old_prediction = int(row["prediction"])
    old_confidence = float(row["confidence"])

    pred_match = (
        result["prediction"] == old_prediction
    )

    conf_diff = abs(
        result["confidence"] - old_confidence
    )

    conf_match = conf_diff <= 1e-4

    frames_match = (
        result["raw_frames"] == int(row["raw_frames"])
        and
        result["active_frames"] == int(row["active_frames"])
    )

    windows_match = (
        result["num_windows"] == int(row["num_windows"])
    )

    prediction_matches += int(pred_match)
    confidence_matches += int(conf_match)
    frame_matches += int(frames_match)
    window_matches += int(windows_match)

    confidence_diffs.append(conf_diff)

    correct += int(
        result["prediction"] == int(row["target"])
    )

    passed = all([
        pred_match,
        conf_match,
        frames_match,
        windows_match,
    ])

    print(
        f"{filename} "
        f"old={old_prediction:2d} "
        f"new={result['prediction']:2d} "
        f"conf_diff={conf_diff:.8f} "
        f"{'PASS' if passed else 'FAIL'}"
    )


# ==================================================
# 최종 검증
# ==================================================

total = len(reference_rows)

print()
print("=" * 70)
print("DEPLOYMENT INFERENCE VERIFICATION")
print("=" * 70)

print("Prediction matches:", prediction_matches, "/", total)
print("Confidence matches:", confidence_matches, "/", total)
print("Frame matches:", frame_matches, "/", total)
print("Window matches:", window_matches, "/", total)

print(
    "Max confidence difference:",
    max(confidence_diffs),
)

print(
    f"Accuracy: {correct}/{total} "
    f"({correct / total * 100:.2f}%)"
)

assert prediction_matches == total
assert confidence_matches == total
assert frame_matches == total
assert window_matches == total
assert correct == 33

print()
print("Deployment inference verification: PASS")
