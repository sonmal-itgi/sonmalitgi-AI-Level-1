from pathlib import Path

import numpy as np


# ==================================================
# 기본 설정
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

ORIGINAL_ROOT = (
    BASE_DIR
    / "numeric_1to12"
    / "landmarks_v3"
)

MIRROR_ROOT = (
    BASE_DIR
    / "numeric_1to12"
    / "landmarks_mirror"
)

LEFT_SLICE = slice(0, 63)
RIGHT_SLICE = slice(63, 126)

LEFT_MASK = 126
RIGHT_MASK = 127


# ==================================================
# Presence 통계
# ==================================================

def presence_statistics(sequence):

    left = sequence[:, LEFT_MASK] > 0.5
    right = sequence[:, RIGHT_MASK] > 0.5

    return np.array([
        np.mean(left),
        np.mean(right),
        np.mean(left & right),
        np.mean(left | right),
        np.mean(~left & ~right),
    ])


# ==================================================
# 좌표 반전 검증
# ==================================================

def compare_coordinates(
    original,
    mirrored,
    original_slice,
    mirror_slice,
    original_mask,
    mirror_mask,
):

    # 반전 전후 모두 검출된 손만 비교
    valid = (
        (original[:, original_mask] > 0.5)
        &
        (mirrored[:, mirror_mask] > 0.5)
    )

    if not np.any(valid):
        return None

    original_hand = original[
        valid,
        original_slice
    ].reshape(-1, 21, 3)

    mirror_hand = mirrored[
        valid,
        mirror_slice
    ].reshape(-1, 21, 3)

    expected = original_hand.copy()

    # 이론적인 Horizontal Flip
    expected[:, :, 0] = (
        1.0 - expected[:, :, 0]
    )

    error = np.abs(
        expected - mirror_hand
    )

    return error.reshape(-1, 3)


# ==================================================
# Split별 검증
# ==================================================

for split in [
    "training",
    "validation",
]:

    print()
    print("=" * 70)
    print(split.upper())
    print("=" * 70)

    original_paths = {
        p.name: p
        for p in (
            ORIGINAL_ROOT / split
        ).glob("*.npy")
    }

    mirror_paths = {
        p.name: p
        for p in (
            MIRROR_ROOT / split
        ).glob("*.npy")
    }

    common_names = sorted(
        original_paths.keys()
        & mirror_paths.keys()
    )

    print("Original files:", len(original_paths))
    print("Mirror files  :", len(mirror_paths))
    print("Matched files :", len(common_names))

    print(
        "Missing mirror:",
        len(original_paths.keys() - mirror_paths.keys())
    )

    print(
        "Extra mirror  :",
        len(mirror_paths.keys() - original_paths.keys())
    )

    shape_errors = []
    frame_errors = []
    invalid_errors = []
    empty_sequences = []
    invalid_masks = []
    missing_coord_errors = []
    no_right_hand = []

    original_stats = []
    mirror_stats = []

    lr_errors = []
    rl_errors = []

    mask_matches = []


    for name in common_names:

        original = np.load(
            original_paths[name]
        )

        mirrored = np.load(
            mirror_paths[name]
        )

        # ------------------------------------------
        # Shape 검증
        # ------------------------------------------

        if (
            original.ndim != 2
            or mirrored.ndim != 2
            or original.shape[1] != 128
            or mirrored.shape[1] != 128
        ):

            shape_errors.append(name)
            continue

        # ------------------------------------------
        # 빈 sequence 검증
        # ------------------------------------------

        if (
            len(original) == 0
            or len(mirrored) == 0
        ):

            empty_sequences.append(name)
            continue

        # ------------------------------------------
        # NaN / Inf 검증
        # ------------------------------------------

        if (
            not np.isfinite(original).all()
            or not np.isfinite(mirrored).all()
        ):

            invalid_errors.append(name)
            continue

        # ------------------------------------------
        # Presence mask 검증
        # ------------------------------------------

        if not np.isin(
            mirrored[:, 126:128],
            [0.0, 1.0],
        ).all():

            invalid_masks.append(name)

        left_missing = mirrored[:, LEFT_MASK] == 0
        right_missing = mirrored[:, RIGHT_MASK] == 0

        if (
            np.any(mirrored[left_missing, LEFT_SLICE] != 0)
            or
            np.any(mirrored[right_missing, RIGHT_SLICE] != 0)
        ):

            missing_coord_errors.append(name)

        if not np.any(
            mirrored[:, RIGHT_MASK] > 0.5
        ):

            no_right_hand.append(name)

        # ------------------------------------------
        # Presence 통계
        # ------------------------------------------

        original_stats.append(
            presence_statistics(original)
        )

        mirror_stats.append(
            presence_statistics(mirrored)
        )

        # ------------------------------------------
        # 프레임 수 검증
        # ------------------------------------------

        if len(original) != len(mirrored):

            frame_errors.append(name)
            continue

        # ------------------------------------------
        # Presence mask 교환 여부
        # ------------------------------------------

        left_to_right = np.mean(
            original[:, LEFT_MASK]
            ==
            mirrored[:, RIGHT_MASK]
        )

        right_to_left = np.mean(
            original[:, RIGHT_MASK]
            ==
            mirrored[:, LEFT_MASK]
        )

        mask_matches.append([
            left_to_right,
            right_to_left,
        ])

        # ------------------------------------------
        # Original Left -> Mirror Right
        # ------------------------------------------

        lr = compare_coordinates(
            original,
            mirrored,
            LEFT_SLICE,
            RIGHT_SLICE,
            LEFT_MASK,
            RIGHT_MASK,
        )

        if lr is not None:
            lr_errors.append(lr)

        # ------------------------------------------
        # Original Right -> Mirror Left
        # ------------------------------------------

        rl = compare_coordinates(
            original,
            mirrored,
            RIGHT_SLICE,
            LEFT_SLICE,
            RIGHT_MASK,
            LEFT_MASK,
        )

        if rl is not None:
            rl_errors.append(rl)


    # ==================================================
    # 기본 검증 결과
    # ==================================================

    print()
    print("[Data integrity]")

    print("Shape errors       :", len(shape_errors))
    print("Frame count errors :", len(frame_errors))
    print("NaN / Inf errors   :", len(invalid_errors))
    print("Empty sequences    :", len(empty_sequences))
    print("Invalid masks      :", len(invalid_masks))
    print("Missing coord errors:", len(missing_coord_errors))
    print("No Right hand      :", len(no_right_hand))

    # ==================================================
    # Presence 비교
    # ==================================================

    if original_stats and mirror_stats:

        original_mean = np.mean(
            original_stats,
            axis=0
        ) * 100

        mirror_mean = np.mean(
            mirror_stats,
            axis=0
        ) * 100

        names = [
            "Left presence",
            "Right presence",
            "Both hands",
            "Any hand",
            "None",
        ]

        print()
        print("[Hand presence comparison]")
        print(f"{'Metric':18s} {'Original':>12s} {'Mirror':>12s}")

        for name, orig, mir in zip(
            names,
            original_mean,
            mirror_mean,
        ):

            print(
                f"{name:18s} "
                f"{orig:11.2f}% "
                f"{mir:11.2f}%"
            )

    # ==================================================
    # Presence mask 교환 검증
    # ==================================================

    if mask_matches:

        match_mean = np.mean(
            mask_matches,
            axis=0
        ) * 100

        print()
        print("[Presence mask agreement]")

        print(
            "Original Left -> Mirror Right:",
            f"{match_mean[0]:.2f}%"
        )

        print(
            "Original Right -> Mirror Left:",
            f"{match_mean[1]:.2f}%"
        )

    # ==================================================
    # 좌표 반전 오차
    # ==================================================

    print()
    print("[Coordinate reflection MAE]")

    for name, errors in [
        ("Original Left -> Mirror Right", lr_errors),
        ("Original Right -> Mirror Left", rl_errors),
    ]:

        print(name)

        if not errors:

            print("No comparable coordinates")
            continue

        errors = np.concatenate(
            errors,
            axis=0
        )

        mae = np.mean(
            errors,
            axis=0
        )

        print(f"  X MAE: {mae[0]:.6f}")
        print(f"  Y MAE: {mae[1]:.6f}")
        print(f"  Z MAE: {mae[2]:.6f}")


print()
print("=" * 70)
print("Verification completed")
print("=" * 70)