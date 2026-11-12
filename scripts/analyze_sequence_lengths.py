from pathlib import Path
import re

import numpy as np


# ==================================================
# 기본 설정
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_DIR = (
    BASE_DIR
    / "numeric_1to12"
    / "landmarks_v3"
    / "training"
)

VAL_DIR = (
    BASE_DIR
    / "numeric_1to12"
    / "landmarks_v3"
    / "validation"
)


# ==================================================
# 클래스 번호 추출
# FS0801 -> 1
# ...
# FS0812 -> 12
# ==================================================

def get_class_number(path):

    match = re.search(
        r"FS(\d{4})",
        path.name,
    )

    if match is None:
        return None

    fs_id = int(
        match.group(1)
    )

    return fs_id - 800


# ==================================================
# 폴더 분석
# ==================================================

def analyze_directory(
    name,
    directory,
):

    rows = []

    for path in sorted(
        directory.glob("*.npy")
    ):

        sequence = np.load(
            path,
            mmap_mode="r",
        )

        if (
            sequence.ndim != 2
            or sequence.shape[1] != 128
        ):
            print(
                "[WARNING]",
                path.name,
                sequence.shape,
            )

            continue

        rows.append(
            (
                path.name,
                get_class_number(path),
                len(sequence),
            )
        )


    lengths = np.array(
        [
            row[2]
            for row in rows
        ],
        dtype=np.int32,
    )


    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print(
        "samples:",
        len(lengths)
    )

    print(
        "min:",
        lengths.min()
    )

    print(
        "max:",
        lengths.max()
    )

    print(
        "mean:",
        f"{lengths.mean():.2f}"
    )

    print(
        "std:",
        f"{lengths.std():.2f}"
    )

    print(
        "median:",
        f"{np.median(lengths):.1f}"
    )


    print()
    print("===== Percentiles =====")

    for percentile in [
        5,
        10,
        25,
        50,
        75,
        90,
        95,
    ]:

        value = np.percentile(
            lengths,
            percentile,
        )

        print(
            f"P{percentile:02d}:",
            f"{value:.1f}"
        )


    # --------------------------------------------------
    # 후보 window별 coverage
    #
    # length >= window이면
    # 최소 하나의 full window를 만들 수 있음
    # --------------------------------------------------

    print()
    print("===== Full-window coverage =====")

    for window in [
        48,
        60,
        64,
        72,
        80,
        90,
        96,
        100,
        112,
        120,
    ]:

        count = np.sum(
            lengths >= window
        )

        rate = (
            count
            / len(lengths)
            * 100
        )

        print(
            f"window={window:3d}: "
            f"{count:3d}/{len(lengths)} "
            f"({rate:5.1f}%)"
        )


    # --------------------------------------------------
    # 클래스별 길이
    # --------------------------------------------------

    print()
    print("===== Per-class =====")

    print(
        f"{'Class':<8}"
        f"{'N':>6}"
        f"{'Min':>8}"
        f"{'Median':>10}"
        f"{'Mean':>10}"
        f"{'Max':>8}"
    )


    for class_number in range(
        1,
        13,
    ):

        class_lengths = np.array(
            [
                length
                for _, label, length
                in rows
                if label == class_number
            ]
        )

        print(
            f"{class_number:<8}"
            f"{len(class_lengths):>6}"
            f"{class_lengths.min():>8}"
            f"{np.median(class_lengths):>10.1f}"
            f"{class_lengths.mean():>10.1f}"
            f"{class_lengths.max():>8}"
        )


    # --------------------------------------------------
    # 가장 짧은 / 긴 sample
    # --------------------------------------------------

    print()
    print("===== 10 shortest =====")

    for filename, label, length in sorted(
        rows,
        key=lambda x: x[2],
    )[:10]:

        print(
            f"{length:4d}  "
            f"class={label:2d}  "
            f"{filename}"
        )


    print()
    print("===== 10 longest =====")

    for filename, label, length in sorted(
        rows,
        key=lambda x: x[2],
        reverse=True,
    )[:10]:

        print(
            f"{length:4d}  "
            f"class={label:2d}  "
            f"{filename}"
        )


# ==================================================
# MAIN
# ==================================================

analyze_directory(
    "TRAINING",
    TRAIN_DIR,
)

analyze_directory(
    "VALIDATION",
    VAL_DIR,
)