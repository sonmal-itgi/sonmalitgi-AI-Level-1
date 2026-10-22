import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

SOURCE_MANIFEST = (
    BASE_DIR
    / "numeric_1to12"
    / "manifest.csv"
)

TRAIN_DIR = (
    BASE_DIR
    / "numeric_1to12"
    / "training"
    / "videos"
)

VAL_DIR = (
    BASE_DIR
    / "numeric_1to12"
    / "validation"
    / "videos"
)

OUTPUT_MANIFEST = (
    BASE_DIR
    / "numeric_1to12"
    / "baseline_v3_manifest.csv"
)


TRAIN_SIGNERS = {
    f"CROWD{i:02d}"
    for i in range(1, 18)
}

VAL_SIGNERS = {
    "CROWD18",
    "CROWD19",
}


rows = []
missing = []


with open(
    SOURCE_MANIFEST,
    "r",
    encoding="utf-8-sig",
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        split = row["split"]
        signer = row["signer"]

        if split == "Training":

            if signer not in TRAIN_SIGNERS:
                continue

            video_path = (
                TRAIN_DIR
                / row["video_name"]
            )

        elif split == "Validation":

            if signer not in VAL_SIGNERS:
                continue

            video_path = (
                VAL_DIR
                / row["video_name"]
            )

        else:
            continue


        if not video_path.exists():
            missing.append(str(video_path))


        new_row = dict(row)

        new_row["video_path"] = str(
            video_path
        )

        rows.append(new_row)


rows.sort(
    key=lambda x: (
        x["split"],
        int(
            x["class_id"]
            .replace("FS", "")
        ),
        x["signer"],
    )
)


fieldnames = [
    "split",
    "class_id",
    "label",
    "signer",
    "angle",
    "video_name",
    "duration",
    "start",
    "end",
    "video_path",
    "json_path",
]


with open(
    OUTPUT_MANIFEST,
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(rows)


training = [
    row
    for row in rows
    if row["split"] == "Training"
]

validation = [
    row
    for row in rows
    if row["split"] == "Validation"
]


print("===== Baseline v2 =====")

print(
    "Training:",
    len(training)
)

print(
    "Validation:",
    len(validation)
)

print(
    "전체:",
    len(rows)
)

print(
    "누락 영상:",
    len(missing)
)

print()
print("Training signers:")
print(
    sorted({
        row["signer"]
        for row in training
    })
)

print()
print("Validation signers:")
print(
    sorted({
        row["signer"]
        for row in validation
    })
)

print()
print("Manifest:")
print(OUTPUT_MANIFEST)


if missing:

    print()
    print("===== 누락 파일 =====")

    for path in missing:
        print(path)