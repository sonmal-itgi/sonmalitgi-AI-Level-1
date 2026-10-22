import re
import zipfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

ZIP_DIR = (
    BASE_DIR
    / "004.수어영상"
    / "1.Training"
    / "원천데이터"
    / "CROWD"
)

OUTPUT_DIR = (
    BASE_DIR
    / "numeric_1to12"
    / "training"
    / "videos"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# 예:
# NIA_SL_FS0801_CROWD04_F.mp4
# NIA_SL_FS0801_CROWD04_F_.mp4
pattern = re.compile(
    r"NIA_SL_FS(\d+)_CROWD(\d+)_F_?\.mp4$",
    re.IGNORECASE,
)


zip_paths = sorted(
    ZIP_DIR.glob("*_crowd_video.zip")
)


print("===== 발견된 ZIP =====")

for path in zip_paths:
    print(path.name)

print()
print("ZIP 수:", len(zip_paths))
print()


total_selected = 0
newly_extracted = 0
already_exists = 0


for zip_path in zip_paths:

    print("=" * 60)
    print("처리 ZIP:", zip_path.name)

    selected = []
    signers = set()

    with zipfile.ZipFile(zip_path, "r") as zf:

        for info in zf.infolist():

            filename = Path(info.filename).name

            match = pattern.match(filename)

            if not match:
                continue

            fs_number = int(
                match.group(1)
            )

            signer_number = int(
                match.group(2)
            )

            if 801 <= fs_number <= 812:

                selected.append(
                    (
                        info,
                        fs_number,
                        signer_number,
                    )
                )

                signers.add(
                    f"CROWD{signer_number:02d}"
                )


        print(
            "Signers:",
            sorted(signers)
        )

        print(
            "선택 영상:",
            len(selected)
        )


        for (
            info,
            fs_number,
            signer_number,
        ) in selected:

            output_name = (
                f"NIA_SL_FS{fs_number:04d}"
                f"_CROWD{signer_number:02d}"
                f"_F.mp4"
            )

            output_path = (
                OUTPUT_DIR
                / output_name
            )

            total_selected += 1


            if output_path.exists():

                already_exists += 1
                continue


            with zf.open(info) as src:
                with open(
                    output_path,
                    "wb"
                ) as dst:

                    while True:

                        chunk = src.read(
                            1024 * 1024
                        )

                        if not chunk:
                            break

                        dst.write(chunk)


            newly_extracted += 1


print()
print("=" * 60)
print("===== 전체 완료 =====")

print(
    "ZIP에서 발견한 대상 영상:",
    total_selected
)

print(
    "새로 추출한 영상:",
    newly_extracted
)

print(
    "이미 존재한 영상:",
    already_exists
)


saved = sorted(
    OUTPUT_DIR.glob("*.mp4")
)

print(
    "Training 폴더 전체 영상 수:",
    len(saved)
)


saved_signers = sorted({
    re.search(
        r"(CROWD\d+)",
        path.name
    ).group(1)

    for path in saved

    if re.search(
        r"(CROWD\d+)",
        path.name
    )
})

print(
    "현재 Training signers:",
    saved_signers
)

print(
    "현재 Training signer 수:",
    len(saved_signers)
)