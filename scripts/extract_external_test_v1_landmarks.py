from pathlib import Path
import csv
import re

import cv2
import mediapipe as mp
import numpy as np


# ==================================================
# 경로
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = (
    BASE_DIR
    / "local_inference"
    / "input"
    / "external_test_v1"
)

OUTPUT_DIR = (
    BASE_DIR
    / "local_inference"
    / "work"
    / "external_test_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "external_test_v1_manifest.csv"
)


# ==================================================
# MediaPipe
# ==================================================

mp_hands = mp.solutions.hands


# ==================================================
# 파일명 파싱
#
# test_0101 -> class 1, repetition 1
# test_1203 -> class 12, repetition 3
# ==================================================

def parse_filename(path):

    match = re.fullmatch(
        r"test_(\d{2})(\d{2})",
        path.stem,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    target = int(
        match.group(1)
    )

    repetition = int(
        match.group(2)
    )

    if not 1 <= target <= 12:
        return None

    if not 1 <= repetition <= 3:
        return None

    return target, repetition


# ==================================================
# 영상 하나 landmark 추출
#
# frame feature:
#
# Left  21 * xyz = 63
# Right 21 * xyz = 63
# Left presence   = 1
# Right presence  = 1
#
# total = 128
# ==================================================

def extract_landmarks(
    video_path,
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "VideoCapture open 실패"
        )


    frames = []

    total_frames = 0

    left_detected = 0
    right_detected = 0
    both_detected = 0
    none_detected = 0


    # 영상마다 tracker를 새로 생성
    # → 이전 영상의 tracking state가 섞이지 않도록 함
    with mp_hands.Hands(

        static_image_mode=False,

        max_num_hands=2,

        model_complexity=1,

        min_detection_confidence=0.5,

        min_tracking_confidence=0.5,

    ) as hands:


        while True:

            ret, frame = cap.read()

            if not ret:
                break


            total_frames += 1


            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )


            result = hands.process(
                rgb
            )


            left = np.zeros(
                63,
                dtype=np.float32,
            )

            right = np.zeros(
                63,
                dtype=np.float32,
            )

            left_present = 0.0
            right_present = 0.0


            # --------------------------------------
            # detected hands
            # --------------------------------------

            if (
                result.multi_hand_landmarks
                and
                result.multi_handedness
            ):

                detected = []

                for (
                    hand_landmarks,
                    handedness,
                ) in zip(
                    result.multi_hand_landmarks,
                    result.multi_handedness,
                ):

                    label = (
                        handedness
                        .classification[0]
                        .label
                    )

                    coords = []

                    for landmark in (
                        hand_landmarks.landmark
                    ):

                        coords.extend([
                            landmark.x,
                            landmark.y,
                            landmark.z,
                        ])

                    detected.append(
                        (
                            label,
                            np.array(
                                coords,
                                dtype=np.float32,
                            ),
                        )
                    )


                # ----------------------------------
                # 일반 handedness 배치
                # ----------------------------------

                for label, coords in detected:

                    if label == "Left":

                        if left_present == 0.0:

                            left = coords

                            left_present = 1.0

                        elif right_present == 0.0:

                            # duplicate handedness fallback
                            right = coords

                            right_present = 1.0


                    elif label == "Right":

                        if right_present == 0.0:

                            right = coords

                            right_present = 1.0

                        elif left_present == 0.0:

                            # duplicate handedness fallback
                            left = coords

                            left_present = 1.0


            # --------------------------------------
            # presence 통계
            # --------------------------------------

            if (
                left_present == 1.0
                and
                right_present == 1.0
            ):

                both_detected += 1

            elif left_present == 1.0:

                left_detected += 1

            elif right_present == 1.0:

                right_detected += 1

            else:

                none_detected += 1


            feature = np.concatenate([
                left,
                right,
                np.array(
                    [
                        left_present,
                        right_present,
                    ],
                    dtype=np.float32,
                ),
            ])


            frames.append(
                feature
            )


    cap.release()


    if total_frames == 0:

        raise RuntimeError(
            "읽은 frame이 0개입니다."
        )


    sequence = np.stack(
        frames
    ).astype(np.float32)


    # ----------------------------------------------
    # 검출률
    # ----------------------------------------------

    left_presence_frames = (
        left_detected
        + both_detected
    )

    right_presence_frames = (
        right_detected
        + both_detected
    )

    any_presence_frames = (
        total_frames
        - none_detected
    )


    stats = {

        "total_frames":
            total_frames,

        "left_presence_rate":
            left_presence_frames
            / total_frames,

        "right_presence_rate":
            right_presence_frames
            / total_frames,

        "any_hand_rate":
            any_presence_frames
            / total_frames,

        "both_rate":
            both_detected
            / total_frames,

        "none_rate":
            none_detected
            / total_frames,
    }


    return sequence, stats


# ==================================================
# 영상 목록
# ==================================================

video_paths = sorted(
    [
        path
        for path in INPUT_DIR.iterdir()

        if (
            path.is_file()
            and
            path.suffix.lower()
            in {
                ".mp4",
                ".mov",
            }
        )
    ],
    key=lambda p: p.stem.lower(),
)


print(
    "영상 개수:",
    len(video_paths)
)


if len(video_paths) != 36:

    print(
        "[WARNING] "
        "예상 영상 수는 36개입니다."
    )


# ==================================================
# 전체 추출
# ==================================================

rows = []


for index, video_path in enumerate(
    video_paths,
    start=1,
):

    parsed = parse_filename(
        video_path
    )


    if parsed is None:

        print(
            f"[SKIP] 잘못된 파일명: "
            f"{video_path.name}"
        )

        continue


    target, repetition = parsed


    output_path = (
        OUTPUT_DIR
        / f"{video_path.stem}.npy"
    )


    print()
    print(
        f"[{index:02d}/{len(video_paths):02d}] "
        f"{video_path.name}"
    )

    print(
        f"target={target}, "
        f"repetition={repetition}"
    )


    try:

        sequence, stats = (
            extract_landmarks(
                video_path
            )
        )


        np.save(
            output_path,
            sequence,
        )


        print(
            "shape:",
            sequence.shape
        )

        print(
            "Left presence :",
            f"{stats['left_presence_rate'] * 100:.1f}%"
        )

        print(
            "Right presence:",
            f"{stats['right_presence_rate'] * 100:.1f}%"
        )

        print(
            "Any hand      :",
            f"{stats['any_hand_rate'] * 100:.1f}%"
        )


        rows.append({

            "filename":
                video_path.name,

            "target":
                target,

            "repetition":
                repetition,

            "extension":
                video_path.suffix.lower(),

            "status":
                "ok",

            "total_frames":
                stats["total_frames"],

            "left_presence_rate":
                stats["left_presence_rate"],

            "right_presence_rate":
                stats["right_presence_rate"],

            "both_rate":
                stats["both_rate"],

            "none_rate":
                stats["none_rate"],

            "landmark_path":
                str(output_path),

        })


    except Exception as e:

        print(
            "[ERROR]",
            e
        )


        rows.append({

            "filename":
                video_path.name,

            "target":
                target,

            "repetition":
                repetition,

            "extension":
                video_path.suffix.lower(),

            "status":
                "error",

            "total_frames":
                "",

            "left_presence_rate":
                "",

            "right_presence_rate":
                "",

            "both_rate":
                "",

            "none_rate":
                "",

            "landmark_path":
                "",

        })


# ==================================================
# CSV 저장
# ==================================================

fieldnames = [

    "filename",

    "target",

    "repetition",

    "extension",

    "status",

    "total_frames",

    "left_presence_rate",

    "right_presence_rate",

    "both_rate",

    "none_rate",

    "landmark_path",
]


with open(
    SUMMARY_PATH,
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
        rows
    )


# ==================================================
# 최종 요약
# ==================================================

success_count = sum(
    row["status"] == "ok"
    for row in rows
)

error_count = sum(
    row["status"] == "error"
    for row in rows
)


print()
print("=" * 70)
print("완료")
print("=" * 70)

print(
    "성공:",
    success_count
)

print(
    "실패:",
    error_count
)

print(
    "Landmark 폴더:",
    OUTPUT_DIR
)

print(
    "Manifest:",
    SUMMARY_PATH
)