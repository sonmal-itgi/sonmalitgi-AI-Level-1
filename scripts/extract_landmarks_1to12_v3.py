import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

MANIFEST_PATH = (
    BASE_DIR
    / "numeric_1to12"
    / "baseline_v3_manifest.csv"
)

OUTPUT_ROOT = (
    BASE_DIR
    / "numeric_1to12"
    / "landmarks_v3"
)

SUMMARY_PATH = (
    BASE_DIR
    / "numeric_1to12"
    / "landmark_manifest_v3.csv"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


mp_hands = mp.solutions.hands


def landmarks_to_vector(hand_landmarks):
    """
    한 손의 21 landmark를
    [x, y, z, x, y, z, ...] 형태의
    63차원 벡터로 변환
    """

    values = []

    for landmark in hand_landmarks.landmark:
        values.extend([
            landmark.x,
            landmark.y,
            landmark.z,
        ])

    return np.array(
        values,
        dtype=np.float32
    )


# --------------------------------------------------
# Manifest 읽기
# --------------------------------------------------

samples = []

with open(
    MANIFEST_PATH,
    "r",
    encoding="utf-8-sig",
) as f:

    reader = csv.DictReader(f)
    samples = list(reader)


print("처리할 영상 수:", len(samples))
print()


summary_rows = []


# --------------------------------------------------
# MediaPipe 초기화
# --------------------------------------------------

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as hands:

    for index, row in enumerate(
        samples,
        start=1
    ):

        split = row["split"]
        class_id = row["class_id"]
        label = row["label"]
        signer = row["signer"]

        video_path = Path(
            row["video_path"]
        )

        start_time = float(
            row["start"]
        )

        end_time = float(
            row["end"]
        )

        # ------------------------------------------
        # 저장 폴더
        # ------------------------------------------

        split_dir = (
            OUTPUT_ROOT
            / split.lower()
        )

        split_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_name = (
            video_path.stem
            + ".npy"
        )

        output_path = (
            split_dir
            / output_name
        )

        # ------------------------------------------
        # 영상 열기
        # ------------------------------------------

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():

            print(
                "[영상 열기 실패]",
                video_path
            )

            continue


        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            start_time * 1000
        )


        sequence = []

        total_frames = 0
        any_hand_frames = 0
        one_hand_frames = 0
        two_hand_frames = 0


        # ------------------------------------------
        # 프레임 처리
        # ------------------------------------------

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            current_time = (
                cap.get(
                    cv2.CAP_PROP_POS_MSEC
                )
                / 1000.0
            )

            if current_time > end_time:
                break


            total_frames += 1


            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb)


            # 한 프레임:
            #
            # 왼손 63
            # 오른손 63
            # left 존재 여부 1
            # right 존재 여부 1
            #
            # 총 128차원

            left_vector = np.zeros(
                63,
                dtype=np.float32
            )

            right_vector = np.zeros(
                63,
                dtype=np.float32
            )

            left_present = 0.0
            right_present = 0.0


            detected_count = 0


            if results.multi_hand_landmarks:

                detected_count = len(
                    results.multi_hand_landmarks
                )

                any_hand_frames += 1

                if detected_count == 1:
                    one_hand_frames += 1

                elif detected_count >= 2:
                    two_hand_frames += 1


                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
                ):

                    vector = (
                        landmarks_to_vector(
                            hand_landmarks
                        )
                    )

                    hand_label = (
                        handedness
                        .classification[0]
                        .label
                    )


                    if hand_label == "Left":

                        # 동일 label이 중복 검출되면
                        # 비어 있는 반대 슬롯 사용
                        if left_present == 0:

                            left_vector = vector
                            left_present = 1.0

                        elif right_present == 0:

                            right_vector = vector
                            right_present = 1.0

                    else:

                        if right_present == 0:

                            right_vector = vector
                            right_present = 1.0

                        elif left_present == 0:

                            left_vector = vector
                            left_present = 1.0


            frame_features = np.concatenate([
                left_vector,
                right_vector,
                np.array(
                    [
                        left_present,
                        right_present,
                    ],
                    dtype=np.float32
                ),
            ])

            sequence.append(
                frame_features
            )


        cap.release()


        # ------------------------------------------
        # Numpy 저장
        # ------------------------------------------

        if sequence:

            sequence_array = np.stack(
                sequence
            )

        else:

            sequence_array = np.empty(
                (0, 128),
                dtype=np.float32
            )


        np.save(
            output_path,
            sequence_array
        )


        if total_frames > 0:

            detection_rate = (
                any_hand_frames
                / total_frames
                * 100
            )

        else:

            detection_rate = 0.0


        summary_rows.append({
            "split": split,
            "class_id": class_id,
            "label": label,
            "signer": signer,
            "video_name": row["video_name"],
            "total_frames": total_frames,
            "any_hand_frames": any_hand_frames,
            "one_hand_frames": one_hand_frames,
            "two_hand_frames": two_hand_frames,
            "detection_rate": f"{detection_rate:.2f}",
            "feature_dim": 128,
            "landmark_path": str(
                output_path
            ),
        })


        print(
            f"[{index:02d}/{len(samples)}] "
            f"{class_id} "
            f"{signer} | "
            f"frames={total_frames} | "
            f"detected={detection_rate:.1f}%"
        )


# --------------------------------------------------
# 결과 Manifest 저장
# --------------------------------------------------

with open(
    SUMMARY_PATH,
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    fieldnames = [
        "split",
        "class_id",
        "label",
        "signer",
        "video_name",
        "total_frames",
        "any_hand_frames",
        "one_hand_frames",
        "two_hand_frames",
        "detection_rate",
        "feature_dim",
        "landmark_path",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        summary_rows
    )


# --------------------------------------------------
# 전체 통계
# --------------------------------------------------

frame_counts = [
    row["total_frames"]
    for row in summary_rows
]

detection_rates = [
    float(row["detection_rate"])
    for row in summary_rows
]


print()
print("=" * 60)
print("Landmark 추출 완료")
print("=" * 60)

print(
    "처리 영상:",
    len(summary_rows)
)

if frame_counts:

    print(
        "최소 프레임:",
        min(frame_counts)
    )

    print(
        "최대 프레임:",
        max(frame_counts)
    )

    print(
        "평균 프레임:",
        f"{np.mean(frame_counts):.1f}"
    )

if detection_rates:

    print(
        "평균 손 검출률:",
        f"{np.mean(detection_rates):.1f}%"
    )

print()
print(
    "Landmark Manifest:",
    SUMMARY_PATH
)