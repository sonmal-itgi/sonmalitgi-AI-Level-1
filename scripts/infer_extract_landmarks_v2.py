import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def landmarks_to_vector(hand_landmarks):
    values = []

    for landmark in hand_landmarks.landmark:
        values.extend([
            landmark.x,
            landmark.y,
            landmark.z,
        ])

    return np.array(
        values,
        dtype=np.float32,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
        help="입력 MP4 경로",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="출력 landmark .npy 경로",
    )

    parser.add_argument(
    "--start",
    type=float,
    default=None,
    help="분석 시작 시간(초)",
    )

    parser.add_argument(
    "--end",
    type=float,
    default=None,
    help="분석 종료 시간(초)",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output)

    if not video_path.exists():
        raise FileNotFoundError(
            f"영상이 없습니다: {video_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"영상을 열 수 없습니다: {video_path}"
        )

    mp_hands = mp.solutions.hands

    sequence = []

    total_frames = 0
    detected_frames = 0

    if args.start is not None:
        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            args.start * 1000
        )

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

            current_time = (
                cap.get(cv2.CAP_PROP_POS_MSEC)
                / 1000.0
            )

            if (
                args.end is not None
                and current_time > args.end
            ):
                break

            total_frames += 1

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            results = hands.process(rgb)

            left_hand = np.zeros(
                63,
                dtype=np.float32,
            )

            right_hand = np.zeros(
                63,
                dtype=np.float32,
            )

            left_present = 0.0
            right_present = 0.0

            if (
                results.multi_hand_landmarks
                and results.multi_handedness
            ):
                detected_frames += 1

                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
                ):
                    label = (
                        handedness
                        .classification[0]
                        .label
                    )

                    vector = landmarks_to_vector(
                        hand_landmarks
                    )

                    if label == "Left":
                        if left_present == 0:
                            left_hand = vector
                            left_present = 1.0

                        elif right_present == 0:
                            right_hand = vector
                            right_present = 1.0

                    else:
                        if right_present == 0:
                            right_hand = vector
                            right_present = 1.0

                        elif left_present == 0:
                            left_hand = vector
                            left_present = 1.0

            feature = np.concatenate([
                left_hand,
                right_hand,
                np.array(
                    [
                        left_present,
                        right_present,
                    ],
                    dtype=np.float32,
                ),
            ])

            sequence.append(feature)

    cap.release()

    if len(sequence) == 0:
        raise RuntimeError(
            "영상에서 프레임을 읽지 못했습니다."
        )

    sequence = np.stack(
        sequence,
        axis=0,
    )

    np.save(
        output_path,
        sequence,
    )

    detection_rate = (
        detected_frames
        / total_frames
        * 100
    )

    print()
    print("=" * 60)
    print("Landmark 추출 완료")
    print("=" * 60)

    print(
        "입력 영상:",
        video_path,
    )

    print(
        "전체 프레임:",
        total_frames,
    )

    print(
        "출력 shape:",
        sequence.shape,
    )

    print(
        f"손 검출률: {detection_rate:.1f}%"
    )

    print(
        "저장:",
        output_path,
    )


if __name__ == "__main__":
    main()