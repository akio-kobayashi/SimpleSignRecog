"""MediaPipe Handsを使った動画ランドマーク抽出。

このモジュールは「1本の動画を126次元の時系列に変換する」処理だけを担当します。
ファイルの列挙、NPZ保存、metadata.csv作成は ``process_videos.py`` が担当します。
"""

import sys
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from .landmark_layout import LANDMARK_DIM, LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET


def _complete_handedness(result) -> tuple[dict[int, str], bool]:
    """MediaPipeの左右ラベルを取得し、欠けている場合は従来規則で補う。"""
    handedness: dict[int, str] = {}
    if result.multi_handedness:
        for hand_data in result.multi_handedness:
            classification = hand_data.classification[0]
            handedness[classification.index] = classification.label

    inferred = False
    number_of_hands = len(result.multi_hand_landmarks or [])
    if number_of_hands == 2 and len(handedness) == 1:
        known_index = next(iter(handedness))
        known_label = handedness[known_index]
        handedness[1 - known_index] = "Left" if known_label == "Right" else "Right"
        inferred = True
    elif number_of_hands == 1 and not handedness:
        handedness[0] = "Right"
        inferred = True
    elif number_of_hands == 2 and not handedness:
        handedness[0] = "Right"
        handedness[1] = "Left"
        inferred = True

    return handedness, inferred


def _create_video_writer(
    output_path: Path, width: int, height: int, fps: float
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def extract_landmarks_from_video(
    video_path: Path,
    draw: bool = False,
    draw_dir: Optional[Path] = None,
    static_image_mode: bool = False,
    max_num_hands: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> tuple[np.ndarray, bool, int]:
    """1本の動画から左右21点ずつのランドマークを抽出する。

    左手は配列の0～62、右手は63～125に格納します。検出できなかった
    座標はNaNのまま残し、後段の欠損処理で補間します。

    Returns:
        ``(landmarks, had_inference, num_frames)``。landmarksの形状は
        ``(フレーム数, 126)``、型はfloat32です。had_inferenceは左右ラベルを
        補ったフレームが一度でもあったことを表します。
    """
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"[WARN] 失敗: {video_path} を開けませんでした。", file=sys.stderr)
        return np.array([]), False, 0

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer = None
    drawn_video_path = None
    if draw:
        if draw_dir is None:
            raise ValueError("draw=True の場合は draw_dir を指定してください。")
        draw_dir = Path(draw_dir)
        draw_dir.mkdir(parents=True, exist_ok=True)
        drawn_video_path = draw_dir / f"{video_path.stem}_hands.mp4"
        writer = _create_video_writer(drawn_video_path, width, height, fps)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    frames = []
    had_inference = False

    try:
        with mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,
        ) as hands:
            progress = tqdm(
                total=total_frames if total_frames > 0 else None,
                desc=f"Processing {video_path.name}",
                unit="f",
            )
            frame_index = 0
            try:
                while True:
                    ok, bgr_frame = capture.read()
                    if not ok:
                        break

                    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                    result = hands.process(rgb_frame)
                    frame_landmarks = np.full(
                        LANDMARK_DIM, np.nan, dtype=np.float32
                    )

                    if result.multi_hand_landmarks:
                        handedness, inferred = _complete_handedness(result)
                        had_inference = had_inference or inferred

                        for hand_index, hand_landmarks in enumerate(
                            result.multi_hand_landmarks
                        ):
                            label = handedness.get(hand_index)
                            if label == "Left":
                                start = LEFT_HAND_OFFSET
                            elif label == "Right":
                                start = RIGHT_HAND_OFFSET
                            else:
                                print(
                                    f"[WARN] 処理されない手: {label} "
                                    f"(ビデオ: {video_path.name}, "
                                    f"フレーム: {frame_index})"
                                )
                                continue

                            for landmark_index, landmark in enumerate(
                                hand_landmarks.landmark
                            ):
                                position = start + landmark_index * 3
                                frame_landmarks[position : position + 3] = (
                                    landmark.x,
                                    landmark.y,
                                    landmark.z,
                                )

                            if writer is not None:
                                mp_drawing.draw_landmarks(
                                    bgr_frame,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_styles.get_default_hand_landmarks_style(),
                                    mp_styles.get_default_hand_connections_style(),
                                )

                    frames.append(frame_landmarks)
                    if writer is not None:
                        writer.write(bgr_frame)
                    frame_index += 1
                    progress.update(1)
            finally:
                progress.close()
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if drawn_video_path is not None:
        print(f"[OK] 描画済み動画: {drawn_video_path}")
    if not frames:
        return np.array([]), had_inference, total_frames
    return np.array(frames, dtype=np.float32), had_inference, total_frames
