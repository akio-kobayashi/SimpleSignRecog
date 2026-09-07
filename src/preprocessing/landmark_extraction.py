"""MediaPipe Tasksで動画から手指ランドマークを抽出する。"""

import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from .landmark_layout import LANDMARK_DIM, LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def _complete_handedness(result) -> tuple[dict[int, str], bool]:
    """Tasks APIの左右ラベルを取得し、欠けている場合だけ補う。"""
    handedness: dict[int, str] = {}
    for hand_index, categories in enumerate(result.handedness or []):
        if categories and categories[0].category_name in {"Left", "Right"}:
            handedness[hand_index] = categories[0].category_name

    inferred = False
    number_of_hands = len(result.hand_landmarks or [])
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


def _draw_hand(frame: np.ndarray, hand_landmarks, width: int, height: int) -> None:
    points = [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in hand_landmarks
    ]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
    for point in points:
        cv2.circle(frame, point, 3, (0, 0, 255), -1)


def extract_landmarks_from_video(
    video_path: Path,
    model_asset_path: Path,
    draw: bool = False,
    draw_dir: Path | None = None,
    static_image_mode: bool = False,
    max_num_hands: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> tuple[np.ndarray, bool, int]:
    """1本の動画を左右21点、計126次元の時系列へ変換する。

    Hand Landmarkerの画像座標 ``(x, y, z)`` を使用する。左手は列0～62、
    右手は列63～125へ格納し、未検出座標はNaNのまま返す。
    """
    video_path = Path(video_path)
    model_asset_path = Path(model_asset_path)
    if not model_asset_path.is_file():
        raise FileNotFoundError(f"Hand Landmarkerモデルがありません: {model_asset_path}")

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

    running_mode = (
        mp.tasks.vision.RunningMode.IMAGE
        if static_image_mode
        else mp.tasks.vision.RunningMode.VIDEO
    )
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_asset_path)),
        running_mode=running_mode,
        num_hands=max_num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    frames = []
    had_inference = False
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
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
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    if static_image_mode:
                        result = detector.detect(image)
                    else:
                        timestamp_ms = round(frame_index * 1000.0 / fps)
                        result = detector.detect_for_video(image, timestamp_ms)

                    frame_landmarks = np.full(LANDMARK_DIM, np.nan, dtype=np.float32)
                    handedness, inferred = _complete_handedness(result)
                    had_inference = had_inference or inferred

                    for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
                        label = handedness.get(hand_index)
                        if label == "Left":
                            start = LEFT_HAND_OFFSET
                        elif label == "Right":
                            start = RIGHT_HAND_OFFSET
                        else:
                            continue

                        for landmark_index, landmark in enumerate(hand_landmarks):
                            position = start + landmark_index * 3
                            frame_landmarks[position : position + 3] = (
                                landmark.x,
                                landmark.y,
                                landmark.z,
                            )
                        if writer is not None:
                            _draw_hand(bgr_frame, hand_landmarks, width, height)

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
    return np.asarray(frames, dtype=np.float32), had_inference, total_frames
