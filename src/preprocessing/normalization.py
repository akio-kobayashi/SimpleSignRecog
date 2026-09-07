"""手指ランドマークの座標正規化。

各関数は左右の手を別々に処理し、入力と同じ ``(フレーム数, 126)``
形式を返します。前回の実験では ``canonical_normalize_landmarks`` を使用します。
"""

import numpy as np

from .landmark_layout import (
    HAND_OFFSETS,
    LEFT_HAND_OFFSET,
    LEFT_HAND_SLICE,
    NUM_LANDMARKS_PER_HAND,
    RIGHT_HAND_SLICE,
)

EPSILON = 1e-6


def _hand_slice(hand_offset: int) -> slice:
    width = NUM_LANDMARKS_PER_HAND * 3
    return slice(hand_offset, hand_offset + width)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """各フレームで手首を原点にし、0番–9番の距離を1にする。"""
    normalized = np.full_like(landmarks, np.nan, dtype=np.float32)

    for frame_index, frame in enumerate(landmarks):
        for hand_offset in HAND_OFFSETS:
            hand_slice = _hand_slice(hand_offset)
            hand = frame[hand_slice].reshape(NUM_LANDMARKS_PER_HAND, 3)
            if np.all(np.isnan(hand)):
                continue

            translated = hand - hand[0]
            scale = np.linalg.norm(translated[9])
            if scale >= EPSILON:
                translated = translated / scale
            normalized[frame_index, hand_slice] = translated.flatten()

    return normalized


def canonical_normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """掌を基準とする座標系へ変換し、位置・大きさ・向きをそろえる。

    - 原点: 手首（0番）
    - X軸: 手首から小指付け根（17番）
    - 掌平面: 人差し指付け根（5番）と小指付け根から決定
    - 大きさ: 0番–17番の距離を1にする
    """
    normalized = np.full_like(landmarks, np.nan, dtype=np.float32)

    for frame_index, frame in enumerate(landmarks):
        for hand_offset in HAND_OFFSETS:
            hand_slice = _hand_slice(hand_offset)
            hand = frame[hand_slice].reshape(NUM_LANDMARKS_PER_HAND, 3)
            if np.all(np.isnan(hand)):
                continue

            translated = hand - hand[0]
            point_5 = translated[5]
            point_17 = translated[17]

            scale = np.linalg.norm(point_17)
            if scale < EPSILON:
                continue
            new_x = point_17 / scale

            point_5_norm = np.linalg.norm(point_5)
            if point_5_norm < EPSILON:
                continue
            new_z_candidate = np.cross(new_x, point_5 / point_5_norm)

            if np.linalg.norm(new_z_candidate) < EPSILON:
                if np.abs(np.dot(new_x, np.array([0, 0, 1]))) < 0.99:
                    new_z_candidate = np.cross(new_x, np.array([0, 0, 1]))
                else:
                    new_z_candidate = np.cross(new_x, np.array([0, 1, 0]))

            new_z = new_z_candidate / np.linalg.norm(new_z_candidate)
            new_y = np.cross(new_z, new_x)
            rotation_matrix = np.stack([new_x, new_y, new_z], axis=0)
            transformed = (translated / scale) @ rotation_matrix.T

            if hand_offset == LEFT_HAND_OFFSET:
                transformed[:, 2] *= -1
            normalized[frame_index, hand_slice] = transformed.flatten()

    return normalized


def normalize_by_current_wrist(landmarks: np.ndarray) -> np.ndarray:
    """各フレームの座標から、そのフレームの手首座標を引く。"""
    normalized = np.full_like(landmarks, np.nan, dtype=np.float32)

    for frame_index, frame in enumerate(landmarks):
        for hand_offset in HAND_OFFSETS:
            hand_slice = _hand_slice(hand_offset)
            hand = frame[hand_slice].reshape(NUM_LANDMARKS_PER_HAND, 3)
            if not np.all(np.isnan(hand)):
                normalized[frame_index, hand_slice] = (hand - hand[0]).flatten()

    return normalized


def normalize_by_first_wrist(landmarks: np.ndarray) -> np.ndarray:
    """各手で最初に検出された手首を、動画全体の原点にする。"""
    normalized = np.full_like(landmarks, np.nan, dtype=np.float32)
    first_wrists = {
        "left": np.full(3, np.nan, dtype=np.float32),
        "right": np.full(3, np.nan, dtype=np.float32),
    }

    for frame in landmarks:
        if np.all(np.isnan(first_wrists["left"])):
            left = frame[LEFT_HAND_SLICE].reshape(NUM_LANDMARKS_PER_HAND, 3)
            if not np.all(np.isnan(left)):
                first_wrists["left"] = left[0].copy()
        if np.all(np.isnan(first_wrists["right"])):
            right = frame[RIGHT_HAND_SLICE].reshape(NUM_LANDMARKS_PER_HAND, 3)
            if not np.all(np.isnan(right)):
                first_wrists["right"] = right[0].copy()
        if not np.all(np.isnan(first_wrists["left"])) and not np.all(
            np.isnan(first_wrists["right"])
        ):
            break

    for frame_index, frame in enumerate(landmarks):
        for name, hand_slice in (
            ("left", LEFT_HAND_SLICE),
            ("right", RIGHT_HAND_SLICE),
        ):
            hand = frame[hand_slice].reshape(NUM_LANDMARKS_PER_HAND, 3)
            wrist = first_wrists[name]
            if not np.all(np.isnan(hand)) and not np.all(np.isnan(wrist)):
                normalized[frame_index, hand_slice] = (hand - wrist).flatten()

    return normalized
