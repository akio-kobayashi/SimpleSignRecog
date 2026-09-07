#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
手指ランドマークデータから特徴量を抽出し、加工するための関数群を定義したモジュール。

「特徴量エンジニアリング」とは、生のデータ（ここでは指の関節座標）から、
機械学習モデルがより学習しやすくなるような「特徴」を計算して作り出す作業のことです。

主な機能:
- データ拡張 (Data Augmentation): 学習データ量を擬似的に増やし、モデルの性能を向上させる
- 欠損値の補間: ランドマークが検出できなかったフレームのデータを補う
- 座標の正規化: 手の大きさやカメラからの距離、位置の違いを吸収する
- データの平滑化: 座標の細かいブレ（ノイズ）を減らし、動きを滑らかにする
- 特徴量の計算: 位置情報だけでなく、速度や加速度、指の形状といった、より豊かな情報を計算する
"""

import numpy as np
import itertools
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

from src.preprocessing.landmark_layout import (
    LEFT_HAND_OFFSET,
    LEFT_HAND_SLICE,
    NUM_LANDMARKS_PER_HAND,
    RIGHT_HAND_OFFSET,
    RIGHT_HAND_SLICE,
)
from src.preprocessing.missing_data import interpolate_missing_data
from src.preprocessing.normalization import (
    canonical_normalize_landmarks,
    normalize_by_current_wrist,
    normalize_by_first_wrist,
    normalize_landmarks,
)

# --- データ拡張 (Data Augmentation) 関数群 ---
# これらは学習時にのみ使用し、学習データのバリエーションを増やすことで、
# 未知のデータに対するモデルの対応能力（汎化性能）を高めることを目的とします。

def augment_rotate(landmarks: np.ndarray) -> np.ndarray:
    """データ拡張：各フレームの手のランドマークを、手首を軸としてランダムに少しだけ回転させる。"""
    augmented_landmarks = landmarks.copy()
    num_frames = landmarks.shape[0]

    for i in range(num_frames):
        for hand_offset in [LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET]:
            hand_slice = slice(hand_offset, hand_offset + NUM_LANDMARKS_PER_HAND * 3)
            hand_data = augmented_landmarks[i, hand_slice].reshape(NUM_LANDMARKS_PER_HAND, 3)

            if np.all(np.isnan(hand_data)):
                continue

            # 手首(0番目のランドマーク)を座標の中心に移動
            wrist_coords = hand_data[0].copy()
            centered_hand = hand_data - wrist_coords

            # X, Y, Z軸周りにそれぞれ-15度から+15度の範囲でランダムな回転を生成
            random_angles = np.random.uniform(-15, 15, 3)
            rotation = R.from_euler('xyz', random_angles, degrees=True)

            # 回転を適用
            rotated_hand = rotation.apply(centered_hand)

            # 元の手首の座標を足し戻す
            final_hand = rotated_hand + wrist_coords

            augmented_landmarks[i, hand_slice] = final_hand.flatten()

    return augmented_landmarks

def augment_noise(landmarks: np.ndarray, scale=0.001) -> np.ndarray:
    """データ拡張：ランドマークの各座標に、ごくわずかなランダムな値（ガウスノイズ）を加える。"""
    noise = np.random.normal(0, scale, landmarks.shape)
    nan_mask = np.isnan(landmarks) # 元のデータがNaNの部分にはノイズを加えない
    noise[nan_mask] = 0
    return landmarks + noise

def augment_flip(landmarks: np.ndarray) -> np.ndarray:
    """データ拡張：左右の手のデータを入れ替え、水平方向に反転させる。右利きの人のデータを左利きのように見せかけることができる。"""
    flipped_landmarks = landmarks.copy()

    # 左手と右手のデータをまるごと入れ替える
    left_hand_data = flipped_landmarks[:, LEFT_HAND_OFFSET : LEFT_HAND_OFFSET + NUM_LANDMARKS_PER_HAND * 3].copy()
    right_hand_data = flipped_landmarks[:, RIGHT_HAND_OFFSET : RIGHT_HAND_OFFSET + NUM_LANDMARKS_PER_HAND * 3].copy()

    flipped_landmarks[:, LEFT_HAND_OFFSET : LEFT_HAND_OFFSET + NUM_LANDMARKS_PER_HAND * 3] = right_hand_data
    flipped_landmarks[:, RIGHT_HAND_OFFSET : RIGHT_HAND_OFFSET + NUM_LANDMARKS_PER_HAND * 3] = left_hand_data

    # 全てのx座標の符号を反転させる（水平反転）
    # x座標は配列の0, 3, 6, ... 番目に格納されている
    flipped_landmarks[:, 0::3] *= -1

    return flipped_landmarks


# --- 平滑化 (Smoothing) ---
def smooth_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Savitzky-Golayフィルタを使って、ランドマークの時系列データを平滑化（スムージング）する関数。
    検出された座標には細かいブレ（ノイズ）が含まれることがあるため、この処理で動きを滑らかにし、
    本質的な動きのパターンを抽出しやすくする。
    """
    # フィルタのパラメータ（これらの値を調整することで、平滑化の度合いが変わる）
    window_length = 5  # 平滑化の際に考慮する前後フレーム数（奇数である必要あり）
    polyorder = 2      # 近似に使う多項式の次数

    # フレーム数がウィンドウサイズより少ないとフィルタを適用できないため、何もしないで返す
    if landmarks.shape[0] < window_length:
        print("[WARN] Not enough frames to apply smoothing filter, skipping.")
        return landmarks

    # 各座標の時系列データ（全フレーム分）に対してフィルタを一度に適用
    smoothed_landmarks = savgol_filter(
        landmarks,
        window_length=window_length,
        polyorder=polyorder,
        axis=0  # 時間軸(フレーム方向)に沿ってフィルタを適用
    )
    return smoothed_landmarks


# --- 特徴量計算 (Feature Calculation) 関数群 ---

def calculate_geometric_features(landmarks: np.ndarray) -> np.ndarray:
    """
    各フレームにおける「形状に関する特徴量」を計算する関数。
    ここでは、親指の先端と他の4本の指の先端との間の距離を計算している。
    これにより、指が開いているか、閉じているか、といった手の「形」に関する情報が特徴量となる。
    """
    num_frames = landmarks.shape[0]
    # 結果を格納する配列 (フレーム数 x 8次元)。8次元なのは、左手4本指 + 右手4本指の距離のため。
    geometric_features = np.zeros((num_frames, 8), dtype=np.float32)

    # 指先のランドマークのインデックス (親指, 人差し指, 中指, 薬指, 小指)
    tip_indices = [4, 8, 12, 16, 20]

    for i in range(num_frames):
        frame_data = landmarks[i]

        # 左右の手を個別に処理
        for hand_idx, hand_offset in enumerate([LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET]):
            hand_data_slice = slice(hand_offset, hand_offset + NUM_LANDMARKS_PER_HAND * 3)
            hand_data = frame_data[hand_data_slice].reshape(NUM_LANDMARKS_PER_HAND, 3)

            if np.all(np.isnan(hand_data)):
                continue

            thumb_tip = hand_data[tip_indices[0]] # 親指先端の座標

            # 親指先端と他の4本の指先端とのユークリッド距離を計算
            for j in range(4):
                other_tip = hand_data[tip_indices[j + 1]]
                distance = np.linalg.norm(thumb_tip - other_tip)
                # 左手は0-3列、右手は4-7列に結果を格納
                geometric_features[i, hand_idx * 4 + j] = distance

    return geometric_features

def calculate_features(landmarks: np.ndarray, raw_landmarks: np.ndarray = None) -> np.ndarray:
    """
    前処理済みのランドマークデータから、最終的な特徴量セットを計算する統合関数。
    生の座標だけでなく、動きや形に関する情報を追加することで、モデルの認識精度向上を目指す。

    Args:
        landmarks (np.ndarray): 正規化・平滑化済みのランドマークデータ。
        raw_landmarks (np.ndarray, optional): 正規化前のランドマークデータ。
                                              グローバルな動き（手首の軌跡）を計算するために使用する。
    """
    # --- 運動特徴量 (Kinematic Features) ---
    # 速度: 前のフレームからの座標の変化量。動きの速さや方向を示す。
    velocity = np.diff(landmarks, axis=0, prepend=landmarks[0:1])
    # 加速度: 速度の変化量。動きの変化の度合いを示す。
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])

    # --- 形状特徴量 (Geometric Features) ---
    geometric = calculate_geometric_features(landmarks)

    # --- グローバル運動特徴量 (Global Motion Features) ---
    # 正規化前の座標を使って、手首の絶対的な動き（速度）を計算する
    if raw_landmarks is not None:
        # 左手の手首(0,1,2番目の要素)と右手の手首(63,64,65番目の要素)の速度を計算
        left_wrist_pos = raw_landmarks[:, 0:3]
        left_wrist_vel = np.diff(left_wrist_pos, axis=0, prepend=left_wrist_pos[0:1])

        right_wrist_pos = raw_landmarks[:, 63:66]
        right_wrist_vel = np.diff(right_wrist_pos, axis=0, prepend=right_wrist_pos[0:1])

        has_global_info = True
    else:
        has_global_info = False

    # --- 全特徴量の結合 ---
    # 元の座標(位置)、速度、加速度、形状特徴量、およびグローバル速度を結合する。

    # 左手の特徴量: [位置(63), 速度(63), 加速度(63), 形状(4), グローバル速度(3)] -> 196次元
    left_components = [
        landmarks[:, LEFT_HAND_SLICE],
        velocity[:, LEFT_HAND_SLICE],
        acceleration[:, LEFT_HAND_SLICE],
        geometric[:, 0:4]
    ]
    if has_global_info:
        left_components.append(left_wrist_vel)
    left_features = np.concatenate(left_components, axis=1)

    # 右手の特徴量: [位置(63), 速度(63), 加速度(63), 形状(4), グローバル速度(3)] -> 196次元
    right_components = [
        landmarks[:, RIGHT_HAND_SLICE],
        velocity[:, RIGHT_HAND_SLICE],
        acceleration[:, RIGHT_HAND_SLICE],
        geometric[:, 4:8]
    ]
    if has_global_info:
        right_components.append(right_wrist_vel)
    right_features = np.concatenate(right_components, axis=1)

    # 最終的に、左手と右手の特徴量を結合する [196 + 196] -> 392次元
    final_features = np.concatenate([left_features, right_features], axis=1)

    return final_features


# --- Gil-Martín et al. (ICAART 2025) 論文の特徴量実装 ---
# 以下の関数群は、"Hand Gesture Recognition Using MediaPipe Landmarks and Deep Learning Networks"
# (Gil-Martín et al., ICAART 2025) で記述された特徴量エンジニアリング手法の実装です。

def calculate_speed_features(landmarks: np.ndarray) -> np.ndarray:
    """
    論文で提案された速度特徴量を計算する。

    処理内容:
    - ランドマーク座標の時間微分を計算する。
    - `np.diff` を使い、フレーム間の座標の差分を求める。
    - 最初のフレームの速度は0とする。
    """
    # 最初のフレームの速度は変化がないため、prependで元の値を先頭に追加し、差分計算後の要素数を保つ
    # これにより、最初のフレームの速度はゼロベクトルになる
    velocity = np.diff(landmarks, axis=0, prepend=landmarks[0:1])
    return velocity


def calculate_anthropometric_features(landmarks: np.ndarray) -> np.ndarray:
    """
    論文で提案された人体測定的特徴量（ランドマーク間の距離）を計算する。

    処理内容:
    - 各手において、21個のランドマークの全てのペア(21 C 2 = 210通り)のユークリッド距離を計算する。
    - これにより、手の形状や指の相対的な位置関係をスケール不変な特徴量として表現する。
    - 出力は (フレーム数, 420) の形状になる (左手210 + 右手210)。
    """
    num_frames = landmarks.shape[0]
    num_pairs = len(list(itertools.combinations(range(NUM_LANDMARKS_PER_HAND), 2))) # 210
    # 左手210 + 右手210 = 420次元
    distances = np.full((num_frames, num_pairs * 2), np.nan, dtype=np.float32)

    landmark_pairs = list(itertools.combinations(range(NUM_LANDMARKS_PER_HAND), 2))

    for i in range(num_frames):
        frame_data = landmarks[i]

        for hand_idx, hand_offset in enumerate([LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET]):
            hand_data_slice = slice(hand_offset, hand_offset + NUM_LANDMARKS_PER_HAND * 3)
            hand_data = frame_data[hand_data_slice].reshape(NUM_LANDMARKS_PER_HAND, 3)

            if np.all(np.isnan(hand_data)):
                continue

            for pair_idx, (p1_idx, p2_idx) in enumerate(landmark_pairs):
                p1 = hand_data[p1_idx]
                p2 = hand_data[p2_idx]
                dist = np.linalg.norm(p1 - p2)
                distances[i, hand_idx * num_pairs + pair_idx] = dist

    return distances

def extract_paper_features(
    landmarks: np.ndarray,
    normalize_mode: str = None,
    speed: bool = False,
    anthropometric: bool = False
) -> np.ndarray:
    """
    Gil-Martín et al. (2025) 論文の特徴量抽出を統合的に実行するラッパー関数。

    Args:
        landmarks (np.ndarray): 元のランドマークデータ。形状は (frames, features)。
        normalize_mode (str, optional): 適用する正規化の種類。
            - 'current_wrist': フレーム毎の手首位置で正規化 (eq:current)。
            - 'first_wrist': 最初のフレームの手首位置で正規化 (eq:first)。
            - None: 正規化を行わない。 Defaults to None.
        speed (bool, optional): 速度特徴量を計算に含めるか。 Defaults to False.
        anthropometric (bool, optional): 人体測定的特徴量（ペア間距離）を計算に含めるか。 Defaults to False.

    Returns:
        np.ndarray: 計算された特徴量を結合した配列。

    Raises:
        ValueError: `normalize_mode`に無効な文字列が指定された場合。
    """

    # 1. 正規化の適用
    if normalize_mode:
        if normalize_mode == 'current_wrist':
            processed_landmarks = normalize_by_current_wrist(landmarks)
        elif normalize_mode == 'first_wrist':
            processed_landmarks = normalize_by_first_wrist(landmarks)
        else:
            raise ValueError(f"Invalid normalize_mode: '{normalize_mode}'. Choose from 'current_wrist', 'first_wrist', or None.")
    else:
        # 正規化しない場合は、元のデータをコピーして使用
        processed_landmarks = landmarks.copy()

    # 2. 特徴量の計算と結合
    # 結合する特徴量リストの初期値として、処理済みのランドマーク座標を設定
    features_to_combine = [processed_landmarks]

    if speed:
        # 速度特徴量を計算
        # 注意: 速度は正規化後の座標から計算するべき
        speed_features = calculate_speed_features(processed_landmarks)
        features_to_combine.append(speed_features)

    if anthropometric:
        # 人体測定的特徴量を計算
        # 注意: 距離はスケール不変なので、正規化前の座標から計算しても良いが、
        # 一貫性のため正規化後の座標から計算する
        anthropometric_features = calculate_anthropometric_features(processed_landmarks)
        features_to_combine.append(anthropometric_features)

    # 3. 全ての特徴量を結合して返す
    if len(features_to_combine) == 1:
        return features_to_combine[0]
    else:
        return np.concatenate(features_to_combine, axis=1)
