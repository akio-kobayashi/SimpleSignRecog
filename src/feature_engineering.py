#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
手指ランドマークデータから特徴量を抽出・生成するモジュール。

機能:
- 欠損値の補間
- 座標の正規化（並進・スケール不変）
- データの平滑化（将来的に実装）
- 運動特徴量・形状特徴量の計算（将来的に実装）
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

# ランドマーク配列のスライス定義
NUM_LANDMARKS_PER_HAND = 21
LEFT_HAND_OFFSET = 0
RIGHT_HAND_OFFSET = NUM_LANDMARKS_PER_HAND * 3

def augment_rotate(landmarks: np.ndarray) -> np.ndarray:
    """Applies a random rotation to each hand's point cloud."""
    augmented_landmarks = landmarks.copy()
    num_frames = landmarks.shape[0]

    for i in range(num_frames):
        for hand_offset in [LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET]:
            hand_slice = slice(hand_offset, hand_offset + NUM_LANDMARKS_PER_HAND * 3)
            hand_data = augmented_landmarks[i, hand_slice].reshape(NUM_LANDMARKS_PER_HAND, 3)

            if np.all(np.isnan(hand_data)):
                continue

            # Center the hand around the wrist before rotating
            wrist_coords = hand_data[0].copy()
            centered_hand = hand_data - wrist_coords

            # Generate a random small rotation
            random_angles = np.random.uniform(-15, 15, 3) # Max 15 degrees rotation on each axis
            rotation = R.from_euler('xyz', random_angles, degrees=True)
            
            # Apply rotation
            rotated_hand = rotation.apply(centered_hand)

            # Add the wrist coordinates back
            final_hand = rotated_hand + wrist_coords
            
            augmented_landmarks[i, hand_slice] = final_hand.flatten()
            
    return augmented_landmarks

def augment_noise(landmarks: np.ndarray, scale=0.001) -> np.ndarray:
    """Adds Gaussian noise to the landmarks."""
    noise = np.random.normal(0, scale, landmarks.shape)
    nan_mask = np.isnan(landmarks)
    noise[nan_mask] = 0
    return landmarks + noise

def augment_flip(landmarks: np.ndarray) -> np.ndarray:
    """Horizontally flips the landmarks and swaps hand labels."""
    flipped_landmarks = landmarks.copy()
    
    # Swap Left and Right hand data
    left_hand_data = flipped_landmarks[:, LEFT_HAND_OFFSET : LEFT_HAND_OFFSET + NUM_LANDMARKS_PER_HAND * 3].copy()
    right_hand_data = flipped_landmarks[:, RIGHT_HAND_OFFSET : RIGHT_HAND_OFFSET + NUM_LANDMARKS_PER_HAND * 3].copy()
    
    flipped_landmarks[:, LEFT_HAND_OFFSET : LEFT_HAND_OFFSET + NUM_LANDMARKS_PER_HAND * 3] = right_hand_data
    flipped_landmarks[:, RIGHT_HAND_OFFSET : RIGHT_HAND_OFFSET + NUM_LANDMARKS_PER_HAND * 3] = left_hand_data

    # Invert the x-coordinate
    # x-coordinates are at indices 0, 3, 6, ...
    flipped_landmarks[:, 0::3] *= -1
    
    return flipped_landmarks


def interpolate_missing_data(landmarks: np.ndarray) -> np.ndarray:
    """
    ランドマークデータの欠損値(NaN)を線形補間する。

    Args:
        landmarks (np.ndarray): (フレーム数, 126) のランドマーク配列。

    Returns:
        np.ndarray: 補間後のランドマーク配列。
    """
    df = pd.DataFrame(landmarks)
    # 時間軸(axis=0)に沿って各列を線形補間
    df_interpolated = df.interpolate(method='linear', limit_direction='both', axis=0)
    return df_interpolated.to_numpy()

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    各フレームのランドマークを正規化する。
    - 手首(0番)を原点とするように平行移動
    - 手首(0番)と中指付け根(9番)の距離でスケーリング

    Args:
        landmarks (np.ndarray): (フレーム数, 126) の補間済みランドマーク配列。

    Returns:
        np.ndarray: 正規化後のランドマーク配列。
    """
    num_frames = landmarks.shape[0]
    normalized_landmarks = np.full_like(landmarks, np.nan, dtype=np.float32)

    for i in range(num_frames):
        frame_data = landmarks[i]
        
        # 左右の手を個別に処理
        for hand_offset in [LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET]:
            hand_data_slice = slice(hand_offset, hand_offset + NUM_LANDMARKS_PER_HAND * 3)
            hand_data = frame_data[hand_data_slice].reshape(NUM_LANDMARKS_PER_HAND, 3)

            # 手が検出されているかチェック (NaNでないか)
            if np.all(np.isnan(hand_data)):
                continue

            # 1. 並進不変性のための処理 (手首を原点に)
            wrist_coords = hand_data[0].copy()
            translated_hand = hand_data - wrist_coords

            # 2. スケール不変性のための処理
            middle_finger_mcp_coords = translated_hand[9]
            scale_dist = np.linalg.norm(middle_finger_mcp_coords)

            # ゼロ除算を避ける
            if scale_dist < 1e-6:
                scaled_hand = translated_hand # スケールが非常に小さい場合はそのまま
            else:
                scaled_hand = translated_hand / scale_dist
            
            normalized_landmarks[i, hand_data_slice] = scaled_hand.flatten()

    return normalized_landmarks

# --- 将来の実装のためのプレースホルダー ---

def smooth_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Savitzky-Golayフィルタを用いてランドマークデータを平滑化する。

    Args:
        landmarks (np.ndarray): (フレーム数, 126) の正規化済みランドマーク配列。

    Returns:
        np.ndarray: 平滑化後のランドマーク配列。
    """
    # フィルタのパラメータ (調整可能)
    window_length = 5  # 奇数である必要がある
    polyorder = 2      # ウィンドウ内でフィットさせる多項式の次数

    # フレーム数がウィンドウサイズより小さい場合はスキップ
    if landmarks.shape[0] < window_length:
        print("[WARN] Not enough frames to apply smoothing filter, skipping.")
        return landmarks

    # 各座標の時系列データに対してフィルタを適用
    smoothed_landmarks = savgol_filter(
        landmarks,
        window_length=window_length,
        polyorder=polyorder,
        axis=0  # 時間軸に沿って適用
    )
    return smoothed_landmarks

def calculate_geometric_features(landmarks: np.ndarray) -> np.ndarray:
    """
    各フレーム内の形状特徴量（指先間の距離）を計算する。

    Args:
        landmarks (np.ndarray): (フレーム数, 126) の正規化・平滑化済みランドマーク配列。

    Returns:
        np.ndarray: 形状特徴量の配列 (フレーム数, 8)。
    """
    num_frames = landmarks.shape[0]
    geometric_features = np.zeros((num_frames, 8), dtype=np.float32)
    
    # 指先のインデックス (Thumb, Index, Middle, Ring, Pinky)
    tip_indices = [4, 8, 12, 16, 20]

    for i in range(num_frames):
        frame_data = landmarks[i]
        
        # 左右の手を個別に処理
        for hand_idx, hand_offset in enumerate([LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET]):
            hand_data_slice = slice(hand_offset, hand_offset + NUM_LANDMARKS_PER_HAND * 3)
            hand_data = frame_data[hand_data_slice].reshape(NUM_LANDMARKS_PER_HAND, 3)

            if np.all(np.isnan(hand_data)):
                continue

            thumb_tip = hand_data[tip_indices[0]]
            
            # 親指先端と他の4本の指先端との距離を計算
            for j in range(4):
                other_tip = hand_data[tip_indices[j + 1]]
                distance = np.linalg.norm(thumb_tip - other_tip)
                # 左手は0-3列、右手は4-7列に格納
                geometric_features[i, hand_idx * 4 + j] = distance
                
    return geometric_features


def calculate_features(landmarks: np.ndarray) -> np.ndarray:
    """
    平滑化されたランドマークから全ての特徴量を計算する。
    - 運動特徴量（速度・加速度）
    - 形状特徴量（指先間距離）
    最終的な特徴量ベクトルは [左手の特徴量, 右手の特徴量] の順に連結される。
    """
    # 運動特徴量
    velocity = np.diff(landmarks, axis=0, prepend=landmarks[0:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])

    # 形状特徴量
    geometric = calculate_geometric_features(landmarks)

    # 左右の手の特徴量をそれぞれ結合し、最終的に左右の手の特徴量を連結する
    # 左手の特徴量: [左手位置, 左手速度, 左手加速度, 左手形状]
    left_features = np.concatenate([
        landmarks[:, LEFT_HAND_SLICE],
        velocity[:, LEFT_HAND_SLICE],
        acceleration[:, LEFT_HAND_SLICE],
        geometric[:, 0:4] # Left geometric features
    ], axis=1) # Shape (T, 63+63+63+4 = 193)

    # 右手の特徴量: [右手位置, 右手速度, 右手加速度, 右手形状]
    right_features = np.concatenate([
        landmarks[:, RIGHT_HAND_SLICE],
        velocity[:, RIGHT_HAND_SLICE],
        acceleration[:, RIGHT_HAND_SLICE],
        geometric[:, 4:8] # Right geometric features
    ], axis=1) # Shape (T, 193)

    # 最終的な特徴量ベクトルは [左手の特徴量, 右手の特徴量] の順に連結
    final_features = np.concatenate([left_features, right_features], axis=1) # Shape (T, 193 + 193 = 386)
    
    return final_features


