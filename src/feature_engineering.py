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

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

# --- 定数定義 ---
# ランドマーク配列を便利に扱うためのスライス（部分配列）を定義

NUM_LANDMARKS_PER_HAND = 21  # 1つの手あたりのランドマーク数
LEFT_HAND_OFFSET = 0         # 左手のデータは配列の0番目から始まる
RIGHT_HAND_OFFSET = NUM_LANDMARKS_PER_HAND * 3  # 右手のデータは配列の63番目から始まる (21*3=63)

# 左手の全座標 (x,y,z * 21個) を取り出すためのスライス
LEFT_HAND_SLICE = slice(LEFT_HAND_OFFSET, LEFT_HAND_OFFSET + NUM_LANDMARKS_PER_HAND * 3)
# 右手の全座標を取り出すためのスライス
RIGHT_HAND_SLICE = slice(RIGHT_HAND_OFFSET, RIGHT_HAND_OFFSET + NUM_LANDMARKS_PER_HAND * 3)


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


# --- 前処理 (Preprocessing) 関数群 ---

def interpolate_missing_data(landmarks: np.ndarray) -> np.ndarray:
    """
    ランドマークデータの欠損値(NaN)を線形補間する関数。
    MediaPipeが手を検出できなかったフレームではデータがNaNになるため、その前後フレームの値から
    「おそらくこの辺りにあっただろう」という値を計算して埋めることで、データを安定させる。
    """
    df = pd.DataFrame(landmarks)
    # `interpolate`はPandasの便利な関数で、欠損値を自動で補間してくれる
    # method='linear': 線形補間（前後の値を直線で結んで中間点を求める）
    # limit_direction='both': 最初や最後のフレームがNaNでも、片側の値を使って埋める
    # axis=0: 時間軸（フレーム方向）に沿って補間する
    df_interpolated = df.interpolate(method='linear', limit_direction='both', axis=0)
    return df_interpolated.to_numpy()

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    各フレームのランドマーク座標を正規化する関数。
    この処理により、モデルは手の大きさや、カメラから手までの距離、画面上の手の位置に影響されにくくなる。
    （＝位置やスケールに対して不変になる）

    処理内容:
    1. 手首(0番)の座標が原点(0,0,0)に来るように、全て点を平行移動する。（位置不変性）
    2. 手首(0番)から中指の付け根(9番)までの距離が1になるように、全ての点の座標を拡大・縮小する。（スケール不変性）
    """
    num_frames = landmarks.shape[0]
    # 結果を格納するための配列を、元データと同じ形状で作成（中身は一旦NaN）
    normalized_landmarks = np.full_like(landmarks, np.nan, dtype=np.float32)

    # 1フレームずつ処理
    for i in range(num_frames):
        frame_data = landmarks[i]
        
        # 左右の手をそれぞれ個別に正規化
        for hand_offset in [LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET]:
            hand_data_slice = slice(hand_offset, hand_offset + NUM_LANDMARKS_PER_HAND * 3)
            hand_data = frame_data[hand_data_slice].reshape(NUM_LANDMARKS_PER_HAND, 3)

            # 手が検出されていないフレーム（全てNaN）はスキップ
            if np.all(np.isnan(hand_data)):
                continue

            # 1. 平行移動 (手首を原点に)
            wrist_coords = hand_data[0].copy() # 手首の座標をコピー
            translated_hand = hand_data - wrist_coords

            # 2. スケーリング (手首から中指付け根までの距離を基準にする)
            middle_finger_mcp_coords = translated_hand[9]
            scale_dist = np.linalg.norm(middle_finger_mcp_coords) # ユークリッド距離を計算

            # ゼロ除算を避けるための安全策
            if scale_dist < 1e-6:
                scaled_hand = translated_hand
            else:
                scaled_hand = translated_hand / scale_dist
            
            # 正規化後のデータを結果配列に格納
            normalized_landmarks[i, hand_data_slice] = scaled_hand.flatten()

    return normalized_landmarks


def canonical_normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    正準変換を用いてランドマーク座標を正規化する関数。
    この処理により、手の回転、位置、スケールに対して不変な特徴量を得る。
    "Real-Time Hand Gesture Monitoring Model Based on MediaPipe's Registerable System"
    で提案された手法に基づく。

    処理内容:
    1. 手首(0番)を原点に移動。
    2. 手のひら平面に基づいて新しい正規直交座標系（基底ベクトル）を定義。
       - X軸: 手首(0) -> 小指付け根(17)
       - Y軸: 手首(0) -> 人差し指付け根(5) を元に、X軸と直交するように計算
       - Z軸: X軸とY軸の外積から計算
    3. 手首(0)から小指付け根(17)までの距離が1になるように全体をスケーリング。
    4. 全てのランドマークを、この新しい「手のひら座標系」の値に変換する。
    """
    num_frames = landmarks.shape[0]
    normalized_landmarks = np.full_like(landmarks, np.nan, dtype=np.float32)

    for i in range(num_frames):
        frame_data = landmarks[i]
        
        for hand_offset in [LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET]:
            hand_data_slice = slice(hand_offset, hand_offset + NUM_LANDMARKS_PER_HAND * 3)
            hand_data = frame_data[hand_data_slice].reshape(NUM_LANDMARKS_PER_HAND, 3)

            if np.all(np.isnan(hand_data)):
                continue

            # 1. 平行移動 (手首を原点に)
            wrist_coords = hand_data[0].copy()
            translated_hand = hand_data - wrist_coords

            # 2. 新しい座標系の基底ベクトルを計算
            p5 = translated_hand[5]   # 人差し指付け根
            p17 = translated_hand[17] # 小指付け根

            # スケールを計算 (0-17間の距離)
            scale_dist = np.linalg.norm(p17)
            if scale_dist < 1e-6:
                # ランドマークが重なっているなど、異常なケースでは処理をスキップ
                continue

            # 新しいX軸 (0->17方向)
            new_x = p17 / scale_dist

            # 新しいZ軸の候補を計算 (X軸と0->5ベクトルの外積)
            p5_norm = np.linalg.norm(p5)
            if p5_norm < 1e-6:
                continue # p5がゼロベクトルならスキップ

            z_candidate = np.cross(new_x, p5 / p5_norm)
            
            # Z軸候補のノルムが非常に小さい場合 (0,5,17がほぼ一直線上の場合)のフォールバック
            if np.linalg.norm(z_candidate) < 1e-6:
                # 代替のベクトル（例：Z軸）と外積をとることで、安定した座標系を構築
                if np.abs(np.dot(new_x, np.array([0,0,1]))) < 0.99:
                    z_candidate = np.cross(new_x, np.array([0,0,1]))
                else: # new_xがZ軸に近すぎる場合
                    z_candidate = np.cross(new_x, np.array([0,1,0]))
            
            # 新しいZ軸を正規化
            new_z = z_candidate / np.linalg.norm(z_candidate)

            # 新しいY軸 (Z軸とX軸の外積で、正規直交基底を完成させる)
            new_y = np.cross(new_z, new_x)

            # 3. 回転行列（基底変換行列）を作成
            rotation_matrix = np.stack([new_x, new_y, new_z], axis=0)

            # 4. スケール正規化と座標変換を適用
            scaled_hand = translated_hand / scale_dist
            transformed_hand = scaled_hand @ rotation_matrix.T
            
            # 5. 左手の場合、Z軸を反転して座標系を右手系に統一
            if hand_offset == LEFT_HAND_OFFSET:
                transformed_hand[:, 2] *= -1

            normalized_landmarks[i, hand_data_slice] = transformed_hand.flatten()

    return normalized_landmarks


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

def calculate_features(landmarks: np.ndarray) -> np.ndarray:
    """
    前処理済みのランドマークデータから、最終的な特徴量セットを計算する統合関数。
    生の座標だけでなく、動きや形に関する情報を追加することで、モデルの認識精度向上を目指す。
    """
    # --- 運動特徴量 (Kinematic Features) ---
    # 速度: 前のフレームからの座標の変化量。動きの速さや方向を示す。
    velocity = np.diff(landmarks, axis=0, prepend=landmarks[0:1])
    # 加速度: 速度の変化量。動きの変化の度合いを示す。
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])

    # --- 形状特徴量 (Geometric Features) ---
    geometric = calculate_geometric_features(landmarks)

    # --- 全特徴量の結合 ---
    # 元の座標(位置)、速度、加速度、形状特徴量を全て結合して、1つの大きな特徴量ベクトルを作成する。
    
    # 左手の特徴量: [左手位置(63), 左手速度(63), 左手加速度(63), 左手形状(4)] -> 193次元
    left_features = np.concatenate([
        landmarks[:, LEFT_HAND_SLICE],
        velocity[:, LEFT_HAND_SLICE],
        acceleration[:, LEFT_HAND_SLICE],
        geometric[:, 0:4] # 左手の形状特徴量
    ], axis=1)

    # 右手の特徴量: [右手位置(63), 右手速度(63), 右手加速度(63), 右手形状(4)] -> 193次元
    right_features = np.concatenate([
        landmarks[:, RIGHT_HAND_SLICE],
        velocity[:, RIGHT_HAND_SLICE],
        acceleration[:, RIGHT_HAND_SLICE],
        geometric[:, 4:8] # 右手の形状特徴量
    ], axis=1)

    # 最終的に、左手と右手の特徴量を結合する [左手特徴量(193), 右手特徴量(193)] -> 386次元
    final_features = np.concatenate([left_features, right_features], axis=1)
    
    return final_features


