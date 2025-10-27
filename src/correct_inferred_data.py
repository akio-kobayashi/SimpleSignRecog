#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NPZファイルに記録された手指ランドマークの左右の割り当てを、
時間的な連続性を利用して修正するスクリプト。

処理内容:
1. metadata.csv を読み込み、quality_flag が 'inferred' のファイルを見つける。
2. 対象のNPZファイルを読み込む。
3. フレームを順番に処理し、手首の位置に基づいてフレーム間で手が入れ替わっていないかチェックする。
4. 入れ替わり（スワップ）が検出された場合、そのフレームの左右のデータを修正する。
5. 修正後のデータを新しいディレクトリに保存する。
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

# 左手: 0-62, 右手: 63-125
LEFT_HAND_SLICE = slice(0, 63)
RIGHT_HAND_SLICE = slice(63, 126)

def calculate_distance(p1, p2):
    """2つの3D点間のユークリッド距離を計算する。"""
    if p1 is None or p2 is None or np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
        return float('inf')
    return np.linalg.norm(p1 - p2)

def determine_and_correct_handedness_globally(landmarks: np.ndarray) -> np.ndarray:
    """
    最初の検出フレームにおける手の空間的な位置に基づいて、動画全体の左右の割り当てを決定・修正する。
    1人称視点の動画で、手が画面の左右に対応する位置から現れることを想定している。
    """
    if landmarks.shape[0] == 0:
        return landmarks

    # 少なくとも片手が検出される最初のフレームを探す
    first_frame_idx = -1
    for i in range(landmarks.shape[0]):
        if not np.all(np.isnan(landmarks[i])):
            first_frame_idx = i
            break
    
    # 有効なフレームが一つもなければ、何もせず返す
    if first_frame_idx == -1:
        return landmarks

    first_frame_data = landmarks[first_frame_idx]
    
    left_hand_data = first_frame_data[LEFT_HAND_SLICE].reshape(21, 3)
    right_hand_data = first_frame_data[RIGHT_HAND_SLICE].reshape(21, 3)

    left_present = not np.all(np.isnan(left_hand_data))
    right_present = not np.all(np.isnan(right_hand_data))

    needs_global_swap = False

    # Case 1: 最初のフレームで両手が検出された
    if left_present and right_present:
        # 各手のX座標の平均値（重心）を計算
        left_centroid_x = np.nanmean(left_hand_data[:, 0])
        right_centroid_x = np.nanmean(right_hand_data[:, 0])
        
        # 「左手」として記録されている手 (left_hand_data) が、
        # 「右手」として記録されている手 (right_hand_data) よりも画面右側にある場合、
        # 動画全体で左右が入れ替わっていると判断する。
        if left_centroid_x > right_centroid_x:
            needs_global_swap = True

    # Case 2: 最初のフレームで片手のみが検出された
    elif left_present and not right_present:
        left_centroid_x = np.nanmean(left_hand_data[:, 0])
        # 「左手」として記録されている手が画面の右半分 (x > 0.5) にある場合、
        # それは実際には右手であると判断する。
        if left_centroid_x > 0.5:
            needs_global_swap = True
            
    elif right_present and not left_present:
        right_centroid_x = np.nanmean(right_hand_data[:, 0])
        # 「右手」として記録されている手が画面の左半分 (x < 0.5) にある場合、
        # それは実際には左手であると判断する。
        if right_centroid_x < 0.5:
            needs_global_swap = True

    # スワップが必要な場合、動画全体の左右のデータを入れ替える
    if needs_global_swap:
        # print(f"  [INFO] Global handedness swap applied.")
        corrected_landmarks = np.copy(landmarks)
        
        # 元のデータを一時的に保持
        original_left_data = corrected_landmarks[:, LEFT_HAND_SLICE].copy()
        original_right_data = corrected_landmarks[:, RIGHT_HAND_SLICE].copy()
        
        # スワップを実行
        corrected_landmarks[:, LEFT_HAND_SLICE] = original_right_data
        corrected_landmarks[:, RIGHT_HAND_SLICE] = original_left_data
        
        return corrected_landmarks
    else:
        # スワップ不要なら元のデータをそのまま返す
        return landmarks

def main():
    ap = argparse.ArgumentParser(description="推測された左右の割り当てを時系列情報で修正する")
    ap.add_argument("-i", "--input_base_dir", required=True, type=Path,
                    help="metadata.csvとprocessed_dataディレクトリを含むベースディレクトリ")
    ap.add_argument("-o", "--output_corrected_dir", required=True, type=Path,
                    help="修正後のNPZファイルを保存するディレクトリ")
    args = ap.parse_args()

    metadata_path = args.input_base_dir / "metadata.csv"
    if not metadata_path.exists():
        print(f"[ERROR] metadata.csvが見つかりません: {metadata_path}", file=sys.stderr)
        sys.exit(1)

    # 出力ディレクトリを作成
    args.output_corrected_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_path)
    
    # quality_flagが'inferred'のものを対象とする
    inferred_df = df[df['quality_flag'] == 'inferred'].copy()

    if inferred_df.empty:
        print("[INFO] 修正対象のファイルはありませんでした。")
        return

    print(f"[INFO] {len(inferred_df)}個のファイルを修正します。")

    for index, row in tqdm(inferred_df.iterrows(), total=inferred_df.shape[0], desc="Correcting files"):
        npz_relative_path = row['npz_path']
        npz_full_path = args.input_base_dir / npz_relative_path
        output_path = args.output_corrected_dir / npz_relative_path

        if not npz_full_path.exists():
            print(f"\n[WARN] NPZファイルが見つかりません: {npz_full_path}", file=sys.stderr)
            continue

        try:
            with np.load(npz_full_path) as data:
                landmarks = data['landmarks']
            
            # 修正処理を実行
            corrected_landmarks = determine_and_correct_handedness_globally(landmarks)

            # 新しいパスに保存
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_path, landmarks=corrected_landmarks)

        except Exception as e:
            print(f"\n[ERROR] ファイル処理中にエラーが発生しました: {npz_full_path}\n{e}", file=sys.stderr)
            print(f"  [INFO] 処理に失敗したため、元のファイルをコピーします: {npz_full_path} -> {output_path}")
            try:
                # Ensure parent directory exists before copying
                output_path.parent.mkdir(parents=True, exist_ok=True)
                # Copy the original file over
                with open(npz_full_path, 'rb') as f_in:
                    content = f_in.read()
                with open(output_path, 'wb') as f_out:
                    f_out.write(content)
            except Exception as copy_e:
                print(f"  [FATAL] 元のファイルのコピーに失敗しました: {copy_e}", file=sys.stderr)

    print("[INFO] すべての処理が完了しました。")


if __name__ == "__main__":
    main()