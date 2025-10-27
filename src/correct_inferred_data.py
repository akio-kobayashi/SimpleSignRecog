#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NPZファイルに記録された手指ランドマークの左右の割り当てを修正し、
一貫性のある完全な新しいデータセットを生成するスクリプト。

処理内容:
1. 元の metadata.csv を読み込む。
2. 'inferred' フラグの付いたファイルを新しいロジックで修正し、新しいディレクトリに保存する。
3. 'clean' フラグの付いたファイルをそのまま新しいディレクトリにコピーする。
4. 新しいデータセット構成を反映した、新しい metadata.csv を生成する。
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
        
        if left_centroid_x > right_centroid_x:
            needs_global_swap = True

    # Case 2: 最初のフレームで片手のみが検出された
    elif left_present and not right_present:
        left_centroid_x = np.nanmean(left_hand_data[:, 0])
        if left_centroid_x > 0.5:
            needs_global_swap = True
            
    elif right_present and not left_present:
        right_centroid_x = np.nanmean(right_hand_data[:, 0])
        if right_centroid_x < 0.5:
            needs_global_swap = True

    if needs_global_swap:
        corrected_landmarks = np.copy(landmarks)
        original_left_data = corrected_landmarks[:, LEFT_HAND_SLICE].copy()
        original_right_data = corrected_landmarks[:, RIGHT_HAND_SLICE].copy()
        corrected_landmarks[:, LEFT_HAND_SLICE] = original_right_data
        corrected_landmarks[:, RIGHT_HAND_SLICE] = original_left_data
        return corrected_landmarks
    else:
        return landmarks

def main():
    ap = argparse.ArgumentParser(description="左右の割り当てを修正し、新しい完全なデータセットを出力する")
    ap.add_argument("-i", "--input_base_dir", required=True, type=Path,
                    help="元となる metadata.csv と processed_data ディレクトリを含むベースディレクトリ")
    ap.add_argument("-o", "--output_base_dir", required=True, type=Path,
                    help="修正後の完全なデータセット（NPZと新しいmetadata.csv）の出力先")
    args = ap.parse_args()

    # --- 1. パスの設定と入力ファイルの検証 ---
    metadata_path = args.input_base_dir / "metadata.csv"
    if not metadata_path.exists():
        print(f"[ERROR] metadata.csvが見つかりません: {metadata_path}", file=sys.stderr)
        sys.exit(1)

    # --- 2. 出力ディレクトリの作成 ---
    args.output_base_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_path)
    new_metadata_rows = []

    print(f"[INFO] {len(df)}個のファイルを処理し、新しいデータセットを作成します。")

    # --- 3. 全てのファイルをループ処理 ---
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Creating new dataset"):
        npz_relative_path_from_input_base = Path(row['npz_path'])
        npz_full_path = args.input_base_dir / npz_relative_path_from_input_base
        output_full_path = args.output_base_dir / npz_relative_path_from_input_base

        if not npz_full_path.exists():
            print(f"\n[WARN] 元ファイルが見つかりません、スキップします: {npz_full_path}", file=sys.stderr)
            continue

        output_full_path.parent.mkdir(parents=True, exist_ok=True)
        
        new_quality_flag = row['quality_flag']

        # --- 4. 'inferred' の場合は修正、'clean' の場合はコピー ---
        if row['quality_flag'] == 'inferred':
            try:
                with np.load(npz_full_path) as data:
                    landmarks = data['landmarks']
                
                corrected_landmarks = determine_and_correct_handedness_globally(landmarks)
                
                np.savez_compressed(output_full_path, landmarks=corrected_landmarks)
                new_quality_flag = 'corrected'

            except Exception as e:
                print(f"\n[ERROR] ファイル処理中にエラー: {npz_full_path}\n{e}", file=sys.stderr)
                print(f"  [INFO] 処理に失敗したため、元のファイルをコピーします: {npz_full_path} -> {output_full_path}")
                with open(npz_full_path, 'rb') as f_in, open(output_full_path, 'wb') as f_out:
                    f_out.write(f_in.read())
                new_quality_flag = 'correction_failed'
        else: # 'clean' またはその他のフラグの場合
            with open(npz_full_path, 'rb') as f_in, open(output_full_path, 'wb') as f_out:
                f_out.write(f_in.read())

        # --- 5. 新しいメタデータを記録 ---
        new_metadata_rows.append({
            "npz_path": str(npz_relative_path_from_input_base),
            "class_label": row['class_label'],
            "original_video_path": row['original_video_path'],
            "quality_flag": new_quality_flag,
            "num_frames": row['num_frames'],
        })

    # --- 6. 新しいメタデータCSVを保存 ---
    if new_metadata_rows:
        new_metadata_df = pd.DataFrame(new_metadata_rows)
        new_metadata_csv_path = args.output_base_dir / "metadata.csv"
        new_metadata_df.to_csv(new_metadata_csv_path, index=False, encoding="utf-8")
        print(f"\n[OK] 新しいメタデータCSVが保存されました: {new_metadata_csv_path}")

    print("[INFO] すべての処理が完了しました。")


if __name__ == "__main__":
    main()