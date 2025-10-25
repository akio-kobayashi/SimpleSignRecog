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

def correct_handedness_temporally(landmarks: np.ndarray) -> np.ndarray:
    """
    時系列情報を用いて、ランドマークデータの左右の入れ替わりを修正する。

    Args:
        landmarks (np.ndarray): (フレーム数, 126) のランドマーク配列。

    Returns:
        np.ndarray: 修正後のランドマーク配列。
    """
    num_frames = landmarks.shape[0]
    corrected_landmarks = np.copy(landmarks)
    
    last_left_wrist = None
    last_right_wrist = None

    for i in range(num_frames):
        current_frame = corrected_landmarks[i]
        
        left_hand_data = current_frame[LEFT_HAND_SLICE]
        right_hand_data = current_frame[RIGHT_HAND_SLICE]

        left_present = not np.all(np.isnan(left_hand_data))
        right_present = not np.all(np.isnan(right_hand_data))

        current_left_wrist = left_hand_data[0:3] if left_present else None
        current_right_wrist = right_hand_data[0:3] if right_present else None

        # 両手が検出され、かつ前のフレームでも両手が検出されていた場合にスワップをチェック
        if left_present and right_present and last_left_wrist is not None and last_right_wrist is not None:
            
            # 現在の左右の手と、前のフレームの左右の手の距離を計算
            dist_ll = calculate_distance(current_left_wrist, last_left_wrist)
            dist_rr = calculate_distance(current_right_wrist, last_right_wrist)
            
            dist_lr = calculate_distance(current_left_wrist, last_right_wrist)
            dist_rl = calculate_distance(current_right_wrist, last_left_wrist)

            # スワップしない場合のコスト vs スワップした場合のコスト
            cost_no_swap = dist_ll + dist_rr
            cost_swap = dist_lr + dist_rl

            if cost_swap < cost_no_swap:
                # スワップしたと判断し、データを入れ替える
                # print(f"  [INFO] Frame {i}: Handedness swap detected and corrected.")
                corrected_landmarks[i, LEFT_HAND_SLICE] = right_hand_data
                corrected_landmarks[i, RIGHT_HAND_SLICE] = left_hand_data
                
                # 入れ替えたので、現在のリストも更新
                current_left_wrist, current_right_wrist = current_right_wrist, current_left_wrist

        # 次のフレームのために、現在のフレームの手首位置を保存
        if left_present:
            last_left_wrist = current_left_wrist
        if right_present:
            last_right_wrist = current_right_wrist
            
    return corrected_landmarks


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
        
        if not npz_full_path.exists():
            print(f"\n[WARN] NPZファイルが見つかりません: {npz_full_path}", file=sys.stderr)
            continue

        try:
            with np.load(npz_full_path) as data:
                landmarks = data['landmarks']
            
            # print(f"\nProcessing: {npz_relative_path}")
            
            # 修正処理を実行
            corrected_landmarks = correct_handedness_temporally(landmarks)

            # 新しいパスに保存
            output_path = args.output_corrected_dir / npz_relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_path, landmarks=corrected_landmarks)
            
            # print(f"  [OK] Saved corrected file to: {output_path}")

        except Exception as e:
            print(f"\n[ERROR] ファイル処理中にエラーが発生しました: {npz_full_path}\n{e}", file=sys.stderr)

    print("[INFO] すべての処理が完了しました。")


if __name__ == "__main__":
    main()
