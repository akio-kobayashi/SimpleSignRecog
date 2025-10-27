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
    1フレーム前の手首位置との距離を元に、現在のフレームで手が入れ替わっているかを判定し、修正する。
    片手しか検出されていない場合も考慮し、より頑健な判定を行う。
    """
    num_frames = landmarks.shape[0]
    if num_frames == 0:
        return landmarks
        
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

        # --- スワップ判定と修正ロジック ---

        # Case 1: 現在のフレームで両手が検出された
        if left_present and right_present:
            # Subcase 1.1: 前のフレームでも両手が検出されていた (2 -> 2)
            if last_left_wrist is not None and last_right_wrist is not None:
                cost_no_swap = calculate_distance(current_left_wrist, last_left_wrist) + calculate_distance(current_right_wrist, last_right_wrist)
                cost_swap = calculate_distance(current_left_wrist, last_right_wrist) + calculate_distance(current_right_wrist, last_left_wrist)
                
                if cost_swap < cost_no_swap:
                    # スワップしたと判断し、データを入れ替える
                    corrected_landmarks[i, LEFT_HAND_SLICE] = right_hand_data
                    corrected_landmarks[i, RIGHT_HAND_SLICE] = left_hand_data
                    # currentの手首位置もスワップして後続の処理に使う
                    current_left_wrist, current_right_wrist = current_right_wrist, current_left_wrist
            
            # Subcase 1.2: 前のフレームでは左手のみ検出されていた (1 -> 2)
            elif last_left_wrist is not None and last_right_wrist is None:
                if calculate_distance(current_right_wrist, last_left_wrist) < calculate_distance(current_left_wrist, last_left_wrist):
                    # 現在の右手が前の左手に近い -> スワップ
                    corrected_landmarks[i, LEFT_HAND_SLICE] = right_hand_data
                    corrected_landmarks[i, RIGHT_HAND_SLICE] = left_hand_data
                    current_left_wrist, current_right_wrist = current_right_wrist, current_left_wrist

            # Subcase 1.3: 前のフレームでは右手のみ検出されていた (1 -> 2)
            elif last_right_wrist is not None and last_left_wrist is None:
                if calculate_distance(current_left_wrist, last_right_wrist) < calculate_distance(current_right_wrist, last_right_wrist):
                    # 現在の左手が前の右手に近い -> スワップ
                    corrected_landmarks[i, LEFT_HAND_SLICE] = right_hand_data
                    corrected_landmarks[i, RIGHT_HAND_SLICE] = left_hand_data
                    current_left_wrist, current_right_wrist = current_right_wrist, current_left_wrist

        # Case 2: 現在のフレームで左手のみが検出された
        elif left_present and not right_present:
            # Subcase 2.1: 前のフレームでは右手のみが検出されていた (1 -> 1, ラベル変化)
            if last_right_wrist is not None and last_left_wrist is None:
                # 手が1つしかなく、ラベルがRight->Leftに変わった場合、同一の手が誤ラベルされた可能性が高い
                # データを右手のスロットに移動する
                corrected_landmarks[i, RIGHT_HAND_SLICE] = left_hand_data
                corrected_landmarks[i, LEFT_HAND_SLICE].fill(np.nan)
                # currentの手首位置もスワップ
                current_right_wrist = current_left_wrist
                current_left_wrist = None

        # Case 3: 現在のフレームで右手のみが検出された
        elif right_present and not left_present:
            # Subcase 3.1: 前のフレームでは左手のみが検出されていた (1 -> 1, ラベル変化)
            if last_left_wrist is not None and last_right_wrist is None:
                # 手が1つしかなく、ラベルがLeft->Rightに変わった場合
                # データを左手のスロットに移動する
                corrected_landmarks[i, LEFT_HAND_SLICE] = right_hand_data
                corrected_landmarks[i, RIGHT_HAND_SLICE].fill(np.nan)
                # currentの手首位置もスワップ
                current_left_wrist = current_right_wrist
                current_right_wrist = None

        # --- 次のフレームのために、現在の（修正後の）手首位置を保存 ---
        last_left_wrist = current_left_wrist
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
        output_path = args.output_corrected_dir / npz_relative_path

        if not npz_full_path.exists():
            print(f"\n[WARN] NPZファイルが見つかりません: {npz_full_path}", file=sys.stderr)
            continue

        try:
            with np.load(npz_full_path) as data:
                landmarks = data['landmarks']
            
            # 修正処理を実行
            corrected_landmarks = correct_handedness_temporally(landmarks)

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
