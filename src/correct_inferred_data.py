#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
このスクリプトは、`process_viedos.py`で生成されたデータセットに含まれる、
左右の手の割り当てが「推測(inferred)」されたデータを自動的に修正し、
より信頼性の高い新しいデータセットを生成することを目的とします。

背景:
`process_viedos.py`では、MediaPipeが左右の手を判別できなかった際に、
状況に応じて左右を推測するロジックが入っています。しかし、その推測は必ずしも正しくありません。
このスクリプトは、その推測ミスを修正するためのものです。

修正ロジックの仮定:
「動画は一人称視点で撮影されており、左手は画面の左半分に、右手は画面の右半分に現れる」
という強い仮定に基づいています。この仮定が成り立たないデータに対しては、意図しない修正を
行ってしまう可能性があるため注意が必要です。

処理内容:
1. 元の `metadata.csv` を読み込みます。
2. `quality_flag` が 'inferred' のファイルに対して、上記の仮定に基づき左右の割り当てをチェックし、
   必要であれば修正して、新しい出力先ディレクトリに保存します。
3. `quality_flag` が 'clean' のファイルは、そのまま新しいディレクトリにコピーします。
4. 全てのファイルの処理後、新しいデータセット構成を反映した `metadata.csv` を出力します。
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

# ランドマーク配列における左手と右手のデータ範囲を定義
# 左手: 0-62 (63次元), 右手: 63-126 (63次元)
LEFT_HAND_SLICE = slice(0, 63)
RIGHT_HAND_SLICE = slice(63, 126)

def calculate_distance(p1, p2):
    """2つの3D点間のユークリッド距離を計算するヘルパー関数（このスクリプトでは現在未使用）。"""
    if p1 is None or p2 is None or np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
        return float('inf')
    return np.linalg.norm(p1 - p2)

def determine_and_correct_handedness_globally(landmarks: np.ndarray) -> np.ndarray:
    """
    動画全体の左右の手の割り当てを、最初の検出フレームの位置に基づいて決定・修正するコア関数。
    """
    if landmarks.shape[0] == 0:
        return landmarks # 空のデータはそのまま返す

    # --- ステップ1: 最初に手が検出されるフレームを探す ---
    first_frame_idx = -1
    for i in range(landmarks.shape[0]):
        # isna(landmarks[i])が全てTrueでなければ、何かしらのデータがある
        if not np.all(np.isnan(landmarks[i])):
            first_frame_idx = i
            break
    
    # 有効なフレームが一つもなければ、何もせず返す
    if first_frame_idx == -1:
        return landmarks

    # --- ステップ2: 最初の検出フレームにおける手の位置を分析 ---
    first_frame_data = landmarks[first_frame_idx]
    
    left_hand_data = first_frame_data[LEFT_HAND_SLICE].reshape(21, 3)
    right_hand_data = first_frame_data[RIGHT_HAND_SLICE].reshape(21, 3)

    left_present = not np.all(np.isnan(left_hand_data))
    right_present = not np.all(np.isnan(right_hand_data))

    needs_global_swap = False # 動画全体で左右を入れ替える必要があるかどうかのフラグ

    # Case 1: 最初のフレームで両手が検出された場合
    if left_present and right_present:
        # 各手のX座標の平均値（重心）を計算。MediaPipeの座標系ではXは0.0(左端)から1.0(右端)の範囲。
        left_centroid_x = np.nanmean(left_hand_data[:, 0])
        right_centroid_x = np.nanmean(right_hand_data[:, 0])
        
        # もし「左手」の重心が「右手」の重心より右にあれば、ラベルが逆になっていると判断
        if left_centroid_x > right_centroid_x:
            needs_global_swap = True

    # Case 2: 最初のフレームで片手のみが検出された場合
    elif left_present and not right_present:
        left_centroid_x = np.nanmean(left_hand_data[:, 0])
        # もし「左手」が画面の右半分 (x > 0.5) にあれば、それは実は「右手」だと判断
        if left_centroid_x > 0.5:
            needs_global_swap = True
            
    elif right_present and not left_present:
        right_centroid_x = np.nanmean(right_hand_data[:, 0])
        # もし「右手」が画面の左半分 (x < 0.5) にあれば、それは実は「左手」だと判断
        if right_centroid_x < 0.5:
            needs_global_swap = True

    # --- ステップ3: 必要であれば、動画全体の左右データを入れ替える ---
    if needs_global_swap:
        print("  [INFO] 左右の割り当てが逆と判断されたため、修正します。")
        corrected_landmarks = np.copy(landmarks)
        # NumPyのスライス機能を使って、左手と右手のデータを効率的に入れ替える
        original_left_data = corrected_landmarks[:, LEFT_HAND_SLICE].copy()
        original_right_data = corrected_landmarks[:, RIGHT_HAND_SLICE].copy()
        corrected_landmarks[:, LEFT_HAND_SLICE] = original_right_data
        corrected_landmarks[:, RIGHT_HAND_SLICE] = original_left_data
        return corrected_landmarks
    else:
        # 修正不要と判断された場合は、元のデータをそのまま返す
        return landmarks

def main():
    """スクリプト実行のメイン関数。"""
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
    new_metadata_rows = [] # 新しいメタデータを格納するリスト

    print(f"[INFO] {len(df)}個のファイルを処理し、新しいデータセットを作成します。")

    # --- 3. 全てのファイルをループ処理 ---
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Creating new dataset"):
        # 各ファイルのパスを構築
        npz_relative_path_from_input_base = Path(row['npz_path'])
        npz_full_path = args.input_base_dir / npz_relative_path_from_input_base
        output_full_path = args.output_base_dir / npz_relative_path_from_input_base

        if not npz_full_path.exists():
            print(f"\n[WARN] 元ファイルが見つかりません、スキップします: {npz_full_path}", file=sys.stderr)
            continue

        # 出力先のサブディレクトリも作成
        output_full_path.parent.mkdir(parents=True, exist_ok=True)
        
        new_quality_flag = row['quality_flag']

        # --- 4. 'inferred' の場合は修正、'clean' の場合はコピー ---
        if row['quality_flag'] == 'inferred':
            try:
                # NPZファイルを読み込み
                with np.load(npz_full_path) as data:
                    landmarks = data['landmarks']
                
                # 左右判定・修正のコア関数を呼び出し
                corrected_landmarks = determine_and_correct_handedness_globally(landmarks)
                
                # 修正後のデータを新しい場所に保存
                np.savez_compressed(output_full_path, landmarks=corrected_landmarks)
                new_quality_flag = 'corrected' # フラグを'修正済み'に更新

            except Exception as e:
                # 万が一エラーが発生した場合は、元のファイルをそのままコピーする
                print(f"\n[ERROR] ファイル処理中にエラー: {npz_full_path}\n{e}", file=sys.stderr)
                print(f"  [INFO] 処理に失敗したため、元のファイルをコピーします: {npz_full_path} -> {output_full_path}")
                with open(npz_full_path, 'rb') as f_in, open(output_full_path, 'wb') as f_out:
                    f_out.write(f_in.read())
                new_quality_flag = 'correction_failed' # フラグを'修正失敗'に更新
        else: # 'clean' またはその他のフラグの場合は、単純にファイルをコピー
            with open(npz_full_path, 'rb') as f_in, open(output_full_path, 'wb') as f_out:
                f_out.write(f_in.read())

        # --- 5. 新しいメタデータを記録 ---
        # ファイルごとの情報を新しいメタデータリストに追加
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