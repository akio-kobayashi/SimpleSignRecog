#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
動画からMediaPipe Handsの手指ランドマークを抽出し、NPZファイルとメタデータCSVを生成するスクリプト。

- 入力: クラスディレクトリ（1-20）を含む入力動画のルートディレクトリ
- 出力:
    - 各動画のランドマークデータを格納したNPZファイル
    - NPZファイルのパスとクラス名を紐付けたメタデータCSV
- 追加: --draw 指定でランドマーク描画済みのMP4を保存

依存:
  pip install mediapipe opencv-python pandas tqdm numpy

使い方例:
  python process_videos.py -i /path/to/root_videos -o ./output_data --draw
"""

import argparse
import os
import sys
import glob
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# MediaPipeライブラリをインポート
import mediapipe as mp

# 処理対象とする動画の拡張子リスト
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm")

# 1フレームあたりのランドマークの次元数
# 1つの手には21個のランドマークがあり、それぞれx, y, zの3次元座標を持つ。
# 両手分なので、21 * 3 * 2 = 126次元となる。
LANDMARK_DIM = 21 * 3 * 2

def find_videos(input_path: Path, pattern: str):
    """指定されたパスの中から、処理対象の動画ファイルを探し出す関数。"""
    if input_path.is_file():
        return [input_path]
    vids = []
    if input_path.is_dir():
        if pattern:
            # パターンに一致するファイルを検索
            vids = [Path(p) for p in glob.glob(str(input_path / pattern), recursive=False)]
        else:
            # ディレクトリ内の全ての動画ファイルを検索
            for p in input_path.iterdir():
                if p.suffix.lower() in VIDEO_EXTS:
                    vids.append(p)
    return sorted(vids)

def init_writer_for_draw(out_path: Path, width: int, height: int, fps: float):
    """ランドマークを描画した動画を保存するためのVideoWriterオブジェクトを作成する関数。"""
    # 動画のコーデック（圧縮形式）を指定
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

def extract_landmarks_from_video(
    video_path: Path,
    draw: bool = False,
    draw_dir: Path = None,
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) -> tuple[np.ndarray, bool, int]:
    """
    一つの動画ファイルからMediaPipe Handsを使って手指のランドマークを抽出するメインの関数。
    手が検出されなかったフレームや、検出されなかった手に対応するデータはNaN（非数）で埋められる。

    Args:
        video_path: 処理対象の動画ファイルのパス
        draw: Trueの場合、ランドマークを描画した動画を別途保存する
        draw_dir: 描画動画の保存先ディレクトリ
        (その他): MediaPipeの動作を制御するパラメータ

    Returns:
        np.ndarray: 抽出されたランドマークデータ（形状: [フレーム数, 126]）
        bool: MediaPipeが左右の手を推測したかどうかを示すフラグ
        int: 動画の総フレーム数
    """
    # MediaPipeの準備
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    # OpenCVで動画ファイルを読み込む
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] 失敗: {video_path} を開けませんでした。", file=sys.stderr)
        return np.array([]), False, 0

    # 動画の情報を取得
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    all_frame_landmarks = []  # 全フレームのランドマークデータを格納するリスト
    had_inference = False  # この動画で左右の手の「推測」が行われたかどうかのフラグ
    writer = None
    out_draw_path = None
    if draw:
        # 描画オプションが有効な場合、保存用の設定を初期化
        draw_dir.mkdir(parents=True, exist_ok=True)
        out_draw_path = draw_dir / f"{video_path.stem}_hands.mp4"
        writer = init_writer_for_draw(out_draw_path, width, height, fps)

    # MediaPipe Handsモデルを初期化
    with mp_hands.Hands(
        static_image_mode=static_image_mode,      # Trueだと毎フレーム検出、Falseだと追跡
        max_num_hands=max_num_hands,              # 検出する最大の手数
        min_detection_confidence=min_detection_confidence, # 手の検出の信頼度の閾値
        min_tracking_confidence=min_tracking_confidence,   # 手の追跡の信頼度の閾値
        model_complexity=1,                       # モデルの複雑度 (0 or 1)
    ) as hands:

        # 進捗バーの表示設定
        pbar = tqdm(total=total_frames if total_frames > 0 else None,
                    desc=f"Processing {video_path.name}", unit="f")
        frame_idx = 0

        # 1フレームずつ動画を処理するループ
        while True:
            ok, bgr = cap.read() # フレームを読み込む
            if not ok:
                break # 動画の終端に達したらループを抜ける
            
            # MediaPipeはRGB画像を期待するため、OpenCVのBGRからRGBに変換
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # 実際に手の検出を実行
            result = hands.process(rgb)

            # このフレームのランドマークを格納する配列をNaNで初期化
            frame_landmarks = np.full(LANDMARK_DIM, np.nan, dtype=np.float32)

            # 手が1つ以上検出された場合
            if result.multi_hand_landmarks:
                handedness_map = {}
                # 検出された各手が「左手」か「右手」かを取得
                if result.multi_handedness:
                    for h_data in result.multi_handedness:
                        handedness_map[h_data.classification[0].index] = h_data.classification[0].label

                # --- 左右の手の推測ロジック --- 
                # MediaPipeが左右どちらの手か確信が持てない場合がある。
                # その場合、状況に応じて「おそらくこうだろう」と推測する処理。
                # 例: 2つの手が映っているのに、片方しか「右手」と分からなかった場合、もう片方は「左手」だろうと推測する。
                if len(result.multi_hand_landmarks) == 2 and len(handedness_map) == 1:
                    known_idx = list(handedness_map.keys())[0]
                    known_label = list(handedness_map.values())[0]
                    unknown_idx = 1 - known_idx
                    inferred_label = "Left" if known_label == "Right" else "Right"
                    handedness_map[unknown_idx] = inferred_label
                    had_inference = True # 推測したことを記録
                elif len(result.multi_hand_landmarks) == 1 and len(handedness_map) == 0:
                    handedness_map[0] = "Right" # 1つしか手がなく、左右不明な場合は、とりあえず「右手」と見なす
                    had_inference = True
                elif len(result.multi_hand_landmarks) == 2 and len(handedness_map) == 0:
                    handedness_map[0] = "Right" # 2つ手が映っていて、両方とも左右不明な場合は、0番を「右手」、1番を「左手」と見なす
                    handedness_map[1] = "Left"
                    had_inference = True
                # --- 推測ロジックここまで ---

                # 検出された各手（hand_lms）のランドマーク情報を処理
                for hand_idx, hand_lms in enumerate(result.multi_hand_landmarks):
                    hand_label = handedness_map.get(hand_idx)

                    start_idx = -1
                    if hand_label == "Left":
                        # 左手の場合、配列の0番目からデータを格納
                        start_idx = 0
                    elif hand_label == "Right":
                        # 右手の場合、配列の63番目（21*3）からデータを格納
                        start_idx = 21 * 3
                    else:
                        # 左右どちらでもないと判断された場合（通常は発生しにくい）
                        print(f"[WARN] 処理されない手: {hand_label} (ビデオ: {video_path.name}, フレーム: {frame_idx})")
                        continue

                    if start_idx != -1:
                        # 21個のランドマークの座標(x, y, z)を配列に格納
                        for lm_id, lm in enumerate(hand_lms.landmark):
                            base_idx = start_idx + lm_id * 3
                            frame_landmarks[base_idx] = lm.x
                            frame_landmarks[base_idx + 1] = lm.y
                            frame_landmarks[base_idx + 2] = lm.z

                    # 描画オプションが有効な場合、元のフレーム画像にランドマークを描画
                    if writer is not None:
                        mp_drawing.draw_landmarks(
                            bgr, hand_lms, mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style())

            # このフレームのランドマークデータをリストに追加
            all_frame_landmarks.append(frame_landmarks)

            # 描画オプションが有効な場合、描画済みのフレームを動画ファイルに書き込む
            if writer is not None:
                writer.write(bgr)

            frame_idx += 1
            pbar.update(1) # 進捗バーを更新
        pbar.close()

    # 動画のリソースを解放
    cap.release()
    if writer is not None:
        writer.release()
        if out_draw_path is not None:
            print(f"[OK] 描画済み動画: {out_draw_path}")

    if not all_frame_landmarks:
        return np.array([]), had_inference, total_frames

    # 全フレームのランドマークデータをNumPy配列に変換して返す
    return np.array(all_frame_landmarks, dtype=np.float32), had_inference, total_frames

def main():
    """スクリプト実行のメイン関数。コマンドライン引数を解釈し、全動画の処理を統括する。"""
    # コマンドライン引数の設定
    ap = argparse.ArgumentParser(description="MediaPipe Handsで動画から手指ランドマークを抽出し、NPZとメタデータCSVを出力")
    ap.add_argument("-i", "--input_root_dir", required=True, type=Path,
                    help="クラスディレクトリ（1-20）を含む入力動画のルートディレクトリ")
    ap.add_argument("-o", "--output_base_dir", required=True, type=Path,
                    help="NPZファイルとメタデータCSVの出力先ベースディレクトリ")
    ap.add_argument("--draw", action="store_true", help="ランドマーク描画済みMP4も保存する")
    ap.add_argument("--drawdir", type=Path, default=None, help="描画動画の出力先（未指定なら output_base_dir/drawn）")
    ap.add_argument("--max-hands", type=int, default=2, help="検出する最大手数")
    ap.add_argument("--min-det", type=float, default=0.5, help="min_detection_confidence")
    ap.add_argument("--min-trk", type=float, default=0.5, help="min_tracking_confidence")
    ap.add_argument("--static", action="store_true", help="静止画モード（追跡なし、毎フレーム検出）")
    args = ap.parse_args()

    # 出力先ディレクトリの準備
    output_npz_dir = args.output_base_dir / "processed_data"
    output_npz_dir.mkdir(parents=True, exist_ok=True)

    draw_dir = args.drawdir if args.drawdir else (args.output_base_dir / "drawn")

    # 後でCSVファイルとして保存するためのメタデータを格納するリスト
    metadata_rows = []

    # クラスディレクトリ（1から20まで）をループ処理
    for class_id in range(1, 21):
        class_dir = args.input_root_dir / str(class_id)
        if not class_dir.is_dir():
            print(f"[WARN] クラスディレクトリが見つかりません: {class_dir}", file=sys.stderr)
            continue

        # クラスディレクトリ内の動画ファイルを検索
        videos_in_class = find_videos(class_dir, "*.mp4") # .mp4ファイルのみを対象と仮定
        if not videos_in_class:
            print(f"[WARN] クラス {class_id} の動画が見つかりません: {class_dir}", file=sys.stderr)
            continue

        # このクラスID用の出力サブディレクトリを作成
        current_class_npz_output_dir = output_npz_dir / str(class_id)
        current_class_npz_output_dir.mkdir(parents=True, exist_ok=True)

        # 見つかった動画を一つずつ処理
        for video_path in videos_in_class:
            print(f"動画を処理中: {video_path}")
            # メインのランドマーク抽出関数を呼び出す
            landmark_data, had_inference, num_frames = extract_landmarks_from_video(
                video_path,
                draw=args.draw,
                draw_dir=draw_dir,
                static_image_mode=args.static,
                max_num_hands=args.max_hands,
                min_detection_confidence=args.min_det,
                min_tracking_confidence=args.min_trk,
            )

            # ランドマークが正常に抽出された場合
            if landmark_data.size > 0:
                # NPZファイルとして保存
                npz_filename = f"{video_path.stem}.npz"
                npz_output_path = current_class_npz_output_dir / npz_filename
                # savez_compressedで圧縮して保存
                np.savez_compressed(npz_output_path, landmarks=landmark_data)
                print(f"[OK] NPZファイルが保存されました: {npz_output_path}")

                # メタデータ（動画の情報）をリストに追加
                quality_flag = "inferred" if had_inference else "clean"
                metadata_rows.append({
                    "npz_path": str(npz_output_path.relative_to(args.output_base_dir)), # ベースディレクトリからの相対パス
                    "class_label": class_id,
                    "original_video_path": str(video_path),
                    "quality_flag": quality_flag, # 左右の手を推測したかどうかのフラグ
                    "num_frames": num_frames, # 総フレーム数
                })
            else:
                print(f"[WARN] ランドマークデータが抽出されませんでした: {video_path}")

    # 全ての動画処理が終わったら、メタデータをCSVファイルとして保存
    if metadata_rows:
        metadata_df = pd.DataFrame(metadata_rows)
        metadata_csv_path = args.output_base_dir / "metadata.csv"
        metadata_df.to_csv(metadata_csv_path, index=False, encoding="utf-8")
        print(f"[OK] メタデータCSVが保存されました: {metadata_csv_path}")
    else:
        print("[WARN] 処理された動画がないため、メタデータCSVは作成されませんでした。")

# このスクリプトが直接実行された場合にmain()関数を呼び出す
if __name__ == "__main__":
    main()