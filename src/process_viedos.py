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

import mediapipe as mp

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm")
# 21 landmarks * 3 coordinates (x,y,z) * 2 hands (Left, Right) = 126
LANDMARK_DIM = 21 * 3 * 2

def find_videos(input_path: Path, pattern: str):
    """指定されたパス内の動画ファイルを検索する。"""
    if input_path.is_file():
        return [input_path]
    vids = []
    if input_path.is_dir():
        if pattern:
            vids = [Path(p) for p in glob.glob(str(input_path / pattern), recursive=False)]
        else:
            for p in input_path.iterdir():
                if p.suffix.lower() in VIDEO_EXTS:
                    vids.append(p)
    return sorted(vids)

def init_writer_for_draw(out_path: Path, width: int, height: int, fps: float):
    """描画済み動画のためのVideoWriterを初期化する。"""
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
) -> np.ndarray:
    """
    動画からMediaPipe Handsのランドマークを抽出し、NumPy配列として返す。
    手が検出されないフレームや手にはNaNを格納する。
    """
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] 失敗: {video_path} を開けませんでした。", file=sys.stderr)
        return np.array([]) # 動画が開けない場合は空の配列を返す

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    all_frame_landmarks = [] # 全フレームのランドマークデータを格納するリスト
    writer = None
    out_draw_path = None
    if draw:
        draw_dir.mkdir(parents=True, exist_ok=True)
        out_draw_path = draw_dir / f"{video_path.stem}_hands.mp4"
        writer = init_writer_for_draw(out_draw_path, width, height, fps)

    with mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        model_complexity=1,
    ) as hands:

        pbar = tqdm(total=total_frames if total_frames > 0 else None,
                    desc=f"Processing {video_path.name}", unit="f")
        frame_idx = 0

        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # 各フレームのランドマークデータをNaNで初期化 (左手63次元, 右手63次元)
            frame_landmarks = np.full(LANDMARK_DIM, np.nan, dtype=np.float32)

            if result.multi_hand_landmarks:
                # handedness（Left/Right）を検出されたランドマークに対応付け
                handedness_map = {}
                if result.multi_handedness:
                    for h_data in result.multi_handedness:
                        label = h_data.classification[0].label # "Left" or "Right"
                        index = h_data.classification[0].index # multi_hand_landmarksのインデックスに対応
                        handedness_map[index] = label

                for hand_idx, hand_lms in enumerate(result.multi_hand_landmarks):
                    hand_label = handedness_map.get(hand_idx, None)

                    # ランドマークデータを格納する配列内の開始インデックスを決定
                    start_idx = -1
                    if hand_label == "Left":
                        start_idx = 0 # 左手は0-62インデックス
                    elif hand_label == "Right":
                        start_idx = 21 * 3 # 右手は63-125インデックス
                    else:
                        # 予期しないラベルの場合、またはmax_num_handsが2より大きい場合
                        print(f"[WARN] 予期しない手のラベル: {hand_label} (ビデオ: {video_path.name}, フレーム: {frame_idx})")
                        continue # この手はスキップ

                    if start_idx != -1:
                        for lm_id, lm in enumerate(hand_lms.landmark):
                            # x, y, z は正規化座標
                            base_idx = start_idx + lm_id * 3
                            frame_landmarks[base_idx] = lm.x
                            frame_landmarks[base_idx + 1] = lm.y
                            frame_landmarks[base_idx + 2] = lm.z

                    # 描画が有効な場合
                    if writer is not None:
                        mp_drawing.draw_landmarks(
                            bgr,
                            hand_lms,
                            mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style()
                        )

            all_frame_landmarks.append(frame_landmarks)

            if writer is not None:
                writer.write(bgr)

            frame_idx += 1
            pbar.update(1)
        pbar.close()

    cap.release()
    if writer is not None:
        writer.release()
        if out_draw_path is not None:
            print(f"[OK] 描画済み動画: {out_draw_path}")

    if not all_frame_landmarks:
        return np.array([]) # 処理されたフレームがない場合は空の配列を返す

    return np.array(all_frame_landmarks, dtype=np.float32)

def main():
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

    output_npz_dir = args.output_base_dir / "processed_data"
    output_npz_dir.mkdir(parents=True, exist_ok=True)

    draw_dir = args.drawdir if args.drawdir else (args.output_base_dir / "drawn")

    metadata_rows = []

    # クラスディレクトリ（1-20）をイテレート
    for class_id in range(1, 21):
        class_dir = args.input_root_dir / str(class_id)
        if not class_dir.is_dir():
            print(f"[WARN] クラスディレクトリが見つかりません: {class_dir}", file=sys.stderr)
            continue

        # クラスディレクトリ内の動画ファイルを検索
        videos_in_class = find_videos(class_dir, "*.mp4") # すべての動画が.mp4と仮定
        if not videos_in_class:
            print(f"[WARN] クラス {class_id} の動画が見つかりません: {class_dir}", file=sys.stderr)
            continue

        # このクラスID用の出力ディレクトリを作成
        current_class_npz_output_dir = output_npz_dir / str(class_id)
        current_class_npz_output_dir.mkdir(parents=True, exist_ok=True)

        for video_path in videos_in_class:
            print(f"動画を処理中: {video_path}")
            landmark_data = extract_landmarks_from_video(
                video_path,
                draw=args.draw,
                draw_dir=draw_dir,
                static_image_mode=args.static,
                max_num_hands=args.max_hands,
                min_detection_confidence=args.min_det,
                min_tracking_confidence=args.min_trk,
            )

            if landmark_data.size > 0:
                npz_filename = f"{video_path.stem}.npz"
                npz_output_path = current_class_npz_output_dir / npz_filename
                # np.savez_compressed を使用してデータを保存
                np.savez_compressed(npz_output_path, landmarks=landmark_data)
                print(f"[OK] NPZファイルが保存されました: {npz_output_path}")

                # メタデータ行を追加
                metadata_rows.append({
                    "npz_path": str(npz_output_path.relative_to(args.output_base_dir)), # 相対パス
                    "class_label": class_id,
                    "original_video_path": str(video_path),
                })
            else:
                print(f"[WARN] ランドマークデータが抽出されませんでした: {video_path}")

    # メタデータCSVを保存
    if metadata_rows:
        metadata_df = pd.DataFrame(metadata_rows)
        metadata_csv_path = args.output_base_dir / "metadata.csv"
        metadata_df.to_csv(metadata_csv_path, index=False, encoding="utf-8")
        print(f"[OK] メタデータCSVが保存されました: {metadata_csv_path}")
    else:
        print("[WARN] 処理された動画がないため、メタデータCSVは作成されませんでした。")

if __name__ == "__main__":
    main()