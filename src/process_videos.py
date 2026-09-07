#!/usr/bin/env python3
"""クラス別の動画をランドマークNPZとmetadata.csvへ変換するCLI。

MediaPipeを使う1動画単位の処理は ``preprocessing.landmark_extraction`` に
分離しています。このファイルでは入力の列挙と保存だけを行います。

実行例:
    python src/process_videos.py -i /path/to/subject_videos -o ./data/subject -m ./hand_landmarker.task
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .preprocessing.landmark_extraction import extract_landmarks_from_video
else:
    from preprocessing.landmark_extraction import extract_landmarks_from_video


def find_mp4_videos(class_dir: Path) -> list[Path]:
    """クラスディレクトリ直下のMP4を名前順で返す。"""
    return sorted(class_dir.glob("*.mp4"))


def create_dataset(
    input_root_dir: Path,
    output_base_dir: Path,
    model_asset_path: Path,
    *,
    draw: bool = False,
    draw_dir: Path | None = None,
    static_image_mode: bool = False,
    max_num_hands: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> pd.DataFrame:
    """クラス1～20の全MP4を処理し、NPZとmetadata.csvを作成する。"""
    input_root_dir = Path(input_root_dir)
    output_base_dir = Path(output_base_dir)
    output_npz_dir = output_base_dir / "processed_data"
    output_npz_dir.mkdir(parents=True, exist_ok=True)
    draw_dir = Path(draw_dir) if draw_dir else output_base_dir / "drawn"

    metadata_rows = []
    for class_id in range(1, 21):
        class_dir = input_root_dir / str(class_id)
        if not class_dir.is_dir():
            print(
                f"[WARN] クラスディレクトリが見つかりません: {class_dir}",
                file=sys.stderr,
            )
            continue

        video_paths = find_mp4_videos(class_dir)
        if not video_paths:
            print(
                f"[WARN] クラス {class_id} の動画が見つかりません: {class_dir}",
                file=sys.stderr,
            )
            continue

        class_output_dir = output_npz_dir / str(class_id)
        class_output_dir.mkdir(parents=True, exist_ok=True)

        for video_path in video_paths:
            print(f"動画を処理中: {video_path}")
            landmarks, had_inference, num_frames = extract_landmarks_from_video(
                video_path,
                model_asset_path=model_asset_path,
                draw=draw,
                draw_dir=draw_dir,
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            if landmarks.size == 0:
                print(f"[WARN] ランドマークを抽出できませんでした: {video_path}")
                continue

            npz_path = class_output_dir / f"{video_path.stem}.npz"
            np.savez_compressed(npz_path, landmarks=landmarks)
            print(f"[OK] NPZファイルを保存しました: {npz_path}")
            metadata_rows.append(
                {
                    "npz_path": str(npz_path.relative_to(output_base_dir)),
                    "class_label": class_id,
                    "original_video_path": str(video_path),
                    "quality_flag": "inferred" if had_inference else "clean",
                    "num_frames": num_frames,
                }
            )

    metadata = pd.DataFrame(
        metadata_rows,
        columns=[
            "npz_path",
            "class_label",
            "original_video_path",
            "quality_flag",
            "num_frames",
        ],
    )
    if metadata.empty:
        print("[WARN] 処理された動画がないため、metadata.csvは作成しません。")
    else:
        metadata_path = output_base_dir / "metadata.csv"
        metadata.to_csv(metadata_path, index=False, encoding="utf-8")
        print(f"[OK] メタデータCSVを保存しました: {metadata_path}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MediaPipe Handsで動画をNPZランドマークへ変換する"
    )
    parser.add_argument(
        "-i", "--input_root_dir", required=True, type=Path,
        help="クラスディレクトリ（1～20）を含む動画ディレクトリ",
    )
    parser.add_argument(
        "-o", "--output_base_dir", required=True, type=Path,
        help="processed_dataとmetadata.csvの出力先",
    )
    parser.add_argument(
        "-m", "--model_asset_path", required=True, type=Path,
        help="MediaPipe Hand Landmarkerの.taskモデル",
    )
    parser.add_argument("--draw", action="store_true", help="検出結果の動画も保存する")
    parser.add_argument("--drawdir", type=Path, default=None, help="描画動画の出力先")
    parser.add_argument("--max-hands", type=int, default=2, help="検出する最大手数")
    parser.add_argument("--min-det", type=float, default=0.5, help="手検出の信頼度閾値")
    parser.add_argument("--min-trk", type=float, default=0.5, help="手追跡の信頼度閾値")
    parser.add_argument("--static", action="store_true", help="毎フレーム検出する")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_dataset(
        args.input_root_dir,
        args.output_base_dir,
        args.model_asset_path,
        draw=args.draw,
        draw_dir=args.drawdir,
        static_image_mode=args.static,
        max_num_hands=args.max_hands,
        min_detection_confidence=args.min_det,
        min_tracking_confidence=args.min_trk,
    )


if __name__ == "__main__":
    main()
