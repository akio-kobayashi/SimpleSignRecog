import matplotlib
matplotlib.use('Agg') # GUIバックエンドなしで動作させるための設定
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional

def plot_spike_distribution(
    all_outputs: List[torch.Tensor],
    all_labels: List[torch.Tensor],
    all_lengths: List[torch.Tensor],
    class_mapping: Optional[Dict[int, str]],
    output_dir: Path,
):
    """
    モデルの時系列出力から、各クラスのスパイク（ピーク）が正規化された時間軸の
    どの位置で発生するかを分析し、クラスごとにヒストグラムとして可視化して保存する。

    Args:
        all_outputs (List[torch.Tensor]): モデル出力のリスト (各要素は (B, T, C) の確率テンソル)
        all_labels (List[torch.Tensor]): 正解ラベルのリスト (各要素は (B,) のテンソル)
        all_lengths (List[torch.Tensor]): 系列長のリスト (各要素は (B,) のテンソル)
        class_mapping (Dict[int, str]): クラスIDとクラス名のマッピング
        output_dir (Path): プロットを保存するディレクトリ
    """
    # データを単一のテンソルに結合
    outputs = torch.cat(all_outputs, dim=0) # (Total_Samples, T_max, C)
    labels = torch.cat(all_labels, dim=0)   # (Total_Samples,)
    lengths = torch.cat(all_lengths, dim=0) # (Total_Samples,)

    # CTC用にクラス数が+1されているので、blankを除いた実際のクラス数を取得
    num_classes = outputs.shape[-1] - 1
    peak_times_by_class = {i: [] for i in range(num_classes)}

    # 各サンプルについてピーク位置を計算
    for i in range(len(labels)):
        label = labels[i].item()
        length = lengths[i].item()
        
        # このサンプルの、正解クラスに対応する確率時系列を取得
        # 出力は (T_max, C) なので、実際の長さまでスライス
        class_probs = outputs[i, :length, label]
        
        if len(class_probs) == 0:
            continue

        # ピークのインデックスを見つける
        peak_index = torch.argmax(class_probs).item()
        
        # 正規化された時間 (0.0 ~ 1.0) を計算
        normalized_peak_time = peak_index / (length - 1) if length > 1 else 0.5
        
        peak_times_by_class[label].append(normalized_peak_time)

    # クラスごとにヒストグラムをプロット
    output_dir.mkdir(exist_ok=True, parents=True)
    for class_id, peak_times in peak_times_by_class.items():
        if not peak_times:
            continue

        class_name = class_mapping.get(class_id, f"Class_{class_id}") if class_mapping else f"Class_{class_id}"
        
        plt.figure(figsize=(10, 6))
        plt.hist(peak_times, bins=20, range=(0, 1), alpha=0.7, edgecolor='black')
        plt.title(f'Spike Peak Distribution for "{class_name}" (n={len(peak_times)})')
        plt.xlabel("Normalized Time (Start -> End)")
        plt.ylabel("Frequency")
        plt.xlim(0, 1)
        
        # ファイル名に使えない文字を置換
        safe_class_name = "".join(c if c.isalnum() else "_" for c in class_name)
        save_path = output_dir / f"spike_dist_{safe_class_name}.png"
        
        plt.savefig(save_path)
        plt.close()

    print(f"スパイク分布のプロットを {output_dir} に保存しました。")
