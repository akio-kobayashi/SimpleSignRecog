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

        # 確率を重みとして、時刻の期待値（加重平均）を計算
        time_indices = torch.arange(length, dtype=torch.float32, device=class_probs.device)
        
        # 確率の合計で正規化（合計が1になるように）
        prob_sum = torch.sum(class_probs)
        if prob_sum > 1e-6: # 確率がほぼゼロの場合は計算をスキップ
            weighted_probs = class_probs / prob_sum
            expected_time = torch.sum(time_indices * weighted_probs)
            
            # 正規化された時間 (0.0 ~ 1.0) を計算
            normalized_expected_time = expected_time.item() / (length - 1) if length > 1 else 0.5
            
            peak_times_by_class[label].append(normalized_expected_time)

    # クラスごとに統計量を計算し、エラーバー付きの点プロットを作成
    output_dir.mkdir(exist_ok=True, parents=True)
    
    class_names = []
    mean_peak_times = []
    std_peak_times = []

    # 計算対象のクラスをソートして、グラフの順序を固定
    sorted_class_ids = sorted(peak_times_by_class.keys())

    for class_id in sorted_class_ids:
        peak_times = peak_times_by_class[class_id]
        if not peak_times:
            continue
        
        class_names.append(class_mapping.get(class_id, f"Class_{class_id}") if class_mapping else f"Class_{class_id}")
        mean_peak_times.append(np.mean(peak_times))
        std_peak_times.append(np.std(peak_times))

    if not class_names:
        print("ピーク時刻のデータがないため、グラフは生成されませんでした。")
        return

    # プロットの作成
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 横倒しのエラーバープロット
    ax.errorbar(
        x=mean_peak_times,
        y=np.arange(len(class_names)),
        xerr=std_peak_times,
        fmt='o', # 点のマーカー
        capsize=5, # エラーバーのキャップサイズ
        linestyle='None', # 点の間を線で結ばない
        markerfacecolor='royalblue',
        markeredgecolor='navy',
        ecolor='gray'
    )
    
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.invert_yaxis()  # 上からクラス0, 1, 2... となるように
    
    ax.set_title('Mean Peak Time of Class Probability by Class', fontsize=16)
    ax.set_xlabel('Normalized Time (Start -> End)', fontsize=12)
    ax.set_ylabel('Class', fontsize=12)
    ax.set_xlim(0, 1)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    save_path = output_dir / "spike_peak_summary.png"
    plt.savefig(save_path)
    plt.close(fig)

    print(f"スパイクピークの統計グラフを {save_path} に保存しました。")
