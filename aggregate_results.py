

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def calculate_metrics_from_cm(cm: np.ndarray):
    """
    混同行列(Confusion Matrix)から各クラスの指標を計算する。
    numpyのベクトル計算を用いて、ループをなくし、可読性と正確性を向上させる。
    """
    num_classes = cm.shape[0]
    
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp

    # 分母が0の場合は0とする
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    
    f1_denom = precision + recall
    f1 = np.divide(2 * precision * recall, f1_denom, out=np.zeros_like(f1_denom, dtype=float), where=f1_denom != 0)

    # Per-class Accuracy (Jaccard) = TP / (TP + FP + FN)
    jaccard_denom = tp + fp + fn
    accuracy = np.divide(tp, jaccard_denom, out=np.zeros_like(tp, dtype=float), where=jaccard_denom != 0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def aggregate_results(results_dir: Path, num_classes: int, mode: str, report_output_path: Path, stats_output_path: Path | None):
    """
    保存された混同行列ファイルから、最終的な統計量とレポートを生成する。
    --modeに応じて出力内容を切り替える。
    """
    print(f"--- '{results_dir}' から混同行列ファイルを読み込んでいます ---")
    
    cm_files = sorted(results_dir.glob("*_cm.csv"))
    if not cm_files:
        print(f"エラー: '{results_dir}' 内に '*_cm.csv' ファイルが見つかりません。")
        return

    all_cms = [pd.read_csv(file, header=None).to_numpy() for file in cm_files]
    print(f"{len(all_cms)} 個のフォールドの混同行列をロードしました。")

    class_names = [f"class_{i}" for i in range(num_classes)]

    # --- 1. 全体を統合した最終レポートを計算 (両方のモードで必須) ---
    print("\n--- 全体を統合した最終レポートを計算しています ---")
    
    total_cm = np.sum(all_cms, axis=0)
    
    if total_cm.shape[0] != num_classes:
        print(f"エラー: 混同行列のサイズ({total_cm.shape[0]})が指定されたクラス数({num_classes})と一致しません。")
        return

    report_metrics = calculate_metrics_from_cm(total_cm)
    support = np.sum(total_cm, axis=1)
    
    report_data = []
    for i in range(num_classes):
        report_data.append({
            "precision": report_metrics["precision"][i],
            "recall": report_metrics["recall"][i],
            "f1-score": report_metrics["f1"][i],
            "accuracy": report_metrics["accuracy"][i],
            "support": support[i]
        })
    
    report_df = pd.DataFrame(report_data, index=class_names)
    
    overall_accuracy = np.sum(np.diag(total_cm)) / np.sum(total_cm) if np.sum(total_cm) > 0 else 0.0
    
    macro_avg_metrics = report_df.loc[class_names].mean()
    macro_avg_metrics['support'] = np.sum(support)
    report_df.loc['macro avg'] = macro_avg_metrics
    
    weighted_avg_metrics = np.average(report_df.loc[class_names, ['precision', 'recall', 'f1-score', 'accuracy']], axis=0, weights=support)
    report_df.loc['weighted avg', ['precision', 'recall', 'f1-score', 'accuracy']] = weighted_avg_metrics
    report_df.loc['weighted avg', 'support'] = np.sum(support)

    print(f"\nOverall Accuracy (Micro): {overall_accuracy:.4f}")

    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_output_path, float_format='%.4f')
    print(f"全体の詳細レポートを保存しました: {report_output_path}")
    print(report_df)

    # --- 2. グラフ描画用の統計量 (csモードとcvモードで処理を分岐) ---
    # cvモードでも統計ファイルを生成する
    if mode == 'cs' or mode == 'cv':
        print(f"\n--- [{mode}モード] グラフ描画用の統計量（平均・標準偏差）を計算しています ---")
        
        per_fold_metrics = defaultdict(list)
        for cm in all_cms:
            if cm.shape[0] != num_classes or cm.shape[1] != num_classes:
                print(f"警告: 予期しない形状の混同行列をスキップします: {cm.shape}")
                continue
            metrics = calculate_metrics_from_cm(cm)
            for name, values in metrics.items():
                per_fold_metrics[name].append(values)

        stats_rows = []
        metric_names = ["accuracy", "precision", "recall", "f1"]
        num_folds = len(all_cms)

        # クラスごとの統計量を計算
        for name in metric_names:
            if name not in per_fold_metrics: continue
            
            stacked_values = np.array(per_fold_metrics[name])
            means = np.mean(stacked_values, axis=0)
            stds = np.std(stacked_values, axis=0)
            for i in range(num_classes):
                row = {
                    "metric": f"test_{name}",
                    "class_label": f"class_{i}",
                    "mean": means[i],
                    "std": stds[i]
                }
                for fold_idx in range(num_folds):
                    row[f"fold_{fold_idx}"] = stacked_values[fold_idx, i]
                stats_rows.append(row)

        # 全体（マクロ平均）の統計量を計算
        for name in metric_names:
            if name not in per_fold_metrics: continue

            stacked_values = np.array(per_fold_metrics[name])
            overall_scores_per_fold = np.mean(stacked_values, axis=1)
            
            mean_of_overalls = np.mean(overall_scores_per_fold)
            std_of_overalls = np.std(overall_scores_per_fold)

            row = {
                "metric": f"test_{name}",
                "class_label": "overall",
                "mean": mean_of_overalls,
                "std": std_of_overalls
            }
            for fold_idx, score in enumerate(overall_scores_per_fold):
                row[f"fold_{fold_idx}"] = score
            stats_rows.append(row)

        stats_df = pd.DataFrame(stats_rows)
        stats_output_path.parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(stats_output_path, index=False, float_format='%.4f')
        print(f"グラフ描画用の統計量を保存しました: {stats_output_path}")
        print(stats_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="混同行列ファイルを集計し、最終的なレポートと統計量を生成します。")
    parser.add_argument("results_dir", type=str, help="混同行列CSVが保存されているディレクトリ")
    parser.add_argument("--mode", type=str, required=True, choices=['cv', 'cs'], help="実行モード ('cv' for K-Fold/LOO, 'cs' for cross-subject)")
    parser.add_argument("--num-classes", type=int, required=True, help="クラス数")
    parser.add_argument("--report-out", type=str, required=True, help="詳細レポートCSVの出力パス")
    parser.add_argument("--stats-out", type=str, required=True, help="統計量CSVの出力パス")
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    if not results_path.is_dir():
        print(f"エラー: 指定されたディレクトリが見つかりません: {results_path}")
    else:
        aggregate_results(results_path, args.num_classes, args.mode, Path(args.report_out), Path(args.stats_out))