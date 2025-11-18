import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def calculate_metrics_from_cm(cm: np.ndarray):
    """
    混同行列(Confusion Matrix)から各クラスの指標を計算する。
    
    Args:
        cm (np.ndarray): (num_classes, num_classes) の形状の混同行列。
                         cm[i, j] は、真のクラスがiで、予測されたクラスがjであるサンプル数。

    Returns:
        dict: precision, recall, f1 のリストを含む辞書。
    """
    num_classes = cm.shape[0]
    metrics = {
        "precision": np.zeros(num_classes),
        "recall": np.zeros(num_classes),
        "f1": np.zeros(num_classes),
        "accuracy": np.zeros(num_classes) # for compatibility
    }

    # TP, FP, FNを計算
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp

    # 各指標を計算
    for i in range(num_classes):
        # Precision = TP / (TP + FP)
        precision_denom = tp[i] + fp[i]
        metrics["precision"][i] = tp[i] / precision_denom if precision_denom > 0 else 0.0
        
        # Recall = TP / (TP + FN)
        metrics["recall"][i] = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        
        # F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
        f1_denom = metrics["precision"][i] + metrics["recall"][i]
        metrics["f1"][i] = 2 * (metrics["precision"][i] * metrics["recall"][i]) / f1_denom if f1_denom > 0 else 0.0

    return metrics

def aggregate_results(results_dir: Path, config: dict):
    """
    保存された混同行列ファイルから、最終的な統計量とレポートを生成する。
    """
    print(f"--- '{results_dir}' から混同行列ファイルを読み込んでいます ---")
    
    cm_files = sorted(results_dir.glob("*_cm.csv"))
    if not cm_files:
        print(f"エラー: '{results_dir}' 内に '*_cm.csv' ファイルが見つかりません。")
        return

    all_cms = [pd.read_csv(file, header=None).to_numpy() for file in cm_files]
    print(f"{len(all_cms)} 個のフォールドの混同行列をロードしました。")

    num_classes = all_cms[0].shape[0]
    output_conf = config.get('output', {})
    class_names = [f"class_{i}" for i in range(num_classes)]

    # --- 1. グラフ描画用の統計量 (平均・標準偏差) を計算 ---
    print("\n--- グラフ描画用の統計量を計算しています ---")
    
    per_fold_metrics = defaultdict(list)
    for cm in all_cms:
        metrics = calculate_metrics_from_cm(cm)
        for name, values in metrics.items():
            per_fold_metrics[name].append(values)

    stats_rows = []
    for name, values_per_fold in per_fold_metrics.items():
        stacked_values = np.array(values_per_fold) # (num_folds, num_classes)
        means = np.mean(stacked_values, axis=0)
        stds = np.std(stacked_values, axis=0)
        for i in range(num_classes):
            stats_rows.append({
                "metric": f"test_{name}",
                "class_label": f"class_{i}",
                "mean": means[i],
                "std": stds[i]
            })

    stats_df = pd.DataFrame(stats_rows)
    stats_output_path = output_conf.get("cv_stats_path", "results_cv_stats.csv")
    stats_df.to_csv(stats_output_path, index=False, float_format='%.4f')
    print(f"グラフ描画用の統計量を保存しました: {stats_output_path}")
    print(stats_df.head())

    # --- 2. 全体を統合した最終レポートを計算 ---
    print("\n--- 全体を統合した最終レポートを計算しています ---")
    
    total_cm = np.sum(all_cms, axis=0)
    
    report_metrics = calculate_metrics_from_cm(total_cm)
    support = np.sum(total_cm, axis=1)
    
    report_data = []
    for i in range(num_classes):
        report_data.append({
            "precision": report_metrics["precision"][i],
            "recall": report_metrics["recall"][i],
            "f1-score": report_metrics["f1"][i],
            "support": support[i]
        })
    
    report_df = pd.DataFrame(report_data, index=class_names)
    
    # 全体指標の計算
    accuracy = np.sum(np.diag(total_cm)) / np.sum(total_cm)
    report_df.loc['accuracy'] = [np.nan, np.nan, accuracy, np.sum(support)]
    
    # マクロ平均
    macro_avg = report_df.loc[class_names].mean()
    macro_avg['support'] = np.sum(support)
    report_df.loc['macro avg'] = macro_avg
    
    # 加重平均
    weighted_avg = np.average(report_df.loc[class_names, ['precision', 'recall', 'f1-score']], axis=0, weights=support)
    report_df.loc['weighted avg', ['precision', 'recall', 'f1-score']] = weighted_avg
    report_df.loc['weighted avg', 'support'] = np.sum(support)

    report_output_path = output_conf.get("classification_report_path", "classification_report.csv")
    report_df.to_csv(report_output_path, float_format='%.4f')
    print(f"全体の詳細レポートを保存しました: {report_output_path}")
    print(report_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="混同行列ファイルを集計し、最終的なレポートと統計量を生成します。")
    parser.add_argument("results_dir", type=str, help="混同行列CSVが保存されているディレクトリ")
    parser.add_argument("--config", type=str, default="config.yaml", help="YAML形式の設定ファイル")
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    if not results_path.is_dir():
        print(f"エラー: 指定されたディレクトリが見つかりません: {results_path}")
    else:
        with open(args.config, "r") as yf:
            config = yaml.safe_load(yf)
        aggregate_results(results_path, config)