

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def calculate_metrics_from_cm(cm: np.ndarray):
    """
    混同行列(Confusion Matrix)から各クラスの指標を計算する。
    """
    num_classes = cm.shape[0]
    metrics = {
        "precision": np.zeros(num_classes),
        "recall": np.zeros(num_classes),
        "f1": np.zeros(num_classes),
    }

    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp

    for i in range(num_classes):
        precision_denom = tp[i] + fp[i]
        recall_denom = tp[i] + fn[i]
        
        metrics["precision"][i] = tp[i] / precision_denom if precision_denom > 0 else 0.0
        metrics["recall"][i] = tp[i] / recall_denom if recall_denom > 0 else 0.0
        
        f1_denom = metrics["precision"][i] + metrics["recall"][i]
        metrics["f1"][i] = 2 * (metrics["precision"][i] * metrics["recall"][i]) / f1_denom if f1_denom > 0 else 0.0

    return metrics

def aggregate_results(results_dir: Path, config: dict, stats_output_path: Path, report_output_path: Path):
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

    num_classes = config['model']['num_classes']
    class_names = [f"class_{i}" for i in range(num_classes)]

    # --- 1. グラフ描画用の統計量 (平均・標準偏差) を計算 ---
    print("\n--- グラフ描画用の統計量を計算しています ---")
    
    per_fold_metrics = defaultdict(list)
    for cm in all_cms:
        # 各フォールドの混同行列が正しいサイズか確認
        if cm.shape[0] != num_classes or cm.shape[1] != num_classes:
            print(f"警告: 予期しない形状の混同行列をスキップします: {cm.shape}")
            continue
        metrics = calculate_metrics_from_cm(cm)
        for name, values in metrics.items():
            per_fold_metrics[name].append(values)

    stats_rows = []
    metric_names = ["acc", "precision", "recall", "f1"] # 'acc'は後で計算
    for name in metric_names:
        if name not in per_fold_metrics: continue
        
        stacked_values = np.array(per_fold_metrics[name])
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
    # ディレクトリを作成
    stats_output_path.parent.mkdir(parents=True, exist_ok=True)
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
    
    accuracy = np.sum(np.diag(total_cm)) / np.sum(total_cm) if np.sum(total_cm) > 0 else 0.0
    report_df.loc['accuracy'] = [np.nan, np.nan, accuracy, np.sum(support)]
    
    macro_avg = report_df.loc[class_names].mean()
    macro_avg['support'] = np.sum(support)
    report_df.loc['macro avg'] = macro_avg
    
    weighted_avg = np.average(report_df.loc[class_names, ['precision', 'recall', 'f1-score']], axis=0, weights=support)
    report_df.loc['weighted avg', ['precision', 'recall', 'f1-score']] = weighted_avg
    report_df.loc['weighted avg', 'support'] = np.sum(support)

    # ディレクトリを作成
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_output_path, float_format='%.4f')
    print(f"全体の詳細レポートを保存しました: {report_output_path}")
    print(report_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="混同行列ファイルを集計し、最終的なレポートと統計量を生成します。")
    parser.add_argument("results_dir", type=str, help="混同行列CSVが保存されているディレクトリ")
    parser.add_argument("--config", type=str, default=None, help="YAML形式の設定ファイル（num_classesの取得に利用）")
    parser.add_argument("--stats-out", type=str, default="results_stats.csv", help="統計量CSVの出力パス")
    parser.add_argument("--report-out", type=str, default="classification_report.csv", help="詳細レポートCSVの出力パス")
    parser.add_argument("--num-classes", type=int, default=None, help="クラス数 (configファイルがない場合の必須項目)")
    args = parser.parse_args()

    config = {}
    num_classes = None

    if args.config:
        with open(args.config, "r") as yf:
            config = yaml.safe_load(yf)
        num_classes = config.get('model', {}).get('num_classes')
        if num_classes:
            print(f"num_classesをconfig.yamlから使用します: {num_classes}")

    if num_classes is None:
        if args.num_classes:
            num_classes = args.num_classes
            config.setdefault('model', {})['num_classes'] = num_classes
            print(f"num_classesを --num-classes 引数から使用します: {num_classes}")
        else:
            raise ValueError("`--config`で指定されたファイルに`num_classes`がない場合、`--num-classes`引数は必須です。")

    results_path = Path(args.results_dir)
    if not results_path.is_dir():
        print(f"エラー: 指定されたディレクトリが見つかりません: {results_path}")
    else:
        aggregate_results(results_path, config, Path(args.stats_out), Path(args.report_out))
