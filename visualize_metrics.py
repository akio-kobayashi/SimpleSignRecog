import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

def plot_metrics(log_dir: Path, metrics: list[str]):
    """
    指定されたログディレクトリから各foldのmetrics.csvを読み込み、
    指定されたメトリクスの学習曲線とfold間の平均をプロットします。
    """
    all_metrics_dfs = []
    fold_dirs = sorted(log_dir.glob('fold_*'))
    if not fold_dirs:
        print(f"エラー: '{log_dir}' 内に 'fold_*' ディレクトリが見つかりません。")
        return

    print(f"{len(fold_dirs)}個のfoldディレクトリが見つかりました。")

    sns.set_theme(style="whitegrid")
    
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        fold_data_for_metric = []

        for fold_dir in fold_dirs:
            metrics_file = fold_dir / 'metrics.csv'
            if not metrics_file.exists():
                print(f"警告: {metrics_file} が見つかりません。スキップします。")
                continue

            df = pd.read_csv(metrics_file)
            
            if metric in df.columns:
                metric_df = df[['epoch', metric]].dropna()
                sns.lineplot(data=metric_df, x='epoch', y=metric, alpha=0.3, label=f"Fold {fold_dir.name.split('_')[-1]}")
                fold_data_for_metric.append(metric_df.set_index('epoch'))

        if not fold_data_for_metric:
            print(f"エラー: メトリクス '{metric}' のデータがどのfoldにも見つかりませんでした。")
            plt.close()
            continue

        combined_df = pd.concat(fold_data_for_metric, axis=1)
        mean_series = combined_df.mean(axis=1)
        std_series = combined_df.std(axis=1)

        plt.plot(mean_series.index, mean_series, lw=2.5, color='black', label='Average')
        plt.fill_between(mean_series.index, mean_series - std_series, mean_series + std_series, color='gray', alpha=0.2, label='Std. Dev.')

        plt.title(f'Training Curve for {metric}', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
        plt.legend(title='Folds')
        plt.grid(True)
        
        output_path = log_dir / f'{metric}_training_curve.png'
        plt.savefig(output_path)
        print(f"グラフを保存しました: {output_path}")
        plt.close()

def plot_final_stats(stats_csv_path: Path, metric: str, output_path: Path):
    """
    統計情報CSVファイルを読み込み、指定されたメトリクスのエラーバー付き棒グラフを作成します。
    """
    if not stats_csv_path.exists():
        print(f"エラー: 統計ファイルが見つかりません: {stats_csv_path}")
        return

    df = pd.read_csv(stats_csv_path)
    
    # 指定されたメトリクスでデータをフィルタリング
    metric_df = df[df['metric'] == metric]
    if metric_df.empty:
        print(f"エラー: メトリクス '{metric}' がファイル内に見つかりません。")
        available_metrics = df['metric'].unique()
        print(f"利用可能なメトリクス: {', '.join(available_metrics)}")
        return

    # 全体 (overall) とクラスごと (per-class) のデータに分割
    overall_stats = metric_df[metric_df['class_label'] == 'overall']
    per_class_stats = metric_df[metric_df['class_label'] != 'overall'].copy()
    
    # クラスラベルから数値部分を抽出してソート
    per_class_stats['class_num'] = per_class_stats['class_label'].str.extract('(\d+)').astype(int)
    per_class_stats = per_class_stats.sort_values('class_num')

    if per_class_stats.empty:
        print(f"警告: メトリクス '{metric}' のクラスごとデータが見つかりません。")
        return

    # --- グラフの描画 ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(16, 9))

    # 棒グラフを作成
    bars = plt.bar(
        per_class_stats['class_label'],
        per_class_stats['mean'],
        yerr=per_class_stats['std'],
        capsize=5, # エラーバーの傘のサイズ
        color=sns.color_palette("viridis", len(per_class_stats)),
        alpha=0.8
    )

    # 全体平均を点線で表示
    if not overall_stats.empty:
        overall_mean = overall_stats['mean'].iloc[0]
        plt.axhline(y=overall_mean, color='r', linestyle='--', label=f'Overall Mean: {overall_mean:.3f}')
        plt.legend()

    plt.title(f'Per-Class {metric.replace("_", " ").title()} with Error Bars', fontsize=18)
    plt.xlabel('Class Label', fontsize=14)
    plt.ylabel(metric.replace("_", " ").title(), fontsize=14)
    plt.xticks(rotation=45, ha='right') # X軸ラベルを回転
    plt.ylim(0, 1.05) # Y軸の範囲を0から1.05に設定
    plt.tight_layout() # レイアウトを調整

    # グラフをファイルに保存
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"グラフを保存しました: {output_path}")
    plt.close()


if __name__ == "__main__":
    # --- メインのパーサー ---
    parser = argparse.ArgumentParser(description="学習結果を可視化するスクリプト。")
    subparsers = parser.add_subparsers(dest='command', required=True, help='実行するコマンド')

    # --- 'plot_curves' サブコマンド ---
    parser_curves = subparsers.add_parser('plot_curves', help='学習曲線（エポックごとの推移）をプロットします。')
    parser_curves.add_argument("log_dir", type=str, help="実験のログが保存されている親ディレクトリ (例: logs/sign_recognition_experiment)")
    parser_curves.add_argument(
        "--metrics",
        nargs='+',
        default=['val_loss', 'val_acc_ctc'],
        help="プロットするメトリクスのリスト (例: val_loss val_acc_ctc)"
    )

    # --- 'plot_stats' サブコマンド ---
    parser_stats = subparsers.add_parser('plot_stats', help='最終結果の統計量（クラスごと精度など）を棒グラフでプロットします。')
    parser_stats.add_argument("stats_csv", type=str, help="統計情報が保存されているCSVファイル (例: results_cv_stats.csv)")
    parser_stats.add_argument("--metric", type=str, required=True, help="プロットするメトリクス名 (例: test_acc_ctc)")
    parser_stats.add_argument("--output", type=str, default="final_stats_plot.png", help="出力するグラフのファイル名")

    args = parser.parse_args()

    # --- コマンドに応じて実行 ---
    if args.command == 'plot_curves':
        log_path = Path(args.log_dir)
        if not log_path.is_dir():
            print(f"エラー: 指定されたディレクトリが見つかりません: {log_path}")
        else:
            plot_metrics(log_path, args.metrics)
    
    elif args.command == 'plot_stats':
        stats_path = Path(args.stats_csv)
        output_path = Path(args.output)
        plot_final_stats(stats_path, args.metric, output_path)