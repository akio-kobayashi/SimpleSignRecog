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

    # グラフのスタイルを設定
    sns.set_theme(style="whitegrid")
    
    # 指定された各メトリクスについてプロットを作成
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        fold_data_for_metric = []

        for fold_dir in fold_dirs:
            metrics_file = fold_dir / 'metrics.csv'
            if not metrics_file.exists():
                print(f"警告: {metrics_file} が見つかりません。スキップします。")
                continue

            df = pd.read_csv(metrics_file)
            
            # メトリクスが存在し、NaNでない行を抽出
            if metric in df.columns:
                metric_df = df[['epoch', metric]].dropna()
                # 各foldの学習曲線をプロット
                sns.lineplot(data=metric_df, x='epoch', y=metric, alpha=0.3, label=f"Fold {fold_dir.name.split('_')[-1]}")
                fold_data_for_metric.append(metric_df.set_index('epoch'))

        if not fold_data_for_metric:
            print(f"エラー: メトリクス '{metric}' のデータがどのfoldにも見つかりませんでした。")
            plt.close()
            continue

        # 全てのfoldのデータを結合して平均と標準偏差を計算
        combined_df = pd.concat(fold_data_for_metric, axis=1)
        mean_series = combined_df.mean(axis=1)
        std_series = combined_df.std(axis=1)

        # 平均値を太線でプロット
        plt.plot(mean_series.index, mean_series, lw=2.5, color='black', label='Average')
        # 標準偏差を塗りつぶしで表示
        plt.fill_between(mean_series.index, mean_series - std_series, mean_series + std_series, color='gray', alpha=0.2, label='Std. Dev.')

        plt.title(f'Training Curve for {metric}', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
        plt.legend(title='Folds')
        plt.grid(True)
        
        # グラフをファイルに保存
        output_path = log_dir / f'{metric}_training_curve.png'
        plt.savefig(output_path)
        print(f"グラフを保存しました: {output_path}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSVロガーから学習曲線を可視化します。")
    parser.add_argument("log_dir", type=str, help="実験のログが保存されている親ディレクトリ (例: lightning_logs/exp_name)")
    parser.add_argument(
        "--metrics",
        nargs='+',
        default=['val_loss', 'val_acc_epoch', 'train_loss_epoch'],
        help="プロットするメトリクスのリスト (例: val_loss val_acc_epoch)"
    )
    args = parser.parse_args()

    log_path = Path(args.log_dir)
    if not log_path.is_dir():
        print(f"エラー: 指定されたディレクトリが見つかりません: {log_path}")
    else:
        plot_metrics(log_path, args.metrics)
