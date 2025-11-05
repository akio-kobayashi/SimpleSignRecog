import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

def get_class_mapping(metadata_path: str) -> dict:
    """
    メタデータファイルからクラス名とラベルのマッピングを読み込む。
    """
    metadata_df = pd.read_csv(metadata_path)
    if 'class_name' in metadata_df.columns and 'class_label' in metadata_df.columns:
        return pd.Series(metadata_df['class_name'].values, index=metadata_df['class_label']).to_dict()
    return {}

def calculate_weighted_stats(distribution: np.ndarray) -> (float, float):
    """
    確率分布から正規化時間における加重平均と加重分散を計算する。
    """
    # 確率分布がゼロサムの場合、計算をスキップ
    dist_sum = np.sum(distribution)
    if dist_sum == 0:
        return np.nan, np.nan

    num_steps = len(distribution)
    # 時間ステップが1の場合のゼロ除算を避ける
    if num_steps > 1:
        normalized_time = np.arange(num_steps) / (num_steps - 1)
    else:
        normalized_time = np.array([0.5]) # ステップが1つなら中央の0.5とする
    
    weights = distribution / dist_sum
    
    mean = np.sum(normalized_time * weights)
    variance = np.sum(weights * (normalized_time - mean)**2)
    
    return mean, variance

def main():
    parser = argparse.ArgumentParser(description="Analyze posteriograms to compute statistics on normalized time.")
    parser.add_argument("input_dir", type=str, help="Directory containing the .npz posteriogram files.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the project config.yaml file.")
    parser.add_argument("--output_csv", type=str, default="posteriogram_time_stats.csv", help="Path to save the output CSV file.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_csv)

    # --- クラスマッピングの準備 ---
    try:
        with open(args.config, 'r') as yf:
            config = yaml.safe_load(yf)
        metadata_path = config['data']['metadata_path']
        class_mapping = get_class_mapping(metadata_path)
    except Exception as e:
        print(f"Warning: Could not load or parse config/metadata: {e}. Using integer labels.")
        class_mapping = {}

    # --- ファイルの処理 ---
    npz_files = sorted(list(input_dir.glob("*.npz")))
    if not npz_files:
        print(f"Error: No .npz files found in directory {input_dir}")
        return

    # 各クラスの統計値を保存する辞書
    class_stats = defaultdict(lambda: {'means': [], 'variances': []})

    print(f"Found {len(npz_files)} files to analyze...")

    for npz_path in npz_files:
        try:
            data = np.load(npz_path)
            posteriogram = data['posteriogram'] # Shape: (T, C+1)
            true_label = data['label'].item()
        except Exception as e:
            print(f"Skipping file {npz_path} due to loading error: {e}")
            continue

        # サンプルの真のクラスに対応する確率時系列を取得
        if true_label < posteriogram.shape[1]:
            prob_over_time = posteriogram[:, true_label]
            
            # このサンプルに対する統計値を計算
            mean, variance = calculate_weighted_stats(prob_over_time)
            
            if not np.isnan(mean):
                # 結果をクラスごとに保存
                class_stats[true_label]['means'].append(mean)
                class_stats[true_label]['variances'].append(variance)
        else:
            print(f"Warning: Skipping sample {npz_path.name}. True label {true_label} is out of bounds for posteriogram with shape {posteriogram.shape}.")


    # --- 統計量の集計 ---
    results = []
    sorted_labels = sorted(class_stats.keys())

    for label in sorted_labels:
        stats = class_stats[label]
        if not stats['means']:
            continue

        class_name = class_mapping.get(label, f"Class {label}")
        
        mean_of_means = np.mean(stats['means'])
        std_of_means = np.std(stats['means'])
        mean_of_variances = np.mean(stats['variances'])
        
        results.append({
            'class_label': label,
            'class_name': class_name,
            'num_samples': len(stats['means']),
            'mean_of_norm_time_mean': mean_of_means,
            'std_of_norm_time_mean': std_of_means,
            'mean_of_norm_time_variance': mean_of_variances
        })

    if not results:
        print("No data processed. Exiting.")
        return

    # 結果をDataFrameに変換してCSVに保存
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False, float_format='%.6f')

    print(f"\nAnalysis complete. Statistics saved to {output_path}")
    print("\n--- Summary ---")
    print(results_df)


if __name__ == "__main__":
    main()
