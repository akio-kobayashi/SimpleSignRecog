import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_class_mapping(metadata_path: str, posteriogram_dim: int) -> dict:
    """
    メタデータファイルからクラス名とラベルのマッピングを読み込む。
    posteriogramの次元数に基づいて、blankクラスを含む完全なマッピングを生成する。
    """
    metadata_df = pd.read_csv(metadata_path)
    class_mapping = {}
    if 'class_name' in metadata_df.columns and 'class_label' in metadata_df.columns:
        class_mapping = pd.Series(metadata_df['class_name'].values, index=metadata_df['class_label']).to_dict()

    # posteriogramの次元数 (クラス数 + 1) に合わせてマッピングを調整
    num_classes_in_posteriogram = posteriogram_dim
    
    # 不足しているラベルを 'unknown_X' として追加
    for i in range(num_classes_in_posteriogram):
        if i not in class_mapping:
            class_mapping[i] = f'unknown_{i}'
            
    # CTC blank tokenのラベルを設定 (最後のクラス)
    class_mapping[num_classes_in_posteriogram - 1] = 'blank'
    
    return class_mapping

def visualize_posteriogram(npz_path: Path, class_mapping: dict, output_dir: Path):
    """
    単一のposteriogramファイルからヒートマップを生成して保存する。
    """
    # データを読み込む
    try:
        data = np.load(npz_path)
        posteriogram = data['posteriogram'] # Shape: (T, C+1)
        true_label = data['label'].item()
    except Exception as e:
        print(f"Error loading or reading {npz_path}: {e}")
        return
    
    # クラス名を取得
    true_class_name = class_mapping.get(true_label, f"Unknown Label: {true_label}")
    
    # 描画
    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.imshow(posteriogram.T, aspect='auto', interpolation='nearest', cmap=cm.viridis, origin='lower')
    
    # 軸ラベルとタイトル
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Classes")
    ax.set_title(f"Posteriogram for '{true_class_name}' (File: {npz_path.name})", fontsize=16)
    
    # Y軸の目盛りを設定
    if class_mapping:
        # ソートされたラベルと名前を取得
        sorted_labels = sorted(class_mapping.keys())
        sorted_names = [class_mapping[k] for k in sorted_labels]
        ax.set_yticks(sorted_labels)
        ax.set_yticklabels(sorted_names)

    # カラーバーを追加
    fig.colorbar(im, ax=ax, label="Probability")
    
    # 保存
    output_path = output_dir / f"{npz_path.stem}.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize posteriograms generated during testing.")
    parser.add_argument("input_path", type=str, help="Path to a single .npz posteriogram file or a directory containing them.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the project config.yaml file to get metadata info.")
    parser.add_argument("--output_dir", type=str, default="posteriogram_visualizations", help="Directory to save the output images.")
    args = parser.parse_args()

    # パスをPathオブジェクトに変換
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    
    # 出力ディレクトリを作成
    output_dir.mkdir(exist_ok=True)

    # 処理するファイルリストを作成
    if input_path.is_dir():
        npz_files = sorted(list(input_path.glob("*.npz")))
        if not npz_files:
            print(f"Error: No .npz files found in directory {input_path}")
            return
    elif input_path.is_file() and input_path.suffix == '.npz':
        npz_files = [input_path]
    else:
        print(f"Error: Provided path {input_path} is not a valid .npz file or directory.")
        return

    # --- クラスマッピングの準備 ---
    # 最初のファイルから次元数を取得
    first_data = np.load(npz_files[0])
    posteriogram_dim = first_data['posteriogram'].shape[1]

    # 設定ファイルからクラスマッピングを読み込む
    try:
        with open(args.config, 'r') as yf:
            config = yaml.safe_load(yf)
        metadata_path = config['data']['metadata_path']
        class_mapping = get_class_mapping(metadata_path, posteriogram_dim)
    except Exception as e:
        print(f"Warning: Could not load or parse config/metadata: {e}. Using integer labels.")
        class_mapping = {i: str(i) for i in range(posteriogram_dim)}
        class_mapping[posteriogram_dim - 1] = 'blank'


    # 各ファイルを可視化
    for npz_path in npz_files:
        visualize_posteriogram(npz_path, class_mapping, output_dir)

if __name__ == "__main__":
    main()
