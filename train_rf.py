


# 必要なライブラリをインポートします
# ---------------------------------
import yaml  # 設定ファイルを読み込むために使用
import random  # 乱数を生成するために使用
import copy  # オブジェクトをコピーするために使用
from argparse import ArgumentParser  # コマンドライン引数を解析するために使用
from pathlib import Path  # ファイルパスを操作するために使用
import numpy as np  # 数値計算、特に配列を扱うために使用
import pandas as pd  # データ分析、特にCSVファイルやデータフレームを扱うために使用
from sklearn.model_selection import StratifiedKFold  # 層化K分割交差検証のために使用
from sklearn.preprocessing import StandardScaler  # データの特徴量を標準化するために使用
from sklearn.ensemble import RandomForestClassifier  # ランダムフォレストモデルを構築するために使用
from sklearn.metrics import classification_report, accuracy_score  # モデルの評価指標を計算するために使用
from tqdm import tqdm  # 処理の進捗状況をプログレスバーで表示するために使用

# 自作のデータセットクラスをインポートします
# -----------------------------------------
from src.dataset import SignDataset

def set_seed(seed: int):
    """
    再現性のために乱数のシードを設定する関数。
    シードを固定することで、何度実行しても同じ結果が得られるようになります。
    
    Args:
        seed (int): 設定するシード値。
    """
    random.seed(seed)
    np.random.seed(seed)

def extract_statistical_features(features: np.ndarray) -> np.ndarray:
    """
    時系列のランドマークデータから統計的特徴量を抽出する関数。
    各ランドマーク座標の平均、標準偏差、最小値、最大値、中央値を計算し、
    それらを連結して1つの特徴量ベクトルを生成します。
    
    Args:
        features (np.ndarray): 形状が (時系列の長さ, 特徴量の数) のNumPy配列。
        
    Returns:
        np.ndarray: 形状が (1, 特徴量の数 * 5) の統計的特徴量ベクトル。
    """
    # もし時系列データが空の場合（例えば、ランドマークが検出されなかった場合）
    if features.shape[0] == 0:
        # ゼロの配列を返します
        return np.zeros(features.shape[1] * 5)
        
    # 各統計量を計算します (axis=0 は時間軸に沿って計算することを意味します)
    mean = np.mean(features, axis=0)  # 平均
    std = np.std(features, axis=0)    # 標準偏差
    min_vals = np.min(features, axis=0)  # 最小値
    max_vals = np.max(features, axis=0)  # 最大値
    median = np.median(features, axis=0) # 中央値
    
    # 計算したすべての統計的特徴量を一つの長いベクトルに連結します
    return np.concatenate([mean, std, min_vals, max_vals, median])

def main(config: dict):
    """
    ランダムフォレストモデルの訓練と評価を行うメインの関数。
    K分割交差検証（K-Fold Cross-Validation）という手法を用いて、
    モデルの性能を評価します。
    
    Args:
        config (dict): config.yamlから読み込まれた設定情報。
    """
    # --- 0. シードの設定 ---
    # 再現性を確保するために、設定ファイルにシードがあれば設定します
    if "seed" in config:
        set_seed(config["seed"])
        print(f"--- シードを {config['seed']} に設定しました ---")

    # --- 1. データセットの読み込みと特徴量抽出 ---
    print("--- データセットを読み込み、特徴量を抽出します ---")
    data_config = config['data']
    # メタデータ（どの動画がどのラベルかなどの情報）をCSVファイルから読み込みます
    metadata_df = pd.read_csv(data_config['metadata_path'])

    # ランダムフォレストでは、データ拡張（augmentation）を行わない固定の特徴量セットを使用します。
    # そのため、データ拡張を無効にした設定を作成します。
    eval_config = copy.deepcopy(config)
    if 'data' not in eval_config: eval_config['data'] = {}
    eval_config['data']['augmentation'] = {}

    # SignDatasetクラスを使って、データセットを準備します
    dataset = SignDataset(
        metadata_df=metadata_df,
        data_base_dir=data_config['source_landmark_dir'], # ランドマークデータが保存されているディレクトリ
        config=eval_config,
        sort_by_length=False  # 時系列の長さでソートしない
    )

    X_list = []  # 特徴量（モデルの入力）を格納するリスト
    y_list = []  # ラベル（正解）を格納するリスト
    
    # tqdmを使って、進捗バーを表示しながらデータセット全体をループ処理します
    for i in tqdm(range(len(dataset)), desc="特徴量を抽出中"):
        # データセットからi番目のデータを取得します
        features_tensor, label = dataset[i]
        # PyTorchのテンソルをNumPy配列に変換します
        features_np = features_tensor.numpy()
        
        # 時系列データから統計的特徴量を抽出します
        stat_features = extract_statistical_features(features_np)
        
        # 抽出した特徴量とラベルをリストに追加します
        X_list.append(stat_features)
        y_list.append(label)

    # リストをNumPy配列に変換します
    X = np.array(X_list)  # モデルの入力データ
    y = np.array(y_list)  # 正解ラベル
    
    print(f"--- 特徴量抽出完了。特徴量行列の形状: {X.shape} ---")

    # --- 2. 交差検証 (Cross-Validation) の設定 ---
    # configファイルから分割数を取得します。デフォルトは5です。
    num_folds = data_config.get('num_folds', 5)
    # 分割数が-1の場合は、Leave-One-Out交差検証（LOOCV）を行います
    is_loocv = (num_folds == -1)

    if is_loocv:
        from sklearn.model_selection import LeaveOneOut
        print("---
 Leave-One-Out 交差検証を設定します ---")
        cv_splitter = LeaveOneOut() # 1つのデータのみをテストデータとし、残りを訓練データとする方法
        n_splits = cv_splitter.get_n_splits(X)
        all_labels = [] # 全てのテストデータの正解ラベルを格納
        all_preds = []  # 全てのテストデータの予測結果を格納
    else:
        print(f"--- 層化 {num_folds} 分割交差検証を設定します ---")
        # StratifiedKFoldは、各クラスの比率を保ったままデータを分割します
        cv_splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=config.get('seed', 42))
        n_splits = num_folds
        all_fold_metrics = [] # 各分割での評価指標を格納

    # cv_splitter.split()で、訓練データとテストデータのインデックスを分割します
    for fold, (train_indices, test_indices) in enumerate(cv_splitter.split(X, y)):
        print(f"\n===== 分割 {fold + 1} / {n_splits} =====")

        # --- 2a. データを訓練用とテスト用に分割 ---
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # --- 2b. 特徴量の標準化 ---
        # StandardScalerを使って、特徴量のスケールを揃えます（平均0, 分散1）
        # 重要：scalerは訓練データにのみ適合させ(fit)、そのscalerを使ってテストデータを変換(transform)します
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- 2c. ランダムフォレストモデルの訓練 ---
        print("--- ランダムフォレストモデルを訓練します ---")
        # n_estimatorsは決定木の数、n_jobs=-1はCPUの全コアを使用することを意味します
        rf_model = RandomForestClassifier(n_estimators=100, random_state=config.get('seed', 42), n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)

        # --- 2d. この分割でのテスト ---
        print(f"--- 分割 {fold + 1} のテスト中 ---")
        # 訓練済みモデルでテストデータのラベルを予測します
        y_pred = rf_model.predict(X_test_scaled)

        if is_loocv:
            # LOOCVの場合、すべての予測結果と正解ラベルを収集します
            all_labels.extend(y_test)
            all_preds.extend(y_pred)
            acc = accuracy_score(y_test, y_pred)
            print(f"分割 {fold + 1} 正解: {acc == 1.0}")
        else:
            # k-foldの場合、各分割ごとに評価レポートを生成・保存します
            # output_dict=Trueで、レポートを辞書形式で取得します
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)
            
            # configファイルから平均化手法を取得します（例: 'macro', 'weighted'）
            avg_mode = config.get('trainer', {}).get('metrics_average_mode', 'macro')
            avg_key = f'{avg_mode} avg'

            # この分割での評価指標を辞書にまとめます
            fold_metrics = {
                'test_accuracy': accuracy,
                'test_f1': report.get(avg_key, {}).get('f1-score', 0),
                'test_precision': report.get(avg_key, {}).get('precision', 0),
                'test_recall': report.get(avg_key, {}).get('recall', 0)
            }
            all_fold_metrics.append(fold_metrics)
            
            print(f"分割 {fold + 1} の正解率: {accuracy:.4f}")
            print(f"分割 {fold + 1} のF1スコア ({avg_mode.capitalize()}): {fold_metrics['test_f1']:.4f}")

    # --- 3. 最終結果の集計と保存/表示 ---
    print("\n===== ランダムフォレスト 交差検証 最終結果 ======")
    # configファイルから出力先のパスを取得します
    output_path = config.get('output', {}).get('rf_results_path')

    if is_loocv:
        # --- LOOCV: 全ての予測結果から単一のレポートを生成 ---
        if not all_labels:
            print("レポートを生成するためのテスト結果がありませんでした。")
            return

        # 全ての予測結果を使って、最終的な評価レポートを作成します
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        results_df = pd.DataFrame(report).transpose()
        
        print("\n--- Leave-One-Out の総合評価指標 ---")
        print(results_df.to_string(float_format='%.4f'))
        
        # 出力パスが指定されていれば、結果をCSVファイルに保存します
        if output_path:
            try:
                results_df.to_csv(output_path, float_format='%.4f')
                print(f"\n総合LOOCVレポートを {output_path} に保存しました")
            except Exception as e:
                print(f"\n結果の保存中にエラーが発生しました: {e}")

    else:
        # --- k-fold: 各分割のレポートを集計し、平均を計算 ---
        if not all_fold_metrics:
            print("レポートを生成するためのテスト結果がありませんでした。")
            return

        # 各分割の評価指標をデータフレームに変換します
        results_df = pd.DataFrame(all_fold_metrics)
        # 平均値を計算し、'average'行として追加します
        results_df.loc['average'] = results_df.mean()

        results_df.index.name = 'fold'
        results_df = results_df.reset_index()
        # 分割のインデックスを 1-based に調整します
        results_df['fold'] = results_df['fold'].apply(lambda x: str(x + 1) if isinstance(x, int) else x)

        print("\n--- 全分割の平均評価指標 ---")
        print(results_df.to_string(float_format='%.4f', index=False))

        # 出力パスが指定されていれば、結果をCSVファイルに保存します
        if output_path:
            try:
                results_df.to_csv(output_path, index=False, float_format='%.4f')
                print(f"\n結果を {output_path} に保存しました")
            except Exception as e:
                print(f"\n結果の保存中にエラーが発生しました: {e}")

# このスクリプトが直接実行された場合にのみ以下のコードが実行されます
if __name__ == "__main__":
    # コマンドライン引数のパーサーを作成します
    parser = ArgumentParser()
    # --config引数を追加します。デフォルトは "config.yaml" です
    parser.add_argument("--config", type=str, default="config.yaml", help="設定YAMLファイルへのパス")
    args = parser.parse_args()

    # 設定ファイルを読み込みます
    with open(args.config, "r") as yf:
        config = yaml.safe_load(yf)

    # メイン関数を実行します
    main(config)
