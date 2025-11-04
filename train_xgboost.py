# 必要なライブラリをインポートします
# ---------------------------------
import yaml
import random
import copy
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# XGBoostライブラリをインポートします
# ----------------------------------
try:
    import xgboost as xgb
except ImportError:
    # もしXGBoostがインストールされていない場合は、エラーメッセージを表示して終了します
    print("XGBoostが見つかりません。 pip install xgboost でインストールしてください。")
    exit()

# 自作のデータセットクラスをインポートします
# -----------------------------------------
from src.dataset import SignDataset

def set_seed(seed: int):
    """
    再現性のために乱数のシードを設定する関数。
    """
    random.seed(seed)
    np.random.seed(seed)

def extract_statistical_features(features: np.ndarray) -> np.ndarray:
    """
    時系列のランドマークデータから統計的特徴量を抽出する関数。
    """
    if features.shape[0] == 0:
        return np.zeros(features.shape[1] * 5)
        
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)
    median = np.median(features, axis=0)
    
    return np.concatenate([mean, std, min_vals, max_vals, median])

def main(config: dict):
    """
    XGBoostモデルの訓練と評価を行うメインの関数。
    K分割交差検証を用いてモデルの性能を評価します。
    """
    # --- 0. シードの設定 ---
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"--- 再現性のためにシードを {seed} に設定しました ---")

    # --- 1. データセットの読み込みと特徴量抽出 ---
    print("--- データセットを読み込み、XGBoost用の特徴量を抽出します ---")
    data_config = config['data']
    metadata_df = pd.read_csv(data_config['metadata_path'])

    # データ拡張を無効にした設定を作成します
    eval_config = copy.deepcopy(config)
    if 'data' not in eval_config: eval_config['data'] = {}
    eval_config['data']['augmentation'] = {}

    # データセットを準備します
    dataset = SignDataset(
        metadata_df=metadata_df,
        data_base_dir=data_config['source_landmark_dir'],
        config=eval_config,
        sort_by_length=False
    )

    X_list = []
    y_list = []
    for i in tqdm(range(len(dataset)), desc="特徴量を抽出中"):
        features_tensor, label = dataset[i]
        features_np = features_tensor.numpy()
        stat_features = extract_statistical_features(features_np)
        X_list.append(stat_features)
        y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"--- 特徴量抽出完了。特徴量行列の形状: {X.shape} ---")

    # --- 2. 交差検証 (Cross-Validation) の設定 ---
    num_folds = data_config.get('num_folds', 5)
    is_loocv = (num_folds == -1)

    if is_loocv:
        from sklearn.model_selection import LeaveOneOut
        print("--- Leave-One-Out 交差検証を設定します ---")
        cv_splitter = LeaveOneOut()
        n_splits = cv_splitter.get_n_splits(X)
        all_labels = []
        all_preds = []
    else:
        print(f"--- 層化 {num_folds} 分割交差検証を設定します ---")
        cv_splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        n_splits = num_folds
        all_fold_metrics = []

    # 交差検証ループ
    for fold, (train_indices, test_indices) in enumerate(cv_splitter.split(X, y)):
        print(f"\n===== 分割 {fold + 1} / {n_splits} =====")

        # --- 2a. データを訓練用とテスト用に分割 ---
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # --- 2b. 特徴量の標準化 ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- 2c. XGBoostモデルの訓練 ---
        print("--- XGBoostモデルを訓練します ---")
        
        # configファイルからXGBoostのパラメータを取得
        trainer_config = config.get('trainer')
        if trainer_config is None:
            trainer_config = {}
        xgboost_params = {}
        
        # GPUを使用するかどうかの設定
        force_cuda = config.get('trainer', {}).get('force_cuda', True)
        if force_cuda:
            # 'device': 'cuda' を設定すると、XGBoostはGPUで学習しようとします
            xgboost_params['device'] = 'cuda'
        else:
            xgboost_params['device'] = xgboost_params.get('device', 'cpu')

        # 学習アルゴリズムは'hist'（ヒストグラムベース）に固定します。これは高速で、GPU/CPUの両方で動作します。
        xgboost_params['tree_method'] = 'hist'            

        # XGBoostモデルを初期化します
        final_xgboost_params = xgboost_params.copy()
        if 'predictor' in final_xgboost_params:
            del final_xgboost_params['predictor']

        xgb_model = xgb.XGBClassifier(
            random_state=seed,      # 再現性のためのシード
            eval_metric='mlogloss', # 評価指標として多クラス対数損失を使用
            **final_xgboost_params  # configから読み込んだその他のパラメータ
        )
        
        try:
            # モデルの学習を実行
            xgb_model.fit(X_train_scaled, y_train)
        except xgb.core.XGBoostError as e:
            # GPUでの学習に失敗した場合のエラーハンドリング
            if 'device=cuda' in str(final_xgboost_params):
                raise RuntimeError(
                    "XGBoostがGPUを利用できませんでした。コンテナ、NVIDIAドライバ、またはXGBoostのビルド設定を確認してください。"
                ) from e
            else:
                raise # その他のエラーはそのまま送出       

        # --- 2d. この分割でのテスト ---
        print(f"--- 分割 {fold + 1} のテスト中 ---")
        y_pred = xgb_model.predict(X_test_scaled)

        if is_loocv:
            # LOOCVの場合、結果を収集
            all_labels.extend(y_test)
            all_preds.extend(y_pred)
            acc = accuracy_score(y_test, y_pred)
            print(f"分割 {fold + 1} 正解: {acc == 1.0}")
        else:
            # k-foldの場合、各分割の評価レポートを生成
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)
            
            avg_mode = config.get('trainer', {}).get('metrics_average_mode', 'macro')
            avg_key = f'{avg_mode} avg'

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
    print("\n===== XGBOOST 交差検証 最終結果 ======")
    output_path = config.get('output', {}).get('xgboost_results_path')

    if is_loocv:
        # --- LOOCV: 全ての予測結果から単一のレポートを生成 ---
        if not all_labels:
            print("レポートを生成するためのテスト結果がありませんでした。")
            return

        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        results_df = pd.DataFrame(report).transpose()
        
        print("\n--- Leave-One-Out の総合評価指標 ---")
        print(results_df.to_string(float_format='%.4f'))
        
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

        results_df = pd.DataFrame(all_fold_metrics)
        results_df.loc['average'] = results_df.mean()

        results_df.index.name = 'fold'
        results_df = results_df.reset_index()
        results_df['fold'] = results_df['fold'].apply(lambda x: str(x + 1) if isinstance(x, int) else x)

        print("\n--- 全分割の平均評価指標 ---")
        print(results_df.to_string(float_format='%.4f', index=False))

        if output_path:
            try:
                results_df.to_csv(output_path, index=False, float_format='%.4f')
                print(f"\n結果を {output_path} に保存しました")
            except Exception as e:
                print(f"\n結果の保存中にエラーが発生しました: {e}")

# このスクリプトが直接実行された場合にのみ以下のコードが実行されます
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="設定YAMLファイルへのパス")
    args = parser.parse_args()

    with open(args.config, "r") as yf:
        config = yaml.safe_load(yf)

    main(config)

