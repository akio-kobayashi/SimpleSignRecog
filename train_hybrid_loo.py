# 必要なライブラリをインポートします
# ---------------------------------
import yaml
import random
import copy
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold, train_test_split, LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, ConcatDataset

# 自作のモジュールをインポートします
# ---------------------------------
from src.dataset import SignDataset, BucketBatchSampler, data_processing
from src.solver import Solver

# --- 再現性のための設定 ---
def set_seed(seed: int):
    """
    再現性のために乱数のシードを設定する関数。
    これにより、何度実行しても同じ結果が得られるようになります。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 決定的な（毎回同じ結果になる）演算を保証します
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """
    DataLoaderの各ワーカープロセスにシードを設定する関数。
    マルチプロセスでデータを読み込む際にも再現性を確保します。
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_feature_dim(feature_config: dict) -> int:
    """
    config.yamlのfeaturesセクションに基づいて、特徴量の次元数を計算する関数。
    設定に応じて、モデルに入力される特徴ベクトルのサイズを決定します。
    """
    # ベースとなる座標の次元数 (x, y, z * 21ランドマーク * 2手)
    BASE_COORD_DIM = 21 * 3 * 2
    
    # 既存の特徴量パイプラインの次元数
    # pos(63*2) + vel(63*2) + acc(63*2) + geo(4*2) のうち、片手分が入力となる
    EXISTING_PIPELINE_DIM = 193 * 2

    normalize_mode = feature_config.get('normalize_mode', 'normalize_landmarks')
    paper_conf = feature_config.get('paper_features', {})
    use_paper_speed = paper_conf.get('speed', False)
    use_paper_anthropometric = paper_conf.get('anthropometric', False)

    # 論文ベースの特徴量計算モードかどうかを判定
    is_paper_mode = normalize_mode in ['current_wrist', 'first_wrist'] or use_paper_speed or use_paper_anthropometric

    if is_paper_mode:
        # 論文ベースのパイプラインの場合
        dim = BASE_COORD_DIM # 座標は常に出力
        if use_paper_speed:
            dim += BASE_COORD_DIM # 速度特徴量を追加
        if use_paper_anthropometric:
            # 21個のランドマークから2つを選ぶ組み合わせの数 (21 C 2 = 210) * 2手
            dim += 210 * 2 # 人体測定学的特徴量を追加
        return dim
    else:
        # 既存のパイプラインの場合
        return EXISTING_PIPELINE_DIM

def create_concat_dataset(df: pd.DataFrame, config: dict, sort_by_length: bool = False) -> ConcatDataset | SignDataset:
    """DataFrameからConcatDatasetを作成するヘルパー関数"""
    if df.empty:
        # 空のデータセットを返す。ConcatDatasetが空リストを許容しないため。
        return SignDataset(pd.DataFrame(columns=df.columns), '', config)

    datasets = []
    # 'base_dir'列でグループ化し、各グループ（＝各話者）ごとにDatasetを作成
    for base_dir, group_df in df.groupby('base_dir'):
        datasets.append(SignDataset(
            metadata_df=group_df.reset_index(drop=True),
            data_base_dir=base_dir,
            config=config,
            # バケットサンプリングは訓練セット全体でソートする必要があるため、
            # ここでのソートは無効化し、後で手動でソートする
            sort_by_length=False 
        ))
    
    if not datasets:
        return SignDataset(pd.DataFrame(columns=df.columns), '', config)
    
    return ConcatDataset(datasets)


def main(config: dict, args: ArgumentParser, checkpoint_path: str | None = None):
    """
    K分割交差検証（K-Fold Cross-Validation）を用いたメインの学習パイプライン。
    """
    # --- 0. コマンドライン引数でconfigを上書き ---
    if args.num_folds is not None:
        config['data']['num_folds'] = args.num_folds
    if args.cm_output_dir is not None:
        config['output']['cm_output_dir'] = args.cm_output_dir
    if args.augment_flip is not None:
        config['data']['augmentation']['augment_flip'] = (args.augment_flip == 'true')
    if args.augment_rotate is not None:
        config['data']['augmentation']['augment_rotate'] = (args.augment_rotate == 'true')
    if args.augment_noise is not None:
        config['data']['augmentation']['augment_noise'] = (args.augment_noise == 'true')

    # --- シードの設定と特徴量次元数の計算 ---
    if "seed" in config:
        set_seed(config["seed"])
        print(f"--- 再現性のためにシードを {config['seed']} に設定しました ---")

    # configに基づいて特徴量の次元数を計算し、config辞書を更新
    feature_dim = get_feature_dim(config.get('features', {}))
    config['model']['input_dim'] = feature_dim
    print(f"--- configに基づき、特徴量の次元数を {feature_dim} と計算しました ---")
    
    # DataLoaderのシャッフルを再現可能にするためのジェネレータ
    g = torch.Generator()
    if "seed" in config:
        g.manual_seed(config["seed"])

    # --- 1. 全データセットの読み込み ---
    print("--- 全話者のデータセットを読み込んでいます ---")
    all_dfs = []
    # コマンドラインから渡された各設定ファイルをループ処理
    for i, config_path in enumerate(args.config):
        print(f"  - {config_path} を読み込み中...")
        with open(config_path, 'r') as f:
            subject_config = yaml.safe_load(f)
        
        data_info = subject_config['data']
        df = pd.read_csv(data_info['metadata_path'])
        df['subject_id'] = i  # 話者IDを整数として追加
        df['base_dir'] = data_info['source_landmark_dir'] # データパスもDFに追加
        all_dfs.append(df)

    metadata_df = pd.concat(all_dfs, ignore_index=True)
    print(f"--- {len(args.config)} 名の話者から合計 {len(metadata_df)} 個のサンプルを読み込みました ---")

    # ベースconfigの 'data' セクションを継続して使用
    data_config = config['data']
    
    # クラスのインデックスとクラス名の対応表を作成（もしあれば）
    class_mapping = None
    if 'class_name' in metadata_df.columns:
        class_mapping = pd.Series(metadata_df['class_name'].values, index=metadata_df['class_label']).to_dict()
        print(f"{len(class_mapping)} 個のクラスが見つかりました。")


    # --- 2. 交差検証 (Cross-Validation) の設定 ---
    num_folds = config['data'].get('num_folds', 5)
    print(f"DEBUG: `num_folds` is set to {num_folds}") # デバッグ用print文
    is_loocv = num_folds <= 1

    if not is_loocv:
        print(f"--- 層化 {num_folds} 分割交差検証を設定します ---")
        cv_splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=config.get('seed', 42))
        cv_iterator = cv_splitter.split(metadata_df, metadata_df['class_label'])
    else:
        print("--- Leave-One-Out 交差検証を設定します ---")
        cv_splitter = LeaveOneOut()
        num_folds = cv_splitter.get_n_splits(metadata_df) # レポート用に分割数を更新
        cv_iterator = cv_splitter.split(metadata_df)

    all_results = [] # 全ての予測結果を格納するリスト ({'true': y, 'pred': p, 'subject_id': sid})
    all_fold_wer_results = [] # WERの結果を格納するリスト

    # 交差検証のループを開始
    for fold, (train_val_indices, test_indices) in enumerate(cv_iterator):
        print(f"\n===== 分割 {fold + 1} / {num_folds} =====")

        # --- 2a. この分割でのデータ分割 (Hybrid LOO) ---
        test_df = metadata_df.iloc[test_indices].reset_index(drop=True)
        train_val_df = metadata_df.iloc[train_val_indices].reset_index(drop=True)

        # テストサンプルの話者IDを取得
        test_subject_id = test_df.iloc[0]['subject_id']

        # 同じ話者の残りのデータを抽出
        same_subject_df = train_val_df[train_val_df['subject_id'] == test_subject_id].reset_index(drop=True)

        # 他の話者の全データを抽出
        other_subjects_df = train_val_df[train_val_df['subject_id'] != test_subject_id].reset_index(drop=True)

        # 同じ話者のデータから検証セットを作成
        if not same_subject_df.empty:
            # 層化分割を試みる
            stratify_col = same_subject_df['class_label'] if len(same_subject_df['class_label'].unique()) > 1 else None
            # 検証スプリットがデータ数より大きい場合に対処
            val_split_ratio = data_config['validation_split_ratio']
            if len(same_subject_df) * val_split_ratio < 1 and len(same_subject_df) > 0:
                val_split_ratio = 1.0 / len(same_subject_df)
            
            if val_split_ratio > 0:
                train_from_same_subject_df, val_df = train_test_split(
                    same_subject_df,
                    test_size=val_split_ratio,
                    random_state=config.get('seed', 42),
                    stratify=stratify_col
                )
                val_df = val_df.reset_index(drop=True)
            else:
                train_from_same_subject_df = same_subject_df
                val_df = pd.DataFrame()
        else:
            # 同じ話者のデータが他にない場合
            train_from_same_subject_df = pd.DataFrame()
            val_df = pd.DataFrame()

        # 最終的な学習データは「他の話者の全データ」＋「同じ話者の残り(検証で使わなかった分)」
        train_df = pd.concat([other_subjects_df, train_from_same_subject_df], ignore_index=True)

        print(f"訓練: {len(train_df)}, 検証: {len(val_df)}, テスト: {len(test_df)}")

        # --- 2b. データセットとデータローダーの作成 ---
        # 検証用とテスト用ではデータ拡張を無効にする
        eval_config = copy.deepcopy(config)
        eval_config['data']['augmentation'] = {}

        # バケットサンプリングのために訓練DFをソート
        if data_config.get('use_bucketing', False):
            if 'num_frames' not in train_df.columns:
                # num_frames列がない場合、バケットサンプリングは無効化
                print("Warning: 'num_frames' column not found. Disabling bucketing for this fold.")
                data_config['use_bucketing'] = False
            else:
                train_df = train_df.sort_values(by='num_frames').reset_index(drop=True)

        # 訓練、検証、テスト用の各データセットを作成
        train_dataset = create_concat_dataset(train_df, config)
        val_dataset = create_concat_dataset(val_df, eval_config)
        test_dataset = create_concat_dataset(test_df, eval_config)

        # データローダーの準備
        if data_config.get('use_bucketing', False):
            # BucketBatchSamplerは .df 属性を持つdatasetを期待するが、ConcatDatasetにはない。
            # そのため、ソート済みの train_df を持つ代理オブジェクトを作成して渡す。
            class SamplerProxy:
                def __init__(self, df):
                    self.df = df
                def __len__(self):
                    return len(self.df)

            proxy_for_sampler = SamplerProxy(train_df)
            train_sampler = BucketBatchSampler(proxy_for_sampler, batch_size=data_config['batch_size'])
            
            # DataLoaderには、実際のデータ取得のため、オリジナルのtrain_dataset(ConcatDataset)を渡す
            train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=data_processing, num_workers=4, worker_init_fn=seed_worker)
        else:
            # 通常のシャッフルを使用する場合
            train_loader = DataLoader(train_dataset, batch_size=data_config['batch_size'], shuffle=True, collate_fn=data_processing, num_workers=4, worker_init_fn=seed_worker, generator=g)

        valid_loader = DataLoader(val_dataset, batch_size=data_config['batch_size'], shuffle=False, collate_fn=data_processing, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=data_config['batch_size'], shuffle=False, collate_fn=data_processing, num_workers=4)

        # --- 2c. モデルとトレーナーの初期化 ---
        if 'scheduler' in config and 'total_steps' not in config['scheduler']:
            # スケジューラの総ステップ数を各分割で再計算
            num_devices = trainer.num_devices if 'trainer' in locals() and hasattr(trainer, 'num_devices') else 1
            effective_batch_size = data_config['batch_size'] * num_devices
            total_steps = (len(train_dataset) // effective_batch_size) * config['trainer']['max_epochs']
            config['scheduler']['total_steps'] = total_steps
            print(f"スケジューラの総ステップ数を {total_steps} に設定しました (分割 {fold + 1}) ")

        # Solverクラス（PyTorch Lightningモジュール）を初期化
        model = Solver(config)

        # この分割用のログとチェックポイントの保存先を定義
        fold_log_dir = Path(config["logger"]["save_dir"]) / config["logger"]["name"]
        
        # TensorBoardとCSVの両方でロギングを行う
        logger = [
            TensorBoardLogger(save_dir=str(fold_log_dir.parent), name=config["logger"]["name"], version=f"fold_{fold}"),
            CSVLogger(save_dir=str(fold_log_dir.parent), name=config["logger"]["name"], version=f"fold_{fold}")
        ]
        
        # チェックポイントの設定
        # config.yamlのdirpathを尊重しつつ、foldごとにサブディレクトリを作成
        base_dir = Path(config["checkpoint"]["dirpath"])
        fold_checkpoint_dir = base_dir / f"fold_{fold}"
        
        # dirpath以外の設定をconfigから取得
        checkpoint_conf = {k: v for k, v in config["checkpoint"].items() if k != 'dirpath'}
        
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=str(fold_checkpoint_dir),
            **checkpoint_conf,
            filename="{epoch}-{val_loss:.2f}"
        )

        # EarlyStoppingコールバックを初期化
        callbacks = [checkpoint_callback]
        if "early_stopping" in config:
            early_stopping_callback = EarlyStopping(**config["early_stopping"])
            callbacks.append(early_stopping_callback)

        # PyTorch LightningのTrainerを初期化
        # Trainerに渡す設定と、Solver/他のロジックで使う設定を分離
        trainer_config = config["trainer"].copy()
        trainer_config.pop("metrics_average_mode", None)
        trainer_config.pop("xgboost_params", None)
        trainer_config.pop("decode_method", None)
        trainer_config.pop("beam_width", None) # beam_widthもTrainerからは削除

        trainer = pl.Trainer(
            callbacks=callbacks, # コールバック（チェックポイント保存など）を設定
            logger=logger, # ロガーのリストを設定
            **trainer_config # 不要な引数を削除したconfigを渡す
        )

        # --- 2d. この分割での訓練とテスト ---
        print(f"--- 分割 {fold + 1} の訓練を開始します ---")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

        print(f"--- 分割 {fold + 1} のテストを実行します ---")

        # テスト前に、posteriogram保存のための情報をSolverに渡す
        model.class_mapping = class_mapping
        # trainer.log_dir は loggerリストの最初のロガーのパスを指す
        model.posteriogram_dir = Path(trainer.log_dir) / "posteriograms"

        # 最良のモデル（val_lossが最小）を使ってテストを実行
        trainer.test(model, dataloaders=test_loader, ckpt_path='best')

        # --- 2e. この分割の結果を収集 ---
        y_true = model.test_labels.cpu().numpy()
        y_pred = model.test_preds.cpu().numpy()
        
        # is_loocv で test_df は常に1行
        test_subject_id = test_df.iloc[0]['subject_id']

        if is_loocv and len(y_true) > 0:
            all_results.append({
                'true': y_true[0],
                'pred': y_pred[0],
                'subject_id': test_subject_id
            })

        # WERの結果を収集 (これはSolver内で計算されるため、ここで収集)
        if hasattr(model, 'wer_results') and not model.wer_results.empty:
            # fold番号を追加
            model.wer_results['fold'] = fold + 1
            all_fold_wer_results.append(model.wer_results)


    # --- 3. 全体の結果を集計・保存 ---
    if is_loocv and all_results:
        print("\n--- 結果集計 ---")
        results_df = pd.DataFrame(all_results)
        output_dir = Path(config["output"]["cm_output_dir"])
        output_dir.mkdir(exist_ok=True, parents=True)
        num_classes = config['model']['num_classes']

        # --- 話者ごとの結果 ---
        subject_accuracies = {}
        for subject_id, group in results_df.groupby('subject_id'):
            # 設定ファイル名から話者名を取得 (例: cv_kobayashi.yaml -> kobayashi)
            config_path = Path(args.config[subject_id])
            subject_name = config_path.stem.replace('cv_', '')

            accuracy = (group['true'] == group['pred']).mean()
            subject_accuracies[subject_name] = accuracy
            print(f"Accuracy for subject {subject_id} ({subject_name}): {accuracy:.4f}")

            # 話者ごとの混同行列を計算・保存
            y_true_subj = group['true']
            y_pred_subj = group['pred']
            cm_subj = confusion_matrix(y_true_subj, y_pred_subj, labels=np.arange(num_classes))
            cm_subj_path = output_dir / f"cm_subject_{subject_id}_{subject_name}.csv"
            pd.DataFrame(cm_subj).to_csv(cm_subj_path, index=False, header=False)
            print(f"  - 話者ごとの混同行列を保存: {cm_subj_path}")

        # --- 全体の結果 ---
        total_accuracy = (results_df['true'] == results_df['pred']).mean()
        print("-" * 20)
        print(f"Total Accuracy: {total_accuracy:.4f}")

        # 全体の混同行列を計算・保存
        cm_total = confusion_matrix(results_df['true'], results_df['pred'], labels=np.arange(num_classes))
        cm_total_path = output_dir / "cm_total.csv"
        pd.DataFrame(cm_total).to_csv(cm_total_path, index=False, header=False)
        print(f"  - 全体の混同行列を保存: {cm_total_path}")


    # --- 4. 完了メッセージ ---
    print("\n===== 全てのフォールドの学習とテストが完了しました =====")
    cm_output_dir = Path(config["output"]["cm_output_dir"])
    print(f"各フォールドの混同行列が '{cm_output_dir}' に保存されました。")
    print("\n次のステップ:")
    print(f"python aggregate_results.py {cm_output_dir} --config {args.config}")
    print("上記のコマンドを実行して、最終的な統計レポートと指標を生成してください。")

    # WERの結果が収集されていれば、ここで集計・保存する
    if all_fold_wer_results:
        output_dir = Path(config["logger"]["save_dir"]) / config["logger"]["name"] / "cv_results"
        output_dir.mkdir(exist_ok=True, parents=True)
        full_wer_df = pd.concat(all_fold_wer_results, ignore_index=True)
        wer_csv_path = output_dir / "cross_validation_wer_detailed.csv"
        full_wer_df.to_csv(wer_csv_path, index=False, float_format='%.4f')
        print(f"\n詳細なWER結果を保存しました: {wer_csv_path}")


# このスクリプトが直接実行された場合にのみ以下のコードが実行されます
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, nargs='+', help="YAML形式の設定ファイル (複数指定可能)")
    parser.add_argument("--checkpoint", type=str, default=None, help="モデルのチェックポイント（CVでは非推奨）")
    # --- 実験用の引数を追加 ---
    parser.add_argument("--num-folds", type=int, default=None, help="CVの分割数 (configを上書き)")
    parser.add_argument("--cm-output-dir", type=str, required=True, help="混同行列の出力先ディレクトリ")
    parser.add_argument("--augment-flip", type=str, choices=['true', 'false'], default=None, help="左右反転Augmentationのオンオフ (configを上書き)")
    parser.add_argument("--augment-rotate", type=str, choices=['true', 'false'], default=None, help="回転Augmentationのオンオフ (configを上書き)")
    parser.add_argument("--augment-noise", type=str, choices=['true', 'false'], default=None, help="ノイズAugmentationのオンオフ (configを上書き)")
    args = parser.parse_args()

    # 高速な行列計算のためのPyTorch設定
    torch.set_float32_matmul_precision("high")

    # 最初の設定ファイルをベースとして読み込む
    with open(args.config[0], "r") as yf:
        config = yaml.safe_load(yf)

    # メイン関数を実行
    main(config, args, checkpoint_path=args.checkpoint)

