

import argparse
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from sklearn.metrics import confusion_matrix
from pathlib import Path
import os
import copy

# ---------------------------------
# train.pyと完全に同じデータセット処理をインポート
from src.dataset import SignDataset, BucketBatchSampler, data_processing
from src.solver import Solver
# ---------------------------------

def get_feature_dim(feature_config: dict) -> int:
    """
    config.yamlのfeaturesセクションに基づいて、特徴量の次元数を計算する関数。
    """
    BASE_COORD_DIM = 21 * 3 * 2
    EXISTING_PIPELINE_DIM = 193 * 2
    normalize_mode = feature_config.get('normalize_mode', 'normalize_landmarks')
    paper_conf = feature_config.get('paper_features', {})
    use_paper_speed = paper_conf.get('speed', False)
    use_paper_anthropometric = paper_conf.get('anthropometric', False)
    is_paper_mode = normalize_mode in ['current_wrist', 'first_wrist'] or use_paper_speed or use_paper_anthropometric
    if is_paper_mode:
        dim = BASE_COORD_DIM
        if use_paper_speed:
            dim += BASE_COORD_DIM
        if use_paper_anthropometric:
            dim += 210 * 2
        return dim
    else:
        return EXISTING_PIPELINE_DIM

def main(args):
    """
    メインの学習・評価処理。
    """
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config.get('seed', 42))

    # configに基づいて特徴量の次元数を計算し、config辞書を更新
    feature_dim = get_feature_dim(config.get('features', {}))
    config['model']['input_dim'] = feature_dim
    print(f"--- configに基づき、特徴量の次元数を {feature_dim} と計算しました ---")

    cs_config = config.get('cross_subject')
    if not cs_config or 'subjects' not in cs_config:
        raise ValueError("`cross_subject.subjects` section not found in config file.")

    # --- 全話者のデータセットを事前にロード ---
    # メタデータのみを先に読み込む
    all_subject_metadata = [
        pd.read_csv(s['metadata_path']) for s in cs_config['subjects']
    ]
    
    num_folds = len(all_subject_metadata)

    # --- 交差検証ループ ---
    for i in range(num_folds):
        print(f"\n{'='*20} FOLD {i+1}/{num_folds} {'='*20}")
        
        # --- データセットの準備 ---
        test_df = all_subject_metadata[i]
        train_val_dfs = [df for j, df in enumerate(all_subject_metadata) if i != j]
        train_val_df = pd.concat(train_val_dfs, ignore_index=True)

        # 訓練・検証データをさらに訓練用と検証用に分割
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=config['data']['validation_split_ratio'],
            random_state=config.get('seed', 42),
            stratify=train_val_df['class_label']
        )
        
        # --- データセットとデータローダーの作成 (train.pyと共通化) ---
        eval_config = copy.deepcopy(config)
        eval_config['data']['augmentation'] = {}

        train_dataset = SignDataset(
            metadata_df=train_df,
            data_base_dir=cs_config['subjects'][0]['source_landmark_dir'], # base_dirは共通と仮定
            config=config
        )
        val_dataset = SignDataset(
            metadata_df=val_df,
            data_base_dir=cs_config['subjects'][0]['source_landmark_dir'],
            config=eval_config
        )
        test_dataset = SignDataset(
            metadata_df=test_df,
            data_base_dir=cs_config['subjects'][0]['source_landmark_dir'],
            config=eval_config
        )

        train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True, collate_fn=data_processing, num_workers=os.cpu_count())
        val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False, collate_fn=data_processing, num_workers=os.cpu_count())
        test_loader = DataLoader(test_dataset, batch_size=config['data']['batch_size'], shuffle=False, collate_fn=data_processing, num_workers=os.cpu_count())
        
        print(f"Training data subjects: {[j for j in range(num_folds) if i != j]}")
        print(f"Testing data subject: {i}")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

        # --- モデル、Trainerの初期化 ---
        if 'scheduler' in config:
            config['scheduler']['total_steps'] = len(train_loader) * config['trainer']['max_epochs']
        
        solver = Solver(config)
        
        logger = TensorBoardLogger(save_dir=config['logger']['save_dir'], name=f"{config['logger']['name']}_cs", version=f"fold_{i}")
        checkpoint_callback = ModelCheckpoint(dirpath=Path(logger.log_dir) / "checkpoints", monitor=config['checkpoint']['monitor'], mode=config['checkpoint']['mode'])
        
        callbacks = [checkpoint_callback, TQDMProgressBar(refresh_rate=10)]
        if "early_stopping" in config:
            early_stopping_callback = EarlyStopping(**config["early_stopping"])
            callbacks.append(early_stopping_callback)

        trainer_config = config["trainer"].copy()
        trainer_config.pop("metrics_average_mode", None)
        trainer_config.pop("xgboost_params", None)
        trainer_config.pop("decode_method", None)
        trainer_config.pop("beam_width", None)

        trainer = pl.Trainer(
            max_epochs=config['trainer']['max_epochs'],
            accelerator=config['trainer']['accelerator'],
            logger=logger,
            callbacks=callbacks,
            log_every_n_steps=config['trainer']['log_every_n_steps'],
            enable_progress_bar=True,
            **trainer_config
        )

        # --- 学習 & テスト ---
        trainer.fit(solver, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(dataloaders=test_loader, ckpt_path='best')
        
        # --- このフォールドの混同行列を計算・保存 ---
        y_true = solver.test_labels.cpu().numpy()
        y_pred = solver.test_preds.cpu().numpy()

        num_classes = config['model']['num_classes']
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
        
        output_dir = Path(cs_config.get("cm_output_dir", "results/confusion_matrices_cs"))
        output_dir.mkdir(exist_ok=True, parents=True)
        fold_results_path = output_dir / f"cs_fold_{i}_cm.csv"

        pd.DataFrame(cm).to_csv(fold_results_path, index=False, header=False)
        print(f"混同行列を保存しました: {fold_results_path}")

    # --- 全てのフォールドが完了 ---
    print("\n===== 全てのフォールドの学習とテストが完了しました =====")
    cm_output_dir = Path(cs_config.get("cm_output_dir", "results/confusion_matrices_cs"))
    print(f"各フォールドの混同行列が '{cm_output_dir}' に保存されました。")
    print("\n次のステップ:")
    print(f"python aggregate_results.py {cm_output_dir} --config {args.config}")
    print("上記のコマンドを実行して、最終的な統計レポートと指標を生成してください。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cross-subject validation training script.")
    parser.add_argument('-c', '--config', type=str, default='config.yaml',
                        help='Path to the configuration file.')
    args = parser.parse_args()
    main(args)
