

import argparse
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pathlib import Path

from src.solver import Solver

class NpzLandmarkDataset(Dataset):
    """
    メタデータCSVとNPZランドマークファイルを読み込むためのPyTorch Datasetクラス。
    """
    def __init__(self, metadata_path: str, landmark_dir: str):
        self.metadata = pd.read_csv(metadata_path)
        self.landmark_dir = Path(landmark_dir)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        npz_path = self.landmark_dir / row['npz_path']
        
        with np.load(npz_path) as data:
            landmarks = data['landmarks'].astype(np.float32)

        label = int(row['class_label'])
        
        # 特徴量とラベルをTensorに変換
        features = torch.from_numpy(landmarks)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # Solverが (features, lengths, labels) のタプルを期待するため、
        # lengthsもダミーで追加
        lengths = torch.tensor(features.shape[0], dtype=torch.long)

        return features, lengths, label_tensor

def collate_fn(batch):
    """
    異なる長さのシーケンスをパディングするためのCollate関数。
    """
    features, lengths, labels = zip(*batch)
    
    # 特徴量をパディング
    padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    
    # ラベルと長さをスタック
    labels = torch.stack(labels)
    lengths = torch.stack(lengths)
    
    return padded_features, lengths, labels

def main(args):
    """
    メインの学習・評価処理。
    """
    # --- 設定ファイルの読み込み ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config.get('seed', 42))

    cs_config = config.get('cross_subject')
    if not cs_config:
        raise ValueError("`cross_subject` section not found in config file.")

    # --- データセットとデータローダーの作成 ---
    # 学習・検証用データ
    train_val_dataset = NpzLandmarkDataset(
        metadata_path=cs_config['train_metadata_path'],
        landmark_dir=cs_config['train_source_landmark_dir']
    )
    
    # 学習データと検証データに分割
    val_ratio = config['data'].get('validation_split_ratio', 0.1)
    n_train_val = len(train_val_dataset)
    n_val = int(n_train_val * val_ratio)
    n_train = n_train_val - n_val
    train_dataset, val_dataset = random_split(train_val_dataset, [n_train, n_val])

    # 評価（テスト）用データ
    test_dataset = NpzLandmarkDataset(
        metadata_path=cs_config['eval_metadata_path'],
        landmark_dir=cs_config['eval_source_landmark_dir']
    )

    batch_size = config['data']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=os.cpu_count())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=os.cpu_count())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=os.cpu_count())
    
    print(f"Training data: {len(train_dataset)} samples")
    print(f"Validation data: {len(val_dataset)} samples")
    print(f"Testing data: {len(test_dataset)} samples")

    # --- モデル(Solver)の初期化 ---
    # スケジューラのtotal_stepsを計算
    if 'scheduler' in config:
        config['scheduler']['total_steps'] = len(train_loader) * config['trainer']['max_epochs']
        
    solver = Solver(config)

    # --- LoggerとCallbackの初期化 ---
    logger = TensorBoardLogger(
        save_dir=config['logger']['save_dir'],
        name=f"{config['logger']['name']}_cross_subject"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config['checkpoint']['dirpath']}/cross_subject/",
        monitor=config['checkpoint']['monitor'],
        save_top_k=config['checkpoint']['save_top_k'],
        mode=config['checkpoint']['mode'],
    )
    progress_bar = TQDMProgressBar()

    # --- Trainerの初期化と実行 ---
    trainer = pl.Trainer(
        max_epochs=config['trainer']['max_epochs'],
        accelerator=config['trainer']['accelerator'],
        logger=logger,
        callbacks=[checkpoint_callback, progress_bar],
        log_every_n_steps=config['trainer']['log_every_n_steps']
    )

    # --- 学習 ---
    print("\n--- Starting Training ---")
    trainer.fit(solver, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # --- 評価 ---
    print("\n--- Starting Testing on Unseen Speaker ---")
    # best_model_path=Noneにすると、fit後の最新のモデルでテストする
    test_results = trainer.test(dataloaders=test_loader, ckpt_path='best')

    print("\n--- Cross-Subject Validation Results ---")
    print(test_results)


if __name__ == '__main__':
    import os
    parser = argparse.ArgumentParser(description="Cross-subject validation training script.")
    parser.add_argument('-c', '--config', type=str, default='config.yaml',
                        help='Path to the configuration file.')
    args = parser.parse_args()
    main(args)

