import argparse
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pathlib import Path
from collections import defaultdict
import os

from src.solver import Solver

class NpzLandmarkDataset(Dataset):
    """
    メタデータCSVとNPZランドマークファイルを読み込むためのPyTorch Datasetクラス。
    """
    def __init__(self, metadata_path: str, landmark_dir: str):
        self.metadata = pd.read_csv(metadata_path)
        self.landmark_dir = Path(landmark_dir)
        # class_labelが0から始まっていない場合や、連続していない場合に対応
        self.class_labels = sorted(self.metadata['class_label'].unique())
        self.class_map = {label: i for i, label in enumerate(self.class_labels)}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        npz_path = self.landmark_dir / row['npz_path']
        
        with np.load(npz_path) as data:
            landmarks = data['landmarks'].astype(np.float32)

        # class_labelを0からの連番にマッピング
        label = self.class_map[row['class_label']]
        
        features = torch.from_numpy(landmarks)
        label_tensor = torch.tensor(label, dtype=torch.long)
        lengths = torch.tensor(features.shape[0], dtype=torch.long)

        return features, lengths, label_tensor

def collate_fn(batch):
    """
    異なる長さのシーケンスをパディングするためのCollate関数。
    """
    features, lengths, labels = zip(*batch)
    padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels)
    lengths = torch.stack(lengths)
    return padded_features, lengths, labels

def main(args):
    """
    メインの学習・評価処理。
    """
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config.get('seed', 42))

    cs_config = config.get('cross_subject')
    if not cs_config or 'subjects' not in cs_config:
        raise ValueError("`cross_subject.subjects` section not found in config file.")

    # --- 全話者のデータセットを事前にロード ---
    all_subject_datasets = [
        NpzLandmarkDataset(
            metadata_path=s['metadata_path'],
            landmark_dir=s['source_landmark_dir']
        ) for s in cs_config['subjects']
    ]
    
    num_folds = len(all_subject_datasets)
    fold_metrics = defaultdict(list)

    # --- 交差検証ループ ---
    for i in range(num_folds):
        print(f"\n{'='*20} FOLD {i+1}/{num_folds} {'='*20}")
        
        # --- データセットの準備 ---
        test_dataset = all_subject_datasets[i]
        train_val_datasets = [ds for j, ds in enumerate(all_subject_datasets) if i != j]
        train_val_dataset = ConcatDataset(train_val_datasets)

        val_ratio = config['data'].get('validation_split_ratio', 0.1)
        n_train_val = len(train_val_dataset)
        n_val = int(n_train_val * val_ratio)
        n_train = n_train_val - n_val
        train_dataset, val_dataset = random_split(train_val_dataset, [n_train, n_val])

        batch_size = config['data']['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=os.cpu_count())
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=os.cpu_count())
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=os.cpu_count())
        
        print(f"Training data subjects: {[j for j in range(num_folds) if i != j]}")
        print(f"Testing data subject: {i}")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

        # --- モデル、Trainerの初期化 ---
        if 'scheduler' in config:
            config['scheduler']['total_steps'] = len(train_loader) * config['trainer']['max_epochs']
        
        solver = Solver(config)
        
        # 毎回新しいログディレクトリとチェックポイントを作成
        logger = TensorBoardLogger(save_dir=config['logger']['save_dir'], name=f"{config['logger']['name']}_cs_fold_{i+1}")
        checkpoint_callback = ModelCheckpoint(dirpath=f"{config['checkpoint']['dirpath']}/cs_fold_{i+1}/", monitor=config['checkpoint']['monitor'], mode=config['checkpoint']['mode'])
        
        trainer = pl.Trainer(
            max_epochs=config['trainer']['max_epochs'],
            accelerator=config['trainer']['accelerator'],
            logger=logger,
            callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=10)],
            log_every_n_steps=config['trainer']['log_every_n_steps'],
            enable_progress_bar=True
        )

        # --- 学習 & テスト ---
        trainer.fit(solver, train_dataloaders=train_loader, val_dataloaders=val_loader)
        test_results = trainer.test(dataloaders=test_loader, ckpt_path='best')
        
        # --- メトリクスの収集 ---
        # test_results[0] には {'test_loss': ..., 'test_acc_ctc': ...} のような辞書が入っている
        results_dict = test_results[0]
        for key, value in results_dict.items():
            fold_metrics[key].append(value)

    # --- 全フォールドのメトリクスを集計 ---
    stats_rows = []
    num_classes = config['model']['num_classes']
    
    # torchmetricsが出力するメトリクス名に合わせて集計
    # 例: 'test_acc_ctc', 'test_f1_macro', 'test_precision_macro' など
    for metric_name, values in fold_metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        
        # メトリック名から指標とクラスを分離 (例: test_acc_ctc_class_5 -> (test_acc_ctc, 5))
        parts = metric_name.split('_')
        class_label = f"class_{parts[-1]}" if parts[-1].isdigit() else "overall"
        base_metric_name = "_".join(parts[:-1]) if class_label != "overall" else metric_name

        stats_rows.append({
            "metric": base_metric_name,
            "class_label": class_label,
            "mean": mean,
            "std": std
        })

    stats_df = pd.DataFrame(stats_rows)
    
    # --- 結果の保存 ---
    output_path = cs_config.get("output_stats_path", "results_cross_subject_stats.csv")
    stats_df.to_csv(output_path, index=False)

    print(f"\n--- Aggregated Cross-Subject Validation Stats ---")
    print(stats_df)
    print(f"\nStatistics saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cross-subject validation training script.")
    parser.add_argument('-c', '--config', type=str, default='config.yaml',
                        help='Path to the configuration file.')
    args = parser.parse_args()
    main(args)