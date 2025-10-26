import yaml
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

# Import our custom modules
from src.dataset import SignDataset, BucketBatchSampler, data_processing
from src.solver import Solver

def main(config: dict, checkpoint_path: str | None = None):
    """
    Main K-Fold Cross-Validation training pipeline.
    """
    # --- 1. Load Full Dataset ---
    print("--- Loading Full Dataset ---")
    data_config = config['data']
    metadata_df = pd.read_csv(data_config['metadata_path'])

    # --- 2. K-Fold Cross-Validation Setup ---
    num_folds = data_config.get('num_folds', 5)
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    all_fold_metrics = []

    for fold, (train_val_indices, test_indices) in enumerate(skf.split(metadata_df, metadata_df['class_label'])):
        print(f"\n===== FOLD {fold + 1} / {num_folds} =====")

        # --- 2a. Split data for this fold ---
        train_val_df = metadata_df.iloc[train_val_indices].reset_index(drop=True)
        test_df = metadata_df.iloc[test_indices].reset_index(drop=True)

        # Split train_val_df further into train and validation
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=data_config['validation_split_ratio'],
            random_state=42,
            stratify=train_val_df['class_label']
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # --- 2b. Create Datasets and DataLoaders for this fold ---
        # Augmentation flags from config
        aug_config = data_config.get('augmentation', {})

        train_dataset = SignDataset(
            metadata_df=train_df,
            data_base_dir=data_config['source_landmark_dir'],
            sort_by_length=data_config.get('use_bucketing', False),
            augment_flip=aug_config.get('augment_flip', False),
            augment_rotate=aug_config.get('augment_rotate', False),
            augment_noise=aug_config.get('augment_noise', False),
            flip_prob=aug_config.get('flip_prob', 0.5)
        )
        val_dataset = SignDataset(
            metadata_df=val_df,
            data_base_dir=data_config['source_landmark_dir'],
            sort_by_length=False, # No sorting for validation
            augment_flip=False, augment_rotate=False, augment_noise=False # No augmentation for validation
        )
        test_dataset = SignDataset(
            metadata_df=test_df,
            data_base_dir=data_config['source_landmark_dir'],
            sort_by_length=False, # No sorting for test
            augment_flip=False, augment_rotate=False, augment_noise=False # No augmentation for test
        )

        if data_config.get('use_bucketing', False):
            train_sampler = BucketBatchSampler(train_dataset, batch_size=data_config['batch_size'])
            train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=data_processing, num_workers=4)
        else:
            train_loader = DataLoader(train_dataset, batch_size=data_config['batch_size'], shuffle=True, collate_fn=data_processing, num_workers=4)

        valid_loader = DataLoader(val_dataset, batch_size=data_config['batch_size'], shuffle=False, collate_fn=data_processing, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=data_config['batch_size'], shuffle=False, collate_fn=data_processing, num_workers=4)

        # --- 2c. Initialize Model and Trainer for this fold ---
        # Inject total_steps for this fold's training into scheduler config
        if 'scheduler' in config and 'total_steps' not in config['scheduler']:
            total_steps = (len(train_dataset) // data_config['batch_size']) * config['trainer']['max_epochs']
            config['scheduler']['total_steps'] = total_steps
            print(f"Scheduler total_steps calculated: {total_steps}")

        # Re-initialize model for each fold to ensure fresh weights
        model = Solver(config)

        # Setup logger and checkpointing for this fold
        logger = TensorBoardLogger(**config["logger"], version=f"fold_{fold}")
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            **config["checkpoint"],
            filename=f"{{epoch}}-{{val_loss:.2f}}-fold_{fold}"
        )

        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            logger=logger,
            **config["trainer"]
        )

        # --- 2d. Train and Test this fold ---
        print(f"--- Training Fold {fold + 1} ---")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

        print(f"--- Testing Fold {fold + 1} ---")
        # trainer.test() returns a list of dictionaries
        test_results = trainer.test(model, dataloaders=test_loader, ckpt_path='best')
        all_fold_metrics.append(test_results[0])

    # --- 3. Aggregate and Print Final Results ---
    print("\n===== CROSS-VALIDATION FINAL RESULTS =====")
    avg_metrics = pd.DataFrame(all_fold_metrics).mean().to_dict()

    print(f"Average Test Accuracy: {avg_metrics.get('test_acc_epoch', 0):.4f}")
    print(f"Average Test F1-Score: {avg_metrics.get('test_f1_epoch', 0):.4f}")
    print(f"Average Test Precision: {avg_metrics.get('test_precision_epoch', 0):.4f}")
    print(f"Average Test Recall: {avg_metrics.get('test_recall_epoch', 0):.4f}")
    print("\nIndividual fold metrics logged in TensorBoard.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="YAML形式の設定ファイル")
    parser.add_argument("--checkpoint", type=str, default=None, help="モデルのチェックポイント（CVでは非推奨）")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    with open(args.config, "r") as yf:
        config = yaml.safe_load(yf)

    main(config, checkpoint_path=args.checkpoint)