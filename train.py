import yaml
import random
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

# Import our custom modules
from src.dataset import SignDataset, BucketBatchSampler, data_processing
from src.solver import Solver

# --- Reproducibility ---
def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """Seeds worker processes for DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(config: dict, checkpoint_path: str | None = None):
    """
    Main K-Fold Cross-Validation training pipeline.
    """
    # --- 0. Set Seed ---
    if "seed" in config:
        set_seed(config["seed"])
        print(f"--- Seed set to {config['seed']} for reproducibility ---")
    
    g = torch.Generator()
    if "seed" in config:
        g.manual_seed(config["seed"])

    # --- 1. Load Full Dataset ---
    print("--- Loading Full Dataset ---")
    data_config = config['data']
    metadata_df = pd.read_csv(data_config['metadata_path'])
    
    # Create a mapping from class index to a readable name if available
    # This assumes the CSV has 'class_label' (int) and 'class_name' (str)
    class_mapping = None
    if 'class_name' in metadata_df.columns:
        class_mapping = pd.Series(metadata_df['class_name'].values, index=metadata_df['class_label']).to_dict()
        print(f"Found {len(class_mapping)} classes.")


    # --- 2. K-Fold Cross-Validation Setup ---
    num_folds = data_config.get('num_folds', 5)
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=config.get('seed', 42))

    all_fold_metrics = []
    all_fold_reports = []

    for fold, (train_val_indices, test_indices) in enumerate(skf.split(metadata_df, metadata_df['class_label'])):
        print(f"\n===== FOLD {fold + 1} / {num_folds} =====")

        # --- 2a. Split data for this fold ---
        train_val_df = metadata_df.iloc[train_val_indices].reset_index(drop=True)
        test_df = metadata_df.iloc[test_indices].reset_index(drop=True)

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=data_config['validation_split_ratio'],
            random_state=config.get('seed', 42),
            stratify=train_val_df['class_label']
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # --- 2b. Create Datasets and DataLoaders for this fold ---
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
            sort_by_length=False,
            augment_flip=False, augment_rotate=False, augment_noise=False
        )
        test_dataset = SignDataset(
            metadata_df=test_df,
            data_base_dir=data_config['source_landmark_dir'],
            sort_by_length=False,
            augment_flip=False, augment_rotate=False, augment_noise=False
        )

        if data_config.get('use_bucketing', False):
            train_sampler = BucketBatchSampler(train_dataset, batch_size=data_config['batch_size'])
            train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=data_processing, num_workers=4, worker_init_fn=seed_worker)
        else:
            train_loader = DataLoader(train_dataset, batch_size=data_config['batch_size'], shuffle=True, collate_fn=data_processing, num_workers=4, worker_init_fn=seed_worker, generator=g)

        valid_loader = DataLoader(val_dataset, batch_size=data_config['batch_size'], shuffle=False, collate_fn=data_processing, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=data_config['batch_size'], shuffle=False, collate_fn=data_processing, num_workers=4)

        # --- 2c. Initialize Model and Trainer for this fold ---
        if 'scheduler' in config and 'total_steps' not in config['scheduler']:
            total_steps = (len(train_dataset) // data_config['batch_size']) * config['trainer']['max_epochs']
            config['scheduler']['total_steps'] = total_steps
            print(f"Scheduler total_steps calculated: {total_steps}")

        model = Solver(config)

        # Define a unique directory for this fold's logs and checkpoints
        fold_log_dir = Path(config["logger"]["save_dir"]) / config["logger"]["name"]
        logger = TensorBoardLogger(save_dir=str(fold_log_dir.parent), name=config["logger"]["name"], version=f"fold_{fold}")
        
        checkpoint_conf = {k: v for k, v in config["checkpoint"].items() if k != 'dirpath'}
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=str(Path(logger.log_dir) / "checkpoints"),
            **checkpoint_conf,
            filename=f"{{epoch}}-{{val_loss:.2f}}"
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
        test_results = trainer.test(model, dataloaders=test_loader, ckpt_path='best')
        all_fold_metrics.append(test_results[0])

        # --- 2e. Generate and Store Detailed Report for this fold ---
        y_true = model.test_labels.cpu().numpy()
        y_pred = model.test_preds.cpu().numpy()
        
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Add class names if available
        if class_mapping:
            report_df['class_name'] = report_df.index.map(lambda x: class_mapping.get(int(x), x))

        report_df['fold'] = fold
        all_fold_reports.append(report_df)
        print(f"Fold {fold + 1} Detailed Report generated.")


    # --- 3. Aggregate, Save, and Print Final Results ---
    print("\n===== CROSS-VALIDATION FINAL RESULTS ======")
    
    # Save detailed per-fold and average results to CSV
    if all_fold_reports:
        # Combine all fold reports
        full_report_df = pd.concat(all_fold_reports)
        
        # Calculate the mean for each metric across all folds
        mean_report_df = full_report_df.groupby(full_report_df.index).mean(numeric_only=True)
        mean_report_df['fold'] = 'mean'
        if class_mapping:
            mean_report_df['class_name'] = mean_report_df.index.map(lambda x: class_mapping.get(int(x), x))

        # Combine into a final report
        final_report_with_avg = pd.concat([full_report_df, mean_report_df])
        
        # Define output path
        output_dir = Path(config["logger"]["save_dir"]) / config["logger"]["name"] / "cv_results"
        output_dir.mkdir(exist_ok=True, parents=True)
        csv_path = output_dir / "cross_validation_detailed_report.csv"

        # Save to CSV
        final_report_with_avg.to_csv(csv_path)
        print(f"\nDetailed cross-validation report saved to: {csv_path}")

    # Print summary of macro averages
    avg_metrics = pd.DataFrame(all_fold_metrics).mean().to_dict()
    print("\n--- Average Metrics Across Folds ---")
    print(f"Average Test Accuracy: {avg_metrics.get('test_acc_epoch', 0):.4f}")
    print(f"Average Test F1-Score (Macro): {avg_metrics.get('test_f1_epoch', 0):.4f}")
    print(f"Average Test Precision (Macro): {avg_metrics.get('test_precision_epoch', 0):.4f}")
    print(f"Average Test Recall (Macro): {avg_metrics.get('test_recall_epoch', 0):.4f}")
    print("\nIndividual fold metrics and checkpoints logged in the respective 'fold_X' directories.")
    # --- 3. Aggregate and Save/Print Final Results ---
    print("\n===== CROSS-VALIDATION FINAL RESULTS ======")
    
    if not all_fold_metrics:
        print("No test results found to generate a report.")
        return

    results_df = pd.DataFrame(all_fold_metrics)
    results_df.loc['average'] = results_df.mean()

    # Rename columns for clarity
    results_df = results_df.rename(columns={
        'test_acc_epoch': 'test_accuracy',
        'test_f1_epoch': 'test_f1',
        'test_precision_epoch': 'test_precision',
        'test_recall_epoch': 'test_recall'
    })

    # Add fold column
    results_df.index.name = 'fold'
    results_df = results_df.reset_index()
    results_df['fold'] = results_df['fold'].apply(lambda x: str(x + 1) if isinstance(x, int) else x)

    # --- 4. Save or Print Results ---
    output_path = config.get('output', {}).get('nn_results_path')
    if output_path:
        file_ext = Path(output_path).suffix
        try:
            if file_ext == '.csv':
                results_df.to_csv(output_path, index=False, float_format='%.4f')
                print(f"\nResults successfully saved to {output_path}")
            elif file_ext == '.md':
                results_df.to_markdown(output_path, index=False, floatfmt='.4f')
                print(f"\nResults successfully saved to {output_path}")
            else:
                print(f"\nUnsupported output format '{file_ext}'. Printing to console instead.")
                print(results_df.to_string(float_format='%.4f', index=False))
        except Exception as e:
            print(f"\nError saving results to {output_path}: {e}")
            print("Printing to console instead:")
            print(results_df.to_string(float_format='%.4f', index=False))
    else:
        print(results_df.to_string(float_format='%.4f', index=False))
    
    print("\nIndividual fold metrics are also logged in TensorBoard.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="YAML形式の設定ファイル")
    parser.add_argument("--checkpoint", type=str, default=None, help="モデルのチェックポイント（CVでは非推奨）")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    with open(args.config, "r") as yf:
        config = yaml.safe_load(yf)

    main(config, checkpoint_path=args.checkpoint)
