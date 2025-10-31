import yaml
import random
import copy
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold, train_test_split, LeaveOneOut
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

def get_feature_dim(feature_config: dict) -> int:
    """
    config.yamlのfeaturesセクションに基づいて、特徴量の次元数を計算する。
    """
    # ベースとなる座標の次元数 (x, y, z * 21ランドマーク * 2手)
    BASE_COORD_DIM = 21 * 3 * 2
    
    # 既存の特徴量パイプラインの次元数
    # calculate_featuresは [pos(63*2) + vel(63*2) + acc(63*2) + geo(4*2)] だが、
    # モデルは片手ずつ処理するので、入力は pos(63)+vel(63)+acc(63)+geo(4) = 193次元。両手で386次元。
    EXISTING_PIPELINE_DIM = 193 * 2

    normalize_mode = feature_config.get('normalize_mode', 'normalize_landmarks')
    paper_conf = feature_config.get('paper_features', {})
    use_paper_speed = paper_conf.get('speed', False)
    use_paper_anthropometric = paper_conf.get('anthropometric', False)

    is_paper_mode = normalize_mode in ['current_wrist', 'first_wrist'] or use_paper_speed or use_paper_anthropometric

    if is_paper_mode:
        # 論文ベースのパイプライン
        dim = BASE_COORD_DIM # 座標は常に出力される
        if use_paper_speed:
            dim += BASE_COORD_DIM # 速度特徴量
        if use_paper_anthropometric:
            # 21 C 2 = 210ペア * 2手
            dim += 210 * 2
        return dim
    else:
        # 既存のパイプライン
        return EXISTING_PIPELINE_DIM

def main(config: dict, checkpoint_path: str | None = None):
    """
    Main K-Fold Cross-Validation training pipeline.
    """
    # --- 0. Set Seed & Calculate Feature Dimension ---
    if "seed" in config:
        set_seed(config["seed"])
        print(f"--- Seed set to {config['seed']} for reproducibility ---")

    # Calculate feature dimension from config and update the config dict
    feature_dim = get_feature_dim(config.get('features', {}))
    config['model']['input_dim'] = feature_dim
    print(f"--- Feature dimension calculated based on config: {feature_dim} ---")
    
    g = torch.Generator()
    if "seed" in config:
        g.manual_seed(config["seed"])

    # --- 1. Load Full Dataset ---
    print("--- Loading Full Dataset ---")
    data_config = config['data']
    metadata_df = pd.read_csv(data_config['metadata_path'])
    
    # Create a mapping from class index to a readable name if available
    class_mapping = None
    if 'class_name' in metadata_df.columns:
        class_mapping = pd.Series(metadata_df['class_name'].values, index=metadata_df['class_label']).to_dict()
        print(f"Found {len(class_mapping)} classes.")


    # --- 2. Cross-Validation Setup ---
    num_folds = data_config.get('num_folds', 5)

    if num_folds > 1:
        print(f"--- Setting up Stratified {num_folds}-Fold Cross-Validation ---")
        cv_splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=config.get('seed', 42))
        cv_iterator = cv_splitter.split(metadata_df, metadata_df['class_label'])
    else:
        print("--- Setting up Leave-One-Out Cross-Validation ---")
        cv_splitter = LeaveOneOut()
        num_folds = cv_splitter.get_n_splits(metadata_df) # Update num_folds for reporting
        cv_iterator = cv_splitter.split(metadata_df)

    all_fold_metrics = []
    # Initialize containers for results based on CV strategy
    if num_folds > 1:
        all_fold_reports = []  # For k-fold
    else:
        all_labels = []  # For LOOCV
        all_preds = []   # For LOOCV

    for fold, (train_val_indices, test_indices) in enumerate(cv_iterator):
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
        # For validation and test, create a config with augmentations disabled
        eval_config = copy.deepcopy(config)
        eval_config['data']['augmentation'] = {}

        train_dataset = SignDataset(
            metadata_df=train_df,
            data_base_dir=data_config['source_landmark_dir'],
            config=config,
            sort_by_length=data_config.get('use_bucketing', False)
        )
        val_dataset = SignDataset(
            metadata_df=val_df,
            data_base_dir=data_config['source_landmark_dir'],
            config=eval_config,
            sort_by_length=False
        )
        test_dataset = SignDataset(
            metadata_df=test_df,
            data_base_dir=data_config['source_landmark_dir'],
            config=eval_config,
            sort_by_length=False
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
            # Recalculate total_steps for each fold to be precise
            num_devices = trainer.num_devices if 'trainer' in locals() and hasattr(trainer, 'num_devices') else 1
            effective_batch_size = data_config['batch_size'] * num_devices
            total_steps = (len(train_dataset) // effective_batch_size) * config['trainer']['max_epochs']
            config['scheduler']['total_steps'] = total_steps
            print(f"Scheduler total_steps calculated for fold {fold + 1}: {total_steps}")

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

        # --- 2e. Collect results based on CV strategy ---
        y_true = model.test_labels.cpu().numpy()
        y_pred = model.test_preds.cpu().numpy()

        if num_folds > 1:
            # For k-fold, generate and store a report for each fold
            if class_mapping:
                class_mapping_int_keys = {int(k): v for k, v in class_mapping.items()}
                target_names = [class_mapping_int_keys.get(i, str(i)) for i in range(config['model']['num_classes'])]
            else:
                target_names = [str(i) for i in range(config['model']['num_classes'])]

            report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            report_df['fold'] = fold + 1
            all_fold_reports.append(report_df)
            print(f"Fold {fold + 1} Detailed Report generated.")
        else:
            # For LOOCV, just collect labels and predictions
            all_labels.extend(y_true)
            all_preds.extend(y_pred)
            print(f"Fold {fold + 1} predictions collected.")


    # --- 3. Aggregate, Save, and Print Final Results ---
    print("\n===== CROSS-VALIDATION FINAL RESULTS ======")
    output_dir = Path(config["logger"]["save_dir"]) / config["logger"]["name"] / "cv_results"
    output_dir.mkdir(exist_ok=True, parents=True)

    if num_folds > 1:
        # --- k-fold: Aggregate fold reports and calculate averages ---
        if not all_fold_reports:
            print("No test results found to generate a report.")
            return

        full_report_df = pd.concat(all_fold_reports)
        
        numeric_cols = full_report_df.select_dtypes(include=np.number).columns.tolist()
        numeric_cols.remove('fold')
        mean_report_df = full_report_df[~full_report_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])].groupby(full_report_df.index)[numeric_cols].mean()
        
        mean_report_df.loc['accuracy', 'support'] = full_report_df[full_report_df.index == 'accuracy']['support'].sum() / num_folds
        mean_report_df.loc['macro avg'] = mean_report_df.mean()
        mean_report_df.loc['weighted avg'] = np.average(mean_report_df.iloc[:-2], weights=mean_report_df['support'].iloc[:-2], axis=0)
        mean_report_df.loc['accuracy', list(mean_report_df.columns.drop('support'))] = np.nan

        mean_report_df['fold'] = 'mean'
        
        final_report_with_avg = pd.concat([full_report_df, mean_report_df.reset_index()])
        
        csv_path = output_dir / "cross_validation_detailed_report.csv"
        final_report_with_avg.to_csv(csv_path, float_format='%.4f')
        print(f"\nDetailed k-fold cross-validation report saved to: {csv_path}")

        avg_metrics = pd.DataFrame(all_fold_metrics).mean().to_dict()
        print(f"\n--- Average Metrics Across {num_folds} Folds ---")
        print(f"Average Test Accuracy: {avg_metrics.get('test_acc_epoch', 0):.4f}")
        print(f"Average Test F1-Score: {avg_metrics.get('test_f1_epoch', 0):.4f}")
        print(f"Average Test Precision: {avg_metrics.get('test_precision_epoch', 0):.4f}")
        print(f"Average Test Recall: {avg_metrics.get('test_recall_epoch', 0):.4f}")

    else:
        # --- LOOCV: Generate a single report from all predictions ---
        if not all_labels:
            print("No test results found to generate a report.")
            return

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        if class_mapping:
            class_mapping_int_keys = {int(k): v for k, v in class_mapping.items()}
            target_names = [class_mapping_int_keys.get(i, str(i)) for i in range(config['model']['num_classes'])]
        else:
            target_names = [str(i) for i in range(config['model']['num_classes'])]

        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
        final_report_df = pd.DataFrame(report).transpose()

        csv_path = output_dir / "leave_one_out_report.csv"
        final_report_df.to_csv(csv_path, float_format='%.4f')
        print(f"\nOverall Leave-One-Out report saved to: {csv_path}")

        print(f"\n--- Overall Metrics for Leave-One-Out ---")
        # In classification_report's dict output, accuracy is a float, not a dict.
        # It's included in the DataFrame under the 'accuracy' index.
        accuracy_series = final_report_df.loc['accuracy']
        accuracy = accuracy_series['support'] # The accuracy value is stored in an unusual place in the DataFrame.
        macro_avg = final_report_df.loc['macro avg']
        
        print(f"Overall Test Accuracy: {accuracy:.4f}")
        print(f"Overall Test F1-Score (Macro): {macro_avg['f1-score']:.4f}")
        print(f"Overall Test Precision (Macro): {macro_avg['precision']:.4f}")
        print(f"Overall Test Recall (Macro): {macro_avg['recall']:.4f}")

    print("\nIndividual fold metrics and checkpoints logged in the respective 'fold_X' directories.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="YAML形式の設定ファイル")
    parser.add_argument("--checkpoint", type=str, default=None, help="モデルのチェックポイント（CVでは非推奨）")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    with open(args.config, "r") as yf:
        config = yaml.safe_load(yf)

    main(config, checkpoint_path=args.checkpoint)
