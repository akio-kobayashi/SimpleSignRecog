


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

try:
    import xgboost as xgb
except ImportError:
    print("XGBoost not found. Please install it using: pip install xgboost")
    exit()

# Import the custom dataset from the existing src directory
from src.dataset import SignDataset

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def extract_statistical_features(features: np.ndarray) -> np.ndarray:
    """
    Extracts statistical features from a time-series of landmarks.
    Input shape: (time, num_features)
    Output shape: (1, num_features * 5)
    """
    if features.shape[0] == 0:
        # Handle empty sequences if they occur
        return np.zeros(features.shape[1] * 5)
        
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)
    median = np.median(features, axis=0)
    
    # Concatenate all statistical features into a single feature vector
    return np.concatenate([mean, std, min_vals, max_vals, median])

def main(config: dict):
    """
    Main K-Fold Cross-Validation pipeline for the XGBoost model.
    """
    # --- 0. Set Seed ---
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"--- Seed set to {seed} for reproducibility ---")

    # --- 1. Load Full Dataset and Extract Features ---
    print("--- Loading Full Dataset and Extracting Features for XGBoost ---")
    data_config = config['data']
    metadata_df = pd.read_csv(data_config['metadata_path'])

    # For XGBoost, we use a fixed feature set without augmentation.
    # Create a config with augmentations disabled.
    eval_config = copy.deepcopy(config)
    if 'data' not in eval_config: eval_config['data'] = {}
    eval_config['data']['augmentation'] = {}

    dataset = SignDataset(
        metadata_df=metadata_df,
        data_base_dir=data_config['source_landmark_dir'],
        config=eval_config,
        sort_by_length=False
    )

    X_list = []
    y_list = []
    # Use tqdm for a progress bar as this can take time
    for i in tqdm(range(len(dataset)), desc="Extracting features"):
        features_tensor, label = dataset[i]
        features_np = features_tensor.numpy()
        
        # Extract statistical features from the time-series data
        stat_features = extract_statistical_features(features_np)
        
        X_list.append(stat_features)
        y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"--- Feature extraction complete. Feature matrix shape: {X.shape} ---")

    # --- 2. Cross-Validation Setup ---
    num_folds = data_config.get('num_folds', 5)
    is_loocv = (num_folds == -1)

    if is_loocv:
        from sklearn.model_selection import LeaveOneOut
        print("--- Setting up Leave-One-Out Cross-Validation ---")
        cv_splitter = LeaveOneOut()
        n_splits = cv_splitter.get_n_splits(X)
        all_labels = []
        all_preds = []
    else:
        print(f"--- Setting up Stratified {num_folds}-Fold Cross-Validation ---")
        cv_splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        n_splits = num_folds
        all_fold_metrics = []

    for fold, (train_indices, test_indices) in enumerate(cv_splitter.split(X, y)):
        print(f"\n===== FOLD {fold + 1} / {n_splits} =====")

        # --- 2a. Split data for this fold ---
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # --- 2b. Scale Features ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- 2c. Train XGBoost Model ---
        print("--- Training XGBoost model ---")
        # Get XGBoost parameters from config, with defaults
        xgboost_params = config.get('trainer', {}).get('xgboost_params', {})
        
        # Determine device based on config and availability
        requested_device = xgboost_params.get('device', 'cpu')
        if requested_device == 'cuda':
            try:
                # A simple way to check if GPU is available to XGBoost
                temp_model = xgb.XGBClassifier(device='cuda', n_estimators=1)
                temp_model.fit(np.array([[0,0]]), np.array([0])) # dummy fit
                print("XGBoost GPU support detected and enabled using device='cuda'.")
                xgboost_params['device'] = 'cuda'
            except xgb.core.XGBoostError:
                print("WARNING: XGBoost GPU support not found. Falling back to CPU. "
                      "Please ensure XGBoost is installed with GPU support and CUDA is configured correctly.")
                xgboost_params['device'] = 'cpu'
        else:
            xgboost_params['device'] = 'cpu' # Ensure device is explicitly set to cpu if not cuda

        # Initialize XGBoost model with parameters from config and fixed ones
        final_xgboost_params = xgboost_params.copy()
        if 'predictor' in final_xgboost_params:
            del final_xgboost_params['predictor']

        xgb_model = xgb.XGBClassifier(
            random_state=seed,
            eval_metric='mlogloss',
            **final_xgboost_params
        )
        xgb_model.fit(X_train_scaled, y_train)

        # --- 2d. Test this fold ---
        print(f"--- Testing Fold {fold + 1} ---")
        y_pred = xgb_model.predict(X_test_scaled)

        if is_loocv:
            # For LOOCV, collect labels and predictions
            all_labels.extend(y_test)
            all_preds.extend(y_pred)
            # Optionally print accuracy for the single sample
            acc = accuracy_score(y_test, y_pred)
            print(f"Fold {fold + 1} Correct: {acc == 1.0}")
        else:
            # For k-fold, generate and store a report for each fold
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
            
            print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")
            print(f"Fold {fold + 1} F1-Score ({avg_mode.capitalize()}): {fold_metrics['test_f1']:.4f}")

    # --- 3. Aggregate and Save/Print Final Results ---
    print("\n===== XGBOOST CROSS-VALIDATION FINAL RESULTS ======")
    output_path = config.get('output', {}).get('xgboost_results_path')

    if is_loocv:
        # --- LOOCV: Generate a single report from all predictions ---
        if not all_labels:
            print("No test results found to generate a report.")
            return

        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        results_df = pd.DataFrame(report).transpose()
        
        print("\n--- Overall Metrics for Leave-One-Out ---")
        print(results_df.to_string(float_format='%.4f'))
        
        if output_path:
            try:
                results_df.to_csv(output_path, float_format='%.4f')
                print(f"\nOverall LOOCV report saved to {output_path}")
            except Exception as e:
                print(f"\nError saving results to {output_path}: {e}")

    else:
        # --- k-fold: Aggregate fold reports and calculate averages ---
        if not all_fold_metrics:
            print("No test results found to generate a report.")
            return

        results_df = pd.DataFrame(all_fold_metrics)
        results_df.loc['average'] = results_df.mean()

        results_df.index.name = 'fold'
        results_df = results_df.reset_index()
        results_df['fold'] = results_df['fold'].apply(lambda x: str(x + 1) if isinstance(x, int) else x)

        print("\n--- Average Metrics Across Folds ---")
        print(results_df.to_string(float_format='%.4f', index=False))

        if output_path:
            try:
                results_df.to_csv(output_path, index=False, float_format='%.4f')
                print(f"\nResults successfully saved to {output_path}")
            except Exception as e:
                print(f"\nError saving results to {output_path}: {e}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config, "r") as yf:
        config = yaml.safe_load(yf)

    main(config)

