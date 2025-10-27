import os
import random
from typing import Tuple, Optional, List
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from torch import Tensor
from torch.utils.data import Sampler

# Import all processing functions from our feature engineering library
from src.feature_engineering import (
    interpolate_missing_data,
    augment_flip,
    augment_noise,
    augment_rotate,
    normalize_landmarks,
    smooth_landmarks,
    calculate_features,
)

class SignDataset(torch.utils.data.Dataset):
    """
    手指動作認識用のカスタムデータセット。
    オンデマンドでデータ拡張と特徴量抽出を行う。
    """
    def __init__(
        self, 
        metadata_df: pd.DataFrame, 
        data_base_dir: str, 
        sort_by_length: bool = False,
        # Augmentation flags
        augment_flip: bool = False,
        augment_rotate: bool = False,
        augment_noise: bool = False,
        flip_prob: float = 0.5,
    ):
        """
        Args:
            metadata_df (pd.DataFrame): メタデータを含むDataFrame。
            data_base_dir (str): "生"のランドマークNPZファイルが格納されているベースディレクトリ。
            sort_by_length (bool): Trueの場合、データをフレーム数でソートする。
            augment_flip (bool): 左右反転拡張を有効にするか。
            augment_rotate (bool): ランダム回転拡張を有効にするか。
            augment_noise (bool): ガウス雑音拡張を有効にするか。
            flip_prob (float): 左右反転の確率。
        """
        super().__init__()
        self.data_base_dir = Path(data_base_dir)
        self.df = metadata_df
        
        # Store augmentation flags
        self.augment_flip = augment_flip
        self.augment_rotate = augment_rotate
        self.augment_noise = augment_noise
        self.flip_prob = flip_prob

        if sort_by_length:
            if 'num_frames' not in self.df.columns:
                raise ValueError("`sort_by_length=True` requires a 'num_frames' column in the metadata CSV.")
            self.df = self.df.sort_values(by='num_frames').reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """指定されたインデックスのデータを取得し、オンデマンドで処理する。"""
        item_row = self.df.iloc[idx]
        
        # 生のランドマークデータをロード
        npz_path = self.data_base_dir / item_row['npz_path']
        with np.load(npz_path) as data:
            landmarks = data['landmarks']
            
        # --- オンデマンド処理パイプライン ---
        # 1. 補間
        processed_landmarks = interpolate_missing_data(landmarks)

        # 2. データ拡張
        if self.augment_flip and random.random() < self.flip_prob:
            processed_landmarks = augment_flip(processed_landmarks)
        if self.augment_rotate and random.random() < 0.5: # 50%の確率で適用
            processed_landmarks = augment_rotate(processed_landmarks)
        if self.augment_noise and random.random() < 0.5: # 50%の確率で適用
            processed_landmarks = augment_noise(processed_landmarks)

        # 3. 正規化
        processed_landmarks = normalize_landmarks(processed_landmarks)

        # 4. 平滑化
        processed_landmarks = smooth_landmarks(processed_landmarks)

        # 5. 特徴量計算
        final_features = calculate_features(processed_landmarks)

        # 6. NaN/infチェック: これらが損失計算時にnanを引き起こすのを防ぐ
        if np.any(np.isnan(final_features)) or np.any(np.isinf(final_features)):
            # 不安定な特徴量が見つかった場合に警告（デバッグ用）
            # print(f"\n[WARN] NaN or Inf found in features for item {idx} ({item_row['npz_path']}). Replacing with 0.")
            final_features = np.nan_to_num(final_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # ラベルを取得 (1-20 -> 0-19)
        label = int(item_row['class_label']) - 1
        
        return torch.from_numpy(final_features.copy()).float(), label


class BucketBatchSampler(Sampler[List[int]]):
    """
    長さに従ってソートされたデータセットから、効率的なバッチを作成するサンプラー。
    """
    def __init__(self, dataset: SignDataset, batch_size: int, drop_last: bool = True):
        if not hasattr(dataset, 'df') or 'num_frames' not in dataset.df.columns:
            raise ValueError("Dataset must have a DataFrame 'df' with a 'num_frames' column.")

        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        indices = list(range(len(self.dataset)))
        self.batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                continue
            self.batches.append(batch)

    def __iter__(self):
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        return len(self.batches)


def data_processing(batch: List[Tuple[Tensor, int]]) -> Tuple[Tensor, Tensor, Tensor]:
    """
    データローダーから渡されたバッチを処理し、パディングを行う collate_fn。
    """
    features, labels = zip(*batch)
    
    lengths = torch.tensor([f.shape[0] for f in features])
    padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_features, lengths, labels
