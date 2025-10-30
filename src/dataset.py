import os
import random
from typing import Tuple, Optional, List
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset, Sampler

# `feature_engineering.py`で定義された、データの前処理や特徴量計算のための関数群をインポート
from src.feature_engineering import (
    interpolate_missing_data,
    augment_flip,
    augment_noise,
    augment_rotate,
    canonical_normalize_landmarks,
    smooth_landmarks,
    calculate_features,
)


class SignDataset(Dataset):
    """
    手指動作認識タスクのためのカスタムデータセットクラス。
    PyTorchの`Dataset`クラスを継承して作成します。

    `Dataset`クラスの役割:
    - データセット全体のサンプル数を報告する (`__len__`メソッド)。
    - 指定されたインデックス(`idx`)のデータサンプルを1つ取得してくる (`__getitem__`メソッド)。

    このクラスでは、データをメモリに全て読み込むのではなく、`__getitem__`が呼ばれるたびに
    ディスクからNPZファイルを1つ読み込み、その場で動的に前処理（データ拡張など）を行います。
    これにより、メモリ使用量を抑えつつ、多様なデータをモデルに供給できます。
    """
    def __init__(
        self, 
        metadata_df: pd.DataFrame, 
        data_base_dir: str, 
        sort_by_length: bool = False,
        # データ拡張を適用するかどうかのフラグ
        augment_flip: bool = False,
        augment_rotate: bool = False,
        augment_noise: bool = False,
        flip_prob: float = 0.5,
    ):
        """
        データセットの初期化処理

        Args:
            metadata_df (pd.DataFrame): 各サンプルの情報（NPZファイルのパス、クラスラベルなど）が書かれたメタデータ。
            data_base_dir (str): NPZファイルが保存されているベースディレクトリ。
            sort_by_length (bool): Trueの場合、データをフレーム数（系列長）でソートする。`BucketBatchSampler`で効率化するために使う。
            augment_flip (bool): 左右反転のデータ拡張を有効にするか。
            augment_rotate (bool): ランダム回転のデータ拡張を有効にするか。
            augment_noise (bool): ノイズ付加のデータ拡張を有効にするか。
            flip_prob (float): 左右反転を適用する確率。
        """
        super().__init__()
        self.data_base_dir = Path(data_base_dir)
        self.df = metadata_df
        
        # データ拡張のフラグをインスタンス変数として保存
        self.augment_flip = augment_flip
        self.augment_rotate = augment_rotate
        self.augment_noise = augment_noise
        self.flip_prob = flip_prob

        # `BucketBatchSampler`を使う場合、あらかじめ系列長でソートしておく
        if sort_by_length:
            if 'num_frames' not in self.df.columns:
                raise ValueError("`sort_by_length=True` requires a 'num_frames' column in the metadata CSV.")
            self.df = self.df.sort_values(by='num_frames').reset_index(drop=True)

    def __len__(self) -> int:
        """データセットに含まれるサンプルの総数を返す。"""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """
        指定されたインデックス `idx` のデータサンプルを1つ取得し、前処理を適用して返す。
        このメソッドは、データローダーによって内部的に呼び出される。
        """
        # メタデータから、idx番目のサンプルの情報を取得
        item_row = self.df.iloc[idx]
        
        # NPZファイルから生のランドマークデータをロード
        npz_path = self.data_base_dir / item_row['npz_path']
        with np.load(npz_path) as data:
            landmarks = data['landmarks']
            
        # --- オンデマンド前処理パイプライン ---
        # 1. 欠損値(NaN)を補間
        processed_landmarks = interpolate_missing_data(landmarks)

        # 2. データ拡張 (学習時のみ有効にする)
        if self.augment_flip and random.random() < self.flip_prob:
            processed_landmarks = augment_flip(processed_landmarks)
        if self.augment_rotate and random.random() < 0.5: # 50%の確率で適用
            processed_landmarks = augment_rotate(processed_landmarks)
        if self.augment_noise and random.random() < 0.5: # 50%の確率で適用
            processed_landmarks = augment_noise(processed_landmarks)

        # 3. 座標の正規化 (位置・スケール不変性の獲得)
        processed_landmarks = canonical_normalize_landmarks(processed_landmarks)

        # 4. データの平滑化 (ノイズの除去)
        processed_landmarks = smooth_landmarks(processed_landmarks)

        # 5. 特徴量の計算 (速度、加速度、形状特徴などを追加)
        final_features = calculate_features(processed_landmarks)

        # 6. NaN/infチェック: 稀に発生する不安定な値を0で置き換え、学習の安定化を図る
        if np.any(np.isnan(final_features)) or np.any(np.isinf(final_features)):
            final_features = np.nan_to_num(final_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # ラベルを取得 (元のラベルは1-20だが、PyTorchでは0から始まるインデックスが一般的なので-1する)
        label = int(item_row['class_label']) - 1
        
        # 最終的な特徴量(NumPy配列)をPyTorchのTensorに変換して、ラベルと共に返す
        return torch.from_numpy(final_features.copy()).float(), label


class BucketBatchSampler(Sampler[List[int]]):
    """
    系列長の近いサンプルをまとめてバッチを作成するためのカスタムサンプラー。
    PyTorchの`Sampler`クラスを継承して作成します。

    背景:
    このプロジェクトのデータは、動画によってフレーム数（系列長）が異なります。
    系列長の大きく異なるデータを同じバッチに入れてしまうと、短いデータにたくさんの
    「パディング」（後述）を追加する必要があり、計算が無駄になります。

    このクラスの役割:
    あらかじめ系列長でソートされたデータセットに対して、隣接するサンプルをグループ化して
    バッチを作成します。これにより、各バッチ内のサンプルは似たような長さを持つことになり、
    パディングの量を最小限に抑え、学習を効率化します。
    """
    def __init__(self, dataset: SignDataset, batch_size: int, drop_last: bool = True):
        if not hasattr(dataset, 'df') or 'num_frames' not in dataset.df.columns:
            raise ValueError("Dataset must have a DataFrame 'df' with a 'num_frames' column.")

        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # バッチを構成するインデックスのリストを作成
        indices = list(range(len(self.dataset)))
        self.batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            # バッチサイズに満たない最後のバッチを捨てるかどうか
            if len(batch) < self.batch_size and self.drop_last:
                continue
            self.batches.append(batch)

    def __iter__(self):
        """エポックごとにバッチの順番をシャッフルして、イテレータとして返す。"""
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        """1エポックあたりのバッチの総数を返す。"""
        return len(self.batches)


def data_processing(batch: List[Tuple[Tensor, int]]) -> Tuple[Tensor, Tensor, Tensor]:
    """
    データローダーが `SignDataset` から集めてきたサンプルのリストを、
    実際にモデルが学習に使える「バッチ」形式に変換するための関数 (collate_fn)。

    処理内容:
    - `__getitem__` が返した (特徴量Tensor, ラベル) のリストを、特徴量Tensorのリストとラベルのリストに分離する。
    - 各サンプルの特徴量Tensorは系列長が異なるため、バッチ内で最も長いものに合わせて、
      短い系列の末尾に0を詰める「パディング」処理を行う。
    - パディング後の特徴量、各サンプルの元の系列長、ラベルをそれぞれ1つの大きなTensorにまとめて返す。
    """
    # (特徴量, ラベル)のリストを、特徴量のリストとラベルのリストに分離
    features, labels = zip(*batch)
    
    # 各サンプルの元の系列長を保存（パディングの影響を受けないようにするため、RNNなどで重要）
    lengths = torch.tensor([f.shape[0] for f in features])
    
    # `pad_sequence` を使って、バッチ内の全サンプルの系列長を最長のものに揃える（短いものは0でパディング）
    padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    
    # ラベルのリストをTensorに変換
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_features, lengths, labels
