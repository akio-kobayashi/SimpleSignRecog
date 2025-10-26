import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class SimpleCNN(nn.Module):
    """
    時系列ランドマークデータのためのシンプルな1D-CNNモデル。
    """
    def __init__(
        self,
        input_dim: int = 386,
        num_classes: int = 20,
        channels: int = 128,
        dropout: float = 0.2
    ):
        """
        Args:
            input_dim (int): 入力特徴量の次元数。
            num_classes (int): 出力クラスの数。
            channels (int): 畳み込み層のチャネル数。
            dropout (float): ドロップアウト率。
        """
        super().__init__()
        
        # 入力特徴量を畳み込み層のチャネル数に射影する
        self.input_proj = nn.Linear(input_dim, channels)

        # 1D畳み込みブロック
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Block 2
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Block 3
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 最終的なクラス分類のための出力層
        self.output_proj = nn.Linear(channels, num_classes)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        """
        順伝播処理。

        Args:
            x (Tensor): 入力テンソル。形状: (Batch, Time, Features)
            lengths (Tensor): 各シーケンスの元の長さ。このモデルでは未使用だが、
                              インターフェースの互換性のために受け取る。

        Returns:
            Tensor: 分類結果のロジット。形状: (Batch, num_classes)
        """
        # (B, T, F) -> (B, T, C)
        x = self.input_proj(x)
        
        # Conv1dは (B, C, T) を期待するため、次元を入れ替える
        # (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        
        # 畳み込みブロックを適用
        # (B, C, T) -> (B, C, T)
        x = self.conv_blocks(x)
        
        # Global Average Pooling (時間軸 T に沿って平均をとる)
        # (B, C, T) -> (B, C)
        x = torch.mean(x, dim=2)
        
        # 出力層
        # (B, C) -> (B, num_classes)
        x = self.output_proj(x)
        
        return x
