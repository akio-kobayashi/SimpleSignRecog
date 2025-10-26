import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class HandEncoder(nn.Module):
    """
    片手の特徴量をエンコードするための1D-CNNエンコーダ。
    """
    def __init__(self, input_hand_dim: int, channels: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(input_hand_dim, channels)
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, F_hand)
        x = self.input_proj(x) # (B, T, C)
        x = x.permute(0, 2, 1) # (B, C, T)
        x = self.conv_blocks(x) # (B, C, T)
        x = torch.mean(x, dim=2) # Global Average Pooling (B, C)
        return x


class TwoStreamCNN(nn.Module):
    """
    左右の手の特徴量を別々に処理し、統合する1D-CNNモデル。
    """
    def __init__(
        self,
        input_hand_dim: int = 193, # 左右それぞれの手の特徴量次元 (63*3 + 4)
        num_classes: int = 20,
        channels: int = 128,
        dropout: float = 0.2
    ):
        """
        Args:
            input_hand_dim (int): 片手の入力特徴量の次元数。
            num_classes (int): 出力クラスの数。
            channels (int): 畳み込み層のチャネル数。
            dropout (float): ドロップアウト率。
        """
        super().__init__()
        
        # 左右の手それぞれにエンコーダを用意 (重みは共有)
        self.hand_encoder = HandEncoder(input_hand_dim, channels, dropout)

        # 統合後の特徴量からクラスを予測する層
        # 左右のエンコーダ出力 (channels * 2) から num_classes へ
        self.output_proj = nn.Linear(channels * 2, num_classes)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        """
        順伝播処理。

        Args:
            x (Tensor): 入力テンソル。形状: (Batch, Time, Features) -> (B, T, 386)
            lengths (Tensor): 各シーケンスの元の長さ。

        Returns:
            Tensor: 分類結果のロジット。形状: (Batch, num_classes)
        """
        # 入力特徴量を左右の手に分割
        # x: (B, T, 386)
        left_features = x[:, :, :193]  # (B, T, 193)
        right_features = x[:, :, 193:] # (B, T, 193)

        # 左右の手をそれぞれエンコード
        encoded_left = self.hand_encoder(left_features)   # (B, C)
        encoded_right = self.hand_encoder(right_features) # (B, C)

        # 左右のエンコード結果を連結
        combined_features = torch.cat([encoded_left, encoded_right], dim=1) # (B, 2 * C)
        
        # 最終的なクラス分類
        logits = self.output_proj(combined_features) # (B, num_classes)
        
        return logits