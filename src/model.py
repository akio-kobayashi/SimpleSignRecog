import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# === 初学者向け解説 ===
# このファイルでは、手指の動きのデータ（時系列データ）から、どのサインかを認識するための
# ニューラルネットワークモデルを定義しています。
# 1次元CNN（従来）と2次元CNN（新規）をオプションで選択できるように拡張しました。
# =======================

class CausalConv1d(nn.Module):
    """
    因果性を考慮した1D畳み込み。オプションで未来のフレーム（右文脈）を限定的に考慮できる。
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, right_context_size=0):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.right_padding = right_context_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: Tensor) -> Tensor:
        # x: (Batch, Channels, Time)
        x = F.pad(x, (self.left_padding, self.right_padding))
        return self.conv(x)


class CausalConv2d(nn.Module):
    """
    時間方向には因果的（過去のみ参照）、特徴量方向には通常のパディングを行う2D畳み込み層。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, right_context_size=0):
        super().__init__()
        # kernel_sizeがintならタプル(Time, Feature)に変換
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.k_time, self.k_feature = kernel_size
        self.dilation = dilation

        # 時間方向のパディング (因果性確保)
        self.time_pad_left = (self.k_time - 1) * dilation
        self.time_pad_right = right_context_size

        # 特徴量方向のパディング (入出力でサイズが変わらないように設定)
        self.feat_pad = (self.k_feature - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=(dilation, 1),
            padding=(0, self.feat_pad)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (Batch, Channels, Time, Features)
        # 時間方向(dim=2)に対して非対称パディング
        x = F.pad(x, (0, 0, self.time_pad_left, self.time_pad_right))
        return self.conv(x)


class HandEncoder(nn.Module):
    """
    片手の特徴量をエンコードするためのCNNエンコーダ。
    cnn_type='1d' または '2d' を選択可能。
    """
    def __init__(self, input_hand_dim: int, channels: int, dropout: float, right_context_size: int, num_blocks: int = 3, cnn_type: str = "1d"):
        super().__init__()
        self.cnn_type = cnn_type.lower()

        if self.cnn_type == "1d":
            self.input_proj = nn.Linear(input_hand_dim, channels)
            self.blocks = nn.ModuleList()
            for _ in range(num_blocks):
                block = nn.Sequential(
                    CausalConv1d(channels, channels, kernel_size=3, right_context_size=right_context_size),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                self.blocks.append(block)

        elif self.cnn_type == "2d":
            # 2D用: 最初はチャネル数1からスタート
            self.first_conv = CausalConv2d(1, channels, kernel_size=3, right_context_size=right_context_size)

            self.blocks = nn.ModuleList()
            for _ in range(num_blocks):
                block = nn.Sequential(
                    CausalConv2d(channels, channels, kernel_size=3, right_context_size=right_context_size),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                self.blocks.append(block)

            # 特徴量方向を1に集約するプーリング層
            self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))

        else:
            raise ValueError(f"Invalid cnn_type: {cnn_type}. Must be '1d' or '2d'.")

    def forward(self, x: Tensor) -> Tensor:
        # x: (Batch, Time, Features)

        if self.cnn_type == "1d":
            x = self.input_proj(x)
            x = x.permute(0, 2, 1) # (Batch, Channels, Time)
            for block in self.blocks:
                identity = x
                x = block(x) + identity
            return x

        elif self.cnn_type == "2d":
            # 1. 画像形式に変換: (Batch, 1, Time, Features)
            x = x.unsqueeze(1)

            # 2. チャンネル数を増やす
            x = self.first_conv(x)

            # 3. 残差ブロックの適用
            for block in self.blocks:
                identity = x
                x = block(x) + identity

            # 4. 特徴量次元を潰す: (Batch, Channels, Time, 1)
            x = self.avg_pool(x)

            # 5. LSTMのために次元を戻す: (Batch, Channels, Time)
            x = x.squeeze(-1)

            return x


class TwoStreamCNN(nn.Module):
    """
    左右の手の特徴量を個別にCNNで処理し、LSTMで統合するモデル。
    """
    def __init__(
        self,
        input_dim: int = 392, # 更新された特徴量次元
        num_classes: int = 20,
        channels: int = 128,
        dropout: float = 0.2,
        right_context_size: int = 0,
        cnn_num_blocks: int = 3,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 1,
        cnn_type: str = "1d",
    ):
        super().__init__()

        if input_dim % 2 != 0:
            raise ValueError("input_dim must be divisible by 2.")
        self.input_hand_dim = input_dim // 2

        self.hand_encoder = HandEncoder(
            input_hand_dim=self.input_hand_dim,
            channels=channels,
            dropout=dropout,
            right_context_size=right_context_size,
            num_blocks=cnn_num_blocks,
            cnn_type=cnn_type
        )

        self.lstm = nn.LSTM(
            input_size=channels * 2,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )

        self.ctc_output_proj = nn.Linear(lstm_hidden_dim, num_classes + 1)
        self.aux_output_proj = nn.Linear(lstm_hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        B, T, _ = x.shape
        left_features = x[:, :, :self.input_hand_dim]
        right_features = x[:, :, self.input_hand_dim:]

        encoded_left = self.hand_encoder(left_features)   # (Batch, Channels, Time)
        encoded_right = self.hand_encoder(right_features) # (Batch, Channels, Time)

        combined = torch.cat([encoded_left, encoded_right], dim=1).permute(0, 2, 1) # (Batch, Time, 2*Channels)

        lstm_out, _ = self.lstm(combined)
        lstm_out_dropped = self.dropout(lstm_out)

        ctc_logits = self.ctc_output_proj(lstm_out_dropped)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=2).permute(1, 0, 2)

        last_step_indices = lengths.long() - 1
        last_output = lstm_out[torch.arange(B, device=x.device), last_step_indices, :]
        aux_logits = self.aux_output_proj(self.dropout(last_output))

        return ctc_log_probs, aux_logits
