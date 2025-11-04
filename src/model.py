import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# === 初学者向け解説 ===
# このファイルでは、手指の動きのデータ（時系列データ）から、どのサインかを認識するための
# ニューラルネットワークモデルを定義しています。
# 今回の変更では、リアルタイム処理（Webカメラで撮影しながら即座に認識するような使い方）を
# 見据えた「因果性」と、時系列データに適した「CTC損失」という考え方を導入しています。
# =======================

class CausalConv1d(nn.Module):
    """
    因果性を考慮した1D畳み込み。オプションで未来のフレーム（右文脈）を限定的に考慮できる。

    【初学者向け解説】
    通常の畳み込みは、ある時点のデータを処理する際に、その前後（過去と未来）のデータを使います。
    しかし、リアルタイム処理では「未来」のデータはまだ存在しないため、使えません。
    「因果畳み込み（Causal Convolution）」は、ある時点のデータを処理する際に、「過去」のデータのみを
    使うように設計された畳み込みです。これにより、1フレームずつデータが入力されても、
    矛盾なく処理を進めることができます。

    ここでは、`F.pad`を使って畳み込みの前に左側（過去）にだけパディング（詰め物）をすることで、
    これを実現しています。`right_context_size`を増やすと、少しだけ未来のフレームを
    「覗き見」することになり、認識精度が上がる可能性がありますが、その分だけ遅延が大きくなります。
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, right_context_size=0):
        super().__init__()
        # 左側のパディング量: (kernel_size - 1) * dilation で因果性を担保
        self.left_padding = (kernel_size - 1) * dilation
        # 右側のパディング量: ユーザーが指定する未来のフレーム数
        self.right_padding = right_context_size
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: Tensor) -> Tensor:
        # x: (Batch, Channels, Time)
        # 左右に非対称なパディングを適用
        x = F.pad(x, (self.left_padding, self.right_padding))
        return self.conv(x)


class HandEncoder(nn.Module):
    """
    片手の特徴量をエンコードするための1D-CNNエンコーダ。
    時系列情報を保持したまま出力する。

    【初学者向け解説】
    入力された手の動きのデータ（時系列）から、畳み込みニューラルネットワーク（CNN）を
    使って、より高度な特徴量を抽出する部分です。
    以前のバージョンでは、ここで抽出した特徴量を時間方向に平均化（Global Average Pooling）して
    一つのベクトルにまとめていましたが、今回は時系列情報を失わないように、
    (Batch, Channels, Time) の形のまま出力します。これにより、どのタイミングで
    どのような特徴が表れたかを後段の層が判断できるようになります。
    """
    def __init__(self, input_hand_dim: int, channels: int, dropout: float, right_context_size: int):
        super().__init__()
        self.input_proj = nn.Linear(input_hand_dim, channels)
        
        # 畳み込みブロックを因果畳み込み（+右文脈）に置き換え
        self.conv_blocks = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size=3, right_context_size=right_context_size),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size=3, right_context_size=right_context_size),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size=3, right_context_size=right_context_size),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (Batch, Time, Features)
        x = self.input_proj(x) # (Batch, Time, Channels)
        x = x.permute(0, 2, 1) # (Batch, Channels, Time) に並べ替え
        x = self.conv_blocks(x) # (Batch, Channels, Time)
        # Global Average Poolingを削除し、時系列情報を保持したまま出力
        return x


class TwoStreamCNN(nn.Module):
    """
    左右の手の特徴量を別々に処理し、統合する1D-CNNモデル。
    CTC損失関数に適した形式で、時系列のロジットを出力する。

    【初学者向け解説】
    このモデルの本体です。「Two-Stream」という名前の通り、左手と右手、2つの流れ（Stream）の
    データを別々のエンコーダ（`HandEncoder`）で処理し、最後に統合する構造になっています。
    """
    def __init__(
        self,
        input_dim: int = 386, # 全体の特徴量次元
        num_classes: int = 20,
        channels: int = 128,
        dropout: float = 0.2,
        right_context_size: int = 0, # 追加：右文脈のサイズ
    ):
        """
        Args:
            input_dim (int): 全体の入力特徴量の次元数。
            num_classes (int): 出力クラスの数（CTCのblankトークンを含まない）。
            channels (int): 畳み込み層のチャネル数。
            dropout (float): ドロップアウト率。
            right_context_size (int): 畳み込みで考慮する未来のフレーム数。0なら完全な因果畳み込み。
        """
        super().__init__()
        
        # 片手の次元を計算
        if input_dim % 2 != 0:
            raise ValueError("input_dim must be divisible by 2.")
        self.input_hand_dim = input_dim // 2
        
        # 左右の手それぞれにエンコーダを用意 (エンコーダの重みは左右で共有される)
        self.hand_encoder = HandEncoder(self.input_hand_dim, channels, dropout, right_context_size)

        # 統合後の特徴量からクラスを予測する層
        # 【初学者向け解説】
        # CTC損失では、「どのクラスでもない」ことを示す "blank" という特別なラベルを使います。
        # そのため、実際のクラス数 `num_classes` に1を加えた数を最終的な出力次元とします。
        self.output_proj = nn.Linear(channels * 2, num_classes + 1)
        

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        """
        順伝播処理。

        Args:
            x (Tensor): 入力テンソル。形状: (Batch, Time, Features)
            lengths (Tensor): 各シーケンスの元の長さ。

        Returns:
            Tensor: CTC損失用のlog-probabilities。形状: (Time, Batch, num_classes + 1)
        """
        # 1. 入力特徴量を左右の手に分割
        left_features = x[:, :, :self.input_hand_dim]
        right_features = x[:, :, self.input_hand_dim:]

        # 2. 左右の手をそれぞれエンコード
        encoded_left = self.hand_encoder(left_features)   # (Batch, Channels, Time)
        encoded_right = self.hand_encoder(right_features) # (Batch, Channels, Time)

        # 3. 左右のエンコード結果をチャンネル次元で連結
        combined_features = torch.cat([encoded_left, encoded_right], dim=1) # (Batch, 2 * Channels, Time)
        
        # 4. 各タイムステップに対してクラス分類器を適用
        #    (B, 2*C, T) -> (B, T, 2*C) に変換
        combined_features = combined_features.permute(0, 2, 1)
        logits = self.output_proj(combined_features) # (Batch, Time, num_classes + 1)
        
        # 5. CTC損失のための後処理
        # 【初学者向け解説】
        #    `nn.CTCLoss` は、入力として「対数確率（log probability）」を要求します。
        #    `F.log_softmax` は、モデルの出力（ロジット）を確率に変換し、さらに対数をとる処理を
        #    効率的かつ数値的に安定して行ってくれます。
        log_probs = F.log_softmax(logits, dim=2)
        
        # 【初学者向け解説】
        #    `nn.CTCLoss` の仕様に合わせて、テンソルの次元の順番を
        #    (Batch, Time, Classes) -> (Time, Batch, Classes) に入れ替えます。
        log_probs = log_probs.permute(1, 0, 2)
        
        return log_probs