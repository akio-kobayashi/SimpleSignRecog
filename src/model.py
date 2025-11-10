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
    片手の特徴量をエンコードするための残差接続付き1D-CNNエンコーダ。
    ブロック数は外部から設定可能。

    【初学者向け解説】
    入力された手の動きのデータ（時系列）から、畳み込みニューラルネットワーク（CNN）を
    使って、より高度な特徴量を抽出する部分です。
    各畳み込みブロックには「残差接続」が導入されており、学習の安定化を図っています。
    ブロックの数は設定ファイルから変更可能です。
    """
    def __init__(self, input_hand_dim: int, channels: int, dropout: float, right_context_size: int, num_blocks: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(input_hand_dim, channels)
        
        # 畳み込みブロックをModuleListに格納し、数を可変にする
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(
                CausalConv1d(channels, channels, kernel_size=3, right_context_size=right_context_size),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.blocks.append(block)

    def forward(self, x: Tensor) -> Tensor:
        # x: (Batch, Time, Features)
        x = self.input_proj(x) # (Batch, Time, Channels)
        x = x.permute(0, 2, 1) # (Batch, Channels, Time) に並べ替え

        # --- 残差ブロックをループで適用 ---
        for block in self.blocks:
            identity = x
            out = block(x)
            x = out + identity # 残差接続

        return x


class TwoStreamCNN(nn.Module):
    """
    左右の手の特徴量を別々に処理し、LSTMで統合する1D-CNN + LSTMモデル。
    学習安定化のため、主損失(CTC)と補助損失(CrossEntropy)を併用する。
    """
    def __init__(
        self,
        input_dim: int = 386, # 全体の特徴量次元
        num_classes: int = 20,
        channels: int = 128,
        dropout: float = 0.2,
        right_context_size: int = 0, # 畳み込みで考慮する未来のフレーム数
        # --- CNNとLSTMの構造に関する新しい引数 ---
        cnn_num_blocks: int = 3,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 1,
    ):
        """
        Args:
            input_dim (int): 全体の入力特徴量の次元数。
            num_classes (int): 出力クラスの数（CTCのblankトークンを含まない）。
            channels (int): 畳み込み層のチャネル数。
            dropout (float): ドロップアウト率。
            right_context_size (int): 畳み込みで考慮する未来のフレーム数。
            cnn_num_blocks (int): HandEncoder内の残差ブロックの数。
            lstm_hidden_dim (int): LSTM隠れ層の次元数。
            lstm_layers (int): LSTM層の数。
        """
        super().__init__()
        
        # 片手の次元を計算
        if input_dim % 2 != 0:
            raise ValueError("input_dim must be divisible by 2.")
        self.input_hand_dim = input_dim // 2
        
        # 左右の手それぞれにエンコーダを用意 (エンコーダの重みは左右で共有される)
        self.hand_encoder = HandEncoder(
            input_hand_dim=self.input_hand_dim, 
            channels=channels, 
            dropout=dropout, 
            right_context_size=right_context_size, 
            num_blocks=cnn_num_blocks # ブロック数を渡す
        )

        # --- 左右の特徴量を統合するLSTM層 ---
        self.lstm = nn.LSTM(
            input_size=channels * 2,  # 左右のチャンネルを連結したものが入力
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True, # 入力形式を (Batch, Time, Features) に
            dropout=dropout if lstm_layers > 1 else 0, # 複数層の場合のみドロップアウト適用
            bidirectional=False # 因果性を保つため単一方向
        )

        # --- 2つの出力ヘッドを定義 ---
        # 主損失(CTC)用の出力層
        self.ctc_output_proj = nn.Linear(lstm_hidden_dim, num_classes + 1)
        # 補助損失(CE)用の出力層
        self.aux_output_proj = nn.Linear(lstm_hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        """
        順伝播処理。
        CTC損失用の時間毎の出力と、補助損失用の最終ステップ出力のタプルを返す。
        """
        B, T, _ = x.shape
        
        # 1. 入力特徴量を左右の手に分割
        left_features = x[:, :, :self.input_hand_dim]
        right_features = x[:, :, self.input_hand_dim:]

        # 2. 左右の手をそれぞれエンコード
        encoded_left = self.hand_encoder(left_features)   # (Batch, Channels, Time)
        encoded_right = self.hand_encoder(right_features) # (Batch, Channels, Time)

        # 3. 左右のエンコード結果をチャンネル次元で連結
        combined_features = torch.cat([encoded_left, encoded_right], dim=1) # (Batch, 2 * Channels, Time)
        
        # 4. LSTMに入力するために次元を並べ替え
        #    (B, 2*C, T) -> (B, T, 2*C)
        combined_features = combined_features.permute(0, 2, 1)

        # 5. LSTM層で時間的依存関係を学習
        #    LSTMからの出力は (B, T, lstm_hidden_dim)
        lstm_out, _ = self.lstm(combined_features)
        lstm_out_dropped = self.dropout(lstm_out)

        # 6. 主出力 (CTC損失用)
        ctc_logits = self.ctc_output_proj(lstm_out_dropped)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=2).permute(1, 0, 2)

        # 7. 補助出力 (CrossEntropy損失用)
        # 各サンプルの実際のシーケンス長に基づいて、LSTMの最後の有効な出力を取得
        row_indices = torch.arange(B, device=x.device)
        last_step_indices = lengths.long() - 1
        last_output = lstm_out[row_indices, last_step_indices, :]
        last_output_dropped = self.dropout(last_output)
        
        aux_logits = self.aux_output_proj(last_output_dropped)
        
        return ctc_log_probs, aux_logits