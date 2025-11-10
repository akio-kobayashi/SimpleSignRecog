
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

# === ST-GCN (Spatio-Temporal Graph Convolutional Network)の実装 ===
# このファイルでは、手指の骨格構造を「グラフ」として捉え、
# 空間的な関節のつながりと、時間的な動きを同時に学習できるST-GCNモデルを定義します。
# 既存のCNN+LSTMモデル(model.py)との互換性を保つため、solver.pyが要求する
# インターフェース（__init__, forwardの引数、戻り値の形状）を維持しています。
# =================================================================

# --- 1. 手の骨格グラフの定義 ---
# 21個のランドマークの接続関係を定義します。
# これはST-GCNが「どこが隣の関節か」を学習するための地図の役割を果たします。
# (参考: MediaPipe Hand landmarker)
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17) # Palm connection
]

class GraphConv(nn.Module):
    """
    空間グラフ畳み込み層 (Spatial Graph Convolutional Layer)。
    
    【初学者向け解説】
    通常の画像に対する畳み込み(CNN)が、ピクセルとその「近傍」の情報を集約するのに対し、
    グラフ畳み込みは、グラフのノード（ここでは関節）とその「隣接」ノードの情報を集約します。
    
    この実装では、A. Kipf & M. Welling の論文 (ICLR 2017) に基づくシンプルな手法を採用しています。
    隣接行列 A と特徴量 H を用いて、A' * H * W という計算を行います。
    A' は正規化された隣接行列で、学習を安定させる役割があります。
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: Tensor, A: Tensor) -> Tensor:
        # x: (N, T, V, C_in)  - N:Batch, T:Time, V:Vertices(Joints), C:Channels
        # A: (V, V) - Adjacency Matrix
        
        x = torch.einsum('ntvc,vw->ntwc', x, A) # 隣接ノードの情報を集約
        x = self.linear(x) # 特徴量を変換
        return x

class STGCNBlock(nn.Module):
    """
    ST-GCNの基本ブロック。空間グラフ畳み込みと時間畳み込みを組み合わせる。
    
    【初学者向け解説】
    このブロックはST-GCNの心臓部です。
    1. GraphConv: まず、各フレーム内での関節間の空間的な関係性を捉えます。
    2. TemporalConv: 次に、時間軸に沿った畳み込みを行い、動きのパターン（速度など）を捉えます。
    
    この「空間→時間」という処理を１つのブロックとして、これを複数積み重ねることで、
    より複雑な時空間パターンを学習していきます。
    """
    def __init__(self, in_channels: int, out_channels: int, temporal_kernel_size: int):
        super().__init__()
        
        # 空間グラフ畳み込み
        self.gcn = GraphConv(in_channels, out_channels)
        
        # 時間畳み込み (通常の2D畳み込みで代用)
        # カーネルサイズを (temporal_kernel_size, 1) にすることで、時間軸方向のみに畳み込みを行う
        self.tcn = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=(temporal_kernel_size, 1),
            padding=((temporal_kernel_size - 1) // 2, 0) # 時間軸方向のみパディング
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # 残差接続のための次元調整
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: Tensor, A: Tensor) -> Tensor:
        # x: (N, T, V, C_in)
        identity = x.permute(0, 3, 1, 2) # (N, C_in, T, V)
        identity = self.residual(identity)

        # 空間畳み込み
        x = self.gcn(x, A) # (N, T, V, C_out)
        
        # 時間畳み込みのために次元を並べ替え: (N, C_out, T, V)
        x = x.permute(0, 3, 1, 2)
        x = self.tcn(x) # (N, C_out, T, V)
        
        x = self.bn(x)
        x = x + identity # 残差接続
        x = self.relu(x)
        
        # 元の次元順序に戻す: (N, T, V, C_out)
        return x.permute(0, 2, 3, 1)


class STGCNModel(nn.Module):
    """
    ST-GCNモデル本体。
    学習安定化のため、主損失(CTC)と補助損失(CrossEntropy)を併用する構成。
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 20,
        channels: int = 64,
        dropout: float = 0.2,
        num_blocks: int = 3,
        temporal_kernel_size: int = 9,
        features_config: dict = None,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 1,
        **kwargs
    ):
        super().__init__()
        
        if features_config is None:
            raise ValueError("STGCNModel requires 'features_config' to be provided.")

        self.features_config = features_config
        self.num_joints = 21
        
        # --- configに基づいて、ノード毎の入力チャンネル数(C)を決定 ---
        self.in_channels = 0
        paper_conf = self.features_config.get('paper_features', {})
        normalize_mode = self.features_config.get('normalize_mode', 'normalize_landmarks')
        
        use_paper_speed = paper_conf.get('speed', False)
        use_paper_anthropometric = paper_conf.get('anthropometric', False)
        is_paper_mode = normalize_mode in ['current_wrist', 'first_wrist'] or use_paper_speed or use_paper_anthropometric

        self.in_channels += 3 

        if not is_paper_mode:
            self.in_channels += 3
            self.in_channels += 3
        elif paper_conf.get('speed'):
            self.in_channels += 3

        # --- グラフ隣接行列の準備 ---
        A = torch.zeros((self.num_joints, self.num_joints), dtype=torch.float32)
        for i, j in HAND_BONES:
            A[i, j] = 1
            A[j, i] = 1
        
        degrees = torch.sum(A, axis=1)
        D_inv_sqrt_vec = torch.pow(degrees, -0.5)
        D_inv_sqrt_vec[torch.isinf(D_inv_sqrt_vec)] = 0.
        D_inv_sqrt_mat = torch.diag(D_inv_sqrt_vec)
        A_norm = D_inv_sqrt_mat @ A @ D_inv_sqrt_mat
        self.register_buffer('A', A_norm)

        # --- モデルの構築 ---
        self.input_proj = nn.Linear(self.in_channels, channels)
        
        self.st_gcn_blocks = nn.ModuleList()
        current_channels = channels
        for _ in range(num_blocks):
            self.st_gcn_blocks.append(
                STGCNBlock(current_channels, channels, temporal_kernel_size)
            )
            current_channels = channels

        self.lstm = nn.LSTM(
            input_size=channels * self.num_joints * 2,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        # --- 出力層 ---
        # 主損失(CTC)用の出力層: (B, T, num_classes + 1)
        self.ctc_output_proj = nn.Linear(lstm_hidden_dim, num_classes + 1)
        # 補助損失(CE)用の出力層: (B, num_classes)
        self.aux_output_proj = nn.Linear(lstm_hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def _unpack_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        B, T, _ = x.shape
        
        paper_conf = self.features_config.get('paper_features', {})
        normalize_mode = self.features_config.get('normalize_mode', 'normalize_landmarks')

        use_paper_speed = paper_conf.get('speed', False)
        use_paper_anthropometric = paper_conf.get('anthropometric', False)
        is_paper_mode = normalize_mode in ['current_wrist', 'first_wrist'] or use_paper_speed or use_paper_anthropometric

        if not is_paper_mode:
            left_flat = x[:, :, :193]
            right_flat = x[:, :, 193:386]

            hands_features = []
            for hand_flat in [left_flat, right_flat]:
                pos = hand_flat[:, :, :63].reshape(B, T, self.num_joints, 3)
                vel = hand_flat[:, :, 63:126].reshape(B, T, self.num_joints, 3)
                acc = hand_flat[:, :, 126:189].reshape(B, T, self.num_joints, 3)
                hand_features = torch.cat([pos, vel, acc], dim=3)
                hands_features.append(hand_features)
            return hands_features[0], hands_features[1]
        else:
            current_offset = 0
            
            pos_dim = self.num_joints * 3 * 2
            pos_feats = x[:, :, current_offset : current_offset + pos_dim]
            current_offset += pos_dim
            
            left_pos = pos_feats[:, :, :pos_dim//2].reshape(B, T, self.num_joints, 3)
            right_pos = pos_feats[:, :, pos_dim//2:].reshape(B, T, self.num_joints, 3)
            
            left_tensors = [left_pos]
            right_tensors = [right_pos]

            if paper_conf.get('speed'):
                vel_dim = self.num_joints * 3 * 2
                vel_feats = x[:, :, current_offset : current_offset + vel_dim]
                current_offset += vel_dim
                
                left_vel = vel_feats[:, :, :vel_dim//2].reshape(B, T, self.num_joints, 3)
                right_vel = vel_feats[:, :, vel_dim//2:].reshape(B, T, self.num_joints, 3)
                left_tensors.append(left_vel)
                right_tensors.append(right_vel)
            
            left_hand_features = torch.cat(left_tensors, dim=3)
            right_hand_features = torch.cat(right_tensors, dim=3)
            
            return left_hand_features, right_hand_features

    def forward(self, x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        """
        順伝播処理。
        CTC損失用の時間毎の出力と、補助損失用の最終ステップ出力のタプルを返す。
        """
        B, T, _ = x.shape
        
        left_hand, right_hand = self._unpack_features(x)
        
        left_out = self.input_proj(left_hand)
        right_out = self.input_proj(right_hand)
        
        for block in self.st_gcn_blocks:
            left_out = block(left_out, self.A)
            right_out = block(right_out, self.A)
            
        left_out = left_out.reshape(B, T, -1)
        right_out = right_out.reshape(B, T, -1)
        combined = torch.cat([left_out, right_out], dim=2)
        
        lstm_out, _ = self.lstm(combined)
        lstm_out_dropped = self.dropout(lstm_out)
        
        # 1. CTC損失用の出力 (時間毎のロジット)
        ctc_logits = self.ctc_output_proj(lstm_out_dropped)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=2).permute(1, 0, 2) # (T, B, C+1)

        # 2. 補助損失用の出力 (最終ステップのロジット)
        row_indices = torch.arange(B, device=x.device)
        last_step_indices = lengths.long() - 1
        last_output = lstm_out[row_indices, last_step_indices, :]
        last_output_dropped = self.dropout(last_output) # Note: use non-dropped output for aux loss? Let's use dropped for consistency.
        aux_logits = self.aux_output_proj(last_output_dropped) # (B, num_classes)
        
        return ctc_log_probs, aux_logits
