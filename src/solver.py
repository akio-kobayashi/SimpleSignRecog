# solver.py
import torch
from torch import Tensor
import torch.nn as nn
import pytorch_lightning as pl
from typing import Tuple, Any, Optional, List
import torchmetrics
import pandas as pd
from pathlib import Path

# 作成したモデルのクラスをインポートします
from src.model import TwoStreamCNN

class Solver(pl.LightningModule):
    """
    PyTorch Lightningのモジュール(LightningModule)です。
    このクラスに、モデル、学習、検証、テストのロジックをすべてまとめます。
    PyTorch Lightningが、このクラスに定義されたメソッドを適切なタイミングで自動的に呼び出してくれます。
    """
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Solverクラスの初期化（コンストラクタ）です。
        モデル、損失関数、評価指標などをここで準備します。

        Args:
            config (dict[str, Any]): config.yamlから読み込んだ設定情報の辞書。
        """
        super().__init__()
        self.config = config
        # save_hyperparameters()を呼ぶと、設定が自動的に保存され、後から参照しやすくなります
        self.save_hyperparameters()

        # 1. モデルのインスタンス化
        # ---------------------------
        model_config = self.config['model'].copy()
        # configから入力次元数を取得し、モデルの設定から削除します
        input_dim = model_config.pop('input_dim', 386)
        # モデルは左右の手を別々に扱うので、入力次元数を2で割って片手分の次元数を渡します
        model_config['input_hand_dim'] = input_dim // 2
        self.model = TwoStreamCNN(**model_config)

        # 2. 損失関数の定義
        # ---------------------
        # CrossEntropyLossは、多クラス分類で一般的に使われる損失関数です
        self.criterion = nn.CrossEntropyLoss()

        # 3. 評価指標の定義
        # ---------------------
        num_classes = self.config['model'].get('num_classes', 20)
        avg_mode = self.config.get('trainer', {}).get('metrics_average_mode', 'macro')

        # torchmetricsライブラリを使って、正解率(Accuracy)を計算するメトリックを定義します
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        # テストステップで使用するメトリックを定義します
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average=avg_mode, zero_division=0)
        self.test_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average=avg_mode, zero_division=0)
        self.test_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average=avg_mode, zero_division=0)
        self.test_cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes) # 混同行列

        # テストセット全体の予測結果と正解ラベルを保存するためのリスト
        self.test_preds: List[Tensor] = []
        self.test_labels: List[Tensor] = []


    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        """
        モデルの順伝播を定義します。
        入力データがモデルをどのように通過するかを記述します。

        Args:
            x (Tensor): 入力特徴量。形状は (バッチサイズ, 時系列長, 特徴量次元数)。
            lengths (Tensor): 元の（パディング前の）時系列長。

        Returns:
            Tensor: モデルの出力（ロジット）。形状は (バッチサイズ, クラス数)。
        """
        return self.model(x, lengths)

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        1回の訓練ステップ（1ミニバッチ分の学習）の処理を定義します。
        """
        features, lengths, labels = batch # バッチからデータを取り出します
        logits = self.forward(features, lengths) # モデルで予測を計算します
        
        loss = self.criterion(logits, labels) # 損失（予測と正解の誤差）を計算します
        acc = self.train_accuracy(logits, labels) # 正解率を計算します
        
        # ログを記録します
        logs = {"train_loss": loss, "train_acc": acc}
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss # 計算した損失を返します (PyTorch Lightningがこれを使って逆伝播を行います)

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        1回の検証ステップの処理を定義します。
        """
        features, lengths, labels = batch
        logits = self.forward(features, lengths)
        
        loss = self.criterion(logits, labels)
        acc = self.val_accuracy(logits, labels)
        
        # 検証ステップのログを記録します
        logs = {"val_loss": loss, "val_acc": acc}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def on_test_epoch_start(self) -> None:
        """テストエポックの開始時に呼ばれるフック。予測とラベルのリストをクリアします。"""
        self.test_preds.clear()
        self.test_labels.clear()

    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        1回のテストステップの処理を定義します。
        """
        features, lengths, labels = batch
        logits = self.forward(features, lengths)
        loss = self.criterion(logits, labels)
        
        # ロジットから最も確率の高いクラスを予測結果とします
        preds = torch.argmax(logits, dim=1)
        
        # このバッチの予測結果と正解ラベルをリストに保存します
        self.test_preds.append(preds)
        self.test_labels.append(labels)

        # 各評価指標を更新します
        self.test_accuracy.update(preds, labels)
        self.test_f1.update(preds, labels)
        self.test_precision.update(preds, labels)
        self.test_recall.update(preds, labels)
        self.test_cm.update(preds, labels)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        """テストエポックの終了時に呼ばれるフック。結果を集計して保存します。"""
        # 全てのバッチの予測結果と正解ラベルを一つのテンソルにまとめます
        if self.test_preds:
            self.test_preds = torch.cat(self.test_preds)
        if self.test_labels:
            self.test_labels = torch.cat(self.test_labels)

        # 最終的な評価指標を計算してログに記録します
        test_acc = self.test_accuracy.compute()
        test_f1 = self.test_f1.compute()
        test_precision = self.test_precision.compute()
        test_recall = self.test_recall.compute()
        self.log_dict({
            "test_acc_epoch": test_acc,
            "test_f1_epoch": test_f1,
            "test_precision_epoch": test_precision,
            "test_recall_epoch": test_recall
        })

        # 混同行列を計算し、CSVファイルとして保存します
        cm = self.test_cm.compute().cpu().numpy()
        df_cm = pd.DataFrame(cm, 
                             index=range(self.config['model']['num_classes']), 
                             columns=range(self.config['model']['num_classes']))
        
        # ロガーのディレクトリにファイルを保存します
        if self.trainer and self.trainer.log_dir:
            cm_path = Path(self.trainer.log_dir) / "confusion_matrix.csv"
            df_cm.to_csv(cm_path)
            print(f"\n混同行列を {cm_path} に保存しました")

    def configure_optimizers(self):
        """
        オプティマイザ（最適化手法）と学習率スケジューラを設定します。
        """
        # RAdamオプティマイザを使用します
        opt = torch.optim.RAdam(self.parameters(), **self.config['optimizer'])
        
        # OneCycleLRスケジューラを使用します。これは学習率を周期的に変化させる手法です。
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(opt, **self.config['scheduler']),
            "interval": "step", # ステップごとに学習率を更新します
        }
        return [opt], [scheduler_config]