import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from pathlib import Path
import numpy as np
import importlib

# from src.model import TwoStreamCNN # 動的にインポートするためコメントアウト

class Solver(pl.LightningModule):
    """
    PyTorch Lightningを使用してモデルの学習、検証、テストのロジックをカプセル化するクラス。
    CTC損失を扱い、時系列アウトプットに対応するように変更されています。
    """
    def __init__(self, config: dict):
        """
        Solverの初期化。
        """
        super().__init__()
        self.save_hyperparameters() # configの内容をハイパーパラメータとして保存
        self.config = config

        # --- モデルの動的初期化 ---
        # configからモデルのパスとクラス名を取得し、動的にインスタンス化する
        model_params = self.config['model'].copy()
        module_path = model_params.pop('module_path')
        class_name = model_params.pop('class_name')
        
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise type(e)(f"Failed to import model '{class_name}' from '{module_path}': {e}")
            
        self.model = model_class(**model_params)

        # --- 損失関数の定義 ---
        # CTC損失を使用。blankトークンはクラスインデックスの末尾 (num_classes) と仮定
        num_classes = self.config['model']['num_classes']
        self.criterion = nn.CTCLoss(blank=num_classes, zero_infinity=True)

        # --- 評価指標のセットアップ ---
        average_mode = self.config['trainer']['metrics_average_mode']
        self.f1 = MulticlassF1Score(num_classes=num_classes, average=average_mode)
        self.acc = MulticlassAccuracy(num_classes=num_classes, average=average_mode)
        self.precision = MulticlassPrecision(num_classes=num_classes, average=average_mode)
        self.recall = MulticlassRecall(num_classes=num_classes, average=average_mode)

        # テスト結果を保存するためのリスト
        self.test_outputs = []
        self.test_labels_for_report = []
        self.test_preds_for_report = []
        self.test_lengths = []

    def forward(self, x, lengths):
        """
        モデルの順伝播。
        """
        return self.model(x, lengths)

    def _calculate_metrics(self, log_probs, labels):
        """
        時系列の予測値(log_probs)と正解ラベルから各評価指標を計算する。
        デコード方式として、Greedy Decode（Best Path Decode）を行い、
        その結果から最も多く出現した非blankクラスを予測結果とする。
        """
        # log_probs: (T, B, C+1)
        # labels: (B,)
        
        # 1. Greedy Decode (Best Path)
        # 各タイムステップで最も確率の高いクラスインデックスを取得
        best_path = torch.argmax(log_probs, dim=2) # (T, B)
        
        # 2. バッチ内の各サンプルに対して多数決で予測クラスを決定
        preds = []
        for i in range(best_path.size(1)): # Batch次元でループ
            path_for_sample = best_path[:, i]
            
            # blankトークン (C) を除外
            non_blank_path = path_for_sample[path_for_sample != self.config['model']['num_classes']]
            
            if non_blank_path.numel() > 0:
                # 最も多く出現したクラスを予測結果とする
                pred = torch.mode(non_blank_path).values
                preds.append(pred)
            else:
                # 非blankの予測が一つもなかった場合
                # (代替案: 最も確率の高い非blankクラスを選ぶなど)
                # ここでは暫定的に最初のクラス(0)を予測とする
                preds.append(torch.tensor(0, device=self.device))

        preds = torch.stack(preds)
        
        # 各評価指標を計算
        f1 = self.f1(preds, labels)
        acc = self.acc(preds, labels)
        precision = self.precision(preds, labels)
        recall = self.recall(preds, labels)
        return f1, acc, precision, recall, preds

    def training_step(self, batch, batch_idx):
        """
        1回の学習ステップ（ミニバッチ）での処理。
        """
        features, lengths, labels = batch
        batch_size = features.size(0)
        
        # モデルからlog_probsを取得: (T, B, num_classes + 1)
        log_probs = self(features, lengths)
        
        # CTC損失の入力長は、モデルの出力シーケンス長
        input_lengths = torch.full(size=(batch_size,), fill_value=log_probs.size(0), dtype=torch.long, device=self.device)
        
        # 今回のタスクでは、各サンプルのターゲット長は1
        target_lengths = torch.ones(batch_size, dtype=torch.long, device=self.device)
        
        loss = self.criterion(log_probs, labels, input_lengths, target_lengths)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        1回の検証ステップでの処理。
        """
        features, lengths, labels = batch
        batch_size = features.size(0)
        
        log_probs = self(features, lengths)
        
        input_lengths = torch.full(size=(batch_size,), fill_value=log_probs.size(0), dtype=torch.long, device=self.device)
        target_lengths = torch.ones(batch_size, dtype=torch.long, device=self.device)
        
        loss = self.criterion(log_probs, labels, input_lengths, target_lengths)
        
        f1, acc, precision, recall, _ = self._calculate_metrics(log_probs, labels)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_precision', precision, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_recall', recall, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        1回のテストステップでの処理。
        """
        features, lengths, labels = batch
        batch_size = features.size(0)
        
        log_probs = self(features, lengths)
        
        input_lengths = torch.full(size=(batch_size,), fill_value=log_probs.size(0), dtype=torch.long, device=self.device)
        target_lengths = torch.ones(batch_size, dtype=torch.long, device=self.device)
        
        loss = self.criterion(log_probs, labels, input_lengths, target_lengths)
        
        f1, acc, precision, recall, preds = self._calculate_metrics(log_probs, labels)
        
        self.log('test_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_f1_epoch', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc_epoch', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_precision_epoch', precision, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_recall_epoch', recall, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        # 分類レポートと可視化のために結果を保存
        self.test_outputs.append(log_probs.permute(1, 0, 2).exp().cpu())
        self.test_lengths.append(lengths.cpu()) # 正規化に使う元の系列長
        self.test_preds_for_report.append(preds.cpu())
        self.test_labels_for_report.append(labels.cpu())
        
        return loss

    def on_test_epoch_end(self):
        """
        テストエポック終了時の処理。
        """
        # classification_reportのために結果を結合 (train.pyで使われる)
        self.test_preds = torch.cat(self.test_preds_for_report)
        self.test_labels = torch.cat(self.test_labels_for_report)

        # Posteriogramを保存
        if hasattr(self, 'posteriogram_dir'):
            output_dir = Path(self.posteriogram_dir)
            output_dir.mkdir(exist_ok=True, parents=True)

            sample_idx = 0
            for batch_idx in range(len(self.test_outputs)):
                probs_batch = self.test_outputs[batch_idx].cpu().numpy()
                labels_batch = self.test_labels_for_report[batch_idx].cpu().numpy()
                
                for i in range(len(labels_batch)):
                    posteriogram = probs_batch[i] # Shape (T, C+1)
                    label = labels_batch[i]
                    
                    # 保存ファイル名を定義
                    filename = output_dir / f"sample_{sample_idx}_label_{label}.npz"
                    
                    # データをNumpy形式で保存
                    np.savez_compressed(
                        filename,
                        posteriogram=posteriogram,
                        label=label
                    )
                    sample_idx += 1

            print(f"Posteriogramsを {output_dir} に保存しました。")
        
        # 次のfoldのためにリストをクリア
        self.test_outputs.clear()
        self.test_lengths.clear()
        self.test_preds_for_report.clear()
        self.test_labels_for_report.clear()

    def configure_optimizers(self):
        """
        オプティマイザと学習率スケジューラを設定します。
        """
        optimizer_config = self.config['optimizer']
        optimizer = optim.Adam(self.parameters(), lr=optimizer_config['lr'])
        
        if 'scheduler' in self.config:
            scheduler_config = self.config['scheduler']
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=scheduler_config['max_lr'],
                total_steps=scheduler_config['total_steps']
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            return optimizer
