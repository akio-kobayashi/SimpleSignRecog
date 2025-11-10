import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from pathlib import Path
import numpy as np
import importlib
from src.error_rate import calculate_wer # calculate_werをインポート
from collections import Counter # Counterをインポート

# from src.model import TwoStreamCNN # 動的にインポートするためコメントアウト

class Solver(pl.LightningModule):
    """
    PyTorch Lightningを使用してモデルの学習、検証、テストのロジックをカプセル化するクラス。
    学習安定化のため、主損失(CTC)と補助損失(CrossEntropy)を併用する。
    """
    def __init__(self, config: dict):
        """
        Solverの初期化。
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.loss_lambda = self.config['trainer'].get('loss_lambda', 0.8) # CTC損失の重み

        # --- モデルの動的初期化 ---
        model_params = self.config['model'].copy()
        module_path = model_params.pop('module_path')
        class_name = model_params.pop('class_name')

        if class_name == 'STGCNModel':
            model_params['features_config'] = self.config.get('features', {})
        
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise type(e)(f"Failed to import model '{class_name}' from '{module_path}': {e}")
            
        self.model = model_class(**model_params)

        # --- 損失関数の定義 ---
        num_classes = self.config['model']['num_classes']
        # 主損失: CTC
        self.ctc_criterion = nn.CTCLoss(blank=num_classes, zero_infinity=True)
        # 補助損失: CrossEntropy
        self.ce_criterion = nn.CrossEntropyLoss()

        # --- 評価指標のセットアップ ---
        average_mode = self.config['trainer']['metrics_average_mode']
        self.f1 = MulticlassF1Score(num_classes=num_classes, average=average_mode)
        self.acc = MulticlassAccuracy(num_classes=num_classes, average=average_mode)
        self.precision = MulticlassPrecision(num_classes=num_classes, average=average_mode)
        self.recall = MulticlassRecall(num_classes=num_classes, average=average_mode)

        # --- CTCデコーダーのセットアップ ---
        self.decode_method = self.config['trainer'].get('decode_method', 'majority_vote')
        if self.decode_method in ['beam_search', 'greedy']:
            try:
                from pyctcdecode import build_ctcdecoder
                labels = [str(i) for i in range(num_classes)] + [''] 
                self.beam_search_decoder = build_ctcdecoder(labels=labels, kenlm_model_path=None, alpha=0.0, beta=0.0)
            except ImportError:
                print("WARN: 'pyctcdecode' is not installed. Beam search decoding will not be available.")
                self.decode_method = 'majority_vote'


        # テスト結果を保存するためのリスト
        self.test_labels_for_report = []
        self.test_preds_for_report = []

    def forward(self, x, lengths):
        """
        モデルの順伝播。
        """
        return self.model(x, lengths)

    def _calculate_metrics(self, ctc_log_probs, labels):
        """
        CTCの出力(log_probs)と正解ラベルから各評価指標を計算する。
        """
        # ctc_log_probs: (T, B, C+1)
        # labels: (B,)
        preds = []
        if self.decode_method in ['beam_search', 'greedy'] and hasattr(self, 'beam_search_decoder'):
            probs = torch.exp(ctc_log_probs).cpu()
            batch_size = probs.size(1)
            beam_width = self.config['trainer'].get('beam_width', 10) if self.decode_method == 'beam_search' else 1

            for i in range(batch_size):
                sample_probs = probs[:, i, :].numpy() 
                beams = self.beam_search_decoder.decode_beams(sample_probs, beam_width=beam_width)
                decoded_text = beams[0][0] 
                if decoded_text:
                    decoded_numbers = [int(c) for c in decoded_text if c.isdigit()]
                    if decoded_numbers:
                        pred = Counter(decoded_numbers).most_common(1)[0][0]
                    else:
                        pred = 0
                else:
                    pred = 0
                preds.append(torch.tensor(pred, device=self.device))
        else: # Fallback to majority_vote
            best_path = torch.argmax(ctc_log_probs, dim=2)
            for i in range(best_path.size(1)):
                path_for_sample = best_path[:, i]
                non_blank_path = path_for_sample[path_for_sample != self.config['model']['num_classes']]
                if non_blank_path.numel() > 0:
                    preds.append(torch.mode(non_blank_path).values)
                else:
                    preds.append(torch.tensor(0, device=self.device))

        preds = torch.stack(preds)
        f1 = self.f1(preds, labels)
        acc = self.acc(preds, labels)
        precision = self.precision(preds, labels)
        recall = self.recall(preds, labels)
        return f1, acc, precision, recall, preds

    def _shared_step(self, batch, batch_idx):
        features, lengths, labels = batch
        batch_size = features.size(0)
        
        ctc_log_probs, aux_logits = self(features, lengths)
        
        # 1. 主損失 (CTC)
        input_lengths = torch.full(size=(batch_size,), fill_value=ctc_log_probs.size(0), dtype=torch.long, device=self.device)
        target_lengths = torch.ones(batch_size, dtype=torch.long, device=self.device)
        loss_ctc = self.ctc_criterion(ctc_log_probs, labels, input_lengths, target_lengths)
        
        # 2. 補助損失 (CrossEntropy)
        loss_ce = self.ce_criterion(aux_logits, labels)
        
        # 3. 損失の結合
        loss = self.loss_lambda * loss_ctc + (1 - self.loss_lambda) * loss_ce
        
        return loss, loss_ctc, loss_ce, ctc_log_probs, labels

    def training_step(self, batch, batch_idx):
        loss, loss_ctc, loss_ce, _, _ = self._shared_step(batch, batch_idx)
        self.log('train_loss', loss)
        self.log('train_loss_ctc', loss_ctc)
        self.log('train_loss_ce', loss_ce)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_ctc, loss_ce, ctc_log_probs, labels = self._shared_step(batch, batch_idx)
        
        f1, acc, precision, recall, _ = self._calculate_metrics(ctc_log_probs, labels)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_loss_ctc', loss_ctc)
        self.log('val_loss_ce', loss_ce)
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_precision', precision)
        self.log('val_recall', recall)
        return loss

    def test_step(self, batch, batch_idx):
        loss, loss_ctc, loss_ce, ctc_log_probs, labels = self._shared_step(batch, batch_idx)
        
        f1, acc, precision, recall, preds = self._calculate_metrics(ctc_log_probs, labels)
        
        self.log('test_loss_epoch', loss)
        self.log('test_f1_epoch', f1)
        self.log('test_acc_epoch', acc)
        self.log('test_precision_epoch', precision)
        self.log('test_recall_epoch', recall)

        self.test_preds_for_report.append(preds.cpu())
        self.test_labels_for_report.append(labels.cpu())
        return loss

    def on_test_epoch_end(self):
        if self.test_preds_for_report:
            self.test_preds = torch.cat(self.test_preds_for_report)
        if self.test_labels_for_report:
            self.test_labels = torch.cat(self.test_labels_for_report)
        
        self.test_preds_for_report.clear()
        self.test_labels_for_report.clear()

    def configure_optimizers(self):
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
