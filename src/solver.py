import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from pathlib import Path
import numpy as np
import importlib
from collections import Counter

# モデルクラスを型チェックのためにインポート
from src.stgcn_model import STGCNModel
from src.model import TwoStreamCNN

class Solver(pl.LightningModule):
    """
    PyTorch Lightningを使用してモデルの学習、検証、テストのロジックをカプセル化するクラス。
    モデルのタイプを判別し、補助損失や複数メトリクスの計算を動的に行う。
    """
    def __init__(self, config: dict):
        """
        Solverの初期化。
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.loss_lambda = self.config['model'].get('loss_lambda', 0.8)
        self.report_target = self.config['model'].get('report_target', 'ctc')

        # --- モデルの動的初期化 ---
        model_params = self.config['model'].copy()
        module_path = model_params.pop('module_path')
        class_name = model_params.pop('class_name')

        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise type(e)(f"Failed to import model '{class_name}' from '{module_path}': {e}")
            
        if class_name in ['STGCNModel', 'TwoStreamCNN']:
            model_params['features_config'] = self.config.get('features', {})

        import inspect
        sig = inspect.signature(model_class.__init__)
        allowed_args = {p.name for p in sig.parameters.values()}
        filtered_model_params = {k: v for k, v in model_params.items() if k in allowed_args}
        self.model = model_class(**filtered_model_params)

        # --- 損失関数の定義 ---
        num_classes = self.config['model']['num_classes']
        self.ctc_criterion = nn.CTCLoss(blank=num_classes, zero_infinity=True)
        if isinstance(self.model, (STGCNModel, TwoStreamCNN)):
            self.ce_criterion = nn.CrossEntropyLoss()

        # --- 評価指標のセットアップ (validation_stepでのプログレスバー表示用) ---
        average_mode = self.config['trainer']['metrics_average_mode']
        num_classes = self.config['model']['num_classes']
        self.ctc_acc = MulticlassAccuracy(num_classes=num_classes, average=average_mode)
        if isinstance(self.model, (STGCNModel, TwoStreamCNN)):
            self.ce_acc = MulticlassAccuracy(num_classes=num_classes, average=average_mode)

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

        # --- テスト結果（生データ）を保持するリスト ---
        self.test_labels_for_report = []
        self.test_preds_for_report = []

    def forward(self, x, lengths):
        return self.model(x, lengths)

    def _calculate_ctc_preds(self, ctc_log_probs, labels):
        """CTC出力から予測ラベルを計算する。"""
        preds = []
        num_classes = self.config['model']['num_classes']
        if self.decode_method in ['beam_search', 'greedy'] and hasattr(self, 'beam_search_decoder'):
            probs = torch.exp(ctc_log_probs).cpu()
            batch_size = probs.size(1)
            beam_width = self.config['trainer'].get('beam_width', 10) if self.decode_method == 'beam_search' else 1
            for i in range(batch_size):
                sample_probs = probs[:, i, :].numpy()
                beams = self.beam_search_decoder.decode_beams(sample_probs, beam_width=beam_width)
                decoded_text = beams[0][0]
                reference_label = labels[i].item()
                pred_label = -1
                try:
                    if len(decoded_text) == 1 and decoded_text.isdigit():
                        pred_label = int(decoded_text)
                except (ValueError, IndexError):
                    pass
                if pred_label != reference_label:
                    pred_label = (reference_label + 1) % num_classes
                preds.append(torch.tensor(pred_label, device=self.device))
        else:
            best_path = torch.argmax(ctc_log_probs, dim=2)
            for i in range(best_path.size(1)):
                path_for_sample = best_path[:, i]
                non_blank_tokens = [p.item() for p in path_for_sample if p.item() != num_classes]
                
                if not non_blank_tokens:
                    pred_label = (labels[i].item() + 1) % num_classes
                else:
                    counts = Counter(non_blank_tokens)
                    pred_label = counts.most_common(1)[0][0]
                preds.append(torch.tensor(pred_label, device=self.device))

        return torch.stack(preds)

    def _calculate_ce_preds(self, aux_logits):
        """CE出力から予測ラベルを計算する。"""
        return torch.argmax(aux_logits, dim=1)

    def _shared_step(self, batch):
        features, lengths, labels = batch
        model_output = self(features, lengths)
        is_dual_head = isinstance(self.model, (STGCNModel, TwoStreamCNN))

        if is_dual_head:
            ctc_log_probs, aux_logits = model_output
            loss_ce = self.ce_criterion(aux_logits, labels)
        else:
            ctc_log_probs = model_output
            aux_logits = None
            loss_ce = torch.tensor(0.0, device=self.device)

        batch_size = features.size(0)
        input_lengths = torch.full((batch_size,), ctc_log_probs.size(0), dtype=torch.long, device=self.device)
        target_lengths = torch.ones(batch_size, dtype=torch.long, device=self.device)
        loss_ctc = self.ctc_criterion(ctc_log_probs, labels, input_lengths, target_lengths)

        if is_dual_head:
            loss = self.loss_lambda * loss_ctc + (1 - self.loss_lambda) * loss_ce
        else:
            loss = loss_ctc

        return loss, loss_ctc, loss_ce, ctc_log_probs, aux_logits, labels

    def training_step(self, batch, batch_idx):
        loss, loss_ctc, loss_ce, _, _, _ = self._shared_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_loss_ctc', loss_ctc, on_step=True, on_epoch=True)
        if isinstance(self.model, (STGCNModel, TwoStreamCNN)):
            self.log('train_loss_ce', loss_ce, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_ctc, loss_ce, ctc_log_probs, aux_logits, labels = self._shared_step(batch)
        
        ctc_preds = self._calculate_ctc_preds(ctc_log_probs, labels)
        # validation_stepではプログレスバー表示用に、__init__で定義された全体指標のみを使用
        acc_ctc = self.ctc_acc(ctc_preds, labels)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc_ctc', acc_ctc, prog_bar=True)
        self.log('val_loss_ctc', loss_ctc)

        if isinstance(self.model, (STGCNModel, TwoStreamCNN)) and aux_logits is not None:
            ce_preds = self._calculate_ce_preds(aux_logits)
            acc_ce = self.ce_acc(ce_preds, labels)
            self.log('val_acc_ce', acc_ce, prog_bar=True)
            self.log('val_loss_ce', loss_ce)
        return loss

    def test_step(self, batch, batch_idx):
        loss, loss_ctc, loss_ce, ctc_log_probs, aux_logits, labels = self._shared_step(batch)
        
        # 予測ラベルを計算
        ctc_preds = self._calculate_ctc_preds(ctc_log_probs, labels)
        
        is_dual_head = isinstance(self.model, (STGCNModel, TwoStreamCNN))
        if is_dual_head and aux_logits is not None:
            ce_preds = self._calculate_ce_preds(aux_logits)
        else:
            ce_preds = None

        # classification_report用のラベルを収集
        if self.report_target == 'ce' and ce_preds is not None:
            self.test_preds_for_report.append(ce_preds.cpu())
        else:
            self.test_preds_for_report.append(ctc_preds.cpu())
        self.test_labels_for_report.append(labels.cpu())
        
        return loss

    def on_test_epoch_end(self):
        # 生の予測結果と正解ラベルを最終的なプロパティとして設定
        if self.test_preds_for_report:
            self.test_preds = torch.cat(self.test_preds_for_report)
        if self.test_labels_for_report:
            self.test_labels = torch.cat(self.test_labels_for_report)
        
        # メモリを解放
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
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        else:
            return optimizer