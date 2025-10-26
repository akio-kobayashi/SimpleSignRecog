# solver.py
import torch
from torch import Tensor
import torch.nn as nn
import pytorch_lightning as pl
from typing import Tuple, Any, Optional, List
import torchmetrics
import pandas as pd
from pathlib import Path

# 作成したモデルとデータセットのコンポーネントをインポート
from src.model import TwoStreamCNN

class Solver(pl.LightningModule):
    def __init__(self, config: dict[str, Any]) -> None:
        """
        PyTorch Lightning Module to encapsulate the training and validation logic.

        Args:
            config (dict[str, Any]): A dictionary containing model, optimizer, and scheduler configs.
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        # 1. Instantiate the Model
        self.model = TwoStreamCNN(**self.config['model'])

        # 2. Define the loss function
        self.criterion = nn.CrossEntropyLoss()

        # 3. Define metrics
        num_classes = self.config['model'].get('num_classes', 20)
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        # Metrics for the test step
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.test_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.test_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.test_cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)


    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input features. Shape: (Batch, Time, Features)
            lengths (Tensor): Original sequence lengths.

        Returns:
            Tensor: Model logits. Shape: (Batch, num_classes)
        """
        return self.model(x, lengths)

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        A single training step.
        """
        features, lengths, labels = batch
        logits = self.forward(features, lengths)
        
        loss = self.criterion(logits, labels)
        acc = self.train_accuracy(logits, labels)
        
        logs = {"train_loss": loss, "train_acc": acc}
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        A single validation step.
        """
        features, lengths, labels = batch
        logits = self.forward(features, lengths)
        
        loss = self.criterion(logits, labels)
        acc = self.val_accuracy(logits, labels)
        
        logs = {"val_loss": loss, "val_acc": acc}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        A single test step.
        """
        features, lengths, labels = batch
        logits = self.forward(features, lengths)
        loss = self.criterion(logits, labels)
        
        # Update metrics
        self.test_accuracy.update(logits, labels)
        self.test_f1.update(logits, labels)
        self.test_precision.update(logits, labels)
        self.test_recall.update(logits, labels)
        self.test_cm.update(logits, labels)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        """Called at the end of the test epoch to aggregate and save results."""
        # Compute and log final metrics
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

        # Compute and save confusion matrix
        cm = self.test_cm.compute().cpu().numpy()
        df_cm = pd.DataFrame(cm, 
                             index=range(self.config['model']['num_classes']), 
                             columns=range(self.config['model']['num_classes']))
        
        # Save to a file in the logger's directory for this run
        cm_path = Path(self.trainer.log_dir) / "confusion_matrix.csv"
        df_cm.to_csv(cm_path)
        print(f"\nConfusion matrix saved to {cm_path}")

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.
        """
        opt = torch.optim.RAdam(self.parameters(), **self.config['optimizer'])
        
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(opt, **self.config['scheduler']),
            "interval": "step",
        }
        return [opt], [scheduler_config]