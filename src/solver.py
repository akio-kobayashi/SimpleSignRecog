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

        # STGCNモデルの場合、特徴量エンジニアリングの設定を渡す
        if class_name == 'STGCNModel':
            model_params['features_config'] = self.config.get('features', {})
        
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

        # --- デコーダーのセットアップ ---
        self.decode_method = self.config['trainer'].get('decode_method', 'majority_vote')
        if self.decode_method in ['beam_search', 'greedy']:
            try:
                from pyctcdecode import build_ctcdecoder
            except ImportError:
                raise ImportError("Beam search decoding requires the 'pyctcdecode' library. Please install it using: pip install pyctcdecode")
            
            # ラベルのリストを作成 (blankを含む)
            # ここでは仮に数字を文字列に変換していますが、実際のクラス名があればそれを使用
            labels = [str(i) for i in range(num_classes)] + [''] 
            
            # decode_method に応じて beam_width を設定
            current_beam_width = self.config['trainer'].get('beam_width', 10) if self.decode_method == 'beam_search' else 1

            self.beam_search_decoder = build_ctcdecoder(
                labels=labels,
                kenlm_model_path=None, # 言語モデルは使用しない
                alpha=0.0,
                beta=0.0
            )

        # テスト結果を保存するためのリスト
        self.test_outputs = []
        self.test_labels_for_report = []
        self.test_preds_for_report = []
        self.test_lengths = []
        self.all_wer_results = [] # WERの結果を保存するためのリスト

    def forward(self, x, lengths):
        """
        モデルの順伝播。
        """
        return self.model(x, lengths)

    def _calculate_metrics(self, log_probs, labels):
        """
        時系列の予測値(log_probs)と正解ラベルから各評価指標を計算する。
        configで指定されたデコード方式(Greedy, Majority Vote, or Beam Search)を用いる。
        """
        # log_probs: (T, B, C+1)
        # labels: (B,)
        decode_method = self.config['trainer'].get('decode_method', 'majority_vote')

        preds = []
        # pyctcdecode を使用するデコード (beam_search または greedy)
        if decode_method in ['beam_search', 'greedy']:
            if not hasattr(self, 'beam_search_decoder'):
                raise RuntimeError("pyctcdecode beam_search_decoder not initialized.")

            probs = torch.exp(log_probs).cpu() # (T, B, C+1)
            batch_size = probs.size(1)
            
            # decode_method に応じて beam_width を設定
            current_beam_width = self.config['trainer'].get('beam_width', 10) if decode_method == 'beam_search' else 1

            # pyctcdecode の decode_beams を使用してデコード
            # decode_beams は (text, score) のタプルのリストを返す
            # 各サンプルに対してビームサーチを実行
            for i in range(batch_size):
                # 各サンプルの確率時系列 (T, C+1)
                sample_probs = probs[:, i, :].numpy() 
                
                # decode_beams に beam_width を渡す
                beams = self.beam_search_decoder.decode_beams(sample_probs, beam_width=current_beam_width)
                
                # 最もスコアの高いビーム (最初のビーム) のテキストを取得
                decoded_text = beams[0][0] 

                # decoded_text は '052' のような文字列になる可能性がある
                # このタスクでは単一のクラスラベルを期待しているので、
                # decoded_text から最も頻繁に出現する数字を抽出する
                
                # decoded_text が空でないことを確認
                if decoded_text:
                    # decoded_text を個々の数字に分解し、intに変換
                    # 例: '052' -> [0, 5, 2]
                    decoded_numbers = [int(c) for c in decoded_text if c.isdigit()]
                    
                    if decoded_numbers:
                        # 最も頻繁に出現する数字を予測とする (Majority Voteと同様の考え方)
                        most_common = Counter(decoded_numbers).most_common(1)
                        pred = most_common[0][0]
                    else:
                        # 数字が見つからない場合はデフォルト値 (0)
                        pred = 0
                else:
                    # decoded_text が空の場合はデフォルト値 (0)
                    pred = 0

                preds.append(torch.tensor(pred, device=self.device))

        elif decode_method == 'majority_vote':
            # --- Majority Vote デコード ---
            best_path = torch.argmax(log_probs, dim=2)  # (T, B)
            for i in range(best_path.size(1)):  # Batch次元でループ
                path_for_sample = best_path[:, i]

                non_blank_path = path_for_sample[path_for_sample != self.config['model']['num_classes']]
                if non_blank_path.numel() > 0:
                    preds.append(torch.mode(non_blank_path).values)
                else:
                    preds.append(torch.tensor(0, device=self.device))
        else:
            raise ValueError(f"Unsupported decode method: {decode_method}")

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
        
        f1, acc, precision, recall, preds = self._calculate_metrics(log_probs, labels)
        
        # --- DEBUG PRINTS (only for the first batch) ---
        if batch_idx == 0:
            print("\n" + "="*20 + f" VALIDATION STEP (Epoch {self.current_epoch}) " + "="*20)
            print(f"Batch index: {batch_idx}")
            print(f"Input features shape: {features.shape}")
            print(f"Input features mean: {torch.mean(features)}")
            print(f"Input features std: {torch.std(features)}")
            print(f"Model output log_probs shape: {log_probs.shape}")
            print(f"Model output log_probs mean: {torch.mean(log_probs)}")
            print(f"Is log_probs NaN: {torch.isnan(log_probs).any()}")
            print(f"Is log_probs Inf: {torch.isinf(log_probs).any()}")
            print(f"Calculated loss: {loss.item()}")
            print(f"Sample labels: {labels[:5].tolist()}")
            print(f"Sample predictions: {preds[:5].tolist()}")
            print("="*60 + "\n")
        # --- END DEBUG PRINTS ---

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

        # WERの計算と保存
        if self.decode_method in ['greedy', 'beam_search']:
            for i in range(batch_size):
                ref_label = [labels[i].item()] # 参照は単一のクラスラベル
                hyp_pred = [preds[i].item()]   # 予測も単一のクラスラベル
                
                # calculate_werはリストを受け取る
                sub, dele, ins, num_w, wer = calculate_wer(ref_label, hyp_pred)
                self.all_wer_results.append({
                    'substitutions': sub,
                    'deletions': dele,
                    'insertions': ins,
                    'num_words': num_w,
                    'wer': wer
                })
        
        return loss

    def on_test_epoch_end(self):
        """
        テストエポック終了時の処理。
        """
        # classification_reportのために結果を結合 (train.pyで使われる)
        self.test_preds = torch.cat(self.test_preds_for_report)
        self.test_labels = torch.cat(self.test_labels_for_report)

        # WERの結果を結合 (train.pyで使われる)
        if self.all_wer_results:
            import pandas as pd
            self.wer_results = pd.DataFrame(self.all_wer_results)
        else:
            import pandas as pd
            self.wer_results = pd.DataFrame() # 結果がない場合は空のDataFrame

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
        self.all_wer_results.clear() # WERの結果もクリア

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
