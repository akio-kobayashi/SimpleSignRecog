

import torch
import yaml
from src.solver import Solver
import pytorch_lightning as pl
import numpy as np

def verify_metrics_calculation():
    """
    Solverの指標計算ロジックが正しく動作するかを検証する。
    ダミーデータを用いて、クラスごとの指標が期待通りに計算されるかを確認する。
    """
    print("--- 指標計算ロジックの検証を開始します ---")

    # 1. configを読み込み、Solverを初期化
    try:
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("エラー: config.yamlが見つかりません。")
        return

    # num_classesをダミーデータに合わせて一時的に変更
    config['model']['num_classes'] = 5
    solver = Solver(config)
    print("Solverを初期化しました。")

    # 2. ダミーデータと期待される結果を定義
    # 2バッチ分のデータを作成
    # バッチ1: labels=[0, 0, 1, 2, 2], preds=[0, 1, 1, 2, 9]
    # バッチ2: labels=[0, 2, 3, 3], preds=[0, 2, 3, 9]
    batches = [
        {
            "preds": torch.tensor([0, 1, 1, 2, 9]),
            "labels": torch.tensor([0, 0, 1, 2, 2]),
        },
        {
            "preds": torch.tensor([0, 2, 3, 9]),
            "labels": torch.tensor([0, 2, 3, 3]),
        },
    ]

    # 期待されるCE（Cross-Entropy）ヘッドのクラスごと指標
    # クラス0: 正解2, 全体2 -> Acc=1.0, Recall=1.0, Precision=1.0, F1=1.0
    # クラス1: 正解1, 全体1 -> Acc=1.0, Recall=1.0, Precision=0.5, F1=0.666
    # クラス2: 正解2, 全体3 -> Acc=0.666, Recall=0.666, Precision=1.0, F1=0.8
    # クラス3: 正解1, 全体2 -> Acc=0.5, Recall=0.5, Precision=1.0, F1=0.666
    # クラス4: 正解0, 全体0 -> 各指標0
    # 予測9は偽陽性(FP)や偽陰性(FN)の計算に影響
    expected_results = {
        'acc':  {'class_0': 1.0, 'class_1': 1.0, 'class_2': 2/3, 'class_3': 0.5, 'class_4': 0.0},
        'precision': {'class_0': 1.0, 'class_1': 0.5, 'class_2': 1.0, 'class_3': 1.0, 'class_4': 0.0},
        'recall': {'class_0': 1.0, 'class_1': 1.0, 'class_2': 2/3, 'class_3': 0.5, 'class_4': 0.0},
        'f1': {'class_0': 1.0, 'class_1': 2/3, 'class_2': 0.8, 'class_3': 2/3, 'class_4': 0.0},
    }
    print("\n--- ダミーデータと期待値を設定 ---")
    print(f"バッチ数: {len(batches)}")
    print("期待されるAccuracy(CE):")
    for cls, val in expected_results['acc'].items():
        print(f"  {cls}: {val:.4f}")


    # 3. PyTorch LightningのTrainerをダミーとして使用
    #    ロギング機能だけを利用する
    trainer = pl.Trainer(logger=True, accelerator='cpu')
    # LightningModuleにtrainerへの参照を持たせる
    solver.trainer = trainer


    # 4. 手動でテストのライフサイクルを模倣
    # on_test_epoch_start() は明示的にないので不要
    
    # test_stepの模倣: 全バッチで指標の状態を更新
    for batch in batches:
        preds, labels = batch["preds"], batch["labels"]
        # CEヘッドの指標のみをテスト
        for name in solver.metrics_overall.keys():
            if name.startswith('ce_'):
                solver.metrics_overall[name].update(preds, labels)
                solver.metrics_per_class[name].update(preds, labels)
    
    # on_test_epoch_endの呼び出し: 最終計算とロギング
    solver.on_test_epoch_end()
    print("\n--- 指標計算ロジックを実行しました ---")

    # 5. 結果の検証
    logged_metrics = trainer.logged_metrics
    all_tests_passed = True

    print("\n--- 検証結果 ---")
    for metric_name, expected_values in expected_results.items():
        print(f"\n--- {metric_name.upper()} ---")
        for i in range(config['model']['num_classes']):
            class_label = f"class_{i}"
            key = f"test_{metric_name}_ce_{class_label}"
            
            if key in logged_metrics:
                actual_value = logged_metrics[key].item()
                expected_value = expected_values.get(class_label, 0.0)
                
                if np.isclose(actual_value, expected_value, atol=1e-4):
                    print(f"✅ {class_label}: 成功 (期待値: {expected_value:.4f}, 実際値: {actual_value:.4f})")
                else:
                    print(f"❌ {class_label}: 失敗 (期待値: {expected_value:.4f}, 実際値: {actual_value:.4f})")
                    all_tests_passed = False
            else:
                print(f"❌ {class_label}: 失敗 (メトリクス '{key}' がログに見つかりません)")
                all_tests_passed = False

    print("\n" + "="*30)
    if all_tests_passed:
        print("✅✅✅ 全ての検証に成功しました！指標計算ロジックは正常です。 ✅✅✅")
    else:
        print("❌❌❌ 検証に失敗した項目があります。ロジックにまだ問題が残っています。 ❌❌❌")
    print("="*30)


if __name__ == '__main__':
    verify_metrics_calculation()
