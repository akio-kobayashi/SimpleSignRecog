

import torch
import yaml
from src.solver import Solver
import numpy as np
from collections import defaultdict

def verify_metrics_calculation():
    """
    Solverの指標計算ロジックが正しく動作するかを検証する。
    ダミーデータを用いて、クラスごとの指標が期待通りに計算されるかを確認する。
    PyTorch LightningのTrainerやlogging機構に依存せず、torchmetricsのロジックを直接テストする。
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
    # 予測に '9' が出てくるため、少なくとも10以上にする必要がある
    config['model']['num_classes'] = 10
    solver = Solver(config)
    print("Solverを初期化しました。")

    # 2. ダミーデータと期待される結果を定義
    batches = [
        {"preds": torch.tensor([0, 1, 1, 2, 9]), "labels": torch.tensor([0, 0, 1, 2, 2])},
        {"preds": torch.tensor([0, 2, 3, 9]), "labels": torch.tensor([0, 2, 3, 3])},
    ]

    # --- 正しい手計算による期待値 ---
    # Labels: [0, 0, 1, 2, 2, 0, 2, 3, 3] (Support: 0:3, 1:1, 2:3, 3:2)
    # Preds:  [0, 1, 1, 2, 9, 0, 2, 3, 9]
    #
    # TP: {0:2, 1:1, 2:2, 3:1}
    # FP: {0:0, 1:1, 2:0, 3:0} (Pred 1 は True 0 からの誤予測)
    # FN: {0:1, 1:0, 2:1, 3:1} (True 0->Pred 1, True 2->Pred 9, True 3->Pred 9)
    #
    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    expected_results = {
        'precision': {'class_0': 1.0, 'class_1': 0.5, 'class_2': 1.0, 'class_3': 1.0, 'class_4': 0.0},
        'recall':    {'class_0': 2/3, 'class_1': 1.0, 'class_2': 2/3, 'class_3': 0.5, 'class_4': 0.0},
        'f1':        {'class_0': 0.8, 'class_1': 2/3, 'class_2': 0.8, 'class_3': 2/3, 'class_4': 0.0},
    }
    print("\n--- ダミーデータと期待値を設定 ---")
    print(f"バッチ数: {len(batches)}")
    print("期待されるRecall(CE):")
    for i in range(5):
        print(f"  class_{i}: {expected_results['recall'].get(f'class_{i}', 0.0):.4f}")

    # 3. 手動でtest_stepのロジックを模倣
    for batch in batches:
        preds, labels = batch["preds"], batch["labels"]
        for name in solver.metrics_per_class.keys():
            if name.startswith('ce_'):
                solver.metrics_per_class[name].update(preds, labels)
    print("\n--- 全ダミーバッチで指標の状態を更新しました ---")

    # 4. 最終計算と結果の検証
    all_tests_passed = True
    print("\n--- 検証結果 ---")
    for metric_name, expected_values in expected_results.items():
        print(f"\n--- {metric_name.upper()} ---")
        
        metric_key = f"ce_{metric_name}"
        if metric_key not in solver.metrics_per_class:
            print(f"❌ エラー: 指標オブジェクト '{metric_key}' がSolverに見つかりません。")
            all_tests_passed = False
            continue
        
        metric_obj = solver.metrics_per_class[metric_key]
        per_class_values = metric_obj.compute()
        
        # num_classesを5に戻してループ（ダミーデータのクラスは0-4の範囲で検証）
        for i in range(5):
            class_label = f"class_{i}"
            actual_value = per_class_values[i].item()
            expected_value = expected_values.get(class_label, 0.0)
            
            if np.isclose(actual_value, expected_value, atol=1e-4):
                print(f"✅ {class_label}: 成功 (期待値: {expected_value:.4f}, 実際値: {actual_value:.4f})")
            else:
                print(f"❌ {class_label}: 失敗 (期待値: {expected_value:.4f}, 実際値: {actual_value:.4f})")
                all_tests_passed = False
        
        metric_obj.reset()

    print("\n" + "="*30)
    if all_tests_passed:
        print("✅✅✅ 全ての検証に成功しました！指標計算ロジックは正常です。 ✅✅✅")
    else:
        print("❌❌❌ 検証に失敗した項目があります。ロジックにまだ問題が残っています。 ❌❌❌")
    print("="*30)


if __name__ == '__main__':
    verify_metrics_calculation()
