#!/bin/bash

# --- 設定 ---
# config.yamlから取得したクラス数
NUM_CLASSES=20
# 実験結果が格納されているベースディレクトリ
BASE_DIR="experiments"
# aggregate_results.pyを実行するコマンド
PYTHON_CMD="python aggregate_results.py"

# --- スクリプト本体 ---

# ベースディレクトリの存在チェック
if [ ! -d "$BASE_DIR" ]; then
  echo "エラー: '$BASE_DIR' ディレクトリが見つかりません。"
  echo "このスクリプトを実行する前に、別マシンから '$BASE_DIR' ディレクトリをカレントディレクトリにコピーしてください。"
  exit 1
fi

# 処理するaugmentationの種類
AUG_TYPES=("no_aug" "flip_only" "all_aug" "rotate_only" "noise_only")

# 処理する被験者
SUBJECTS=("kobayashi" "yamashita")

echo "--- 結果の集計を開始します ---"

# 各augmentation設定でループ
for aug in "${AUG_TYPES[@]}"; do
  AUG_DIR="$BASE_DIR/$aug"
  if [ -d "$AUG_DIR" ]; then
    echo ""
    echo "================================================="
    echo "--- 処理中: $AUG_DIR"
    echo "================================================="

    # 1. Cross-Subject ('cs') の集計
    # このディレクトリに *_cm.csv が直接含まれていると仮定
    if ls "$AUG_DIR"/*_cm.csv 1> /dev/null 2>&1; then
      echo "[Mode: cs] を '$AUG_DIR' で実行します..."
      $PYTHON_CMD "$AUG_DIR" \
        --mode cs \
        --num-classes $NUM_CLASSES \
        --report-out "$AUG_DIR/cross_subject_report.csv" \
        --stats-out "$AUG_DIR/cross_subject_stats.csv"
      echo "-> '$AUG_DIR/cross_subject_report.csv' と 'cross_subject_stats.csv' を生成しました。"
    else
        echo "警告: '$AUG_DIR' に cross-subject 用の *_cm.csv ファイルが見つかりません。スキップします。"
    fi
    echo ""

    # 2. 被験者ごとの Cross-Validation ('cv') の集計
    for subject in "${SUBJECTS[@]}"; do
      SUBJECT_DIR="$AUG_DIR/$subject"
      if [ -d "$SUBJECT_DIR" ]; then
        if ls "$SUBJECT_DIR"/*_cm.csv 1> /dev/null 2>&1; then
          echo "[Mode: cv] を '$SUBJECT_DIR' で実行します..."
          $PYTHON_CMD "$SUBJECT_DIR" \
            --mode cv \
            --num-classes $NUM_CLASSES \
            --report-out "$SUBJECT_DIR/cv_kfold_report.csv"
          echo "-> '$SUBJECT_DIR/cv_kfold_report.csv' を生成しました。"
        else
          echo "警告: '$SUBJECT_DIR' に *_cm.csv ファイルが見つかりません。スキップします。"
        fi
      else
        echo "情報: ディレクトリが見つからないためスキップします: $SUBJECT_DIR"
      fi
      echo ""
    done
  else
    echo "情報: ディレクトリが見つからないためスキップします: $AUG_DIR"
  fi
done

echo "================================================="
echo "--- 全ての集計処理が完了しました ---"
