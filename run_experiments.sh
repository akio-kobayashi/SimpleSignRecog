#!/bin/bash
# エラーが発生したら、その時点でスクリプトを停止する
set -e

# --- 基本設定 ---
# 全ての実験結果を保存する親ディレクトリ
EXPERIMENT_BASE_DIR="experiments"
# クラス数
NUM_CLASSES=20
# K-Foldの分割数
K_FOLDS=5
# ベースとなるconfigファイル
CONFIG_FILE="config.yaml"

# --- 実験パターンの定義 ---
# ここで試したいAugmentationの組み合わせを定義します。
# 書式: "実験名" "引数"
declare -a EXPERIMENTS=(
    "no_aug" "--augment-flip false --augment-rotate false --augment-noise false"
    "flip_only" "--augment-flip true --augment-rotate false --augment-noise false"
    "rotate_only" "--augment-flip false --augment-rotate true --augment-noise false"
    "noise_only" "--augment-flip false --augment-rotate false --augment-noise true"
    "all_aug" "--augment-flip true --augment-rotate true --augment-noise true"
)

# --- 実験ループ ---
# EXPERIMENTS配列を2つずつ処理 (名前と引数)
for (( i=0; i<${#EXPERIMENTS[@]}; i+=2 )); do
    EXP_NAME=${EXPERIMENTS[i]}
    AUG_ARGS=${EXPERIMENTS[i+1]}
    
    echo "=================================================="
    echo ">> STARTING EXPERIMENT: ${EXP_NAME}"
    echo ">> Augmentations: ${AUG_ARGS}"
    echo "=================================================="

    # --- 1. train.py (K-Fold CV) の実行 ---
    MODE_CV="cv_kfold"
    # この実験の混同行列の保存先 (例: experiments/flip_only/cv_kfold)
    CM_DIR_CV="${EXPERIMENT_BASE_DIR}/${EXP_NAME}/${MODE_CV}"
    
    echo "--- Running K-Fold CV (${K_FOLDS} folds) ---"
    # 学習の実行
    python train.py \
        -c ${CONFIG_FILE} \
        --num-folds ${K_FOLDS} \
        ${AUG_ARGS} \
        --cm-output-dir ${CM_DIR_CV}
    
    # 集計の実行
    python aggregate_results.py ${CM_DIR_CV} \
        --num-classes ${NUM_CLASSES} \
        --report-out "${EXPERIMENT_BASE_DIR}/${EXP_NAME}/${MODE_CV}_report.csv" \
        --stats-out "${EXPERIMENT_BASE_DIR}/${EXP_NAME}/${MODE_CV}_stats.csv"

    # --- 2. train_cross_subject.py (話者間CV) の実行 ---
    MODE_CS="cross_subject"
    CM_DIR_CS="${EXPERIMENT_BASE_DIR}/${EXP_NAME}/${MODE_CS}"

    echo "--- Running Cross-Subject CV ---"
    # 学習の実行
    python train_cross_subject.py \
        -c ${CONFIG_FILE} \
        ${AUG_ARGS} \
        --cm-output-dir ${CM_DIR_CS}

    # 集計の実行
    python aggregate_results.py ${CM_DIR_CS} \
        --num-classes ${NUM_CLASSES} \
        --mode cs \
        --report-out "${EXPERIMENT_BASE_DIR}/${EXP_NAME}/${MODE_CS}_report.csv" \
        --stats-out "${EXPERIMENT_BASE_DIR}/${EXP_NAME}/${MODE_CS}_stats.csv"

done

echo "=================================================="
echo ">> All experiments finished successfully."
echo ">> Results are saved in '${EXPERIMENT_BASE_DIR}' directory."
echo "=================================================="