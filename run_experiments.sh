#!/bin/bash
# エラーが発生したら、その時点でスクリプトを停止する
set -e

# --- 実行モードの選択 ---
# 第1引数で実行する実験タイプを指定 (loo, hybrid, cs, all)
# 指定がない場合は 'all' をデフォルトとする
MODE=${1:-all}
echo "Running experiment mode: $MODE"
echo ""

# --- 基本設定 ---
# 全ての実験結果を保存する親ディレクトリ
EXPERIMENT_BASE_DIR="experiments"
# クラス数
NUM_CLASSES=20
# K-Foldの分割数
K_FOLDS=-1
# ベースとなるconfigファイル
CONFIG_FILE_K=kobayashi_cv.yaml
CONFIG_FILE_Y=yamashita_cv.yaml
CONFIG_CR_FILE=cross.yaml

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

    # --- 1. Standard LOO (train.py) ---
    if [[ "$MODE" == "loo" || "$MODE" == "all" ]]; then
        echo "--- Running Standard LOO Experiments ---"
        for SPEAKER in kobayashi yamashita; do
            echo "--- Processing for speaker: $SPEAKER ---"
            case "$SPEAKER" in
                kobayashi)
            CONFIG_FILE="$CONFIG_FILE_K"
            ;;
                yamashita)
            CONFIG_FILE="$CONFIG_FILE_Y"
            ;;
                *)
            # 予期しない話者の場合はエラーとして終了
            echo "Error: Unknown speaker '$SPEAKER'"
            exit 1
            ;;
        esac
        echo "Using config file: $CONFIG_FILE"
        
        MODE_LOO="standard_loo"
        # この実験の混同行列の保存先
        CM_DIR_LOO="${EXPERIMENT_BASE_DIR}/${EXP_NAME}/${SPEAKER}/${MODE_LOO}"
        
        echo "--- Running Standard LOO for $SPEAKER ---"
        # 学習の実行
        python train.py \
                   -c ${CONFIG_FILE} \
                   --num-folds ${K_FOLDS} \
                   ${AUG_ARGS} \
                   --cm-output-dir ${CM_DIR_LOO}
        
        # 集計の実行 (train.pyもcm_total.csvを出力するようになったため、それを対象にする)
        python aggregate_results.py ${CM_DIR_LOO} \
                --num-classes ${NUM_CLASSES} \
                --mode cv \
                --target-file "cm_total.csv" \
                --report-out "${EXPERIMENT_BASE_DIR}/${EXP_NAME}/${SPEAKER}/${MODE_LOO}_report.csv"
        done
    fi

    # --- 2. Hybrid LOO (train_hybrid_loo.py) ---
    if [[ "$MODE" == "hybrid" || "$MODE" == "all" ]]; then
        echo "--- Running Hybrid LOO Experiment ---"
        MODE_HYBRID="hybrid_loo"
        CM_DIR_HYBRID="${EXPERIMENT_BASE_DIR}/${EXP_NAME}/${MODE_HYBRID}"

        # 学習の実行
        python train_hybrid_loo.py \
            -c ${CONFIG_FILE_K} ${CONFIG_FILE_Y} \
            --num-folds -1 \
            ${AUG_ARGS} \
            --cm-output-dir ${CM_DIR_HYBRID}

        # 集計の実行
        python aggregate_results.py ${CM_DIR_HYBRID} \
            --num-classes ${NUM_CLASSES} \
            --mode cv \
            --target-file "cm_total.csv" \
            --report-out "${EXPERIMENT_BASE_DIR}/${EXP_NAME}/${MODE_HYBRID}_report.csv"
    fi

    # --- 3. Cross-Subject CV (train_cross_subject.py) ---
    if [[ "$MODE" == "cs" || "$MODE" == "all" ]]; then
        echo "--- Running Cross-Subject CV Experiment ---"
        MODE_CS="cross_subject"
        CM_DIR_CS="${EXPERIMENT_BASE_DIR}/${EXP_NAME}/${MODE_CS}"

        # 学習の実行
        python train_cross_subject.py \
            -c ${CONFIG_CR_FILE} \
            ${AUG_ARGS} \
            --cm-output-dir ${CM_DIR_CS}

        # 集計の実行
        python aggregate_results.py ${CM_DIR_CS} \
            --num-classes ${NUM_CLASSES} \
            --mode cs \
            --report-out "${EXPERIMENT_BASE_DIR}/${EXP_NAME}/${MODE_CS}_report.csv" \
            --stats-out "${EXPERIMENT_BASE_DIR}/${EXP_NAME}/${MODE_CS}_stats.csv"
    fi
    fi

done

echo "=================================================="
echo ">> All experiments finished successfully."
echo ">> Results are saved in '${EXPERIMENT_BASE_DIR}' directory."
echo "=================================================="
