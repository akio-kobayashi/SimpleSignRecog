#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 SUBJECT_ID [process_videos.py options]" >&2
    echo "SUBJECT_ID: subject_01, subject_02, or subject_03" >&2
    exit 2
fi

subject_id=$1
shift

case "$subject_id" in
    subject_01|subject_02|subject_03) ;;
    *)
        echo "SUBJECT_ID must be subject_01, subject_02, or subject_03" >&2
        exit 2
        ;;
esac

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

data_root=${EGO_SIGN_DATA_ROOT:-/srv/share/ego_sign_recog}
input_dir="$data_root/videos/$subject_id"
output_root="$data_root/processed"

if [[ ! -d "$input_dir" ]]; then
    echo "Input directory does not exist: $input_dir" >&2
    exit 1
fi

raw_output_dir="$output_root/data/$subject_id"
corrected_output_dir="$output_root/corrected/$subject_id"
model_dir="$output_root/models"
model_path="$model_dir/hand_landmarker.task"
mkdir -p "$raw_output_dir" "$corrected_output_dir" "$model_dir"

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
requirements="$project_dir/requirements-preprocessing.txt"
uv_run=(uv run --isolated --python 3.13 --with-requirements "$requirements")

if [[ ! -s "$model_path" ]]; then
    model_url=https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
    temp_model="$model_path.tmp.$$"
    trap 'rm -f "$temp_model"' EXIT
    "${uv_run[@]}" python -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$model_url" "$temp_model"
    mv "$temp_model" "$model_path"
    trap - EXIT
fi

cd "$project_dir"
"${uv_run[@]}" python src/process_videos.py \
    --input_root_dir "$input_dir" \
    --output_base_dir "$raw_output_dir" \
    --model_asset_path "$model_path" \
    "$@"

"${uv_run[@]}" python src/correct_inferred_data.py \
    --input_base_dir "$raw_output_dir" \
    --output_base_dir "$corrected_output_dir"

echo "Raw data: $raw_output_dir"
echo "Corrected data: $corrected_output_dir"
