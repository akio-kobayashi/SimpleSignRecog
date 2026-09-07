#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 INPUT_DIR OUTPUT_ROOT SUBJECT_ID [process_videos.py options]" >&2
    exit 2
fi

input_dir=$1
output_root=$2
subject_id=$3
shift 3

if [[ ! -d "$input_dir" ]]; then
    echo "Input directory does not exist: $input_dir" >&2
    exit 1
fi
if [[ -z "$subject_id" || "$subject_id" == */* ]]; then
    echo "SUBJECT_ID must be a non-empty directory name" >&2
    exit 2
fi

mkdir -p "$output_root"
input_dir=$(realpath "$input_dir")
output_root=$(realpath "$output_root")
raw_output_dir="$output_root/data/$subject_id"
corrected_output_dir="$output_root/corrected/$subject_id"
mkdir -p "$raw_output_dir" "$corrected_output_dir"

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
image_name=${SIMPLE_SIGN_PREPROCESSING_IMAGE:-simple-sign-preprocessing:latest}

docker build \
    --file "$project_dir/docker/Dockerfile.preprocessing" \
    --tag "$image_name" \
    "$project_dir"

docker run --rm \
    --user "$(id -u):$(id -g)" \
    --volume "$input_dir:/input:ro" \
    --volume "$raw_output_dir:/output" \
    "$image_name" \
    --input_root_dir /input \
    --output_base_dir /output \
    --model_asset_path /opt/models/hand_landmarker.task \
    "$@"

docker run --rm \
    --user "$(id -u):$(id -g)" \
    --entrypoint python \
    --volume "$raw_output_dir:/input:ro" \
    --volume "$corrected_output_dir:/output" \
    "$image_name" \
    src/correct_inferred_data.py \
    --input_base_dir /input \
    --output_base_dir /output

echo "Raw data: $raw_output_dir"
echo "Corrected data: $corrected_output_dir"
