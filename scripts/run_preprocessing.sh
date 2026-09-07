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

data_root=${EGO_SIGN_DATA_ROOT:-/srv/share/ego_sign_recog}
input_dir="$data_root/videos/$subject_id"
output_root="$data_root/processed"

if [[ ! -d "$input_dir" ]]; then
    echo "Input directory does not exist: $input_dir" >&2
    exit 1
fi

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
