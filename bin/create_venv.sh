#!/usr/bin/env bash
set -euo pipefail

# ----- 既定設定（必要なら変更） -----
DEFAULT_PY_VER="3.12.3"
TORCH_INDEX="https://download.pytorch.org/whl/cu126"
VENV_DIR=".venv"
REQ_FILE=""
# ------------------------------------

usage() {
  echo "Usage: $0 [-p PY_VER] [-r REQUIREMENTS_OR_PYPROJECT]"
  exit 1
}

while getopts ":p:r:" opt; do
  case $opt in
    p) PY_VER="$OPTARG" ;;
    r) REQ_FILE="$OPTARG" ;;
    *) usage ;;
  esac
done

PY_VER="${PY_VER:-$DEFAULT_PY_VER}"
export PATH="$HOME/.local/bin:$PATH"

echo "[venv] uv の存在確認"
command -v uv >/dev/null 2>&1 || { echo "uv が見つかりません。先に init.sh を実行してください。"; exit 1; }

echo "[venv] Python ${PY_VER} の導入確認"
uv python install "${PY_VER}"
PY_PATH="$(uv python find ${PY_VER})"
echo "[venv] 使用する Python: ${PY_PATH}"

echo "[venv] プロジェクト仮想環境 ${VENV_DIR} を作成"
uv venv -p "${PY_VER}" "${VENV_DIR}"
# シェル種に依らず有効化
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# 依存インストール
if [[ -n "${REQ_FILE}" ]]; then
  echo "[venv] ${REQ_FILE} から依存を同期（lockがあれば uv sync を推奨）"
  case "${REQ_FILE}" in
    *.txt)
      uv pip install -r "${REQ_FILE}"
      ;;
    pyproject.toml)
      # lock運用が理想： uv lock → uv sync
      uv pip install -r <(uv pip compile "${REQ_FILE}")
      ;;
    *)
      echo "未対応のファイル形式です: ${REQ_FILE}"
      exit 1
      ;;
  esac
else
  echo "[venv] 既定セットをインストール（GPU対応）"
  uv pip install --index-url "${TORCH_INDEX}" \
    torch torchvision torchaudio

  uv pip install \
    pytorch-lightning torchmetrics \
    einops numpy tqdm pandas scipy numba h5py seaborn \
    opencv-python ffmpeg-python \
    librosa soundfile \
    mediapipe \
    tensorboard \
    transformers \
    onnxruntime-gpu
fi

echo "[venv] 動作確認"
python - <<'PY'
import torch, onnxruntime as ort, cv2, mediapipe as mp
print("CUDA:", torch.version.cuda, "| GPU available:", torch.cuda.is_available())
print("ONNX Runtime Providers:", ort.get_available_providers())
PY

cat <<'MSG'

[venv] 完了：
- プロジェクト仮想環境 .venv を作成しました
- 次回は「source ./bin/.venv/bin/activate」で再利用できます
- 依存を固定する場合は「uv lock」「uv sync」の運用を推奨します

MSG
