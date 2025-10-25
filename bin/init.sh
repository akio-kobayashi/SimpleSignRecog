#!/usr/bin/env bash
set -euo pipefail

# ----- 設定（必要なら変更） -----
PY_VER="${PY_VER:-3.12.3}"                  # 永続化するPythonバージョン
UV_URL="https://astral.sh/uv/install.sh"    # uv installer
# 共有キャッシュを管理者が用意しているなら、実行時に --bind で結び付ける（後述）
# --------------------------------

echo "[init] PATHにユーザローカルを追加"
export PATH="$HOME/.local/bin:$PATH"

echo "[init] uv の導入確認"
if ! command -v uv >/dev/null 2>&1; then
  echo "[init] uv をインストール"
  curl -LsSf "${UV_URL}" | sh
fi

echo "[init] uv バージョン: $(uv --version)"

echo "[init] Python ${PY_VER} を永続インストール（ユーザ空間）"
uv python install "${PY_VER}"

PY_PATH="$(uv python find ${PY_VER})"
echo "[init] 解決された Python パス: ${PY_PATH}"

# 初回速度を上げるためにキャッシュディレクトリを作成
mkdir -p "$HOME/.cache/uv" "$HOME/.local/share/uv/python"

cat <<'MSG'

[init] 完了：
- uv を導入
- Python を $HOME/.local/share/uv/python/ 以下に永続配置
- 次は create_project_venv.sh をプロジェクトルートで実行

MSG
