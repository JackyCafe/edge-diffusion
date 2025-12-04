#!/usr/bin/env bash
set -e

echo "[postCreate] Env check..."
python - <<'PY'
import os, torch, sys
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
print("Dataset mount exists:", os.path.exists("/data/ai-model"))
print("Workspace root exists:", os.path.isdir("/workspaces"))
PY

# 避免 git safe.directory 警告
git config --global --add safe.directory "*" || true
# 建議的子資料夾
mkdir -p /data/outputs/{checkpoints,logs,tensorboard,runs}
echo "Outputs dir: $(ls -1 /data/outputs | xargs echo)"
echo "[postCreate] done."
