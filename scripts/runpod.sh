#!/usr/bin/env bash
# RunPod A100 setup + launch. Run from the repo root, which must live on the persistent volume (/workspace).
#
#   bash scripts/runpod.sh setup    # install deps, run unit tests, GPU smoke run (~5 min) to exercise the CUDA/bf16 path
#   bash scripts/runpod.sh sweep    # full experiment in the background (survives SSH disconnects); log: logs/sweep.log
#   bash scripts/runpod.sh status   # progress: finished stages, latest eval of the newest run, GPU usage
#   bash scripts/runpod.sh analyze  # plots + stats into results/
set -euo pipefail

cd "$(dirname "$0")/.."
case "$(pwd)" in
  /workspace/*) ;;
  *) echo "WARNING: $(pwd) is not under /workspace; outputs will be lost if the pod restarts." >&2 ;;
esac

# Keep model/dataset downloads on the persistent volume too.
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
# Containers often report all host cores while allowing only a few; cap CPU threads to avoid oversubscription.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( $(nproc) < 8 ? $(nproc) : 8 ))}"
PY="${PY:-python}"

case "${1:-}" in
  setup)
    nvidia-smi --query-gpu=name,memory.total --format=csv
    # Use the template's preinstalled CUDA torch; pip only adds what is missing.
    $PY -m pip install -q -e ".[dev]"
    $PY -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, '| bf16:', torch.cuda.is_bf16_supported())"
    $PY -m pytest -q
    rm -rf runs_smoke data_smoke results_smoke
    $PY scripts/06_sweep.py --config configs/base.yaml --config configs/smoke.yaml
    $PY scripts/07_analyze.py --config configs/base.yaml --config configs/smoke.yaml
    echo "Setup OK. GPU smoke results in results_smoke/. Next: bash scripts/runpod.sh sweep"
    ;;
  sweep)
    mkdir -p logs
    nohup $PY -u scripts/06_sweep.py >> logs/sweep.log 2>&1 &
    echo "Sweep started (pid $!). Follow with: tail -f logs/sweep.log"
    ;;
  status)
    grep -E "^\[(done|fail|skip)\]|Selected|WARNING" logs/sweep.log 2>/dev/null || echo "no sweep log yet"
    latest=$(ls -td runs/*/ 2>/dev/null | head -1 || true)
    if [ -n "$latest" ] && [ -f "$latest/metrics.jsonl" ]; then
      echo "latest run: $latest"
      tail -1 "$latest/metrics.jsonl" | $PY -c "import json,sys; r=json.loads(sys.stdin.read()); print({k: r.get(k) for k in ['step','reward_mean','kl_exact','t_train','t_total','best_val_acc','loss']})"
    fi
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
    ;;
  analyze)
    $PY scripts/07_analyze.py
    ;;
  *)
    sed -n '2,8p' "$0"; exit 1 ;;
esac
