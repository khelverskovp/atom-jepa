#!/bin/bash -l
# Run a command in the LUMI ROCm/PyTorch container (see scripts/lumi/README.md):
#   bash scripts/lumi/lumi.sh setup          one-time venv on top of the image
#   bash scripts/lumi/lumi.sh python ...     a command from the repository root
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ATOM_JEPA_ENV:-$ROOT/.env}"
: "${ATOM_JEPA_WORK:?}" "${ATOM_JEPA_IMAGE:?}"
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=2
export WANDB_MODE="${WANDB_MODE:-disabled}" PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export WANDB_DIR="${WANDB_DIR:-$ATOM_JEPA_WORK/cache/wandb}"
export XDG_CACHE_HOME="$ATOM_JEPA_WORK/cache" HF_HOME="$ATOM_JEPA_WORK/cache/huggingface"
export TORCH_HOME="$ATOM_JEPA_WORK/cache/torch" TRITON_CACHE_DIR="$ATOM_JEPA_WORK/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$ATOM_JEPA_WORK/cache/inductor" PIP_CACHE_DIR="$ATOM_JEPA_WORK/cache/pip"
export TMPDIR="$ATOM_JEPA_WORK/tmp"
export SINGULARITYENV_PYTHONPATH="$ROOT:$ATOM_JEPA_WORK/venv/lib/python3.12/site-packages:/opt/venv/lib/python3.12/site-packages"
export SINGULARITYENV_PYTHONNOUSERSITE=1 SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
if [[ "${1:-}" == setup ]]; then
    singularity exec "$ATOM_JEPA_IMAGE" python -m venv --system-site-packages "$ATOM_JEPA_WORK/venv"
    singularity exec "$ATOM_JEPA_IMAGE" python - <<'PY'
import torch
assert torch.version.hip, 'A ROCm image is required'
print(torch.__version__, torch.version.hip)
PY
    singularity exec "$ATOM_JEPA_IMAGE" "$ATOM_JEPA_WORK/venv/bin/python" -m pip install -r "$ROOT/scripts/lumi/requirements.txt" -c "$ROOT/scripts/lumi/constraints.txt"
    exit
fi
COMMAND="${1:?Usage: lumi.sh setup | python ...}"; shift
if [[ -x "$ATOM_JEPA_WORK/venv/bin/$COMMAND" ]]; then COMMAND="$ATOM_JEPA_WORK/venv/bin/$COMMAND"; fi
cd "$ROOT"
EXTRA_BINDS=()
if [[ -f "$ATOM_JEPA_WORK/lib/libXrender.so.1" ]]; then
    EXTRA_BINDS=(--bind "$ATOM_JEPA_WORK/lib/libXrender.so.1:/usr/lib/x86_64-linux-gnu/libXrender.so.1:ro")
fi
exec singularity exec "${EXTRA_BINDS[@]}" "$ATOM_JEPA_IMAGE" "$COMMAND" "$@"
