# Second-stage pretraining on LUMI (AMD MI250X)

Atom-JEPA-Bio was trained on LUMI: 32 MI250X GCDs (4 nodes x 8), 32 molecules per GCD,
with `conf/pretrain2_lumi.yaml`. That profile changes only how the model executes:

- `no_radial`: selective BF16. The SO(2) and linear projections run in BF16; the
  radial networks, the parameters and the outputs stay FP32 (`atom_jepa/mixed_precision.py`).
- packed SO(2) GEMMs and the reassociated S2-grid activation.
- a Triton kernel for the S2-grid product on ROCm (`atom_jepa/rocm_grid.py`; lmax = mmax = 2).

The objective, batch and schedule are those of `conf/pretrain2.yaml`. For an FP32
reference run on AMD, keep `--config-name pretrain2_lumi` and set
`misc.precision_policy=fp32 misc.optimize_grid_mlp=false misc.optimize_packed_so2=false
misc.optimize_rocm_grid=false`.

## Setup

Use the LUMI ROCm/PyTorch container with Python 3.12, PyTorch 2.11 and ROCm 7.2.
`constraints.txt` pins the packages installed next to the container's PyTorch and
Triton. Copy `env.example` to `.env` in the repository root and set the container
path, the scratch workspace and the compute account. Then, inside a GPU allocation:

```bash
bash scripts/lumi/lumi.sh setup
```

## Train

From the repository root, check the resolved configuration, then submit:

```bash
source .env
bash scripts/lumi/lumi.sh python -m pretraining.train2 --config-name pretrain2_lumi \
    data.root=/path/to/pcba init.ckpt_path=/path/to/train_state_stage1.pt --cfg job --resolve
sbatch --account="$ATOM_JEPA_ACCOUNT" scripts/lumi/train2.sbatch \
    data.root=/path/to/pcba init.ckpt_path=/path/to/train_state_stage1.pt
```

`train2.sbatch` allocates one GCD. For more, override the Slurm resources (Atom-JEPA-Bio:
`--nodes=4 --gpus-per-node=8`) and set `ATOM_JEPA_RANKS_PER_NODE` to the GCDs per node.
The launcher starts one torchrun parent per node. Batch size and loader workers are per
rank, so more ranks mean a larger global batch; `data.lr` is not rescaled for you.

Validation and checkpoints run at the end of every epoch. Resume from a completed
epoch with `misc.resume=true misc.resume_path=/path/to/train_state_<run>_epochNNNN.pt`,
keeping the data, assays, schedule, world size and execution settings unchanged. This
continues the training state; it does not replay the distributed RNG streams exactly.

## Inference and fine-tuning on AMD

The `atom_jepa` package runs in FP32 on ROCm by default (`AtomJEPA.from_pretrained`);
`precision="no_radial"` selects the selective BF16 path. For the fine-tuning scripts,
use `finetune.cuequivariance=false finetune.precision=no_radial finetune.compile_blocks=false`.
These switches never change the learned parameters or the checkpoint keys.
