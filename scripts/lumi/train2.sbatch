#!/bin/bash -l
#SBATCH --job-name=train2
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=mi250:1
#SBATCH --cpus-per-task=7
#SBATCH --threads-per-core=1
#SBATCH --mem=60G
#SBATCH --time=01:00:00
# Second-stage pretraining on LUMI: one torchrun parent per node, one rank per GCD.
# Atom-JEPA-Bio used 4 nodes x 8 GCDs (32 ranks); for that, override --nodes=4,
# --gpus-per-node=8 (and memory/CPUs/time) and set ATOM_JEPA_RANKS_PER_NODE=8.
#   sbatch --account="$ATOM_JEPA_ACCOUNT" scripts/lumi/train2.sbatch \
#       data.root=/path/to/pcba init.ckpt_path=/path/to/train_state_stage1.pt
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export ATOM_JEPA_RANKS_PER_NODE="${ATOM_JEPA_RANKS_PER_NODE:-1}"
export ATOM_JEPA_MASTER_PORT="${ATOM_JEPA_MASTER_PORT:-29500}"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export ATOM_JEPA_MASTER_ADDR="${nodes[0]}"
# One torchrun parent per node; one child per allocated GCD. Preserve visibility.
srun --export=ALL --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --cpu-bind=cores \
    bash -c 'exec bash scripts/lumi/lumi.sh python -m torch.distributed.run \
        --nnodes="$SLURM_NNODES" --nproc_per_node="$ATOM_JEPA_RANKS_PER_NODE" \
        --node_rank="$SLURM_PROCID" --master_addr="$ATOM_JEPA_MASTER_ADDR" \
        --master_port="$ATOM_JEPA_MASTER_PORT" -m pretraining.train2 \
        --config-name pretrain2_lumi "$@"' train2-launch "$@"
