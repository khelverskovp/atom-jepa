# ADMET workflows

Run commands from the repository root. The main training and HPO entry points
remain in `finetuning.admet`.

## Layout

| Location | Responsibility |
| --- | --- |
| `baseline.py`, `hpo.py`, `finetune_*.py` | Main feature-based and encoder fine-tuning workflows |
| `training/` | PyTorch and LightGBM training, losses, regularization, distributed execution, utilities |
| `training/train.py` | Coordinates multi-task training |
| `training/preparation.py` | Data loaders, train-only normalization, task indices, and label weights |
| `training/state.py` | Model and optimizer construction, learning rates, and training resume |
| `training/epoch.py` | Forward loss, optimizer steps, and one training epoch |
| `finetuning/execution.py` | Encoder precision, grid-MLP optimization, and block compilation |
| `training/checkpoints.py` | Checkpoint selection, metadata, and saving |
| `training/single_task.py` | Single-task training, validation selection, and train+validation refitting |
| `training/helpers.py` | Scaling, tensor reduction, logging, schedules, gradient norms, and checkpoint helpers |
| `training/evaluation.py` | Epoch validation, test/final evaluation, metric logging, and result assembly |
| `models/` | Encoder wrappers, readout heads, and feature-vector MLP |
| `features/` | Offline Mol-JEPA embedding generation |
| `metrics/` | Shared scoring functions, classification metrics, interval metrics, reporting scales |
| `reporting/` | Combine completed runs and score external predictions |
| `experiments/` | Additional fine-tuning HPO, sweeps, MoleculeNet, and validation grids |
| `../../data/datasets/admet/` | Dataset loaders, feature extraction, and dataset classes |
| `../../data/datasets/admet/download/` | Dataset download commands |

Each Python subdirectory is a package. Supporting modules have moved; import
them using their new locations, for example
`from finetuning.admet.training.train import train_multitask`.

## Download datasets

```bash
python -m data.datasets.admet.download.biogen_adme
python -m data.datasets.admet.download.chembl_mt
python -m data.datasets.admet.download.expansionrx
```

The TDC benchmark group downloads on first use through `load_admet_group`.
Use `--help` on a download command to see its destination and overwrite options.

## Atom-JEPA execution speedups

Biogen, ExpansionRX, ChEMBL-MT, and TDC fine-tuning enable these defaults:

```yaml
finetune:
  batch_size: 32
  grad_checkpointing: false
  precision: bf16
  cuequivariance: true
  optimize_grid_mlp: true
  compile_blocks: true
  compile_mode: default
  compile_dynamic: true
```

The standard fine-tuning entry points now install the tested `combo_node`
operations directly: compact gather/scaling/rotation, attention-weighted inverse
rotation/scatter, gated spherical-grid products, initial rotation/scatter,
per-node attention embeddings, and packed SO2 GEMMs. Configuration is per model;
other workflows retain their existing execution unless explicitly enabled.

Install the tested dependencies for PyTorch/CUDA 13:

```bash
python -m pip install -r finetuning/requirements-cue-cu13.txt
```

For CUDA 12, use the corresponding cu12 operator wheels with a compatible
PyTorch installation. `finetune.cuequivariance=false` selects native PyTorch
operations. CPU execution skips CUDA fusions. Missing CUDA dependencies and
unsupported encoders fail explicitly; the supported contraction layout is
`lmax=mmax=2` with gated SwiGLU grid products and inactive grid dropout.

Fused operators contain only derived constants and introduce no checkpoint
keys or learned parameters. Native and newly integrated fused checkpoints load
strictly into either execution path. Resume a
standard-driver run with `+finetune.resume=true` and the same checkpoint directory.

These defaults change execution and batch size. Learning rates, EMA decay,
weight decay, and L2-SP remain dataset configuration choices; the Biogen LR
experiments are not applied globally. To reproduce the first batch-32 Biogen
schedule, explicitly set `finetune.lr=0.0001414213562373095`,
`finetune.mtl_lr=0.0001414213562373095`,
`finetune.min_lr=0.000000282842712474619`, and
`finetune.ema_decay=0.9227446944279201`, together with the desired scaffold split,
seed, and `train_on_val` settings.

The QM9 execution changes also apply to ADMET: encoder BF16 with FP32 geometry,
normalization reductions, readout and loss; reordered grid-MLP channel linears;
in-place transformer-block compilation; explicit Wigner inputs for compilation
and activation checkpointing; the GraphDropPath fast path using the collator's
graph count; and device-side loss/prediction accumulation. The backbone changes
are shared with QM9 rather than duplicated here.

Training and evaluation use the same encoder precision, including EMA and
frozen-encoder epochs. Parameters and optimizer state remain FP32. Compilation
happens after resume and EMA copying, preserving parameter identities and
checkpoint keys. CPU runs use FP32 and skip compilation. CUDA BF16 requires a
supported GPU; compiler errors are surfaced rather than silently ignored.
Compilation adds startup cost and may recompile after unfreezing or shape changes.
Prediction buffers stay on the device until the end of each epoch/evaluation,
so their memory use scales with the number of conformer predictions and tasks.

CUDA training uses fused AdamW and batched EMA/L2-SP parameter updates. Loading
older optimizer checkpoints preserves their moments and learning rates while
selecting the current device's optimizer implementation. CPU training retains
ordinary AdamW. EMA synchronizes audited fixed Equiformer bases/indices once
after resume, and still copies mutable or unknown buffers on every update.
L2-SP still skips frozen parameters. With cuEquivariance enabled, gradient
clipping raises on non-finite BF16/FP32 gradients; FP16 retains GradScaler
overflow handling.

Tune batch size and activation checkpointing together using molecules/second,
not step time alone. Check peak memory on large molecules as well as typical
batches, and compare validation error before changing a publication run's batch
size. The larger-batch experiments use compiled BF16 blocks and these updates
throughout; frozen-feature caching and task-head vectorization are excluded.

For the FP32/eager execution path, use:

```bash
python -m finetuning.admet.finetune_biogen_adme \
    finetune.ckpt_path=/path/to/encoder.pt \
    finetune.precision=fp32 finetune.optimize_grid_mlp=false \
    finetune.compile_blocks=false finetune.cuequivariance=false
```

`precision` takes precedence over legacy `amp`/`amp_dtype` settings. Configs
without `precision` still accept those settings; FP16 uses a gradient scaler.
Feature encoders without an Equiformer body skip block/grid optimizations.
BF16 and reordered arithmetic can change rounding and training trajectories;
measure steady-state speed and validation quality before comparing full runs.

Run `python -m unittest discover -s finetuning/admet/tests -v` for regressions.
The CUDA BF16/Inductor test skips when no CUDA GPU is available.
The cuEquivariance tests check output/gradient parity, per-model isolation,
checkpoint compatibility, single-task heads, and empty-edge graphs. Integration
was also tested through a complete Biogen training/evaluation epoch and a
standard-driver checkpoint resume on L40S. Full-dataset end-to-end reruns of
ExpansionRX, ChEMBL-MT, and TDC were not part of this integration check.

## HPO

```bash
python -m finetuning.admet.hpo \
    dataset=biogen_adme data.split=cluster model=morgan_rdkit_lgbm \
    hpo.n_trials=500 hpo.final_full=false
```

Feature-based HPO currently implements Biogen ADME only. `hpo.final_full=true`
runs the final seed evaluation after selecting parameters on validation data.
Encoder fine-tuning HPO is a separate, optional command:
`python -m finetuning.admet.experiments.hpo_finetune`.

## Train and evaluate

```bash
python -m finetuning.admet.baseline dataset=biogen_adme model=morgan_rdkit_lgbm
python -m finetuning.admet.baseline dataset=expansionrx model=morgan_rdkit_lgbm
python -m finetuning.admet.baseline dataset=chembl_mt model=morgan_rdkit_lgbm
python -m finetuning.admet.baseline dataset=admet_tdc model=morgan_rdkit_lgbm

python -m finetuning.admet.finetune_biogen_adme finetune.ckpt_path=/path/to/encoder.pt
python -m finetuning.admet.finetune_expansionrx finetune.ckpt_path=/path/to/encoder.pt
python -m finetuning.admet.finetune_chembl_mt finetune.ckpt_path=/path/to/encoder.pt
python -m finetuning.admet.finetune_admet finetune.ckpt_path=/path/to/encoder.pt
```

These drivers include test evaluation. Existing Hydra dataset, model, seed,
split, and checkpoint overrides are unchanged. Add `--cfg job` to inspect a
driver's composed configuration without training.

## Aggregate completed runs

```bash
python -m finetuning.admet.reporting.aggregate_runs --help
python -m finetuning.admet.reporting.aggregate_biogen_adme_seeds --help
python -m finetuning.admet.reporting.aggregate_admet_tdc --help
python -m finetuning.admet.reporting.aggregate_admet_tdc_seeds --help
```

External KERMT prediction scoring is available as
`python -m finetuning.admet.reporting.score_kermt_predictions`.

## Mol-JEPA embeddings

The embedding generator lives at `finetuning/admet/features/moljepa_embed.py`.
Run it with the separate Mol-JEPA environment as described in its module
docstring. The baseline driver reads the cached embeddings through
`finetuning/admet/features/moljepa_features.py`. No Mol-JEPA model presets are
included in `conf/model/`.
