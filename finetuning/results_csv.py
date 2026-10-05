"""
One-row-per-run results log, shared by finetuning/qm9/finetune.py and finetuning/probing/qm9_frozen_encoder_probe.py.

Both scripts append to the SAME csv (misc.results_csv) with the same schema, so
a 12-target x 3-encoder x {finetune, probe} sweep collects into a single table:

    import pandas as pd
    df = pd.read_csv("results/qm9_results.csv")
    df.pivot_table(index="target", columns=["script", "encoder"], values="test_mae")

Concurrency: SLURM array tasks all append to one file. Each write is a single
short line guarded by an exclusive flock, so interleaving cannot corrupt a row
and only one job can win the header write.
"""

import csv
import os
from datetime import datetime, timezone
from typing import Dict, Optional

try:                        # POSIX only; degrade to unlocked appends elsewhere
    import fcntl
except ImportError:         # pragma: no cover
    fcntl = None


# Shared schema. Columns a given script has no value for are left blank rather
# than dropped, so both scripts' rows line up in one file.
FIELDNAMES = [
    "timestamp",        # UTC, ISO-8601
    "script",           # "finetune" | "probe_head"
    "encoder",          # checkpoint tag, e.g. qm9_y1_atgt_eq_Lmax6_resume1
    "target",           # QM9 target
    "unit",             # native unit of the target
    "val_mae",          # best val MAE (native units)
    "test_mae",         # test MAE at the best-val-MAE epoch
    "best_epoch",
    "epochs_run",       # epochs actually completed (early stopping may cut this)
    "n_head_params",    # trainable params in the readout head
    "layers",           # probe only: which encoder layers feed the head
    "f_in",             # head scalar input width
    "f_vec",            # head l=1 input width (mu only)
    "pool",
    "loss",
    "monitor",
    "standardize",
    "use_atom_ref",
    "seed",
    "split_seed",
    "n_train",
    "n_val",
    "n_test",
    "run_name",
    "ckpt_path",
    "slurm_job_id",
]


def encoder_tag(ckpt_path: str) -> str:
    """checkpoint/context_encoder_qm9_atgt_resume1.pt -> qm9_atgt_resume1"""
    stem = os.path.splitext(os.path.basename(str(ckpt_path)))[0]
    prefix = "context_encoder_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def append_row(path: Optional[str], row: Dict) -> Optional[str]:
    """Append one run to `path`, writing the header if the file is new.

    Returns the path written, or None when `path` is falsy (logging disabled).
    Never raises: a results-logging failure must not lose a finished run, so
    problems are reported and swallowed."""
    if not path:
        return None
    try:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)

        row = dict(row)
        row.setdefault("timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        row.setdefault("slurm_job_id", os.environ.get("SLURM_JOB_ID", ""))
        task = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task is not None and row.get("slurm_job_id"):
            row["slurm_job_id"] = f"{os.environ.get('SLURM_ARRAY_JOB_ID', row['slurm_job_id'])}_{task}"

        with open(path, "a", newline="") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES,
                                        extrasaction="ignore", restval="")
                if os.fstat(f.fileno()).st_size == 0:
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return path
    except Exception as e:                                   # pragma: no cover
        print(f"[results] WARNING: could not append to {path}: "
              f"{type(e).__name__}: {e}", flush=True)
        return None