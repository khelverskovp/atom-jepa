"""
ExpansionRX multi-task ADME benchmark: data loading only (see
data/datasets/admet/multitask_finetune.py for the encoder-ready Dataset).

9-task regression benchmark from the OpenADMET ExpansionRX challenge
(huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data,
CC-BY-4.0). Run `python -m data.datasets.admet.download.expansionrx` first to download into
data/expansionrx/.

Format: wide CSV, `Molecule Name` + `SMILES` + 9 task columns (LogD, KSOL,
HLM CLint, MLM CLint, Caco-2 Permeability Papp A>B, Caco-2 Permeability
Efflux, MPPB, MBPB, MGMB). Missing label = empty cell (NaN via pandas -- same
convention as data/datasets/admet/chembl_mt.py). Much denser than ChEMBL-MT: mean 4.74/9
labels per molecule (vs. 1.12/25), though MGMB (4.2% train coverage) and MBPB
(18.3%) are still sparse enough that their per-task metrics will be noisy.

Only LogD is already a log quantity; the other 8 endpoints are raw
(un-transformed) values, including exact zeros in HLM/MLM CLint (~3-4% of
their valid labels) -- see finetuning.admet.training.loss.fwd_target_mt's docstring for why
this needs no LLOQ-floor workaround with this repo's log1p-based transform.

Split: train (5,326 rows) / test (2,282 rows) is FIXED and TEMPORAL -- verified
zero SMILES overlap, and `Molecule Name`'s numeric ID (`E-0001321`..`E-0020100`
in train, `E-0020101`..`E-0027239` in test) is 100% monotonically increasing
with row order, with every test ID exceeding every train ID. Re-splitting
would leak future molecules into training. No validation split ships with the
data (unlike ChEMBL-MT's fold_0/fold_1), so get_train_valid carves one from
the TAIL of train (already ID-ordered, no re-sorting needed) -- preserving the
same "never validate on the past relative to train" discipline.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

SMILES_COL = "SMILES"
LABEL_COLS = ["LogD", "KSOL", "HLM CLint", "MLM CLint", "Caco-2 Permeability Papp A>B",
             "Caco-2 Permeability Efflux", "MPPB", "MBPB", "MGMB"]
# all endpoints except LogD are raw-scale and need log1p; LogD is already log-scale.
LOG_TRANSFORM_TASKS = [t for t in LABEL_COLS if t != "LogD"]

# Endpoints the Contrastive KERMT paper (arXiv:2606.11508) log10s before
# scoring -- its ExpansionRX table names them LogD, Log_KSOL, Log_HLM_CLint,
# Log_MLM_CLint, Log_Caco2_Papp_A>B, Log_Caco2_Efflux, Log_MPPB, Log_MBPB,
# Log_MGMB, i.e. everything but LogD. The membership coincides with
# LOG_TRANSFORM_TASKS above, but the two lists mean different things and are
# deliberately kept separate: LOG_TRANSFORM_TASKS is the TRAINING target
# transform (log1p, chosen because it is defined at the exact zeros in HLM/MLM
# CLint), whereas this is only a REPORTING ruler used to put our MAEs on the
# same axis as the paper's. See finetuning/admet/metrics/report_scale.py.
KERMT_LOG10_TASKS = [t for t in LABEL_COLS if t != "LogD"]

# Per-endpoint test MAE reported for the best Contrastive KERMT configuration
# (Appendix G, Figure 12; overall 0.359), for side-by-side reference. Same
# temporal split and same 5-seed protocol we use -- see this module's docstring.
KERMT_REFERENCE_MAE = {
    "LogD": 0.393,
    "KSOL": 0.373,
    "HLM CLint": 0.413,
    "MLM CLint": 0.410,
    "Caco-2 Permeability Papp A>B": 0.453,
    "Caco-2 Permeability Efflux": 0.447,
    "MPPB": 0.228,
    "MBPB": 0.220,
    "MGMB": 0.290,
}
KERMT_REFERENCE_MACRO = 0.359


@dataclass
class ExpansionRXData:
    tasks: List[str]
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def load(data_dir: str) -> ExpansionRXData:
    """Read the 2 CSVs once."""
    def _read(name: str) -> pd.DataFrame:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found -- run `python -m data.datasets.admet.download.expansionrx` first."
            )
        df = pd.read_csv(path)
        missing = [c for c in [SMILES_COL] + LABEL_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{path}: missing expected column(s) {missing} -- got {list(df.columns)}. "
                f"If this has 10 endpoint columns instead of 9, you loaded "
                f"expansion_data_raw.csv (censored values, RLM CLint) by mistake."
            )
        return df

    train_df = _read("expansion_data_train.csv")
    test_df = _read("expansion_data_test.csv")
    return ExpansionRXData(tasks=LABEL_COLS, train_df=train_df, test_df=test_df)


def get_train_valid(data: ExpansionRXData, val_frac: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carve validation from the TAIL of train (rows are already in temporal/ID
    order -- see module docstring), so the internal split respects the same
    "never validate on data from before training" discipline as the official
    train/test split."""
    n = len(data.train_df)
    n_val = max(1, int(round(n * val_frac)))
    train_sub = data.train_df.iloc[: n - n_val].reset_index(drop=True)
    valid_sub = data.train_df.iloc[n - n_val:].reset_index(drop=True)
    return train_sub, valid_sub
