"""Pretraining dataset registry.

To pretrain on a new dataset:
  1. write data/datasets/<name>.py with a Dataset class whose samples follow the
     format in data/README.md, a class attribute `periodic` (True for crystals),
     and a `from_config(data_cfg)` classmethod;
  2. add it to PRETRAINING_DATASETS below;
  3. add conf/data/<name>.yaml and run with `data=<name>`.
"""

from data.datasets.alexandria import AlexandriaDataset
from data.datasets.qm9 import QM9Dataset
from data.datasets.unimol import UniMolDataset

PRETRAINING_DATASETS = {
    "qm9": QM9Dataset,
    "unimol": UniMolDataset,
    "alexandria": AlexandriaDataset,
}


def build_pretraining_dataset(data_cfg):
    """Build the dataset named by data_cfg.dataset."""
    name = str(data_cfg.dataset).lower()
    if name not in PRETRAINING_DATASETS:
        raise ValueError(f"unknown data.dataset {name!r}; expected one of {sorted(PRETRAINING_DATASETS)}")
    return PRETRAINING_DATASETS[name].from_config(data_cfg)
