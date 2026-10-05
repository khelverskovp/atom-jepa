"""CPU regressions for ADMET configuration, feature caches, and training paths."""

import pickle
import tempfile
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from data.datasets.admet.feature_finetune import MolecularFeatureDataset, feature_collate
from finetuning.admet.baseline import fit_apply_feature_scaler
from finetuning.admet.config import config_dict, split_sizes
from finetuning.admet.features.datasets import concat_feature_datasets
from finetuning.admet.features.jepa_activations import build_or_load_jepa_activations
from finetuning.admet.features.moljepa_features import load_moljepa_features, moljepa_cache_path
from finetuning.admet.metrics.core import resolve_task, _spearman
from finetuning.admet.models.baseline_model import FeatureVectorEncoder
from finetuning.admet.models.model import MultiTaskReadout
from finetuning.admet.training.gbm import train_multitask_gbm
from finetuning.admet.training.train import train_multitask, train_seed


@dataclass
class EncoderConfig:
    max_radius: float = 4.0
    num_channels: int = 4
    max_num_elements: int = 128
    grad_checkpointing: bool = False


class TinyEncoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer = torch.nn.Linear(1, cfg.num_channels)

    def encode_nodes(self, batch):
        return self.layer(batch['atomic_numbers'].float().unsqueeze(-1)), None


def config(directory):
    return OmegaConf.create({
        'finetune': {'epochs': 1, 'batch_size': 4, 'ema': True, 'ema_decay': .9,
                     'train_from_scratch': True, 'save_best': False, 'save_last': False,
                     'mtl_loss': False, 'plot_every': 0, 'lr': .001},
        'data': {'num_workers': 0},
        'misc': {'device': 'cpu', 'checkpoint_dir': directory, 'log_every': 100},
        'baseline': {'feature_scaling': 'quantile', 'n_quantiles': 4,
                     'lgbm': {'n_estimators': 3, 'min_child_samples': 1,
                              'n_jobs': 1, 'verbosity': -1, 'early_stopping_rounds': 0}},
    })


def datasets():
    df = pd.DataFrame({'smiles': ['CC', 'CCC', 'CO', 'CN'],
                       'reg': [1., 2., 3., 4.], 'binary': [0., 1., 0., 1.]})
    features: dict[str, np.ndarray | None] = {s: np.array([i, i + 1], dtype=np.float32)
                for i, s in enumerate(df['smiles'])}
    train, valid, test = [MolecularFeatureDataset(df, features, ['reg', 'binary']) for _ in range(3)]
    return train, valid, test


class ADMETRegressions(unittest.TestCase):
    def test_config_mappings_and_split_sizes(self):
        self.assertEqual(config_dict(OmegaConf.create({'x': 1, 'y': '${x}'})), {'x': 1, 'y': 1})
        self.assertEqual(config_dict({}), {})
        with self.assertRaises(TypeError):
            config_dict(OmegaConf.create([1, 2]))
        self.assertEqual(split_sizes([.8, .1, .1]), (.8, .1, .1))
        with self.assertRaises(ValueError):
            split_sizes([.8, .2])

    def test_train_fitted_scaling_and_concatenation(self):
        train, valid, test = datasets()
        with tempfile.TemporaryDirectory() as directory:
            fit_apply_feature_scaler(config(directory), train, [valid, test])
        combined = concat_feature_datasets(train, valid)
        np.testing.assert_array_equal(combined.feature_matrix(),
                                      np.concatenate([train.feature_matrix(), valid.feature_matrix()]))
        self.assertEqual(len(combined), 8)
        self.assertEqual(combined.keep_index, [])
        self.assertTrue(np.isfinite(combined.labels.numpy()).all())

    def test_moljepa_cache_variants_and_missing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = moljepa_cache_path(directory, 'tiny')
            with open(path, 'wb') as stream:
                pickle.dump({'CC': {'cls': np.ones(2), 'predictions': np.ones((3, 2))},
                             'bad': None}, stream)
            cls = load_moljepa_features('tiny', ['CC', 'bad'], directory)
            self.assertIsNone(cls['bad'])
            np.testing.assert_array_equal(cls['CC'], np.ones(2))
            modalities = load_moljepa_features('tiny', ['CC'], directory, output='modalities')
            np.testing.assert_array_equal(modalities['CC'], np.ones(6))
            with self.assertRaises(KeyError):
                load_moljepa_features('tiny', ['missing'], directory)

    def test_atom_jepa_cache_separates_higher_order_features(self):
        conformers = {'CC': (np.array([6, 6]), [np.array([[0., 0., 0.], [1., 0., 0.]])])}
        with tempfile.TemporaryDirectory() as directory:
            with patch('finetuning.admet.features.jepa_activations.build_or_load_conformers',
                       return_value=conformers), \
                 patch('finetuning.admet.training.utils.pooled_node_features') as pool:
                for higher_order, width in ((False, 4), (True, 8)):
                    pool.return_value = np.ones((1, width), dtype=np.float32)
                    result = build_or_load_jepa_activations(
                        'tiny', ['CC'], None, EncoderConfig(), directory, layers='last',
                        higher_order=higher_order, conformer_cache_dir=directory)
                    value = result['CC']
                    assert value is not None
                    self.assertEqual(value.shape, (width,))
                    self.assertEqual(pool.call_args.kwargs['higher_order'], higher_order)
                self.assertEqual(pool.call_count, 2)
                build_or_load_jepa_activations(
                    'tiny', ['CC'], None, EncoderConfig(), directory, layers='last',
                    higher_order=False, conformer_cache_dir=directory)
                self.assertEqual(pool.call_count, 2)

    def test_readout_fusion_modes_and_higher_order(self):
        batch = {'node_graph_index': torch.arange(3), 'num_graphs': 3,
                 'mol_features': torch.randn(3, 2), 'mol_features_valid': torch.ones(3)}
        for mode in ('none', 'early_concat', 'projected_concat', 'film', 'late'):
            for members in (1, 2):
                with self.subTest(mode=mode, members=members):
                    model = MultiTaskReadout(4, 2, feature_dim=2, fusion=mode,
                                             n_members=members, lmax=1, use_higher_order=True)
                    out = model(torch.randn(3, 4), batch, node_full=torch.randn(3, 4, 4))
                    self.assertEqual(tuple(out.shape), (3, 2))
                    out.sum().backward()

    def test_multitask_training_and_completed_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory)
            cfg.finetune.save_last = True
            cfg.finetune.select_best = False
            data = datasets()
            def run():
                return train_multitask(cfg, *data, ['reg', 'binary'],
                                      lambda: FeatureVectorEncoder(2, 4, num_layers=1),
                                      4, feature_collate, torch.device('cpu'), False,
                                      dataset_name='tiny', run_tag='s1', seed=1,
                                      task_kinds=['regression', 'binary'],
                                      return_test_preds=True)
            first = run()
            cfg.finetune.resume = True
            resumed = run()
            self.assertEqual(resumed['best_epoch'], 0)
            np.testing.assert_allclose(first['test_per_mol_preds'], resumed['test_per_mol_preds'])

    def test_gbm_mixed_tasks_and_refit(self):
        with tempfile.TemporaryDirectory() as directory:
            data = datasets()
            result = train_multitask_gbm(
                config(directory), *data, ['reg', 'binary'], dataset_name='tiny',
                run_tag='s1', seed=1, task_kinds=['regression', 'binary'],
                refit_train_ds=concat_feature_datasets(data[0], data[1]), return_test_preds=True)
            self.assertTrue(np.isfinite(result['test_per_mol_preds']).all())

    def test_single_task_backward_and_refit_with_conformers(self):
        frame = pd.DataFrame({'Drug': ['CC', 'CCC', 'CO', 'CN'], 'Y': [1., 2., 3., 4.]})
        conformers = {s: (np.array([6, 6]), [np.array([[0., 0., 0.], [1., 0., 0.]]),
                                           np.array([[0., 0., 0.], [1.1, 0., 0.]])])
                      for s in frame['Drug']}
        settings = dict(pool='mean', standardize=True, log_transform=False, loss='mse',
                        head_dropout=0., lr=.001, batch_size=4, grad_clip=10.,
                        head_lr_mult=1., freeze_encoder_epochs=0, lr_warmup_epochs=0,
                        grad_checkpointing=False)
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory)
            task = resolve_task('tiny', frame['Y'].to_numpy(), metric_override='mae')
            with patch('finetuning.admet.training.single_task.EquiformerV3Encoder', TinyEncoder), \
                 patch('finetuning.admet.training.single_task.get_train_valid', return_value=(frame, frame)):
                for refit in (None, 1):
                    result = train_seed(cfg, None, 'tiny', 1, task, settings, conformers,
                                        EncoderConfig(), None, frame, frame, torch.device('cpu'),
                                        False, refit_epochs=refit)
                    self.assertTrue(np.isfinite(result['preds']).all())
                    self.assertEqual(result['preds_conf'].shape, (4, 2))

    def test_spearman_scalar(self):
        self.assertAlmostEqual(_spearman([1, 2, 3], [3, 2, 1]), -1.)


if __name__ == '__main__':
    unittest.main()
