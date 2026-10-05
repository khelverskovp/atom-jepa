"""Train-only preprocessing and binary class-weight regression tests."""

import tempfile
import unittest

import torch
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler

from finetuning.admet.models.baseline_model import FeatureVectorEncoder
from finetuning.admet.tests.test_regressions import config, datasets
from data.datasets.admet.feature_finetune import feature_collate
from finetuning.admet.training.distributed import DistState
from finetuning.admet.training.loss import binary_positive_weights, fit_target_stats_mt
from finetuning.admet.training.preparation import prepare_data
from finetuning.admet.training.state import prepare_model


class PreparationTests(unittest.TestCase):
    def test_binary_weights_ignore_missing_labels_and_leave_regression_unweighted(self):
        labels = torch.tensor([[0., 0., 10.], [0., 0., 20.], [1., float('nan'), 30.],
                               [float('nan'), 0., float('nan')], [0., 0., 50.]])
        kinds = ['binary', 'binary', 'regression']
        auto = binary_positive_weights(labels, kinds, 'auto')
        self.assertIsNotNone(auto)
        torch.testing.assert_close(auto, torch.tensor([3., 1., 1.]))
        fixed = binary_positive_weights(labels, kinds, 2.5)
        torch.testing.assert_close(fixed, torch.tensor([2.5, 2.5, 1.]))
        self.assertIsNone(binary_positive_weights(labels, kinds, 'none'))
        self.assertIsNone(binary_positive_weights(labels, None))
        self.assertIsNone(binary_positive_weights(labels, ['regression'] * 3))

    def test_normalization_uses_training_labels_only(self):
        train, valid, test = datasets()
        valid.labels[:, 0] += 1000
        test.labels[:, 0] += 2000
        kinds = ['regression', 'binary']
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory)
            state = DistState(False, 0, 0, 1, torch.device('cpu'))
            prepared = prepare_data(cfg, train, valid, test, ['reg', 'binary'], feature_collate,
                                    device=state.device, dist_state=state, task_kinds=kinds)
        mean, std = fit_target_stats_mt(train.labels, True, None, task_kinds=kinds)
        torch.testing.assert_close(prepared.y_mean, mean)
        torch.testing.assert_close(prepared.y_std, std)
        self.assertEqual(prepared.reg_task_idx, [0])
        self.assertIs(prepared.val_loader.dataset, valid)
        self.assertIs(prepared.test_loader.dataset, test)
        self.assertIsInstance(prepared.train_loader.sampler, RandomSampler)
        self.assertIsInstance(prepared.val_loader.sampler, SequentialSampler)

    def test_distributed_preparation_shards_only_training(self):
        train, valid, test = datasets()
        with tempfile.TemporaryDirectory() as directory:
            dist = DistState(True, 1, 1, 2, torch.device('cpu'))
            prepared = prepare_data(config(directory), train, valid, test, ['reg', 'binary'],
                                    feature_collate, device=dist.device, dist_state=dist)
        self.assertIsInstance(prepared.train_loader.sampler, DistributedSampler)
        self.assertIsInstance(prepared.val_loader.sampler, SequentialSampler)
        self.assertIsInstance(prepared.test_loader.sampler, SequentialSampler)
        assert isinstance(prepared.train_loader.sampler, DistributedSampler)
        assert isinstance(prepared.val_loader.sampler, SequentialSampler)
        self.assertEqual(len(prepared.train_loader.sampler), 2)
        self.assertEqual(len(prepared.val_loader.sampler), 4)

    def test_model_setup_freezes_encoder_and_builds_independent_ema(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory)
            cfg.finetune.freeze_encoder_epochs = 1
            dist = DistState(False, 0, 0, 1, torch.device('cpu'))
            data = prepare_data(cfg, *datasets(), ['reg', 'binary'], feature_collate,
                                device=dist.device, dist_state=dist)
            state = prepare_model(cfg, data, lambda: FeatureVectorEncoder(2, 4, num_layers=1),
                                  4, 2, device=dist.device, dist_state=dist,
                                  task_kinds=['regression', 'binary'], run_label='test')
        self.assertTrue(all(not p.requires_grad for p in state.core.encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in state.core.head.parameters()))
        self.assertIsNotNone(state.ema_model)
        assert state.ema_model is not None
        for trained, averaged in zip(state.core.parameters(), state.ema_model.parameters()):
            self.assertNotEqual(trained.data_ptr(), averaged.data_ptr())
            torch.testing.assert_close(trained, averaged)
            self.assertFalse(averaged.requires_grad)


if __name__ == '__main__':
    unittest.main()
