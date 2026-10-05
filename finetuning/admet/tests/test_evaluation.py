"""Regression checks for evaluation order, distributed gating, and pruning."""

import unittest
from unittest.mock import Mock, patch

import optuna
import torch

from finetuning.admet.training.distributed import DistState
from finetuning.admet.training.evaluation import evaluate_final, evaluate_test, validate_epoch


class EvaluationTests(unittest.TestCase):
    def state(self, rank=0):
        return DistState(rank != 0, rank, rank, 2 if rank else 1, torch.device('cpu'))

    def test_validation_scores_only_validation_and_reports_selection(self):
        score = Mock(return_value={'macro': 1.2, 'other': .8})
        trial = Mock()
        trial.should_prune.return_value = False
        report, macro, values = validate_epoch(
            score, 'model', 'validation', dist_state=self.state(),
            selection={'mae': {'key': 'macro'}, 'other': {'key': 'other'}},
            skip=False, trial=trial, epoch=3)
        score.assert_called_once_with('model', 'validation')
        self.assertEqual(macro, 1.2)
        self.assertEqual(values, {'mae': 1.2, 'other': .8})
        self.assertEqual(report, score.return_value)
        trial.report.assert_called_once_with(1.2, 3)

    def test_skipped_validation_and_worker_rank_do_not_evaluate(self):
        score = Mock()
        report, _, values = validate_epoch(
            score, 'model', 'validation', dist_state=self.state(), selection={}, skip=True)
        self.assertIsNone(report)
        self.assertIsNone(values)
        with patch('finetuning.admet.training.evaluation.ddp.broadcast_obj',
                   side_effect=[.7, {'mae': .7}]) as broadcast:
            report, macro, values = validate_epoch(
                score, 'model', 'validation', dist_state=self.state(1), selection={}, skip=False)
            self.assertEqual(broadcast.call_count, 2)
        self.assertIsNone(report)
        self.assertEqual(macro, .7)
        self.assertEqual(values, {'mae': .7})
        self.assertIsNone(evaluate_test(score, 'model', 'test', is_main=False))
        score.assert_not_called()

    def test_pruning_stops_after_validation(self):
        trial = Mock()
        trial.should_prune.return_value = True
        score = Mock(return_value={'macro': 2.})
        with self.assertRaises(optuna.TrialPruned):
            validate_epoch(score, 'model', 'validation', dist_state=self.state(),
                           selection={}, skip=False, trial=trial, epoch=1)
        score.assert_called_once_with('model', 'validation')
        trial.report.assert_called_once_with(2., 1)

    def test_final_validation_precedes_test_and_enables_curves(self):
        score = Mock(return_value={'macro': 1., 'per_task': {'target': 1.}})
        val, test = evaluate_final(
            score, 'model', 'validation', 'test', ftc={}, is_main=True,
            use_wandb=False, dataset_name='tiny', run_tag='s1', return_val_preds=True)
        self.assertEqual(score.call_count, 2)
        self.assertEqual(score.call_args_list[0].args, ('model', 'validation'))
        self.assertEqual(score.call_args_list[0].kwargs,
                         {'compute_curve': True, 'return_per_graph': True})
        self.assertEqual(score.call_args_list[1].args, ('model', 'test'))
        self.assertEqual(score.call_args_list[1].kwargs, {'compute_curve': True})
        self.assertEqual(val, test)


if __name__ == '__main__':
    unittest.main()
