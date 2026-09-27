"""Behavioral checks for leakage, mixture arithmetic and synthetic controls."""
from contextlib import redirect_stderr, redirect_stdout
import csv
from datetime import date, timedelta
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp

from marksix.cli import main, run
from marksix.data import load_draws, load_draws_bytes, write_synthetic
from marksix.evaluation import MIXTURE_PRIOR, block_uncertainty, holm, summarize, walk_forward


def synthetic(n=90, seed=17, planted=False):
    rng = np.random.default_rng(seed)
    y = np.zeros((n, 49))
    for row in y:
        indices = np.r_[0, rng.choice(np.arange(1, 49), 5, replace=False)] if planted else rng.choice(49, 6, replace=False)
        row[indices] = 1
    return y


def hot_ball(n, seed, p=0.3):
    """Ball 1 appears with probability p; the remaining mains are uniform."""
    rng = np.random.default_rng(seed)
    y = np.zeros((n, 49))
    for row in y:
        if rng.random() < p:
            row[np.r_[0, rng.choice(np.arange(1, 49), 5, replace=False)]] = 1
        else:
            row[rng.choice(np.arange(1, 49), 6, replace=False)] = 1
    return y


class EvaluationTests(unittest.TestCase):
    def test_target_and_future_do_not_change_forecasts(self):
        y = synthetic()
        altered = y.copy()
        altered[65:] = synthetic(25, seed=131)
        original = walk_forward(y, warmup=50)
        changed = walk_forward(altered, warmup=50)
        # Forecast for target 65 must also be independent of its own outcome.
        np.testing.assert_allclose(original['marginals'][:16], changed['marginals'][:16], atol=0, rtol=0)
        np.testing.assert_allclose(original['mixture_weights'][:16], changed['mixture_weights'][:16], atol=0, rtol=0)
        np.testing.assert_allclose(original['log_probabilities'][:15], changed['log_probabilities'][:15], atol=0, rtol=0)

    def test_arithmetic_mixture_scores_and_weight_replay(self):
        result = walk_forward(synthetic(55), warmup=30)
        lp, weights = result['log_probabilities'], result['mixture_weights']
        np.testing.assert_allclose(lp[:, 3], logsumexp(np.log(weights) + lp[:, :3], axis=1), atol=1e-12)
        expected = MIXTURE_PRIOR.copy()
        for row, saved in zip(lp, weights):
            np.testing.assert_allclose(saved, expected, atol=1e-14)
            posterior = expected * np.exp(.25 * (row[:3] - row[:3].max()))
            posterior /= posterior.sum()
            expected = .99 * posterior + .01 * MIXTURE_PRIOR

    def test_reset_removes_old_regime_from_new_forecasts(self):
        y = synthetic(80)
        changed = y.copy()
        changed[:60] = synthetic(60, seed=123, planted=True)
        a = walk_forward(y, warmup=50, reset_index=60)
        b = walk_forward(changed, warmup=50, reset_index=60)
        np.testing.assert_allclose(a['marginals'][10:], b['marginals'][10:], atol=0, rtol=0)
        np.testing.assert_allclose(a['mixture_weights'][10], MIXTURE_PRIOR)
        np.testing.assert_allclose(a['marginals'][10], np.full((4, 49), 6/49), atol=1e-12)

    def test_multiple_machine_boundaries_remove_all_earlier_state(self):
        y = synthetic(90)
        altered = y.copy()
        altered[:75] = synthetic(75, seed=99, planted=True)
        a = walk_forward(y, warmup=50, reset_indices=(60, 75))
        b = walk_forward(altered, warmup=50, reset_indices=(60, 75))
        np.testing.assert_allclose(a['marginals'][25:], b['marginals'][25:], atol=0, rtol=0)
        np.testing.assert_allclose(a['mixture_weights'][25:], b['mixture_weights'][25:], atol=0, rtol=0)
        np.testing.assert_allclose(a['mixture_weights'][25], MIXTURE_PRIOR)
        for invalid in [(float('nan'),), (60, True), (60, 60.0), 60]:
            with self.assertRaises(ValueError):
                walk_forward(y, reset_indices=invalid)

    def test_strong_synthetic_signal_is_detectable(self):
        result = walk_forward(synthetic(140, planted=True), warmup=60)
        gain = result['log_probabilities'][:, 1:] - result['log_probabilities'][:, :1]
        self.assertTrue(np.all(gain.sum(axis=0) > 20))

    def test_bootstrap_matches_loop_reference(self):
        x = np.random.default_rng(3).normal(size=(45, 3))
        for block_length in (1, 4, 8):
            lo, hi, p = block_uncertainty(x, seed=11, replicates=150, block_length=block_length)
            blocks = -(-len(x) // block_length)
            starts = np.random.default_rng(11).integers(len(x), size=(150, blocks))
            means = np.array([
                np.mean([x[(start + j) % len(x)] for start in row for j in range(block_length)][:len(x)], axis=0)
                for row in starts])
            np.testing.assert_allclose(np.r_[[lo], [hi]], np.quantile(means, [.025, .975], axis=0))
            np.testing.assert_allclose(p, (1 + (means - x.mean(0) >= x.mean(0)).sum(0)) / 151)
            small_types = block_uncertainty(x, seed=11, replicates=np.uint8(150), block_length=np.int8(block_length))
            np.testing.assert_array_equal(np.r_[small_types], np.r_[lo, hi, p])

    def test_summary_inference_on_planted_and_fair_draws(self):
        y = hot_ball(240, seed=0)
        summary = summarize(walk_forward(y, warmup=60), y)
        self.assertEqual(summary[0]['total_log_gain'], 0.)
        self.assertEqual(summary[0]['mean_95pct_block_ci'], [0., 0.])
        self.assertEqual(summary[0]['one_sided_bootstrap_p'], 1.)
        p = [row['one_sided_bootstrap_p'] for row in summary[1:]]
        np.testing.assert_allclose([row['holm_p'] for row in summary[1:]], holm(p))
        for row in summary[1:]:
            lo, hi = row['mean_95pct_block_ci']
            self.assertTrue(0 < lo <= row['mean_log_gain'] <= hi)
            self.assertLess(row['holm_p'], 0.05)
        fair = synthetic(240)
        for row in summarize(walk_forward(fair, warmup=60), fair)[1:]:
            self.assertGreater(row['holm_p'], 0.05)

    def test_uniform_control_and_probability_constraints(self):
        y = synthetic(75)
        result = walk_forward(y, warmup=60)
        np.testing.assert_allclose(result['marginals'].sum(axis=-1), 6, atol=1e-11)
        self.assertTrue(np.all((result['marginals'] >= 0) & (result['marginals'] <= 1)))
        summary = summarize(result, y)
        self.assertAlmostEqual(summary[0]['total_log_gain'], 0., places=10)
        self.assertAlmostEqual(summary[0]['mean_brier_gain'], 0., places=10)
        self.assertEqual(summary[0]['holm_p'], 1.)

    def test_short_samples_do_not_emit_confident_inference(self):
        with self.assertRaises(ValueError):
            block_uncertainty(np.ones((8, 3)))
        y = synthetic(68, planted=True)
        summary = summarize(walk_forward(y, warmup=60), y)
        self.assertIsNone(summary[1]['mean_95pct_block_ci'])
        self.assertIsNone(summary[1]['holm_p'])
        with self.assertRaises(ValueError):
            walk_forward(synthetic(65).astype(complex) + 1j, warmup=60)

    def test_holm_reference_and_invalid_arguments(self):
        np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])
        with self.assertRaises(ValueError):
            walk_forward(synthetic(10), warmup=10)
        with self.assertRaises(ValueError):
            walk_forward(synthetic(10), warmup=5, reset_index=-1)
        with self.assertRaises(ValueError):
            holm([float('nan')])


class DataAndRunTests(unittest.TestCase):
    def test_synthetic_regeneration_and_strict_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/'a.csv', Path(directory)/'b.csv'
            write_synthetic(a, draws=25)
            write_synthetic(b, draws=25)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(load_draws(a).outcomes.shape, (25, 49))
            with self.assertRaises(FileExistsError):
                write_synthetic(a, draws=25)

    def test_rejects_disordered_duplicate_and_invalid_draws(self):
        header = ['date', 'draw_id', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'extra']
        good = ['2000-01-01', 'A', 1, 2, 3, 4, 5, 6, 7]
        cases = [
            [good, ['1999-12-31', 'B', 1, 2, 3, 4, 5, 6, 7]],
            [good, ['2000-01-02', 'A', 1, 2, 3, 4, 5, 6, 7]],
            [['2000-01-01', 'A', 1, 2, 3, 4, 5, 5, 7]],
            [['2000-01-01', 'A', 1, 2, 3, 4, 5, 50, 7]],
            [['2000-01-01', 'A', 1, 2, 3, 4, 5, 6, 6]],
            [['2000-01-01', 'A', 1, 2, 3, 4, 5, '6.5', 7]],
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'invalid.csv'
            for rows in cases:
                with self.subTest(rows=rows):
                    with path.open('w', newline='') as handle:
                        writer = csv.writer(handle); writer.writerow(header); writer.writerows(rows)
                    with self.assertRaises(ValueError):
                        load_draws(path)

    def test_malformed_csv_reports_clean_errors_with_physical_lines(self):
        header = 'date,draw_id,n1,n2,n3,n4,n5,n6,extra\n'
        cases = {
            'short row': (header + '2000-01-01,A,1,2,3,4,5,6,7\n2000-01-02,B,1,2,3,4,5,6\n', 'line 3: Malformed'),
            'blank lines': (header + '\n\n\n2000-01-01,A,1,2,3,4,5,5,7\n', 'line 5:'),
            'oversized field': (header + '2000-01-01,' + 'A' * 200000 + ',1,2,3,4,5,6,7\n', 'at line 2:'),
            'oversized header': ('A' * 200000 + ',' + header, 'at line 1:'),
        }
        for name, (text, message) in cases.items():
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, message):
                    load_draws_bytes(text.encode())

    def test_cli_rejects_negative_seed_and_applies_every_reset_date(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit) as exit_:
                main(['demo', '--output', str(root/'negative'), '--draws', '80', '--seed', '-1'])
            self.assertEqual(exit_.exception.code, 2)
            self.assertIn('seed must be a non-negative integer', stderr.getvalue())
            self.assertFalse((root/'negative').exists())
            with redirect_stdout(io.StringIO()):
                main(['demo', '--output', str(root/'resets'), '--reset-date', '2002-10-03',
                      '--reset-date', '2002-12-03'])
            provenance = json.loads((root/'resets'/'provenance.json').read_text())
            self.assertEqual(provenance['reset_date'], ['2002-10-03', '2002-12-03'])
            self.assertEqual(provenance['reset_dates'], ['2002-10-03', '2002-12-03'])
            self.assertEqual(provenance['reset_indices'], [91, 152])

    def test_run_normalizes_seed_and_reset_date_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_synthetic(root/'input.csv', draws=100)
            run(root/'input.csv', root/'a', seed=np.int64(5), warmup=np.int64(60),
                reset_date=('2002-08-01',))
            provenance = json.loads((root/'a'/'provenance.json').read_text())
            self.assertEqual((provenance['seed'], provenance['warmup_draws']), (5, 60))
            self.assertEqual(provenance['reset_date'], ['2002-08-01'])
            for invalid in [5, [date(2002, 8, 1)], [None], ['2002-8-1']]:
                with self.subTest(reset_date=invalid), self.assertRaises(ValueError):
                    run(root/'input.csv', root/'b', reset_date=invalid)
            self.assertFalse((root/'b').exists())

    def test_failed_publish_leaves_no_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_synthetic(root/'input.csv', draws=65)
            original_rename = Path.rename
            calls = []

            def fail_second_move(path, target):
                calls.append(path)
                if len(calls) == 2:
                    raise OSError('simulated failure')
                return original_rename(path, target)

            with patch.object(Path, 'rename', fail_second_move), self.assertRaises(OSError):
                run(root/'input.csv', root/'output')
            self.assertEqual(sorted(p.name for p in root.iterdir()), ['input.csv'])
            run(root/'input.csv', root/'output')
            self.assertEqual(len(list((root/'output').iterdir())), 3)

    def test_evaluation_preserves_inputs_and_prior_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root/'input.csv', root/'output'
            write_synthetic(source, draws=75)
            before = source.read_bytes()
            summary = run(source, output)
            self.assertEqual(summary['dataset_kind'], 'user_supplied_unverified')
            self.assertEqual(source.read_bytes(), before)
            saved = (output/'summary.json').read_bytes()
            with self.assertRaises(ValueError):
                run(source, output)
            self.assertEqual(saved, (output/'summary.json').read_bytes())
            self.assertFalse((output/'synthetic_draws.csv').exists())
            self.assertNotIn(str(root), (output/'provenance.json').read_text())

    def test_known_machine_change_is_automatic_with_missing_boundary_day(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'input.csv'
            y = synthetic(75)
            # Omit the actual first-machine day: resetting cannot depend on exact match.
            days = [date(2026, 3, 1) + timedelta(days=i) for i in range(76)]
            days = [d for d in days if d.isoformat() != '2026-05-05'][:75]
            with source.open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['date', 'draw_id', *(f'n{i}' for i in range(1, 7))])
                for i, (day, row) in enumerate(zip(days, y)):
                    writer.writerow([day.isoformat(), str(i), *(np.flatnonzero(row)+1)])
            result = run(source, root/'output', warmup=50, reset_date='2026-04-01')
            provenance = json.loads((root/'output'/'provenance.json').read_text())
            self.assertIn('2026-05-05', provenance['reset_dates'])
            self.assertIn('2026-04-01', provenance['reset_dates'])
            self.assertFalse(result['betting_eligible'])
            with (root/'output'/'scores.csv').open(newline='') as handle:
                scores = list(csv.DictReader(handle))
            first_new = [row for row in scores if row['date'] == '2026-05-06']
            self.assertEqual(len(first_new), 4)
            for row in first_new:
                self.assertAlmostEqual(float(row['log_gain']), 0., places=11)

    def test_draws_before_six_of_49_format_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for first, accepted in [('2002-07-02', False), ('2002-07-04', True)]:
                source, output = root/f'{first}.csv', root/f'out-{first}'
                with source.open('w', newline='') as handle:
                    writer = csv.writer(handle)
                    writer.writerow(['date', 'draw_id', *(f'n{i}' for i in range(1, 7))])
                    for i, row in enumerate(synthetic(70)):
                        day = date.fromisoformat(first) + timedelta(days=i)
                        writer.writerow([day.isoformat(), str(i), *(np.flatnonzero(row) + 1)])
                if accepted:
                    run(source, output)
                    self.assertTrue(output.exists())
                else:
                    with self.assertRaisesRegex(ValueError, 'fewer than 49 numbers'):
                        run(source, output, reset_date='2002-07-04')
                    self.assertFalse(output.exists())

    def test_invalid_input_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_synthetic(root/'input.csv', draws=10)
            with self.assertRaises(ValueError):
                run(root/'input.csv', root/'output', warmup=60)
            self.assertFalse((root/'output').exists())

    def test_evaluation_uses_one_snapshot_when_input_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, replacement, output = root/'input.csv', root/'replacement.csv', root/'output'
            write_synthetic(source, draws=65)
            write_synthetic(replacement, draws=80, seed=99)
            original = source.read_bytes()

            def replace_during_evaluation(*args, **kwargs):
                replacement.replace(source)
                return walk_forward(*args, **kwargs)

            with patch('marksix.cli.walk_forward', side_effect=replace_during_evaluation):
                summary = run(source, output, dataset_kind='synthetic_demo')

            self.assertNotEqual(source.read_bytes(), original)
            self.assertEqual(summary['input_draws'], 65)
            self.assertEqual(summary['evaluation_draws'], 5)
            self.assertEqual((output/'synthetic_draws.csv').read_bytes(), original)
            provenance = json.loads((output/'provenance.json').read_text())
            self.assertEqual(provenance['input_sha256'], hashlib.sha256(original).hexdigest())
