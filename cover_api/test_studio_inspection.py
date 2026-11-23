"""Offline regression tests: python -m unittest cover_api.test_studio_inspection"""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cover_api import studio


class InspectionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ds = Path(tmp.name) / 'voice'
        self.ds.mkdir()
        self.manager = studio.StudioManager({'data_dir': str(self.ds.parent / 'jobs'),
                                             'rvc': {'python': sys.executable}})

    def inspect(self, metrics, n=1, duration=None):
        for i in range(n):
            (self.ds / f'{i:03}.wav').touch()
        def run(argv, **kw):
            self.plan = json.loads(kw['input'])
            self.worker = argv[2]
            compile(self.worker, '<inspection>', 'exec')
            return subprocess.CompletedProcess(argv, 0, json.dumps(metrics))
        with patch.object(studio, '_probe_duration', return_value=duration or 1200 / n), patch.object(studio.subprocess, 'run', side_effect=run):
            report = self.manager.inspect_dataset(str(self.ds))
        self.checks = {c['key']: c for c in report['checks']}
        self.assertEqual(set(self.checks), {'duration', 'silence', 'clipping', 'high_notes', 'coverage'})
        for c in report['checks']:
            self.assertEqual(set(c), {'key', 'level', 'title', 'detail'})
        return report

    def test_no_measurements(self):
        for metrics in ({}, {'sampled': 0, 'segments': 0, 'attempted': 3, 'frames': 0}):
            report = self.inspect(metrics)
            self.assertEqual((report['overall'], report['active_min']), ('bad', 0))
            self.assertNotIn('ok', [c['level'] for c in report['checks']])

    def test_silence(self):
        report = self.inspect({'sampled': 1, 'segments': 3, 'attempted': 3, 'frames': 120, 'active': 0})
        self.assertEqual(report['overall'], 'bad')
        self.assertEqual(self.checks['silence']['level'], 'bad')

    def test_partial_failure(self):
        report = self.inspect({'sampled': 2, 'segments': 5, 'attempted': 6, 'frames': 100,
                               'active': 99, 'failed': ['001.wav'], 'f0_median': 175}, n=2)
        self.assertEqual(report['overall'], 'warn')
        self.assertIn('001.wav', self.checks['coverage']['detail'])
        self.assertIn('5/6', self.checks['coverage']['detail'])
        self.assertNotIn('ok', [c['level'] for c in report['checks']])

    def test_even_file_sampling_and_healthy_control(self):
        report = self.inspect({'sampled': 24, 'segments': 72, 'attempted': 72, 'frames': 100,
                               'active': 100, 'f0_median': 175}, n=48)
        indices = [int(Path(path).stem) for path, _ in self.plan]
        self.assertEqual((len(set(indices)), indices[0], indices[-1]), (24, 0, 47))
        self.assertLessEqual(max(b-a for a, b in zip(indices, indices[1:])), 3)
        self.assertEqual(report['overall'], 'ok')

    def test_worker_reads_head_middle_tail_and_survives_decode_failure(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('numpy optional outside RVC environment')
        self.inspect({}, n=2, duration=120)
        starts = []
        def decode(argv, **kw):
            starts.append(float(argv[argv.index('-ss') + 1]))
            if argv[argv.index('-i') + 1].endswith('001.wav'):
                raise subprocess.TimeoutExpired(argv, 8)
            return subprocess.CompletedProcess(argv, 0, np.zeros(16000 * 8, dtype='<f4').tobytes())
        stdout = io.StringIO()
        with patch.object(sys, 'argv', ['worker', 'ffmpeg']), patch.object(sys, 'stdin', io.StringIO(json.dumps(self.plan))), patch.object(subprocess, 'run', side_effect=decode), contextlib.redirect_stdout(stdout):
            exec(self.worker, {})
        metrics = json.loads(stdout.getvalue())
        self.assertEqual(starts, [0, 56, 112] * 2)
        self.assertEqual((metrics['sampled'], metrics['segments'], metrics['attempted']), (1, 3, 6))
        self.assertEqual(metrics['failed'], ['001.wav'])
        self.assertGreater(metrics['frames'], 0)
        self.assertEqual(metrics['active'], 0)


if __name__ == '__main__':
    unittest.main()
