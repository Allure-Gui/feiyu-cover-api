"""Run from either API project root with unittest discover -s <this directory>."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cover_api import pipeline


class VocalActivityTests(unittest.TestCase):
    def detect(self, stderr, duration=100, returncode=0):
        result = subprocess.CompletedProcess([], returncode, stderr=stderr)
        with patch.object(pipeline.subprocess, 'run', return_value=result) as run:
            activity = pipeline._vocal_activity({'mix': {}}, Path('vocals.wav'), duration)
        run.assert_called_once()
        self.assertIn('silencedetect=noise=-45.0dB:d=0.8', run.call_args.args[0])
        return activity

    def test_active_spans_include_beginning_and_end(self):
        self.assertEqual(self.detect('silence_start: 4\nsilence_end: 80 | silence_duration: 76'), (.24, 20))
        self.assertEqual(self.detect('silence_start: 4\nsilence_end: 8\nsilence_start: 20'), (.16, 12))
        self.assertEqual(self.detect(''), (1, 100))
        self.assertEqual(self.detect('silence_start: 0\nsilence_end: 100'), (0, 0))

    def test_failed_scan_is_unknown(self):
        self.assertIsNone(self.detect('silence_start: 0', returncode=1))
        self.assertIsNone(self.detect('silence_start: nan'))
        self.assertIsNone(self.detect('silence_end: 30'))
        with patch.object(pipeline.subprocess, 'run', side_effect=subprocess.TimeoutExpired('ffmpeg', 300)):
            self.assertIsNone(pipeline._vocal_activity({'mix': {}}, Path('vocals.wav'), 100))
        with patch.object(pipeline.subprocess, 'run') as run:
            self.assertIsNone(pipeline._vocal_activity({'mix': {}}, Path('vocals.wav'), 0))
            run.assert_not_called()

    def test_thresholds_and_continuity(self):
        for activity, minimum, reject in [
            (None, .30, False), ((.099, 9), .30, True),
            ((.10, 3), .30, False), ((.20, 2.9), .30, True),
            ((.30, 1), .30, False), ((.05, 1), .05, False),
            ((.049, 5), .05, True), ((0, 0), 0, False),
        ]:
            with self.subTest(activity=activity, minimum=minimum):
                self.assertEqual(pipeline._reject_vocal_activity(activity, minimum), reject)


class MixFilterTests(unittest.TestCase):
    def filters(self, overrides=None, **kwargs):
        cfg = {'mix': {'auto_gain': False, **(overrides or {})}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, '_run') as run:
            output = Path(tmp) / 'cover.mp3'
            output.touch()
            pipeline._mix(cfg, Path('v.wav'), Path('i.wav'), output, Path('mix.log'), **kwargs)
            args = run.call_args.args[0]
            return args[args.index('-filter_complex') + 1]

    def test_default_chain_order(self):
        graph = self.filters()
        stages = ['highpass=f=90', 'equalizer=f=280:', 'agate=', 'equalizer=f=3200:',
                  'equalizer=f=6200:', 'treble=', 'acompressor=threshold=-18dB',
                  'acompressor=threshold=-9dB', 'aecho=', 'equalizer=f=3000:', 'sidechaincompress=']
        indices = [graph.index(stage) for stage in stages]
        self.assertEqual(indices, sorted(indices))

    def test_disabling_eq_keeps_independent_dynamics(self):
        graph = self.filters(eq_on=False, reverb_on=False)
        for token in ['highpass=', 'equalizer=', 'treble=', 'aecho=']:
            self.assertNotIn(token, graph)
        self.assertIn('agate=', graph)
        self.assertEqual(graph.count('acompressor='), 2)
        self.assertIn('sidechaincompress=', graph)

    def test_dynamics_switches_are_independent(self):
        for gate, compress in [(False, True), (True, False), (False, False)]:
            with self.subTest(gate=gate, compress=compress):
                graph = self.filters({'vocal_gate_on': gate, 'vocal_compress': compress})
                self.assertEqual('agate=' in graph, gate)
                self.assertEqual('acompressor=' in graph, compress)
                self.assertIn('equalizer=f=3200:', graph)
