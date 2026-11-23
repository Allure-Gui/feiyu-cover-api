"""清理回归测试，仅操作 TemporaryDirectory。"""
import json
import os
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

if __package__:
    from .cleanup import clean, clean_all, collect_skip_ids
else:
    from cleanup import clean, clean_all, collect_skip_ids


class CleanupTests(unittest.TestCase):
    def test_force_preserves_every_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = []
            for name in ('abcdef123456', 'abcdef654321', 'my-folder'):
                folder = root / name
                folder.mkdir()
                for filename in ('cover.mp3', 'job.json', 'spec.json'):
                    path = folder / filename
                    path.write_bytes(b'keep')
                    files.append(path)
            for name in ('sep/stem.wav', 'final/cover.mp3'):
                file = root / 'cache' / name
                file.parent.mkdir(parents=True)
                file.write_bytes(b'cache')
            forgotten = []
            result = clean_all(root, {'abcdef654321'}, forgotten.extend)
            self.assertEqual((result['removed_jobs'], result['removed_files']), (0, 2))
            self.assertEqual(forgotten, [])
            self.assertEqual(list((root / 'cache').iterdir()), [])
            for path in files:
                self.assertEqual(path.read_bytes(), b'keep')

    def test_threshold_only_prunes_confirmed_terminal_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = SimpleNamespace(_active='active', _jobs={'in-flight': SimpleNamespace(status='running')},
                                      _lock=threading.RLock(), _queue=queue.Queue())
            manager._queue.put('000000000002')
            skip = collect_skip_ids(manager)
            self.assertEqual(skip, {'active', 'in-flight', '000000000002'})
            for n, status in enumerate(('done', 'done', 'done', 'running', 'queued', 'unknown')):
                folder = root / f'{n:012x}'
                folder.mkdir()
                (folder / 'job.json').write_text(json.dumps({'id': folder.name, 'status': status}))
                (folder / 'spec.json').write_text('{}')
                os.utime(folder, (1000 + n, 1000 + n))
            for name in ('abcdefabcdef', 'library'):
                (root / name).mkdir()
            forgotten = []
            result = clean(root, {'keep_jobs': 1}, skip, forgotten.extend)
            self.assertEqual(forgotten, ['000000000000'])
            self.assertEqual(result['removed_jobs'], 1)
            self.assertEqual({p.name for p in root.iterdir()},
                             {f'{i:012x}' for i in range(1, 6)} | {'abcdefabcdef', 'library'})


if __name__ == '__main__':
    unittest.main()
