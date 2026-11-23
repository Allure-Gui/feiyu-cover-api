import sys
import tempfile
import unittest
from pathlib import Path

from cover_api.config import validate


class OptionalIndexTest(unittest.TestCase):
    def test_explicitly_missing_index_is_different_from_no_index(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder, 'voice.pth')
            model.touch()
            rvc = {'python': sys.executable, 'model': str(model)}
            self.assertEqual(validate({'rvc': rvc}), [])
            rvc['index'] = ''
            self.assertEqual(validate({'rvc': rvc}), [])
            index = Path(folder, 'voice.index')
            rvc['index'] = str(index)
            self.assertEqual(len(validate({'rvc': rvc})), 1)
            index.touch()
            self.assertEqual(validate({'rvc': rvc}), [])
