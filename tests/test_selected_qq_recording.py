import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from cover_api import sources


class SelectedQQRecording(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cfg = {'source': {'qq_exact_recording': True, 'qq_cookie': 'musicid=123', 'cookie': 'mock-secret',
                               'bili_cookie': 'mock-bili', 'qq_cookie_file': str(self.root / 'absent')}}
        self.spec = {'selected_track': {'origin': 'qq', 'songmid': 'testMid', 'gd_id': 'other'}}
        self.lookup = self.mock(sources.qq_recording, 'lookup')
        self.legacy = self.mock(sources, '_try_download_candidate')
        self.search = self.mock(sources, 'search')

    def mock(self, obj, name):
        ctx = patch.object(obj, name)
        self.addCleanup(ctx.stop)
        return ctx.start()

    def resolve(self):
        return sources.resolve(self.spec, self.cfg, self.root)

    def test_authorized_full_media(self):
        url = 'https://ws.stream.qqmusic.qq.com/test.mp3?label=bilibili'
        self.lookup.return_value = ({'interval': 180}, [{'result': 0, 'url': url, 'trial': None}])
        def media(request):
            self.assertEqual(str(request.url), url)
            self.assertNotIn('cookie', request.headers)
            return httpx.Response(200, content=b'x' * 70000)
        client = httpx.Client
        with patch.object(sources.httpx, 'Client', side_effect=lambda **kw: client(transport=httpx.MockTransport(media), **kw)):
            probe = self.mock(sources, '_probe_audio_duration')
            probe.return_value = 180.5
            path, info = self.resolve()
            self.assertTrue(path.is_file())
            self.assertEqual(info['songmid'], 'testMid')
            self.assertEqual(self.lookup.call_args.args[1:], ('musicid=123', 'testMid'))
            for duration in (30, 0, 240):
                probe.return_value = duration
                with self.assertRaisesRegex(sources.SourceError, '时长'):
                    self.resolve()
                self.assertFalse(path.exists())
        self.legacy.assert_not_called()
        self.search.assert_not_called()

    def test_refusals_distinct(self):
        download = self.mock(sources, 'download')
        self.lookup.return_value = ({'interval': 180}, [
            {'result': 104003}, {'result': 0}, {'result': 42}, {},
            {'result': 0, 'url': 'https://example.invalid', 'trial': True}])
        with self.assertRaises(sources.SourceError) as result:
            self.resolve()
        for message in ('104003', '地址为空', '其它返回码 42', '授权状态未知', '试听'):
            self.assertIn(message, str(result.exception))
        download.assert_not_called()
        self.legacy.assert_not_called()
        self.search.assert_not_called()

    def test_guest_preview_and_other_origin_keep_legacy_route(self):
        self.legacy.return_value = (self.root, {})
        for cookie, preview, origin in [('', 0, 'qq'), ('musicid=123', 30, 'qq'), ('musicid=123', 0, 'kugou')]:
            self.cfg['source']['qq_cookie'] = cookie
            self.spec.update(preview_seconds=preview)
            self.spec['selected_track']['origin'] = origin
            self.assertEqual(self.resolve(), self.legacy.return_value)
        self.lookup.assert_not_called()
