"""HTTP contract tests; no GPU worker or remote services required."""
import asyncio
import io
import os
import shutil
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from cover_api import app as api, source_uploads as uploads, sources
from cover_api.jobs import JobManager


@pytest.fixture
def setup(tmp_path, monkeypatch):
    manager = JobManager({'data_dir': str(tmp_path), 'limits': {'max_queue': 8}}, lambda *_: None)
    store = uploads.UploadStore(manager)
    monkeypatch.setattr(api, 'manager', manager)
    monkeypatch.setattr(api, 'UPLOADS', store)
    monkeypatch.setitem(api.CFG, 'api_key', 'test-key')
    monkeypatch.setattr(api.STUDIO, 'is_busy', lambda: False)
    client = TestClient(api.app)  # no lifespan: do not start the worker
    client.headers['Authorization'] = 'Bearer test-key'
    yield client, store
    client.close()


HEADERS = {'Content-Type': 'application/octet-stream', 'X-Audio-Extension': 'wav'}


def test_auth_headers_and_opaque_ids(setup):
    client, store = setup
    url = '/sources/uploads'
    assert client.post(url, content=b'a', headers={**HEADERS, 'Authorization': 'Bearer wrong'}).status_code == 401
    assert client.post(url, content=b'a', headers={**HEADERS, 'X-Audio-Extension': '../wav'}).status_code == 415
    assert client.post(url, content=b'a', headers={**HEADERS, 'Content-Type': 'application/json'}).status_code == 415
    assert client.post(url, content=b'', headers=HEADERS).status_code == 400
    for value, status in [('../../track.wav', 422), ('b' * 32, 404)]:
        assert client.post('/cover', json={'song': 'audio.wav', 'upload_id': value}).status_code == status
    assert client.post('/cover', json={'upload_id': 'a' * 32, 'audio_path': '../x.wav'}).status_code == 422
    assert list(store.root.iterdir()) == []


def test_stream_limits_and_partial_cleanup(setup, monkeypatch):
    _, store = setup
    class Stream:
        headers = {k.lower(): v for k, v in HEADERS.items()}
        async def stream(self):
            yield b'abc'
            yield b'd'
            pytest.fail('must stop reading oversized stream')
    assert uploads.MAX_BYTES == 32 * 1024 * 1024
    monkeypatch.setattr(uploads, 'MAX_BYTES', 3)
    for length in [None, '1', '4']:
        stream = Stream()
        stream.headers = dict(stream.headers)
        if length:
            stream.headers['content-length'] = length
        with pytest.raises(HTTPException) as err:
            asyncio.run(store.receive(stream, 'ffprobe'))
        assert err.value.status_code == 413
        assert not store.writing
        assert not list(store.root.glob('*.part'))


def test_expired_shared_upload_survives_all_jobs(setup, monkeypatch):
    client, store = setup
    monkeypatch.setattr(uploads, '_probe', lambda *_: None)
    monkeypatch.setattr(uploads, 'MAX_FILES', 1)
    result = client.post('/sources/uploads', content=b'file bytes', headers=HEADERS)
    assert result.status_code == 200
    assert set(result.json()) == {'upload_id'}
    token = result.json()['upload_id']
    assert client.post('/sources/uploads', content=b'other', headers=HEADERS).status_code == 429
    spec = {'song': 'original.wav', 'upload_id': token}
    with ThreadPoolExecutor(2) as pool:
        responses = list(pool.map(lambda _: client.post('/cover', json=spec), range(2)))
    assert [r.status_code for r in responses] == [202, 202]
    first, second = [store.manager.get(r.json()['id']) for r in responses]
    path, meta = sources.resolve(first.spec, {}, store.manager.root / first.id)
    assert path.parent == store.root and path.read_bytes() == b'file bytes'
    assert meta['title'] == 'original.wav' and meta['origin'] == 'upload'
    assert Path(second.spec['audio_path']) == path
    old = time.time() - uploads.TTL - 1
    os.utime(path, (old, old))
    assert client.post('/cover', json=spec).status_code == 404
    first.status, second.status = 'done', 'running'
    store.clean()
    assert path.exists()
    second.status = 'done'
    store.clean()
    assert not path.exists()


@pytest.mark.skipif(not shutil.which('ffprobe'), reason='requires ffprobe')
def test_real_audio_probe(setup, monkeypatch):
    client, store = setup
    monkeypatch.setitem(api.CFG['mix'], 'ffprobe', shutil.which('ffprobe'))
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav:
        wav.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
        wav.writeframes(b'\0\0' * 800)
    for data, ext, status in [(b'not audio', 'wav', 422), (buf.getvalue(), 'mp3', 422), (buf.getvalue(), 'wav', 200)]:
        assert client.post('/sources/uploads', content=data, headers={**HEADERS, 'X-Audio-Extension': ext}).status_code == status
    assert len(list(store.root.iterdir())) == 1
