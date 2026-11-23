"""Bounded upload staging for the single-worker cover API."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import subprocess
import threading
import time
from pathlib import Path

from fastapi import HTTPException, Request

MAX_BYTES = 32 * 1024 * 1024
MAX_FILES = 16  # includes partial uploads: at most 512 MiB on disk
TTL = 3600
FORMATS = {'mp3': 'mp3', 'wav': 'wav', 'flac': 'flac', 'm4a': 'mov',
           'ogg': 'ogg', 'opus': 'ogg', 'aac': 'aac'}
ID_PATTERN = r'[a-f0-9]{32}'


def _probe(path, extension, executable):
    # Limit demuxers and protocols: never interpret an upload as a playlist.
    try:
        proc = subprocess.run(
            [executable, '-v', 'error', '-protocol_whitelist', 'file',
             '-format_whitelist', FORMATS[extension], '-select_streams', 'a:0',
             '-show_entries', 'stream=codec_name:format=format_name', '-of', 'json', str(path)],
            capture_output=True, timeout=10, check=False)
        info = json.loads(proc.stdout) if proc.returncode == 0 else {}
        streams = info.get('streams') or []
        codec = streams[0].get('codec_name') if streams else None
        if (codec and FORMATS[extension] in info['format']['format_name'].split(',')
                and (extension != 'opus' or codec == 'opus')):
            return
    except OSError as exc:
        raise HTTPException(503, '服务端无法探测音频格式') from exc
    except (ValueError, KeyError, subprocess.TimeoutExpired):
        pass
    raise HTTPException(422, '音频内容无效或与扩展名不符')


class UploadStore:
    def __init__(self, manager):
        self.manager = manager
        self.root = manager.root.resolve() / 'source_uploads'
        self.lock = threading.RLock()
        self.writing = set()

    def _files(self):
        if self.root.is_symlink() or self.root.resolve() != self.root:
            raise RuntimeError('音源上传目录无效')
        self.root.mkdir(mode=0o700, exist_ok=True)
        return [p for p in self.root.iterdir() if re.fullmatch(ID_PATTERN, p.stem)
                and p.suffix[1:] in (*FORMATS, 'part')]

    def clean(self):
        with self.lock:
            with self.manager._lock:
                pinned = {job.spec.get('upload_id') for job in self.manager._jobs.values()
                          if job.status in ('queued', 'running')}
            for path in self._files():
                if path.stem not in self.writing and path.stem not in pinned:
                    if path.lstat().st_mtime <= time.time() - TTL:
                        path.unlink(missing_ok=True)

    async def periodic(self):
        while True:
            try:
                await asyncio.to_thread(self.clean)
            except (OSError, RuntimeError):
                logging.getLogger(__name__).exception('音源暂存清理失败')
            await asyncio.sleep(60)

    async def receive(self, request: Request, ffprobe: str):
        ext = request.headers.get('x-audio-extension', '')
        if request.headers.get('content-type', '').partition(';')[0].strip().lower() != 'application/octet-stream' or ext not in FORMATS:
            raise HTTPException(415, '需要 application/octet-stream 和有效的 X-Audio-Extension')
        length = request.headers.get('content-length')
        if length is not None:
            if not re.fullmatch(r'[0-9]{1,20}', length):
                raise HTTPException(400, 'Content-Length 无效')
            if int(length) > MAX_BYTES:
                raise HTTPException(413, '单个音频不能超过 32 MiB')
        with self.lock:
            self.clean()
            if len(self._files()) >= MAX_FILES:
                raise HTTPException(429, '音源暂存已满，请稍后再试')
            token = secrets.token_hex(16)
            temp = self.root / (token + '.part')
            handle = temp.open('xb')
            self.writing.add(token)
        try:
            size = 0
            with handle:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, '单个音频不能超过 32 MiB')
                    handle.write(chunk)
            if not size:
                raise HTTPException(400, '上传音频为空')
            await asyncio.to_thread(_probe, temp, ext, ffprobe)
            with self.lock:
                temp.touch()
                temp.rename(temp.with_suffix('.' + ext))
            return {'upload_id': token}
        finally:
            with self.lock:
                self.writing.discard(token)
                temp.unlink(missing_ok=True)

    def submit(self, spec):
        token = spec['upload_id']
        if not re.fullmatch(ID_PATTERN, token):
            raise HTTPException(422, '音源 ID 无效')
        # Cleanup cannot slip between lookup and registration in the job table.
        with self.lock:
            for path in self._files():
                if (path.stem == token and path.suffix[1:] in FORMATS and not path.is_symlink()
                        and path.is_file() and path.stat().st_mtime > time.time() - TTL):
                    return self.manager.submit({**spec, 'audio_path': str(path)})
        raise HTTPException(404, '音源不存在或已过期，请重新上传')
