"""Content identities and atomic, cleanup-compatible render cache entries."""
import contextlib
import hashlib
import json
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def select_assets(cfg, voice):
    model = Path(voice.get('model') or cfg.get('model') or '').expanduser().resolve()
    # A model override supplies its own optional index, even when explicitly empty.
    index = voice.get('index', '' if voice.get('model') else cfg.get('index'))
    index = Path(index).expanduser().resolve() if index else None
    if not model.is_file():
        raise ValueError(f'RVC 模型不存在: {model}')
    if index is not None and not index.is_file():
        raise ValueError(f'RVC 索引不存在: {index}')
    return model, index


def dry_cache_key(audio_digest, vocals_digest, model_digest, index_digest, embedder, settings):
    return hashlib.sha256(json.dumps([
        'rvc-v2', audio_digest, vocals_digest, model_digest, index_digest, embedder, settings,
    ], sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def final_cache_key(dry_key, instrumental_digest, mix_settings):
    return hashlib.sha256(json.dumps([
        'mix-v11', dry_key, instrumental_digest, mix_settings,
    ], sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def store_render(entry, audio, meta, *, name='audio'):
    """Publish audio and pitch metadata together, under a single rename."""
    temp = None
    try:
        entry.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=entry.parent, delete=False) as stage:
            temp = Path(stage.name)
            with zipfile.ZipFile(stage, 'w') as archive:
                archive.write(audio, name)
                archive.writestr('meta', json.dumps(meta, ensure_ascii=False))
        os.replace(temp, entry)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        logging.getLogger(__name__).warning('写缓存失败: %s', exc)
    finally:
        if temp:
            with contextlib.suppress(OSError):
                temp.unlink(missing_ok=True)


def restore_render(entry, audio, *, name='audio'):
    temp = None
    try:
        with zipfile.ZipFile(entry) as archive:
            meta = json.loads(archive.read('meta'))
            if not isinstance(meta, dict) or type(meta.get('effective_pitch')) is not int:
                return None
            if archive.getinfo(name).file_size <= 0:
                return None
            audio.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as src, tempfile.NamedTemporaryFile(dir=audio.parent, delete=False) as dst:
                temp = Path(dst.name)
                shutil.copyfileobj(src, dst)
        os.replace(temp, audio)
        return meta
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None
    finally:
        if temp:
            with contextlib.suppress(OSError):
                temp.unlink(missing_ok=True)
