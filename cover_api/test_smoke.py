"""API 冒烟测试：不依赖 GPU，验证路由、校验、状态机。"""
import os
import sys
import tempfile
from pathlib import Path

# 相对本文件定位项目根（cover_api/ 的上一级），避免写死某台机器的工作区路径
PROJECT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT)
os.environ["FEIYU_COVER_DATA_DIR"] = tempfile.mkdtemp(prefix="feiyu-jobs-")
os.environ.setdefault("FEIYU_COVER_API_KEY", "")

import cover_api.pipeline as pipeline


def fake_runner(job, mgr):
    mgr.update(job, step="converting", progress=0.5, message="假装在换声")


pipeline.runner = fake_runner

from fastapi.testclient import TestClient  # noqa: E402
from cover_api.app import app  # noqa: E402

ok = []
fail = []


def check(name, cond, extra=""):
    (ok if cond else fail).append(name)
    print(("  PASS " if cond else "  FAIL ") + name + (f"  {extra}" if extra else ""))


with TestClient(app) as client:
    r = client.get("/health")
    check("GET /health", r.status_code == 200 and r.json()["status"] == "ok", r.text[:80])

    r = client.get("/capabilities")
    body = r.json()
    check("GET /capabilities", r.status_code == 200 and "problems" in body,
          f"ready={body.get('ready')}")

    r = client.post("/cover", json={"song": "大鱼 周深"})
    check("POST /cover 202", r.status_code == 202, r.text[:120])
    payload = r.json()
    job_id = payload.get("id", "")
    check("返回 job_id", bool(job_id), job_id)
    check("返回排队信息", "queue_position" in payload and "eta_seconds" in payload)

    r = client.get(f"/jobs/{job_id}")
    check("GET /jobs/{id}", r.status_code == 200 and r.json()["id"] == job_id)

    r = client.get("/jobs/deadbeef0000")
    check("不存在的任务返回 404", r.status_code == 404)

    r = client.get(f"/jobs/{job_id}/audio")
    check("未完成时取音频返回 409", r.status_code in (404, 409), str(r.status_code))

    r = client.post("/cover", json={"pitch": 5})
    check("缺少音源返回 422", r.status_code == 422)

    r = client.post("/cover", json={"song": "x", "bogus": 1})
    check("多余字段被拒绝", r.status_code == 422)

    r = client.get("/jobs")
    check("GET /jobs", r.status_code == 200 and "jobs" in r.json())

    r = client.get("/system/status")
    check("GET /system/status", r.status_code == 200 and r.json().get("ok") is True and "rvc_worker" in r.json())

    r = client.post("/system/unload")
    check("POST /system/unload", r.status_code == 200 and r.json().get("ok") is True)

    r = client.post("/jobs/deadbeef0000/cancel")
    check("POST /jobs/{id}/cancel 不存在返回 404", r.status_code == 404)

    r = client.post("/cover", json={"bvid": "BV1faak6AEzm", "preview_seconds": 60, "voice": {"auto_pitch": True, "f0_autotune": True, "vocal_reverb": True}})
    check("POST /cover 新参数(bvid/preview/auto_pitch/autotune/reverb)", r.status_code == 202)

    r = client.get("/studio")
    check("GET /studio 独立上传页", r.status_code == 200 and "text/html" in r.headers.get("content-type", ""))

    r = client.get("/studio/status")
    body = r.json()
    check("GET /studio/status", r.status_code == 200 and "task" in body and "datasets" in body)

    r = client.post("/studio/cancel")
    check("POST /studio/cancel 空闲返回 ok=False", r.status_code == 200 and r.json().get("ok") is False)

    r = client.get("/jobs/deadbeef0000/highlight")
    check("GET /jobs/{id}/highlight 不存在返回 404", r.status_code == 404)

    r = client.post("/studio/inspect", json={"dataset": "/nonexistent/dataset_xyz"})
    check("POST /studio/inspect 目录不存在返回 422", r.status_code == 422)

print()
print(f"通过 {len(ok)} 项，失败 {len(fail)} 项")
if fail:
    print("失败项:", fail)
    sys.exit(1)
