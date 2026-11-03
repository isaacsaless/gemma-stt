# -*- coding: utf-8 -*-
"""Web app server: audio -> transcript (faster-whisper) + semantic search.

Run: python server.py  (then open http://localhost:8425)
"""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent))
from stt.search import TextIndexer
from stt.transcribe import Engine, clear_gpu
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models" / "embeddinggemma-2"
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="gemma-stt")
_lock = threading.Lock()  # one GPU job at a time


class Job:
    def __init__(self, path: Path, name: str):
        self.id = uuid.uuid4().hex[:8]
        self.path = path
        self.name = name
        self.status = "queued"
        self.message = "na fila"
        self.result: dict | None = None
        self.error: str | None = None
        self.events: list[str] = []  # ready-to-send SSE blocks
        self.created = time.time()

    def tick(self, msg: str):
        self.message = msg
        self.events.append(f"event: progress\ndata: {msg}\n\n")


_jobs: dict[str, Job] = {}
_engine: Engine | None = None
_indexer: TextIndexer | None = None
_active: dict = {}  # per-serve transcript index keyed by job id


def get_engine() -> tuple[Engine, TextIndexer]:
    global _engine, _indexer
    if _engine is None:
        _engine = Engine(str(MODEL_DIR))
        _indexer = TextIndexer(_engine.embedder)
    return _engine, _indexer


class TranscribeReq(BaseModel):
    path: str | None = None


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), name: str | None = None):
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{Path(file.filename or 'audio').name}"
    with open(dest, "wb") as fh:
        while chunk := await file.read(1 << 20):
            fh.write(chunk)
    return start_job(dest, file.filename or dest.name)


@app.post("/api/transcribe")
async def transcribe(req: TranscribeReq):
    if not req.path:
        raise HTTPException(400, "path required")
    p = Path(req.path)
    if not p.is_file():
        raise HTTPException(404, f"não encontrado: {p}")
    return start_job(p, p.name)


def start_job(path: Path, name: str) -> dict:
    job = Job(path, name)
    _jobs[job.id] = job
    threading.Thread(target=_run_job_wrapped, args=(job,), daemon=True).start()
    return {"id": job.id, "name": name}


def _run_job_wrapped(job: Job):
    t0 = time.perf_counter()
    try:
        with _lock:
            engine, indexer = get_engine()
            job.status = "running"
            result = engine.transcribe(job.path, progress=job.tick)
        clear_gpu()
        job.status = "indexing"
        job.tick("indexando para busca semântica")
        with _lock:
            idx = indexer.index(result["words"])
        _active[job.id] = idx
        clear_gpu()
        result["realtime_x"] = round(result["duration"] / max(1e-6, time.perf_counter() - t0), 1)
        job.result = result
        job.status = "done"
        job.tick("pronto")
    except Exception as e:  # noqa: BLE001
        import traceback
        job.status = "error"
        job.error = "".join(traceback.format_exception_only(type(e), e)).strip()
        job.events.append(f"event: error\ndata: {job.error}\n\n")


@app.get("/api/job/{job_id}")
async def job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404)
    return {
        "id": job.id, "name": job.name, "status": job.status,
        "message": job.message, "error": job.error,
        "result": job.result,
    }


@app.get("/api/job/{job_id}/events")
async def job_events(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404)
    import asyncio

    async def gen():
        idx = 0
        while True:
            await asyncio.sleep(0.25)
            while idx < len(job.events):
                yield job.events[idx]
                idx += 1
            if job.status in ("done", "error"):
                yield "event: end\ndata: {}\n\n"
                return

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/job/{job_id}/audio")
async def job_audio(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404)
    from fastapi.responses import FileResponse
    return FileResponse(str(job.path), filename=job.name)


class SearchReq(BaseModel):
    q: str
    job_id: str


@app.post("/api/search")
async def search(req: SearchReq):
    idx = _active.get(req.job_id)
    if idx is None:
        return {"hits": []}
    if not req.q.strip():
        return {"hits": []}
    with _lock:
        hits = _indexer.search(idx, req.q)  # type: ignore[union-attr]
    return {"hits": hits}


@app.get("/api/status")
async def status():
    dev = _engine.device if _engine else None
    return {"server": "ok", "device": dev}


class ConcatReq(BaseModel):
    job_id: str
    fmt: str = "txt"


@app.post("/api/export/{job_id}")
async def export(job_id: str, req: ConcatReq):
    job = _jobs.get(job_id)
    if not job or job.result is None:
        raise HTTPException(404)
    r = dict(job.result)
    if req.fmt == "json":
        return r
    txt = r["text"]
    return StreamingResponse(iter([txt]), media_type="text/plain")


app.mount("/", StaticFiles(directory=str(ROOT / "static"), html=True), name="static")


def main():
    import uvicorn
    print("\n  gemma-stt rodando: http://localhost:8425\n")
    uvicorn.run(app, host="0.0.0.0", port=8425, log_level="warning")


if __name__ == "__main__":
    main()
