"""HTTP API: submit a path or an upload, poll status, fetch results.
The Gradio UI is mounted at /ui and uses the same queue."""
import logging
import os
import shutil
import zipfile
from typing import Literal

import gradio as gr
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

from . import higgsfield as hf
from . import processing
from .genstore import GenStore
from .genworker import GenWorker
from .jobs import Job, JobQueue
from .models import REPOS, ModelCache

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

DATA_DIR = os.environ.get("DATA_DIR", "/data")
JOBS_DIR = os.path.join(DATA_DIR, "jobs")
# Path jobs may only read/write under these roots (colon separated).
ALLOWED_ROOTS = [
    os.path.realpath(p)
    for p in (os.environ.get("ALLOWED_ROOTS", "") + ":" + DATA_DIR).split(":")
    if p
]

Encoder = Literal["da3-small", "da3-base", "da3-large"]
Compression = Literal["piz", "zip16", "zip", "none"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
models = ModelCache(DEVICE)
jobs = JobQueue(models)

# Higgsfield generation: persistent history + submit/poll/download worker
GEN_DIR = os.path.join(DATA_DIR, "generate")
os.makedirs(GEN_DIR, exist_ok=True)
gen_client = hf.Client()
gen_store = GenStore(os.path.join(GEN_DIR, "history.db"))
gen_worker = GenWorker(gen_store, gen_client, os.path.join(GEN_DIR, "outputs"),
                       max_concurrent=int(os.environ.get("HIGGSFIELD_MAX_CONCURRENT", "4")))

app = FastAPI(title="Depth Anything V3 service")


@app.on_event("startup")
def _start():
    os.makedirs(JOBS_DIR, exist_ok=True)
    jobs.start()
    gen_worker.start()
    if gen_client.configured and not os.environ.get("UI_PASSWORD"):
        logging.warning("HIGGSFIELD key set but UI_PASSWORD is empty: Generate submissions are blocked")


class JobError(ValueError):
    """Bad job request; becomes a 400 in the API and a message in the UI."""


class JobOptions(BaseModel):
    encoder: Encoder = "da3-base"
    first: int | None = Field(None, description="first frame (1-based for videos)")
    last: int | None = None
    process_res: int = Field(504, ge=140, le=2016, description="DA3 processing resolution")
    batch_size: int = Field(1, ge=1, le=64, description="frames per inference call")
    compression: Compression = "piz"
    preview: bool = True
    fps: float | None = None


class PathJob(JobOptions):
    input: str = Field(..., description="video, image, or sequence (shot.####.exr / %04d)")
    output_dir: str | None = Field(None, description="default: <input dir>/da3_depth")


# ---------------------------------------------------------------- shared logic

def _check_allowed(path):
    real = os.path.realpath(path)
    if not any(real == r or real.startswith(r + os.sep) for r in ALLOWED_ROOTS):
        raise JobError(f"path outside allowed roots {ALLOWED_ROOTS}: {path}")


def job_dict(job):
    return job.to_dict(position=jobs.position(job))


def submit_path_job(req: PathJob) -> Job:
    if processing.split_sequence(req.input) is None and not os.path.isfile(req.input):
        raise JobError(f"input not found (is its directory mounted?): {req.input}")
    out_dir = req.output_dir or os.path.join(os.path.dirname(req.input), "da3_depth")
    _check_allowed(req.input)
    _check_allowed(out_dir)
    try:
        if not processing.split_sequence(req.input):
            # a single numbered frame implies its sequence
            req.input = processing.sequence_from_file(req.input) or req.input
        if not len(processing.Source(req.input, req.first, req.last)):
            raise processing.InputError("no frames")
    except processing.InputError as e:
        raise JobError(str(e))
    job = Job(output_dir=out_dir, **req.model_dump(exclude={"output_dir"}))
    return jobs.submit(job)


def submit_upload_job(fileobj, filename, opts: JobOptions) -> Job:
    """`fileobj` is a readable binary file; it is copied into the job dir."""
    job = Job(input="", output_dir="", uploaded=True, **opts.model_dump())
    in_dir = os.path.join(JOBS_DIR, job.id, "in")
    os.makedirs(in_dir)
    name = os.path.basename(filename or "upload")
    dest = os.path.join(in_dir, name)
    with open(dest, "wb") as f:
        shutil.copyfileobj(fileobj, f)
    try:
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(dest) as z:
                for member in z.namelist():
                    target = os.path.realpath(os.path.join(in_dir, member))
                    if not target.startswith(os.path.realpath(in_dir) + os.sep):
                        raise processing.InputError(f"bad path in zip: {member}")
                z.extractall(in_dir)
            os.remove(dest)
            dest = processing.sequence_from_dir(in_dir)
        job.input = dest
        job.output_dir = os.path.join(JOBS_DIR, job.id, "out")
        processing.Source(job.input, opts.first, opts.last)
    except (processing.InputError, zipfile.BadZipFile) as e:
        shutil.rmtree(os.path.join(JOBS_DIR, job.id), ignore_errors=True)
        raise JobError(str(e))
    return jobs.submit(job)


def result_zip(job) -> str:
    """Zip of everything the job wrote (EXRs + preview); built once."""
    if job.status != "done":
        raise JobError(f"job is {job.status}")
    zip_path = os.path.join(JOBS_DIR, job.id, "result.zip")
    if not os.path.exists(zip_path):
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        stem = os.path.basename(job.result["exr_pattern"]).split(".")[0]
        with zipfile.ZipFile(zip_path + ".tmp", "w", zipfile.ZIP_STORED) as z:
            for f in sorted(os.listdir(job.output_dir)):
                if f.startswith(stem):
                    z.write(os.path.join(job.output_dir, f), f)
        os.replace(zip_path + ".tmp", zip_path)
    return zip_path


# ---------------------------------------------------------------- routes

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui")


@app.get("/health")
def health():
    gpu = None
    if DEVICE == "cuda":
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": "%d.%d" % torch.cuda.get_device_capability(0),
            "arch_list": torch.cuda.get_arch_list(),
        }
    return {
        "device": DEVICE,
        "torch": torch.__version__,
        "gpu": gpu,
        "loaded_model": models.encoder,
        "queued": sum(j.status == "queued" for j in jobs.jobs.values()),
        "running": [j.id for j in jobs.jobs.values() if j.status == "running"],
    }


@app.get("/models")
def list_models():
    return {"encoders": list(REPOS), "loaded": models.encoder}


@app.post("/jobs", status_code=201)
def submit_path(req: PathJob):
    try:
        return job_dict(submit_path_job(req))
    except JobError as e:
        raise HTTPException(400, str(e))


@app.post("/jobs/upload", status_code=201)
def submit_upload(
    file: UploadFile = File(..., description="video, image, or .zip of an image sequence"),
    encoder: Encoder = Form("da3-base"),
    first: int | None = Form(None),
    last: int | None = Form(None),
    process_res: int = Form(504),
    batch_size: int = Form(1),
    compression: Compression = Form("piz"),
    preview: bool = Form(True),
    fps: float | None = Form(None),
):
    opts = JobOptions(encoder=encoder, first=first, last=last, process_res=process_res,
                      batch_size=batch_size, compression=compression, preview=preview, fps=fps)
    try:
        return job_dict(submit_upload_job(file.file, file.filename, opts))
    except JobError as e:
        raise HTTPException(400, str(e))


@app.get("/jobs")
def list_jobs():
    return [job_dict(j) for j in jobs.list()]


def _job(job_id):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    return job_dict(_job(job_id))


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    job = _job(job_id)
    jobs.cancel(job)
    return job_dict(job)


@app.get("/jobs/{job_id}/result")
def download_result(job_id: str):
    job = _job(job_id)
    try:
        path = result_zip(job)
    except JobError as e:
        raise HTTPException(409, str(e))
    return FileResponse(path, filename=f"{job.id}_depth.zip")


@app.get("/jobs/{job_id}/preview")
def download_preview(job_id: str):
    job = _job(job_id)
    if job.status != "done" or not (job.result or {}).get("preview"):
        raise HTTPException(404, "no preview")
    return FileResponse(job.result["preview"], media_type="video/mp4")


from .ui import CSS, THEME, build_ui  # noqa: E402  (needs the helpers above)

# One shared password for the whole UI (the REST API stays open for the Nuke node).
_auth = (os.environ.get("UI_USER", "da3"), os.environ["UI_PASSWORD"]) if os.environ.get("UI_PASSWORD") else None
app = gr.mount_gradio_app(app, build_ui(), path="/ui", allowed_paths=ALLOWED_ROOTS, auth=_auth,
                          auth_message="Depth Anything V3 · shared studio password",
                          theme=THEME, css=CSS)
