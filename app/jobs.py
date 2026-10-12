"""In-memory FIFO job queue with a single GPU worker thread."""
import logging
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field

from . import processing

log = logging.getLogger(__name__)


@dataclass
class Job:
    input: str
    output_dir: str
    encoder: str = "da3-base"
    first: int | None = None
    last: int | None = None
    process_res: int = 504
    batch_size: int = 1
    compression: str = "piz"
    preview: bool = True
    fps: float | None = None
    uploaded: bool = False

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "queued"  # queued | running | done | failed | cancelled
    done: int = 0
    total: int = 0
    error: str | None = None
    result: dict | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self, position=None):
        d = {k: v for k, v in self.__dict__.items() if k != "cancel_event"}
        d["queue_position"] = position
        return d


class JobQueue:
    def __init__(self, models, keep=500):
        self.models = models
        self.keep = keep
        self.jobs: dict[str, Job] = {}
        self._pending: queue.Queue[Job] = queue.Queue()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, name="da3-worker", daemon=True)

    def start(self):
        self._thread.start()

    def submit(self, job):
        with self._lock:
            self.jobs[job.id] = job
            self._prune()
        self._pending.put(job)
        return job

    def get(self, job_id):
        return self.jobs.get(job_id)

    def position(self, job):
        if job.status != "queued":
            return None
        queued = [j for j in self.jobs.values() if j.status == "queued"]
        return queued.index(job) + 1 if job in queued else None

    def list(self):
        return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)

    def cancel(self, job):
        job.cancel_event.set()
        if job.status == "queued":
            job.status = "cancelled"
            job.finished = time.time()

    def _prune(self):
        finished = [j for j in self.jobs.values() if j.status in ("done", "failed", "cancelled")]
        for j in sorted(finished, key=lambda j: j.created)[: max(0, len(self.jobs) - self.keep)]:
            del self.jobs[j.id]

    def _worker(self):
        while True:
            job = self._pending.get()
            if job.cancel_event.is_set():
                continue
            job.status = "running"
            job.started = time.time()
            try:
                source = processing.Source(job.input, job.first, job.last)
                job.total = len(source)
                model = self.models.get(job.encoder)

                def progress(done, total):
                    job.done, job.total = done, total

                job.result = processing.process(
                    model, source, job.output_dir, job.encoder,
                    process_res=job.process_res, batch_size=job.batch_size,
                    compression=job.compression, preview=job.preview, fps=job.fps,
                    on_progress=progress, cancelled=job.cancel_event.is_set,
                )
                job.status = "cancelled" if job.cancel_event.is_set() else "done"
            except Exception as e:
                log.error("job %s failed:\n%s", job.id, traceback.format_exc())
                job.status = "failed"
                job.error = f"{type(e).__name__}: {e}"
            finally:
                job.finished = time.time()
