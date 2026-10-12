"""Background thread: submits pending Higgsfield jobs within the account's
concurrency limit, polls accepted ones, and downloads finished outputs."""
import logging
import os
import random
import threading
import time
import traceback

import requests

from . import higgsfield as hf

log = logging.getLogger(__name__)


class GenWorker:
    def __init__(self, store, client, out_dir, max_concurrent=4):
        self.store = store
        self.client = client
        self.out_dir = out_dir
        self.max_concurrent = max_concurrent
        self._pause_until = 0.0
        self._thread = threading.Thread(target=self._run, name="higgsfield-worker", daemon=True)

    def start(self):
        os.makedirs(self.out_dir, exist_ok=True)
        # a crash mid-submit leaves "submitting": resubmit with the same idempotency key
        for j in self.store.query("status='submitting'"):
            self.store.update(j["id"], status="pending")
        self._thread.start()

    # ------------------------------------------------------------ actions

    def cancel(self, job_id):
        job = self.store.get(job_id)
        if job is None:
            return "no such job"
        if job["status"] == "pending":
            self.store.update(job_id, status="canceled", finished=time.time())
            return "canceled (not yet submitted)"
        if job["status"] == "queued" and job["cancel_url"]:
            try:
                self.client.cancel(job["cancel_url"])
            except hf.HiggsfieldError as e:
                return f"cancel refused: {e}"
            self.store.update(job_id, next_poll=0)
            return "cancel requested"
        return f"cannot cancel a job that is {job['status']}"

    def build_arguments(self, job):
        """params + media fields, uploading local files to Higgsfield."""
        args = dict(job["params"] or {})
        props = hf.WORKFLOWS[job["endpoint"]]["schema"].get("properties", {})
        for field, items in (job["media"] or {}).items():
            urls = [it["url"] if it.get("url") else self.client.upload(it["path"]) for it in items]
            if hf.field_kind(props.get(field, {})) == "uri":
                args[field] = urls[0]
            else:
                args[field] = urls
        return args

    # ------------------------------------------------------------ loop

    def _run(self):
        while True:
            try:
                self._tick()
            except Exception:
                log.error("worker tick failed:\n%s", traceback.format_exc())
            time.sleep(1.0)

    def _tick(self):
        now = time.time()
        if self.client.configured and now >= self._pause_until:
            active = len(self.store.query("status IN ('queued','in_progress')", limit=10000))
            for job in self.store.query("status='pending' AND next_poll<=?", (now,), order="created ASC",
                                        limit=max(0, self.max_concurrent - active)):
                if not self._submit(job):
                    break
        for job in self.store.query("status IN ('queued','in_progress') AND next_poll<=?", (now,),
                                    order="next_poll ASC", limit=50):
            self._poll(job)
        for job in self.store.query("status='completed' AND output_path IS NULL AND output_url IS NOT NULL "
                                    "AND next_poll<=?", (now,), limit=5):
            self._download(job)

    def _submit(self, job):
        """Returns False when submissions should stop for this tick."""
        self.store.update(job["id"], status="submitting", error=None)
        try:
            args = self.build_arguments(job)
            try:
                est = self.client.estimate(job["endpoint"], args)
                est_fields = {"est_credits": float(est.get("credits", 0)), "est_usd": float(est.get("usd", 0))}
            except hf.HiggsfieldError as e:
                log.warning("estimate failed for %s: %s", job["id"], e)
                est_fields = {}
            body, cid = self.client.submit(job["endpoint"], args, idempotency_key=job["id"])
        except hf.HiggsfieldError as e:
            if e.is_concurrency_limit:
                self.store.update(job["id"], status="pending")
                self._pause_until = time.time() + 10
                return False
            if e.status is None or e.status >= 500 or e.status in (423, 503):
                # ambiguous / temporary: retry later with the same idempotency key
                self.store.update(job["id"], status="pending", error=str(e), next_poll=time.time() + 30)
                return True
            self.store.update(job["id"], status="error", error=str(e), correlation_id=e.correlation_id,
                              finished=time.time())
            return True
        except Exception as e:  # e.g. local file missing
            self.store.update(job["id"], status="error", error=f"{type(e).__name__}: {e}", finished=time.time())
            return True
        self.store.update(
            job["id"], status=body.get("status", "queued"), request_id=body.get("request_id"),
            status_url=body.get("status_url"), cancel_url=body.get("cancel_url"), correlation_id=cid,
            sent_args=args, submitted=time.time(), next_poll=time.time() + 2, poll_delay=2, error=None,
            **est_fields,
        )
        return True

    def _poll(self, job):
        delay = min(job["poll_delay"] * 1.5, 10.0)
        try:
            body = self.client.status(job["status_url"])
        except hf.HiggsfieldError as e:
            if e.status in (401, 404):
                self.store.update(job["id"], status="error", error=str(e), finished=time.time())
            else:
                self.store.update(job["id"], next_poll=time.time() + delay, poll_delay=delay)
            return
        status = body.get("status", job["status"])
        fields = {"status": status, "last_response": body, "next_poll": time.time() + delay + random.uniform(0, 0.5),
                  "poll_delay": delay}
        if status in hf.TERMINAL:
            fields.update(finished=time.time(), next_poll=0, error=body.get("error"),
                          output_url=hf.output_url(body) if status == "completed" else None)
        self.store.update(job["id"], **fields)

    def _download(self, job):
        url = job["output_url"]
        ext = os.path.splitext(url.split("?")[0])[1] or ".mp4"
        dest = os.path.join(self.out_dir, f"{job['id']}{ext}")
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest + ".part", "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            os.replace(dest + ".part", dest)
            self.store.update(job["id"], output_path=dest, error=None)
        except Exception as e:
            log.warning("download failed for %s: %s", job["id"], e)
            self.store.update(job["id"], error=f"download failed, retrying: {e}", next_poll=time.time() + 60)
