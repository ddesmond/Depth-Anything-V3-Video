"""SQLite history of Higgsfield generation jobs (survives restarts)."""
import json
import sqlite3
import threading
import time
import uuid

LOCAL_STATES = {"pending", "submitting"}  # not yet accepted by Higgsfield
ACTIVE_STATES = {"queued", "in_progress"}  # accepted, polling
CHARGED_STATES = {"completed"}  # failed / nsfw / canceled are refunded

_SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id TEXT PRIMARY KEY,
    name TEXT,
    created REAL
);
CREATE TABLE IF NOT EXISTS gen_jobs (
    id TEXT PRIMARY KEY,
    batch_id TEXT,
    parent_id TEXT,          -- job this was resubmitted / upscaled from
    kind TEXT,               -- draft | full | upscale | resubmit
    endpoint TEXT,
    title TEXT,
    family TEXT,
    prompt TEXT,
    seed INTEGER,
    quality TEXT,            -- resolution / mode value sent
    params TEXT,             -- JSON: arguments without media fields
    media TEXT,              -- JSON: {field: [{"label", "path"|"url", "da3_job"}]}
    sent_args TEXT,          -- JSON: exact body sent (with uploaded URLs)
    status TEXT,
    error TEXT,
    request_id TEXT,
    status_url TEXT,
    cancel_url TEXT,
    correlation_id TEXT,
    est_credits REAL,
    est_usd REAL,
    created REAL,
    submitted REAL,
    finished REAL,
    output_url TEXT,
    output_path TEXT,
    last_response TEXT,
    next_poll REAL DEFAULT 0,
    poll_delay REAL DEFAULT 2
);
CREATE INDEX IF NOT EXISTS gen_jobs_status ON gen_jobs(status);
CREATE INDEX IF NOT EXISTS gen_jobs_created ON gen_jobs(created);
"""

_JSON_COLS = ("params", "media", "sent_args", "last_response")


class GenStore:
    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(_SCHEMA)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.commit()

    @staticmethod
    def _row(r):
        if r is None:
            return None
        d = dict(r)
        for c in _JSON_COLS:
            if d.get(c):
                d[c] = json.loads(d[c])
        return d

    def new_batch(self, name):
        bid = uuid.uuid4().hex[:8]
        with self.lock:
            self.db.execute("INSERT INTO batches VALUES (?,?,?)", (bid, name, time.time()))
            self.db.commit()
        return bid

    def add_job(self, **fields):
        fields.setdefault("id", uuid.uuid4().hex[:12])
        fields.setdefault("created", time.time())
        fields.setdefault("status", "pending")
        for c in _JSON_COLS:
            if c in fields and not isinstance(fields[c], str):
                fields[c] = json.dumps(fields[c])
        cols = ",".join(fields)
        with self.lock:
            self.db.execute(f"INSERT INTO gen_jobs ({cols}) VALUES ({','.join('?' * len(fields))})",
                            list(fields.values()))
            self.db.commit()
        return fields["id"]

    def update(self, job_id, **fields):
        for c in _JSON_COLS:
            if c in fields and fields[c] is not None and not isinstance(fields[c], str):
                fields[c] = json.dumps(fields[c])
        sets = ",".join(f"{k}=?" for k in fields)
        with self.lock:
            self.db.execute(f"UPDATE gen_jobs SET {sets} WHERE id=?", [*fields.values(), job_id])
            self.db.commit()

    def get(self, job_id):
        with self.lock:
            return self._row(self.db.execute("SELECT * FROM gen_jobs WHERE id=?", (job_id,)).fetchone())

    def query(self, where="1=1", args=(), order="created DESC", limit=500):
        with self.lock:
            rows = self.db.execute(f"SELECT * FROM gen_jobs WHERE {where} ORDER BY {order} LIMIT ?",
                                   (*args, limit)).fetchall()
        return [self._row(r) for r in rows]

    def batch_names(self):
        with self.lock:
            return {r["id"]: r["name"] for r in self.db.execute("SELECT id, name FROM batches")}

    def spend(self):
        """Estimated spend: completed jobs by day and by model, plus in-flight."""
        with self.lock:
            by_day = self.db.execute(
                "SELECT date(finished, 'unixepoch', 'localtime') AS day, COUNT(*) n, "
                "SUM(est_usd) usd, SUM(est_credits) credits FROM gen_jobs WHERE status='completed' "
                "GROUP BY day ORDER BY day DESC LIMIT 60").fetchall()
            by_model = self.db.execute(
                "SELECT title, COUNT(*) n, SUM(est_usd) usd, SUM(est_credits) credits FROM gen_jobs "
                "WHERE status='completed' GROUP BY title ORDER BY usd DESC").fetchall()
            inflight = self.db.execute(
                "SELECT COUNT(*) n, SUM(est_usd) usd FROM gen_jobs "
                "WHERE status IN ('pending','submitting','queued','in_progress')").fetchone()
            totals = self.db.execute(
                "SELECT COUNT(*) n, SUM(est_usd) usd, SUM(est_credits) credits FROM gen_jobs "
                "WHERE status='completed'").fetchone()
        return [dict(r) for r in by_day], [dict(r) for r in by_model], dict(inflight), dict(totals)
