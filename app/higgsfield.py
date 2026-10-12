"""Higgsfield API client and helpers over the bundled model catalog
(app/higgsfield_catalog.json, rebuilt by scripts/update_higgsfield_catalog.py)."""
import json
import mimetypes
import os
import re
import threading
import time

import requests

BASE_URL = "https://api.higgsfield.ai"
CONSOLE_URL = "https://console.higgsfield.ai"
TERMINAL = {"completed", "failed", "nsfw", "canceled"}

_CATALOG_PATH = os.path.join(os.path.dirname(__file__), "higgsfield_catalog.json")


class HiggsfieldError(RuntimeError):
    def __init__(self, message, status=None, correlation_id=None):
        super().__init__(message)
        self.status = status
        self.correlation_id = correlation_id

    @property
    def is_concurrency_limit(self):
        return self.status == 400 and "concurrent" in str(self).lower()


# ---------------------------------------------------------------- catalog

def load_catalog():
    with open(_CATALOG_PATH) as f:
        return json.load(f)


CATALOG = load_catalog()
WORKFLOWS = {
    wf["endpoint"]: {**wf, "family": fam["name"], "vendor": fam["vendor"]}
    for fam in CATALOG
    for wf in fam["workflows"]
}


def workflow_label(endpoint):
    wf = WORKFLOWS.get(endpoint)
    return wf["title"] if wf else endpoint


def field_kind(prop):
    """Map a JSON-schema property to the form widget kind used by the UI."""
    t = prop.get("type")
    if "enum" in prop:
        return "enum"
    if t == "boolean":
        return "bool"
    if t in ("integer", "number"):
        return "number"
    if t == "array":
        items = prop.get("items") or {}
        if items.get("type") == "object":
            return "json"
        return "uri_list" if items.get("format") == "uri" else "str_list"
    if t == "string" and prop.get("format") == "uri":
        return "uri"
    return "text"


_TIER_ORDER = {"480p": 480, "540p": 540, "580p": 580, "720p": 720, "768p": 768, "1080p": 1080,
               "2k": 1440, "1440p": 1440, "4k": 2160, "2160p": 2160,
               "std": 1, "standard": 1, "lite": 1, "fast": 1, "turbo": 2, "pro": 3}


def tier_rank(value):
    v = str(value).lower()
    if v in _TIER_ORDER:
        return _TIER_ORDER[v]
    m = re.match(r"(\d+)p$", v)
    return int(m.group(1)) if m else 0


def quality_field(endpoint):
    """The field that selects output quality: `resolution`, else Kling-style `mode`."""
    props = WORKFLOWS[endpoint]["schema"].get("properties", {})
    for name in ("resolution", "mode"):
        if name in props and "enum" in props[name]:
            return name, sorted(props[name]["enum"], key=tier_rank)
    return None, []


_TIER_TOKENS = ("std", "standard", "pro", "4k", "turbo", "fast", "lite")


def sibling_tiers(endpoint):
    """Same model at another quality tier published as its own endpoint
    (e.g. kling-video/v3.0/standard/... -> .../pro/..., .../4k/...)."""
    out = []
    parts = re.split(r"([/-])", endpoint)
    for i, tok in enumerate(parts):
        if tok.lower() not in _TIER_TOKENS:
            continue
        for alt in _TIER_TOKENS:
            if alt == tok.lower():
                continue
            cand = "".join(parts[:i] + [alt] + parts[i + 1:])
            if cand in WORKFLOWS and cand not in out:
                out.append(cand)
    return out


def endpoint_tier(endpoint):
    """Rank of the tier token in an endpoint ID (std < turbo < pro < 4k); 0 if none."""
    return max((tier_rank(t) if t.lower() != "4k" else 4 for t in re.split(r"[/-]", endpoint)
                if t.lower() in _TIER_TOKENS), default=0)


def has_seed(endpoint):
    return "seed" in WORKFLOWS[endpoint]["schema"].get("properties", {})


def seed_range(endpoint):
    p = WORKFLOWS[endpoint]["schema"]["properties"]["seed"]
    return max(1, int(p.get("minimum", 1))), int(p.get("maximum", 2147483646))


# ---------------------------------------------------------------- client

class Client:
    def __init__(self, key_id=None, key_secret=None, base_url=None):
        self.key_id = key_id if key_id is not None else os.environ.get("HIGGSFIELD_KEY_ID", "")
        self.key_secret = key_secret if key_secret is not None else os.environ.get("HIGGSFIELD_KEY_SECRET", "")
        self.base_url = (base_url or os.environ.get("HIGGSFIELD_BASE_URL") or BASE_URL).rstrip("/")
        self.session = requests.Session()
        self._uploads = {}  # (path, mtime) -> (public_url, uploaded_at)
        self._upload_lock = threading.Lock()

    @property
    def configured(self):
        return bool(self.key_id and self.key_secret)

    def _call(self, method, url, timeout=60, **kwargs):
        if not self.configured:
            raise HiggsfieldError("No Higgsfield API key: set HIGGSFIELD_KEY_ID and HIGGSFIELD_KEY_SECRET in .env")
        if not url.startswith("http"):
            url = self.base_url + url
        headers = {"Authorization": f"Key {self.key_id}:{self.key_secret}", **kwargs.pop("headers", {})}
        try:
            r = self.session.request(method, url, headers=headers, timeout=timeout, **kwargs)
        except requests.RequestException as e:
            raise HiggsfieldError(f"network error: {e}")
        cid = r.headers.get("X-Correlation-ID")
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = r.text[:300]
            if isinstance(detail, list):  # validation errors
                detail = "; ".join(f"{'.'.join(map(str, d.get('loc', [])[1:]))}: {d.get('msg')}" for d in detail)
            raise HiggsfieldError(f"HTTP {r.status_code}: {detail}", r.status_code, cid)
        return (r.json() if r.content else {}), cid

    def submit(self, endpoint, arguments, idempotency_key):
        body, cid = self._call("POST", f"/{endpoint}", json=arguments,
                               headers={"Idempotency-Key": idempotency_key})
        return body, cid

    def estimate(self, endpoint, arguments):
        body, _ = self._call("POST", f"/estimate/{endpoint}", json=arguments, timeout=30)
        return body  # {"credits": "1.500", "usd": "0.094"}

    def status(self, status_url):
        body, _ = self._call("GET", status_url, timeout=30)
        return body

    def cancel(self, cancel_url):
        self._call("POST", cancel_url, timeout=30)

    def presets(self, endpoint):
        body, _ = self._call("GET", f"/models/{endpoint}/presets", timeout=30)
        return body.get("items", body if isinstance(body, list) else [])

    def upload(self, path, max_age=40 * 60):
        """Upload a local file via a presigned URL; reuses recent uploads."""
        key = (os.path.realpath(path), os.path.getmtime(path))
        with self._upload_lock:
            hit = self._uploads.get(key)
            if hit and time.time() - hit[1] < max_age:
                return hit[0]
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        info, _ = self._call("POST", "/files/generate-upload-url", json={"content_type": ctype})
        with open(path, "rb") as f:
            r = requests.put(info["upload_url"], data=f, headers=info.get("upload_headers", {}), timeout=600)
        if r.status_code >= 400:
            raise HiggsfieldError(f"upload failed: HTTP {r.status_code} {r.text[:200]}", r.status_code)
        with self._upload_lock:
            self._uploads[key] = (info["public_url"], time.time())
        return info["public_url"]


def output_url(status_body):
    """The main output URL of a completed request (video, else image/audio)."""
    for key in ("video", "audio"):
        if isinstance(status_body.get(key), dict) and status_body[key].get("url"):
            return status_body[key]["url"]
    for key in ("videos", "images", "audios"):
        if status_body.get(key):
            return status_body[key][0].get("url")
    return None
