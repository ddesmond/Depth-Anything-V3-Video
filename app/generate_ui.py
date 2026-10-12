"""Gradio tabs for sending DA3 results to Higgsfield video models:
Generate (form built from the model catalog), History, Funds."""
import itertools
import json
import os
import random
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import gradio as gr

from . import higgsfield as hf
from . import media

STATUS_FILTERS = ["all", "active", "completed", "failed / error", "canceled"]
HISTORY_COLUMNS = ["id", "created", "batch", "kind", "model", "quality", "seed", "status", "est $", "prompt"]
MEDIA_KINDS = ["depth mp4", "plate mp4", "preview mp4"]
# Shown on page load so the prompt and parameters are visible straight away.
DEFAULT_WORKFLOW = "wan/v2.7/reference-to-video"
VARIANT_SPLIT = re.compile(r"^\s*---\s*$", re.M)


# ---------------------------------------------------------------- form model

def _form_fields():
    """Every (name, kind) that appears in any workflow, in a stable order."""
    seen = {}
    for wf in hf.WORKFLOWS.values():
        for name, prop in wf["schema"].get("properties", {}).items():
            kind = "preset" if name == "preset_id" else hf.field_kind(prop)
            seen.setdefault((name, kind), prop)
    first = ["prompt", "negative_prompt", "multi_prompt", "video_url", "video_urls", "image_url",
             "image_urls", "first_frame_url", "last_frame_url", "end_image_url", "last_image_url",
             "audio_url", "audio_urls", "preset_id", "resolution", "mode", "duration", "aspect_ratio", "seed"]
    rank = {n: i for i, n in enumerate(first)}
    return sorted(seen, key=lambda k: (rank.get(k[0], len(first)), k[0], k[1]))


FIELDS = _form_fields()


def _prop_kind(endpoint, name):
    prop = hf.WORKFLOWS[endpoint]["schema"].get("properties", {}).get(name)
    if prop is None:
        return None, None
    return prop, ("preset" if name == "preset_id" else hf.field_kind(prop))


_MEDIA_LABELS = {
    "video_url": "Source video", "video_urls": "Reference videos", "image_url": "Image",
    "image_urls": "Reference images", "first_frame_url": "First frame", "last_frame_url": "Last frame",
    "end_image_url": "End frame", "last_image_url": "Last frame", "audio_url": "Audio",
    "audio_urls": "Reference audio", "file_url": "Document", "link_url": "Web page",
}
_GENERIC_INFO = {"write your prompt here", "text instructions for the generation or edit."}


def _label(name, prop):
    if name in _MEDIA_LABELS:
        return _MEDIA_LABELS[name]
    title = (prop.get("title") or "").strip()
    return title if title and title.lower() != name else name.replace("_", " ").capitalize()


def _info(name, prop):
    """Hint text only when it says more than the label."""
    desc = (prop.get("description") or "").strip()
    if not desc or desc.lower() in _GENERIC_INFO or desc.lower() == _label(name, prop).lower():
        return None
    return desc[:200]


def _family_choices():
    return [fam["name"] for fam in hf.CATALOG]


def _workflow_choices(family):
    for fam in hf.CATALOG:
        if fam["name"] == family:
            return [(wf["title"], wf["endpoint"]) for wf in fam["workflows"]]
    return []


def _search_text(endpoint):
    """What the model search matches: names, vendor, endpoint, notes and capability tags."""
    wf = hf.WORKFLOWS[endpoint]
    props = wf["schema"].get("properties", {})
    tags = [k for k in props]
    if hf.has_seed(endpoint):
        tags.append("seed")
    if any(k in props for k in ("video_url", "video_urls")):
        tags.append("video input")
    if any(k in props for k in ("image_url", "image_urls", "first_frame_url")):
        tags.append("image input")
    _, tiers = hf.quality_field(endpoint)
    tags += [str(t) for t in tiers]
    return " ".join([wf["family"], wf["vendor"], wf["title"], endpoint, *wf["notes"], *tags]).lower()


SEARCH_INDEX = {e: _search_text(e) for e in hf.WORKFLOWS}


ALL_FAMILIES = "All families"


def _short_label(endpoint):
    wf = hf.WORKFLOWS[endpoint]
    return f"{wf['family']} · {wf['title'].split(' — ')[-1]}"


def _word_match(word, tokens):
    # whole words, so "seed" doesn't match "Seedance"; longer words may be prefixes ("restyl" -> restyle)
    return word in tokens or (len(word) >= 5 and any(t.startswith(word) for t in tokens))


SEARCH_TOKENS = {e: set(re.split(r"[^a-z0-9.]+", t)) for e, t in SEARCH_INDEX.items()}


def search_workflows(query, family=ALL_FAMILIES):
    """Workflows matching every word of the query, optionally within one family."""
    words = query.lower().split()
    return [(_short_label(e), e) for e, tokens in SEARCH_TOKENS.items()
            if (family in (None, "", ALL_FAMILIES) or hf.WORKFLOWS[e]["family"] == family)
            and all(_word_match(w, tokens) for w in words)]


def _human_time(t):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else ""


def build_tabs(svc):
    store, worker, client = svc.gen_store, svc.gen_worker, svc.gen_client
    media_dir = os.path.join(svc.DATA_DIR, "generate", "media")
    password_set = bool(os.environ.get("UI_PASSWORD"))

    # ------------------------------------------------------------ helpers

    def workflow_info(endpoint):
        """Model card: what the workflow takes and produces."""
        if not endpoint:
            return "Pick a model in the list."
        wf = hf.WORKFLOWS[endpoint]
        props = wf["schema"].get("properties", {})
        required = set(wf["schema"].get("required", []))
        accepts = [f"{_label(k, p)}{' (required)' if k in required else ''}" for k, p in props.items()
                   if hf.field_kind(p) in ("uri", "uri_list")]
        qf, tiers = hf.quality_field(endpoint)
        dur = props.get("duration", {})
        dur_txt = (", ".join(map(str, dur["enum"])) + " s" if "enum" in dur else
                   f"{dur.get('minimum', '?')}–{dur.get('maximum', '?')} s" if dur else "follows the input")
        lines = [f"### {_short_label(endpoint)}",
                 f"{wf['vendor']} · `{endpoint}` · [API docs]({wf['doc']})", "",
                 f"- **Inputs:** {', '.join(accepts) if accepts else 'text only'}",
                 f"- **Quality tiers:** {', '.join(map(str, tiers)) if tiers else 'one tier'}",
                 f"- **Duration:** {dur_txt}",
                 f"- **Seed:** {'yes, so upscales can reuse it' if hf.has_seed(endpoint) else 'no, so reruns differ'}"]
        if wf["notes"]:
            lines += ["", f"<details><summary>Usage notes ({len(wf['notes'])})</summary>", "",
                      *[f"- {n}" for n in wf["notes"]], "", "</details>"]
        return "\n".join(lines)

    def field_updates(endpoint, values=None):
        """gr.update per form component for this workflow (defaults or `values`)."""
        props = hf.WORKFLOWS[endpoint]["schema"].get("properties", {}) if endpoint else {}
        required = set(hf.WORKFLOWS[endpoint]["schema"].get("required", [])) if endpoint else set()
        updates = []
        for name, kind in FIELDS:
            prop, wf_kind = _prop_kind(endpoint, name) if endpoint else (None, None)
            if prop is None or wf_kind != kind:
                updates.append(gr.update(visible=False))
                continue
            label = _label(name, prop) + (" *" if name in required else "")
            info = _info(name, prop)
            if values is not None and name in values:
                value = values[name]
            elif kind in ("uri", "uri_list"):
                value = []
            elif kind == "json":
                value = json.dumps(prop.get("default", []), indent=1) if "default" in prop else ""
            elif kind == "str_list":
                value = ", ".join(prop.get("default", []))
            else:
                value = prop.get("default")
            if kind in ("text",) and value is None:
                value = ""
            u = {"visible": True, "label": label, "info": info, "value": value}
            if kind == "enum":
                u["choices"] = prop["enum"]
            if name == "seed":
                u["info"] = "Blank = random, stored with the job"
            updates.append(gr.update(**u))
        return updates

    def pool_choices(pool):
        return [p["label"] for p in pool]

    def media_updates(endpoint, pool):
        """Refresh media pickers' choices after the pool changes."""
        out = []
        for name, kind in FIELDS:
            if kind in ("uri", "uri_list"):
                out.append(gr.update(choices=pool_choices(pool)))
            else:
                out.append(gr.update())
        return out

    def da3_choices():
        return [(f"{j.id} · {j.encoder} · {os.path.basename(j.input)}", j.id)
                for j in svc.jobs.list() if j.status == "done" and j.result]

    # ------------------------------------------------------------ spec building

    def collect(endpoint, pool, values):
        """Form values -> list of job specs (batch expanded)."""
        if not endpoint:
            raise gr.Error("Pick a model workflow.")
        wf = hf.WORKFLOWS[endpoint]
        props = wf["schema"].get("properties", {})
        required = set(wf["schema"].get("required", []))
        by_label = {p["label"]: p for p in pool}

        def media_item(v):
            if v in by_label:
                return {"label": v, "path": by_label[v]["path"], "da3_job": by_label[v].get("da3_job")}
            if str(v).startswith(("http://", "https://")):
                return {"label": v, "url": v}
            raise gr.Error(f"Unknown media '{v}': add it to the media list or paste a full https URL.")

        params, media_fields, batch_axes = {}, {}, []
        prompts = [None]
        for (name, kind), value in zip(FIELDS, values):
            prop, wf_kind = _prop_kind(endpoint, name)
            if prop is None or wf_kind != kind:
                continue
            if kind in ("uri", "uri_list"):
                items = [media_item(v) for v in (value or [])]
                if not items:
                    continue
                if kind == "uri" and len(items) > 1:
                    batch_axes.append((name, items))  # one job per item
                else:
                    media_fields[name] = items
                continue
            if name == "prompt":
                variants = [v.strip() for v in VARIANT_SPLIT.split(value or "") if v.strip()]
                prompts = variants or [None]
                continue
            if kind == "bool":
                params[name] = bool(value)
            elif kind == "number":
                if value is None or value == "":
                    continue
                params[name] = int(value) if prop.get("type") == "integer" else float(value)
            elif kind == "enum":
                if value is None or value == "":
                    continue
                match = [e for e in prop["enum"] if str(e) == str(value)]
                if not match:
                    raise gr.Error(f"{name}: '{value}' is not one of {prop['enum']}")
                params[name] = match[0]
            elif kind == "str_list":
                vals = [v.strip() for v in (value or "").split(",") if v.strip()]
                if vals:
                    params[name] = vals
            elif kind == "json":
                if (value or "").strip():
                    try:
                        params[name] = json.loads(value)
                    except ValueError as e:
                        raise gr.Error(f"{name}: invalid JSON ({e})")
            elif kind == "preset":
                if value:
                    m = re.search(r"\(([0-9a-f-]{36})\)$", str(value))
                    params[name] = m.group(1) if m else str(value).strip()
            elif value not in (None, ""):
                params[name] = value

        specs = []
        axis_names = [a[0] for a in batch_axes]
        for prompt, combo in itertools.product(prompts, itertools.product(*[a[1] for a in batch_axes])):
            p = dict(params)
            if prompt is not None:
                p["prompt"] = prompt
            m = dict(media_fields)
            for name, item in zip(axis_names, combo):
                m[name] = [item]
            missing = [r for r in required if r not in p and r not in m]
            if missing:
                raise gr.Error(f"Required: {', '.join(sorted(missing))}")
            if hf.has_seed(endpoint) and "seed" not in p:
                lo, hi_ = hf.seed_range(endpoint)
                p["seed"] = random.randint(lo, hi_)
            specs.append({"endpoint": endpoint, "params": p, "media": m})
        return specs

    def apply_draft(spec):
        qf, tiers = hf.quality_field(spec["endpoint"])
        if qf:
            spec["params"][qf] = tiers[0]
        return spec

    def spec_row(spec):
        qf, _ = hf.quality_field(spec["endpoint"])
        p = spec["params"]
        media_desc = "; ".join(f"{k}: {', '.join(i['label'] for i in v)}" for k, v in spec["media"].items())
        return dict(endpoint=spec["endpoint"], title=hf.workflow_label(spec["endpoint"]),
                    family=hf.WORKFLOWS[spec["endpoint"]]["family"], prompt=p.get("prompt", ""),
                    seed=p.get("seed"), quality=str(p.get(qf, "")) if qf else "", media_desc=media_desc)

    # ------------------------------------------------------------ handlers: generate

    def on_filter(query, family, current):
        hits = search_workflows(query or "", family)
        keep = current if current in [e for _, e in hits] else None
        return gr.update(choices=hits, value=keep), f"{len(hits)} of {len(hf.WORKFLOWS)} workflows"

    def on_workflow(endpoint):
        return [workflow_info(endpoint), *field_updates(endpoint),
                gr.update(visible=bool(endpoint and "preset_id" in hf.WORKFLOWS[endpoint]["schema"]["properties"]))]

    def add_da3(job_ids, kinds, pool, progress=gr.Progress()):
        pool = list(pool)
        if not job_ids or not kinds:
            raise gr.Error("Pick DA3 jobs and what to add.")
        for jid in job_ids:
            job = svc.jobs.get(jid)
            if job is None or job.status != "done":
                raise gr.Error(f"DA3 job {jid} is not finished (DA3 jobs are kept in memory until restart).")
            for kind in kinds:
                progress(None, desc=f"{jid}: {kind}")
                try:
                    if kind == "depth mp4":
                        path = media.depth_video(job, media_dir, job.fps)
                    elif kind == "plate mp4":
                        path = media.plate_video(job, media_dir, job.fps)
                    else:
                        path = (job.result or {}).get("preview")
                        if not path:
                            raise gr.Error(f"{jid} has no preview mp4")
                except Exception as e:
                    if isinstance(e, gr.Error):
                        raise
                    raise gr.Error(f"{jid} {kind}: {e}")
                label = f"{len(pool) + 1}. {jid} {kind} ({os.path.basename(job.input)})"
                pool.append({"label": label, "path": path, "da3_job": jid})
        return pool, pool_table(pool)

    def add_uploads(files, pool):
        pool = list(pool)
        os.makedirs(os.path.join(media_dir, "uploads"), exist_ok=True)
        for f in files or []:
            dest = os.path.join(media_dir, "uploads", f"{int(time.time() * 1000)}_{os.path.basename(f)}")
            shutil.copy(f, dest)
            pool.append({"label": f"{len(pool) + 1}. {os.path.basename(f)}", "path": dest})
        return pool, pool_table(pool), None

    def pool_table(pool):
        """Gallery items (videos and images); audio is listed by name only."""
        return [(p["path"], p["label"]) for p in pool
                if os.path.splitext(p["path"])[1].lower() in (".mp4", ".mov", ".png", ".jpg", ".jpeg", ".webp", ".gif")]

    def clear_pool():
        return [], []

    def load_presets(endpoint):
        try:
            items = client.presets(endpoint)
        except hf.HiggsfieldError as e:
            raise gr.Error(str(e))
        choices = [f"{it.get('name') or it.get('title') or 'preset'} ({it['id']})" for it in items if it.get("id")]
        idx = [i for i, (n, k) in enumerate(FIELDS) if n == "preset_id"][0]
        out = [gr.update() for _ in FIELDS]
        out[idx] = gr.update(choices=choices, value=choices[0] if choices else None)
        return out

    def estimate(endpoint, pool, draft, *values):
        specs = collect(endpoint, pool, values)
        if draft:
            specs = [apply_draft(s) for s in specs]
        if not client.configured:
            raise gr.Error("No Higgsfield API key yet: set HIGGSFIELD_KEY_ID / HIGGSFIELD_KEY_SECRET in .env "
                           "and restart.")

        def one(spec):
            job = {"endpoint": spec["endpoint"], "params": spec["params"], "media": spec["media"]}
            try:
                return client.estimate(spec["endpoint"], worker.build_arguments(job))
            except hf.HiggsfieldError as e:
                return {"error": str(e)}

        with ThreadPoolExecutor(4) as ex:
            results = list(ex.map(one, specs))
        total = sum(float(r.get("usd", 0)) for r in results if "usd" in r)
        lines = [f"**{len(specs)} job(s), estimated ${total:.3f}**", "",
                 "| # | prompt | quality | seed | media | estimate |", "|---|---|---|---|---|---|"]
        for i, (s, r) in enumerate(zip(specs, results), 1):
            row = spec_row(s)
            est = f"${float(r['usd']):.3f} ({r.get('credits')} cr)" if "usd" in r else f"error: {r.get('error')}"
            lines.append(f"| {i} | {row['prompt'][:60]} | {row['quality']} | {row['seed'] or ''} | "
                         f"{row['media_desc'][:80]} | {est} |")
        return "\n".join(lines)

    def submit(endpoint, pool, draft, batch_name, *values):
        if client.configured and not password_set:
            raise gr.Error("Set UI_PASSWORD in .env before submitting: without it anyone on the LAN "
                           "can spend Higgsfield credits.")
        specs = collect(endpoint, pool, values)
        if draft:
            specs = [apply_draft(s) for s in specs]
        batch_id = None
        if len(specs) > 1:
            batch_id = store.new_batch(batch_name.strip() or
                                       f"{_human_time(time.time())} · {hf.workflow_label(endpoint)} · {len(specs)} jobs")
        ids = []
        for s in specs:
            row = spec_row(s)
            ids.append(store.add_job(batch_id=batch_id, kind="draft" if draft else "full", endpoint=s["endpoint"],
                                     title=row["title"], family=row["family"], prompt=row["prompt"],
                                     seed=row["seed"], quality=row["quality"], params=s["params"],
                                     media=s["media"]))
        note = "" if client.configured else " They will wait as **pending** until an API key is set."
        return (f"Queued {len(ids)} job(s){f' in batch `{batch_id}`' if batch_id else ''}: "
                f"{', '.join(ids)}.{note} See the History tab.")

    # ------------------------------------------------------------ handlers: history

    def history_rows(search, status_filter):
        where, args = ["1=1"], []
        if status_filter == "active":
            where.append("status IN ('pending','submitting','queued','in_progress')")
        elif status_filter == "completed":
            where.append("status='completed'")
        elif status_filter == "failed / error":
            where.append("status IN ('failed','nsfw','error')")
        elif status_filter == "canceled":
            where.append("status='canceled'")
        if search:
            where.append("(prompt LIKE ? OR title LIKE ? OR id LIKE ? OR batch_id LIKE ? OR endpoint LIKE ?)")
            args += [f"%{search}%"] * 5
        names = store.batch_names()
        rows = []
        for j in store.query(" AND ".join(where), args):
            rows.append([j["id"], _human_time(j["created"]),
                         names.get(j["batch_id"], "") if j["batch_id"] else "", j["kind"], j["title"],
                         j["quality"] or "", j["seed"] or "", j["status"],
                         f"{j['est_usd']:.3f}" if j["est_usd"] is not None else "",
                         (j["prompt"] or "")[:80]])
        return rows

    def upscale_targets(job):
        if not job:
            return []
        out = []
        qf, tiers = hf.quality_field(job["endpoint"])
        cur = job["params"].get(qf) if qf else None
        for t in tiers:
            if hf.tier_rank(t) > hf.tier_rank(cur):
                out.append((f"{qf} = {t}", f"{job['endpoint']}|{qf}|{t}"))
        for sib in hf.sibling_tiers(job["endpoint"]):
            if hf.endpoint_tier(sib) > hf.endpoint_tier(job["endpoint"]):
                out.append((f"switch to {hf.workflow_label(sib)}", f"{sib}||"))
        return out

    def job_detail(job_id):
        job = store.get(job_id or "")
        if not job:
            return ("Click a job in the table to see its details, output and actions.", None, None,
                    gr.update(value=None, visible=False), gr.update(choices=[], value=None))
        lines = [f"**{job['id']}** · {job['title']} · **{job['status']}**",
                 f"kind: {job['kind']}" + (f" (from `{job['parent_id']}`)" if job["parent_id"] else ""),
                 f"created {_human_time(job['created'])}" +
                 (f", submitted {_human_time(job['submitted'])}" if job["submitted"] else "") +
                 (f", finished {_human_time(job['finished'])}" if job["finished"] else ""),
                 f"seed: {job['seed'] if job['seed'] is not None else 'none (model has no seed)'} · "
                 f"quality: {job['quality'] or 'n/a'} · estimate: "
                 + (f"${job['est_usd']:.3f} ({job['est_credits']} credits)" if job["est_usd"] is not None else "n/a"),
                 f"request id: `{job['request_id'] or '-'}`"]
        if job["error"]:
            lines.append(f"error: `{job['error']}`")
        lines += ["", "**Prompt**", "", job["prompt"] or "_(none)_"]
        record = {k: job[k] for k in ("id", "batch_id", "parent_id", "kind", "endpoint", "prompt", "seed",
                                      "quality", "params", "media", "sent_args", "status", "error", "request_id",
                                      "correlation_id", "est_credits", "est_usd", "created", "submitted",
                                      "finished", "output_url", "output_path")}
        out = job["output_path"] if job["output_path"] and os.path.exists(job["output_path"]) else None
        targets = upscale_targets(job)
        return ("\n\n".join(lines[:5]) + "\n\n" + "\n".join(lines[5:]), record,
                out if out and out.endswith(".mp4") else None, gr.update(value=out, visible=bool(out)),
                gr.update(choices=targets, value=targets[0][1] if targets else None))

    def clone_job(job, kind, endpoint=None, overrides=None, batch_id=None):
        endpoint = endpoint or job["endpoint"]
        props = hf.WORKFLOWS[endpoint]["schema"].get("properties", {})
        params = {k: v for k, v in job["params"].items() if k in props}
        params.update(overrides or {})
        media_fields = {k: v for k, v in (job["media"] or {}).items() if k in props}
        qf, _ = hf.quality_field(endpoint)
        return store.add_job(batch_id=batch_id or job["batch_id"], parent_id=job["id"], kind=kind,
                             endpoint=endpoint, title=hf.workflow_label(endpoint),
                             family=hf.WORKFLOWS[endpoint]["family"], prompt=job["prompt"],
                             seed=params.get("seed"), quality=str(params.get(qf, "")) if qf else "",
                             params=params, media=media_fields)

    def _guard():
        if client.configured and not password_set:
            raise gr.Error("Set UI_PASSWORD in .env before submitting.")

    def resubmit(job_id):
        _guard()
        job = store.get(job_id or "")
        if not job:
            raise gr.Error("Pick a job first.")
        return f"Resubmitted as `{clone_job(job, 'resubmit')}` (same model, prompt, seed and inputs)."

    def _upscale(job, target):
        endpoint, qf, value = target.split("|")
        return clone_job(job, "upscale", endpoint=endpoint, overrides={qf: value} if qf else None)

    def upscale(job_id, target):
        _guard()
        job = store.get(job_id or "")
        if not job or not target:
            raise gr.Error("Pick a job and an upscale target.")
        return f"Upscale queued as `{_upscale(job, target)}`."

    def add_pick(job_id, picks):
        picks = [p for p in (picks or "").split() if p]
        if job_id and job_id not in picks:
            picks.append(job_id)
        return " ".join(picks)

    def upscale_picks(picks, quality_value):
        _guard()
        ids = [p for p in (picks or "").split() if p]
        if not ids:
            raise gr.Error("Add jobs to the picks first.")
        done = []
        batch_id = store.new_batch(f"{_human_time(time.time())} · upscale picks · {len(ids)} jobs") \
            if len(ids) > 1 else None
        for jid in ids:
            job = store.get(jid)
            if not job:
                continue
            targets = [v for _, v in upscale_targets(job)]
            if quality_value == "highest":
                same = [t for t in targets if t.startswith(job["endpoint"] + "|")]
                other = [t for t in targets if not t.startswith(job["endpoint"] + "|")]
                target = same[-1] if same else (other[-1] if other else None)
            else:
                target = next((t for t in targets if t.endswith("|" + quality_value)), None)
            if target is None:
                continue
            new = clone_job(job, "upscale", endpoint=target.split("|")[0],
                            overrides={target.split("|")[1]: target.split("|")[2]} if target.split("|")[1] else None,
                            batch_id=batch_id)
            done.append(new)
        skipped = len(ids) - len(done)
        return (f"Queued {len(done)} upscale(s): {', '.join(done)}."
                + (f" {skipped} skipped (no such tier on that model)." if skipped else "")), ""

    def cancel(job_id):
        return worker.cancel(job_id or "")

    def duplicate(job_id):
        job = store.get(job_id or "")
        if not job:
            raise gr.Error("Pick a job first.")
        endpoint = job["endpoint"]
        family = hf.WORKFLOWS[endpoint]["family"]
        pool, values = [], {}
        for field, items in (job["media"] or {}).items():
            labels = []
            for it in items:
                if it.get("url"):
                    labels.append(it["url"])
                    continue
                label = f"{len(pool) + 1}. {os.path.basename(it['path'])}"
                pool.append({"label": label, "path": it["path"], "da3_job": it.get("da3_job")})
                labels.append(label)
            values[field] = labels
        for k, v in job["params"].items():
            prop, kind = _prop_kind(endpoint, k)
            if kind == "json":
                v = json.dumps(v, indent=1)
            elif kind == "str_list":
                v = ", ".join(v)
            values[k] = v
        updates = field_updates(endpoint, values)
        for i, (name, kind) in enumerate(FIELDS):
            if kind in ("uri", "uri_list") and updates[i]["visible"]:
                updates[i]["choices"] = pool_choices(pool)
        return [gr.update(value=ALL_FAMILIES), gr.update(choices=search_workflows(""), value=endpoint),
                workflow_info(endpoint), pool, pool_table(pool), *updates,
                gr.update(visible="preset_id" in hf.WORKFLOWS[endpoint]["schema"]["properties"]),
                gr.Tabs(selected="generate")]

    # ------------------------------------------------------------ handlers: funds

    def funds():
        by_day, by_model, inflight, totals = store.spend()
        key = "configured" if client.configured else "**not set** (add HIGGSFIELD_KEY_ID / HIGGSFIELD_KEY_SECRET)"
        md = (f"**Estimated spend on completed jobs: ${totals['usd'] or 0:.2f}** "
              f"({totals['n']} jobs, {totals['credits'] or 0:.1f} credits). "
              f"Queued or running: {inflight['n']} jobs, about ${inflight['usd'] or 0:.2f}.\n\n"
              f"Costs are Higgsfield's per-request estimates; failed, moderated and canceled jobs are refunded "
              f"and not counted. The real balance is only shown in the "
              f"[Higgsfield console]({hf.CONSOLE_URL}).\n\n"
              f"API key: {key} · max concurrent submissions: {worker.max_concurrent} · "
              f"UI password: {'set' if password_set else '**not set**'}")
        day_rows = [[d["day"], d["n"], f"{d['usd'] or 0:.2f}", f"{d['credits'] or 0:.1f}"] for d in by_day]
        model_rows = [[m["title"], m["n"], f"{m['usd'] or 0:.2f}", f"{m['credits'] or 0:.1f}"] for m in by_model]
        return md, day_rows, model_rows

    # ------------------------------------------------------------ layout

    media_kinds = ("uri", "uri_list")
    prompt_names = ("prompt", "negative_prompt", "multi_prompt")
    common_names = ("resolution", "mode", "duration", "aspect_ratio", "seed", "preset_id")
    common_bools = ("generate_audio", "sound")
    comps = [None] * len(FIELDS)

    def make(i):
        name, kind = FIELDS[i]
        if kind == "enum":
            # choices are set per workflow; values are checked against the schema in collect()
            c = gr.Dropdown(label=name, allow_custom_value=True, visible=False, min_width=140)
        elif kind == "number":
            c = gr.Number(label=name, visible=False, min_width=140)
        elif kind == "bool":
            c = gr.Checkbox(label=name, visible=False)
        elif kind in media_kinds:
            c = gr.Dropdown(label=name, multiselect=True, allow_custom_value=True, visible=False)
        elif kind == "json":
            c = gr.Code(label=name, language="json", visible=False)
        elif kind == "preset":
            c = gr.Dropdown(label=name, allow_custom_value=True, visible=False, min_width=220)
        elif name in ("prompt", "negative_prompt"):
            c = gr.Textbox(label=name, lines=6 if name == "prompt" else 2, visible=False,
                           placeholder="Describe the shot. Separate variants with a line containing only ---"
                           if name == "prompt" else None)
        else:
            c = gr.Textbox(label=name, visible=False)
        comps[i] = c

    def group(pred):
        return [i for i, (n, k) in enumerate(FIELDS) if pred(n, k)]

    media_idx = group(lambda n, k: k in media_kinds)
    prompt_idx = group(lambda n, k: n in prompt_names and k not in media_kinds)
    common_idx = sorted(group(lambda n, k: n in common_names and k not in media_kinds),
                        key=lambda i: common_names.index(FIELDS[i][0]))
    bool_idx = group(lambda n, k: n in common_bools)
    adv_idx = [i for i in range(len(FIELDS)) if i not in set(media_idx + prompt_idx + common_idx + bool_idx)]

    with gr.Tab("Generate", id="generate"):
        status = []
        status.append("API key **set**" if client.configured else
                      "**No API key**: jobs wait as *pending* until `HIGGSFIELD_KEY_ID` / `_SECRET` are in `.env`")
        status.append("password **on**" if password_set else "**no UI password**")
        gr.Markdown("Send DA3 depth or plates to Higgsfield video models. Depth goes in as a **reference "
                    "video**, so models may not follow its layout exactly. · " + " · ".join(status),
                    elem_classes="status-line")
        pool = gr.State([])
        with gr.Row(equal_height=False):
            # ---- left: model + media library
            with gr.Column(scale=4, min_width=340):
                gr.Markdown("### 1 · Model")
                model_search = gr.Textbox(show_label=False, placeholder="Search: seed · video input · kling 4k · "
                                                                        "motion · restyle · reference")
                with gr.Row():
                    family = gr.Dropdown([ALL_FAMILIES, *_family_choices()], value=ALL_FAMILIES,
                                         show_label=False, scale=3)
                    search_hits = gr.Markdown(f"{len(hf.WORKFLOWS)} of {len(hf.WORKFLOWS)} workflows",
                                              elem_classes="hits")
                workflow = gr.Radio(choices=search_workflows(""), value=DEFAULT_WORKFLOW, show_label=False,
                                    elem_classes="model-list")
                info = gr.Markdown(elem_classes="card")

                gr.Markdown("### 2 · Media library")
                with gr.Tabs():
                    with gr.Tab("From DA3 jobs"):
                        da3_pick = gr.Dropdown(choices=da3_choices(), multiselect=True, allow_custom_value=True,
                                               label="Finished DA3 jobs")
                        da3_kinds = gr.CheckboxGroup(MEDIA_KINDS, value=["depth mp4"], label="Add as")
                        with gr.Row():
                            refresh_da3 = gr.Button("Refresh", size="sm")
                            add_da3_btn = gr.Button("Add to library", size="sm", variant="primary")
                    with gr.Tab("Upload"):
                        uploads = gr.File(file_count="multiple", type="filepath",
                                          label="Images, mp4 or audio")
                        add_up_btn = gr.Button("Add to library", size="sm", variant="primary")
                pool_view = gr.Gallery(label="Library · add clips above, then pick them in step 3",
                                       columns=4, height=150, allow_preview=True, object_fit="contain")
                clear_btn = gr.Button("Clear library", size="sm")

            # ---- right: inputs, prompt, settings, submit
            with gr.Column(scale=6, min_width=420):
                gr.Markdown("### 3 · Inputs\nPick items from the library. Several picks in a one-file slot "
                            "make a batch (one job each).")
                for i in media_idx:
                    make(i)
                gr.Markdown("### 4 · Prompt")
                for i in prompt_idx:
                    make(i)
                gr.Markdown("### 5 · Settings")
                with gr.Row():
                    for i in common_idx:
                        make(i)
                with gr.Row():
                    for i in bool_idx:
                        make(i)
                presets_btn = gr.Button("Load style presets", size="sm", visible=False)
                with gr.Accordion("Advanced settings", open=False):
                    for i in adv_idx:
                        make(i)
                gr.Markdown("### 6 · Review and submit")
                with gr.Row():
                    draft = gr.Checkbox(value=True, label="Draft at the lowest quality tier",
                                        info="Cheap first pass; upscale the good ones from History")
                    batch_name = gr.Textbox(label="Batch name", placeholder="optional", scale=2)
                with gr.Row():
                    est_btn = gr.Button("Estimate cost")
                    submit_btn = gr.Button("Submit", variant="primary")
                gen_out = gr.Markdown("Estimate first to see the cost per job, then submit. "
                                      "Progress and results appear in **History**.", elem_classes="card")

    with gr.Tab("History", id="history"):
        with gr.Row():
            search = gr.Textbox(show_label=False, placeholder="Search prompt, model, job or batch id", scale=4)
            status_filter = gr.Dropdown(STATUS_FILTERS, value="all", show_label=False, scale=1, min_width=140)
            sel = gr.Textbox(show_label=False, placeholder="selected job id", scale=1, min_width=140)
        hist = gr.Dataframe(headers=HISTORY_COLUMNS, interactive=False, wrap=True, max_height=340,
                            label="Jobs · click a row to open it")
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                detail = gr.Markdown("Click a job in the table to see its details, output and actions.",
                                     elem_classes="card")
                out_video = gr.Video(label="Output", interactive=False, height=320)
            with gr.Column(scale=2, min_width=300):
                with gr.Group():
                    gr.Markdown("**Run again**", elem_classes="group-title")
                    with gr.Row():
                        resub_btn = gr.Button("Resubmit as is", size="sm")
                        dup_btn = gr.Button("Edit as new job", size="sm")
                with gr.Group():
                    gr.Markdown("**Upscale this job**", elem_classes="group-title")
                    target = gr.Dropdown(show_label=False, choices=[])
                    up_btn = gr.Button("Upscale", variant="primary", size="sm")
                with gr.Group():
                    gr.Markdown("**Picks** · collect good drafts, upscale together", elem_classes="group-title")
                    picks = gr.Textbox(show_label=False, placeholder="picked job ids")
                    pick_btn = gr.Button("Add selected job", size="sm")
                    pick_quality = gr.Dropdown(["highest", "720p", "1080p", "pro", "4k"], value="highest",
                                               label="Upscale picks to")
                    pick_up_btn = gr.Button("Upscale picks", variant="primary", size="sm")
                with gr.Row():
                    out_file = gr.DownloadButton("Download output", visible=False, size="sm")
                    cancel_btn = gr.Button("Cancel job", variant="stop", size="sm")
                action_out = gr.Markdown()
        with gr.Accordion("Full record (JSON)", open=False):
            record = gr.JSON(show_label=False)
        shown = gr.State("")

    with gr.Tab("Funds", id="funds"):
        funds_md = gr.Markdown(elem_classes="card")
        with gr.Row():
            by_day = gr.Dataframe(headers=["day", "jobs", "est USD", "credits"], interactive=False,
                                  label="Completed, by day")
            by_model = gr.Dataframe(headers=["model", "jobs", "est USD", "credits"], interactive=False,
                                    label="Completed, by model")
        funds_btn = gr.Button("Refresh", size="sm")

    # ------------------------------------------------------------ events

    form_outputs = [info, *comps, presets_btn]
    model_search.input(on_filter, [model_search, family, workflow], [workflow, search_hits])
    family.input(on_filter, [model_search, family, workflow], [workflow, search_hits])
    workflow.input(on_workflow, workflow, form_outputs)
    refresh_da3.click(lambda: gr.update(choices=da3_choices()), None, da3_pick)
    da3_pick.focus(lambda: gr.update(choices=da3_choices()), None, da3_pick)
    add_da3_btn.click(add_da3, [da3_pick, da3_kinds, pool], [pool, pool_view]) \
        .then(media_updates, [workflow, pool], comps)
    add_up_btn.click(add_uploads, [uploads, pool], [pool, pool_view, uploads]) \
        .then(media_updates, [workflow, pool], comps)
    clear_btn.click(clear_pool, None, [pool, pool_view]).then(media_updates, [workflow, pool], comps)
    presets_btn.click(load_presets, workflow, comps)
    est_btn.click(estimate, [workflow, pool, draft, *comps], gen_out)
    submit_btn.click(submit, [workflow, pool, draft, batch_name, *comps], gen_out)

    def refresh_history(search_v, status_v, job_id, shown_key):
        job = store.get(job_id or "")
        key = f"{job_id}:{job['status'] if job else ''}:{job['output_path'] if job else ''}"
        rows = history_rows(search_v, status_v)
        if key == shown_key:
            return rows, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), shown_key
        return (rows, *job_detail(job_id), key)

    hist_outputs = [hist, detail, record, out_video, out_file, target, shown]
    gr.Timer(3).tick(refresh_history, [search, status_filter, sel, shown], hist_outputs, show_progress="hidden")
    for ev in (search.submit, status_filter.change, sel.change):
        ev(refresh_history, [search, status_filter, sel, shown], hist_outputs, show_progress="hidden")
    hist.select(_select_first, None, sel)
    resub_btn.click(resubmit, sel, action_out)
    up_btn.click(upscale, [sel, target], action_out)
    cancel_btn.click(cancel, sel, action_out)
    pick_btn.click(add_pick, [sel, picks], picks)
    pick_up_btn.click(upscale_picks, [picks, pick_quality], [action_out, picks])
    funds_btn.click(funds, None, [funds_md, by_day, by_model])
    gr.Timer(15).tick(funds, None, [funds_md, by_day, by_model], show_progress="hidden")
    return {"load_form": on_workflow, "workflow": workflow, "form_outputs": form_outputs,
            "dup_btn": dup_btn, "duplicate": duplicate, "sel": sel,
            "dup_outputs": [family, workflow, info, pool, pool_view, *comps, presets_btn],
            "funds": funds, "funds_outputs": [funds_md, by_day, by_model]}


def _select_first(evt: gr.SelectData):
    return evt.row_value[0]
