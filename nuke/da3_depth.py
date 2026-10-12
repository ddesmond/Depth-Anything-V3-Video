"""DA3 Depth node for Nuke.

Sends the upstream Read node's file to the Depth Anything V3 service, polls the
job in a background thread and creates a Read node on the resulting EXR depth
sequence (depth.Z, half float).

Only uses the standard library, so it runs in Nuke's bundled Python 3.
"""
import json
import threading
import time
import urllib.error
import urllib.request

import nuke

DEFAULT_SERVER = "http://localhost:8000"
ENCODERS = ["da3-base", "da3-small", "da3-large"]
COMPRESSIONS = ["piz", "zip16"]
POLL_SECONDS = 2.0


# ------------------------------------------------------------------ http

def _request(method, url, payload=None, timeout=10):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode()).get("detail")
        except Exception:
            detail = e.reason
        raise RuntimeError("HTTP %s: %s" % (e.code, detail))
    except urllib.error.URLError as e:
        raise RuntimeError("cannot reach %s (%s)" % (url, e.reason))


# ------------------------------------------------------------------ node

def create():
    """Create a DA3Depth node connected to the selected node."""
    node = nuke.createNode("NoOp")
    node.setName("DA3Depth")
    node["tile_color"].setValue(0x2E6E9EFF)

    tab = nuke.Tab_Knob("da3", "DA3 Depth")
    node.addKnob(tab)
    node.addKnob(nuke.String_Knob("server", "server"))
    node["server"].setValue(DEFAULT_SERVER)
    node.addKnob(nuke.Enumeration_Knob("encoder", "model", ENCODERS))
    res = nuke.Int_Knob("process_res", "process res")
    res.setValue(504)
    res.setTooltip("DA3 processing resolution (longest side). Higher = sharper, slower.")
    node.addKnob(res)
    batch = nuke.Int_Knob("batch_size", "batch size")
    batch.setValue(1)
    batch.setTooltip("Frames per inference call. >1 gives more consistent depth scale "
                     "across frames but needs more VRAM.")
    node.addKnob(batch)
    node.addKnob(nuke.Enumeration_Knob("compression", "EXR compression", COMPRESSIONS))
    preview = nuke.Boolean_Knob("preview", "write preview mp4")
    preview.setValue(True)
    preview.setFlag(nuke.STARTLINE)
    node.addKnob(preview)

    use_range = nuke.Boolean_Knob("use_range", "custom frame range")
    use_range.setFlag(nuke.STARTLINE)
    node.addKnob(use_range)
    node.addKnob(nuke.Int_Knob("first_frame", "first"))
    last = nuke.Int_Knob("last_frame", "last")
    last.clearFlag(nuke.STARTLINE)
    node.addKnob(last)
    out = nuke.File_Knob("output_dir", "output dir")
    out.setTooltip("Empty = <input dir>/da3_depth")
    node.addKnob(out)

    node.addKnob(nuke.Text_Knob("div1", ""))
    submit_btn = nuke.PyScript_Knob("submit", "Submit", "import da3_depth; da3_depth.submit(nuke.thisNode())")
    submit_btn.setFlag(nuke.STARTLINE)
    node.addKnob(submit_btn)
    node.addKnob(nuke.PyScript_Knob("cancel", "Cancel", "import da3_depth; da3_depth.cancel(nuke.thisNode())"))
    node.addKnob(nuke.String_Knob("job_id", "job id"))
    node["job_id"].setEnabled(False)
    node.addKnob(nuke.Text_Knob("status", "status", "idle"))
    return node


def _upstream_read(node):
    seen = set()
    stack = [node.input(0)] if node.input(0) else []
    while stack:
        n = stack.pop()
        if n is None or n.name() in seen:
            continue
        seen.add(n.name())
        if n.Class() == "Read":
            return n
        stack.extend(n.input(i) for i in range(n.inputs()))
    return None


def _set_status(node, text):
    def apply():
        try:
            node["status"].setValue(text)
        except ValueError:
            pass  # node was deleted
    nuke.executeInMainThread(apply)


def submit(node):
    read = _upstream_read(node)
    if read is None:
        nuke.message("DA3Depth: connect a Read node upstream.")
        return

    server = node["server"].value().rstrip("/")
    if node["use_range"].value():
        first, last = int(node["first_frame"].value()), int(node["last_frame"].value())
    else:
        first, last = int(read["first"].value()), int(read["last"].value())

    payload = {
        "input": nuke.filename(read),
        "encoder": node["encoder"].value(),
        "first": first,
        "last": last,
        "process_res": int(node["process_res"].value()),
        "batch_size": max(1, int(node["batch_size"].value())),
        "compression": node["compression"].value(),
        "preview": bool(node["preview"].value()),
    }
    if node["output_dir"].value():
        payload["output_dir"] = node["output_dir"].value()

    try:
        job = _request("POST", server + "/jobs", payload)
    except RuntimeError as e:
        nuke.message("DA3Depth: %s" % e)
        return

    node["job_id"].setValue(job["id"])
    node["status"].setValue("queued (#%s)" % job.get("queue_position"))
    threading.Thread(target=_poll, args=(node, server, job["id"]), daemon=True).start()


def cancel(node):
    job_id = node["job_id"].value()
    if not job_id:
        return
    try:
        _request("DELETE", "%s/jobs/%s" % (node["server"].value().rstrip("/"), job_id))
    except RuntimeError as e:
        nuke.message("DA3Depth: %s" % e)


def _poll(node, server, job_id):
    while True:
        time.sleep(POLL_SECONDS)
        try:
            job = _request("GET", "%s/jobs/%s" % (server, job_id))
        except RuntimeError as e:
            _set_status(node, "poll error: %s" % e)
            continue
        status = job["status"]
        if status == "queued":
            _set_status(node, "queued (#%s)" % job.get("queue_position"))
        elif status == "running":
            _set_status(node, "running %d/%d" % (job["done"], job["total"]))
        elif status == "done":
            _set_status(node, "done: %d frames" % job["result"]["frames_written"])
            nuke.executeInMainThread(_create_read, (node, job))
            return
        else:
            _set_status(node, "%s %s" % (status, job.get("error") or ""))
            return


def _create_read(node, job):
    res = job["result"]
    read = nuke.nodes.Read(file=res["exr_pattern"], first=res["first"], last=res["last"],
                           origfirst=res["first"], origlast=res["last"])
    read["raw"].setValue(True)
    read["on_error"].setValue("nearest frame")
    read["label"].setValue("DA3 %s" % job["encoder"])
    read.setXYpos(node.xpos() + 150, node.ypos())
