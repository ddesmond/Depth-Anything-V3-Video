"""Frame sources (video / image sequence / single image), DA3 inference and
EXR + preview writers."""
import glob
import os
import re

import cv2
import matplotlib
import numpy as np
import OpenEXR

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mxf"}
IMAGE_EXTS = {".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dpx"}

EXR_COMPRESSION = {
    "piz": OpenEXR.PIZ_COMPRESSION,
    "zip16": OpenEXR.ZIP_COMPRESSION,  # 16 scanlines per block
    "zip": OpenEXR.ZIPS_COMPRESSION,  # 1 scanline per block
    "none": OpenEXR.NO_COMPRESSION,
}

# '####', '%04d' or '@@@@' frame tokens, as used by Nuke Read nodes
_SEQ_TOKEN = re.compile(r"(#+|%0?(\d*)d|@+)")
_TRAILING_DIGITS = re.compile(r"^(.*?)(\d+)(\.[^.]+)$")


class InputError(ValueError):
    pass


# ---------------------------------------------------------------- sources

def split_sequence(path):
    """Return (prefix, padding, suffix) if path contains a frame token, else None."""
    name = os.path.basename(path)
    matches = list(_SEQ_TOKEN.finditer(name))
    if not matches:
        return None
    m = matches[-1]
    token = m.group(1)
    if token.startswith("%"):
        padding = int(m.group(2) or 1)
    else:
        padding = len(token)
    directory = os.path.dirname(path)
    return os.path.join(directory, name[: m.start()]), padding, name[m.end():]


def sequence_from_file(path):
    """`shot.1001.exr` -> `shot.####.exr` if sibling frames exist."""
    m = _TRAILING_DIGITS.match(os.path.basename(path))
    if not m:
        return None
    prefix, digits, suffix = m.groups()
    pattern = os.path.join(os.path.dirname(path), f"{prefix}{'#' * len(digits)}{suffix}")
    return pattern if len(list_sequence(pattern)) > 1 else None


def sequence_from_dir(directory):
    """Pick the image sequence inside a directory (used for uploaded zips)."""
    files = sorted(
        os.path.join(root, f)
        for root, _, names in os.walk(directory)
        for f in names
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    if not files:
        raise InputError("no images found in upload")
    return sequence_from_file(files[0]) or files[0]


def list_sequence(pattern, first=None, last=None):
    prefix, padding, suffix = split_sequence(pattern)
    rx = re.compile(re.escape(os.path.basename(prefix)) + r"(-?\d+)" + re.escape(suffix) + "$")
    frames = []
    for f in glob.glob(glob.escape(prefix) + "*" + glob.escape(suffix)):
        m = rx.match(os.path.basename(f))
        if not m:
            continue
        n = int(m.group(1))
        if (first is None or n >= first) and (last is None or n <= last):
            frames.append((n, f))
    return sorted(frames), padding


def _srgb_encode(lin):
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def read_exr_rgb(path):
    """Read an EXR as 8-bit sRGB (scene-linear assumed)."""
    with OpenEXR.File(path) as f:
        channels = f.channels()
        for key in ("RGBA", "RGB"):
            if key in channels:
                rgb = channels[key].pixels[..., :3]
                break
        else:
            names = {n.split(".")[-1]: n for n in channels}
            if all(c in names for c in "RGB"):
                rgb = np.stack([channels[names[c]].pixels for c in "RGB"], axis=-1)
            else:
                ch = next(iter(channels.values())).pixels
                rgb = np.repeat(ch[..., None], 3, axis=-1) if ch.ndim == 2 else ch[..., :3]
    return (_srgb_encode(rgb.astype(np.float32)) * 255 + 0.5).astype(np.uint8)


def read_image_rgb(path):
    if path.lower().endswith(".exr"):
        return read_exr_rgb(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise InputError(f"cannot read image: {path}")
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class Source:
    """Iterates (frame_number, rgb_uint8) over a video, sequence or single image.

    Frame numbers are 1-based for videos and single images (matching a Nuke
    Read node), and the real file numbers for sequences."""

    def __init__(self, path, first=None, last=None):
        self.path = path
        self.fps = None
        self.padding = 4
        ext = os.path.splitext(path)[1].lower()

        if split_sequence(path):
            self.kind = "sequence"
            self.frames, self.padding = list_sequence(path, first, last)
            if not self.frames:
                raise InputError(f"no frames match {path} in range {first}-{last}")
            base = os.path.basename(split_sequence(path)[0]).rstrip("._-")
        elif not os.path.isfile(path):
            raise InputError(f"input not found: {path}")
        elif ext in VIDEO_EXTS:
            self.kind = "video"
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise InputError(f"cannot open video: {path}")
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) or None
            cap.release()
            lo, hi = first or 1, min(last or count, count)
            if lo > hi:
                raise InputError(f"frame range {lo}-{hi} outside video (1-{count})")
            self.frames = [(n, None) for n in range(lo, hi + 1)]
            base = os.path.splitext(os.path.basename(path))[0]
        elif ext in IMAGE_EXTS:
            self.kind = "image"
            self.frames = [(first or 1, path)]
            base = os.path.splitext(os.path.basename(path))[0]
        else:
            raise InputError(f"unsupported input type: {ext}")

        self.basename = base or "frames"

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        if self.kind != "video":
            for n, f in self.frames:
                yield n, read_image_rgb(f)
            return
        cap = cv2.VideoCapture(self.path)
        try:
            first = self.frames[0][0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, first - 1)
            for n, _ in self.frames:
                ok, bgr = cap.read()
                if not ok:
                    break
                yield n, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()


# ---------------------------------------------------------------- outputs

def write_exr_depth(path, depth, compression="piz"):
    header = {"compression": EXR_COMPRESSION[compression], "type": OpenEXR.scanlineimage}
    with OpenEXR.File(header, {"depth.Z": depth.astype(np.float16)}) as f:
        f.write(path)


class Preview:
    """Side-by-side RGB | colourised depth mp4, capped at 720p height."""

    def __init__(self, path, fps):
        self.path = path
        self.fps = fps
        self.writer = None
        self.cmap = matplotlib.colormaps.get_cmap("Spectral_r")

    def add(self, rgb, depth):
        h, w = depth.shape
        scale = min(1.0, 720 / h)
        size = (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2)
        lo, hi = np.percentile(depth, (1, 99))
        norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
        vis = (self.cmap(norm)[..., :3] * 255).astype(np.uint8)
        frame = np.hstack([cv2.resize(rgb, size), cv2.resize(vis, size)])
        if self.writer is None:
            self.writer = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (frame.shape[1], frame.shape[0])
            )
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.writer is not None:
            self.writer.release()


# ---------------------------------------------------------------- job

def process(model, source, out_dir, encoder, process_res=504, batch_size=1,
            compression="piz", preview=True, fps=None, on_progress=None, cancelled=None):
    """Run DA3 over `source`, writing EXR depth frames (and a preview) to out_dir.

    Returns a dict describing the outputs."""
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{source.basename}_depth_{encoder}"
    pad = source.padding
    exr_pattern = os.path.join(out_dir, f"{stem}.{'#' * pad}.exr")
    preview_path = os.path.join(out_dir, f"{stem}_preview.mp4") if preview else None
    pv = Preview(preview_path, fps or source.fps or 24) if preview else None

    done = 0
    written = []

    def flush(batch):
        nonlocal done
        pred = model.inference([rgb for _, rgb in batch], process_res=process_res)
        depth = np.asarray(pred.depth, dtype=np.float32).reshape(len(batch), *pred.depth.shape[-2:])
        for (n, rgb), d in zip(batch, depth):
            h, w = rgb.shape[:2]
            if d.shape != (h, w):
                d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
            write_exr_depth(os.path.join(out_dir, f"{stem}.{n:0{pad}d}.exr"), d, compression)
            if pv:
                pv.add(rgb, d)
            written.append(n)
        done += len(batch)
        if on_progress:
            on_progress(done, len(source))

    try:
        batch = []
        for n, rgb in source:
            if cancelled and cancelled():
                break
            batch.append((n, rgb))
            if len(batch) >= batch_size:
                flush(batch)
                batch = []
        if batch and not (cancelled and cancelled()):
            flush(batch)
    finally:
        if pv:
            pv.close()

    return {
        "exr_pattern": exr_pattern,
        "first": written[0] if written else None,
        "last": written[-1] if written else None,
        "frames_written": len(written),
        "preview": preview_path if written and preview else None,
    }
