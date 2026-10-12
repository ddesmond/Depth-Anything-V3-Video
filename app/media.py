"""Turn a finished DA3 job into mp4s that video APIs accept (Higgsfield takes
video/mp4 only): a greyscale depth video and an sRGB plate video."""
import os

import cv2
import imageio_ffmpeg
import numpy as np
import OpenEXR

from . import processing

MAX_HEIGHT = 1080


def _even_size(w, h):
    scale = min(1.0, MAX_HEIGHT / h)
    return int(w * scale) // 2 * 2, int(h * scale) // 2 * 2


def _write(path, frames, fps):
    writer = None
    try:
        for frame in frames:
            if writer is None:
                writer = imageio_ffmpeg.write_frames(
                    path + ".part.mp4", (frame.shape[1], frame.shape[0]), fps=fps, codec="libx264",
                    pix_fmt_out="yuv420p", quality=8, macro_block_size=2,
                    output_params=["-movflags", "+faststart"])
                writer.send(None)
            writer.send(np.ascontiguousarray(frame))
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise processing.InputError("no frames to write")
    os.replace(path + ".part.mp4", path)
    return path


def _exr_frames(job):
    pattern = job.result["exr_pattern"]
    pad = pattern.count("#")
    for n in range(job.result["first"], job.result["last"] + 1):
        f = pattern.replace("#" * pad, f"{n:0{pad}d}")
        if os.path.exists(f):
            yield f


def _read_depth(path):
    with OpenEXR.File(path) as f:
        return f.channels()["depth.Z"].pixels.astype(np.float32)


def depth_video(job, out_dir, fps=None):
    """Greyscale depth mp4, near = white, normalised once over the whole shot
    (1st-99th percentile of sampled frames) so it doesn't flicker."""
    out = os.path.join(out_dir, f"{job.id}_depth.mp4")
    if os.path.exists(out):
        return out
    files = list(_exr_frames(job))
    if not files:
        raise processing.InputError("job has no EXR frames")
    sample = [_read_depth(f) for f in files[:: max(1, len(files) // 16)]]
    lo, hi = np.percentile(np.concatenate([s.ravel()[::7] for s in sample]), (1, 99))
    h, w = sample[0].shape
    size = _even_size(w, h)

    def frames():
        for f in files:
            d = np.clip((_read_depth(f) - lo) / max(hi - lo, 1e-6), 0, 1)
            g = ((1.0 - d) * 255).astype(np.uint8)
            g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
            yield np.repeat(g[..., None], 3, axis=-1)

    os.makedirs(out_dir, exist_ok=True)
    return _write(out, frames(), fps or _source_fps(job) or 24)


def plate_video(job, out_dir, fps=None):
    """The job's input frames as an sRGB mp4 (EXR plates get the sRGB curve)."""
    out = os.path.join(out_dir, f"{job.id}_plate.mp4")
    if os.path.exists(out):
        return out
    src = processing.Source(job.input, job.result["first"], job.result["last"])

    def frames():
        for _, rgb in src:
            yield cv2.resize(rgb, _even_size(rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)

    os.makedirs(out_dir, exist_ok=True)
    return _write(out, frames(), fps or src.fps or 24)


def _source_fps(job):
    try:
        return processing.Source(job.input, job.first, job.last).fps
    except Exception:
        return None
