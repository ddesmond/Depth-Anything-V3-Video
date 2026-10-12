# Depth Anything V3 – Simple Video Inference

A streamlined, standalone Python script to generate depth maps from video files using the state-of-the-art [Depth Anything V3](https://github.com/ByteDance-Seed/Depth-Anything-3) model.

Unlike the official repository demos—which often require extracting frames or setting up complex pipelines—this script enables **direct video-to-video processing** with optional side-by-side visualization.

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)
![Torch](https://img.shields.io/badge/PyTorch-CUDA-orange)
![License](https://img.shields.io/badge/License-Apache_2.0-green)

---

## Features

- 🎥 **Direct Video Support**: Process `.mp4`, `.avi`, `.mov` files without manual frame extraction.
- 🖼️ **Side-by-Side Visualization**: Automatically stacks RGB input with the generated depth map.
- 🚀 **Model Selection**: Switch between `da3-small`, `da3-base`, and `da3-large` via CLI.
- 🛠️ **Robust Inference**: Handles V3 API batch dimensions and normalization automatically.
- 🎨 **Visualization Options**: Spectral heatmap or raw grayscale depth output.

---

## Docker service (API + queue + Nuke node)

A GPU container that runs DA3 behind a small HTTP API with a FIFO job queue (one
job at a time, model kept loaded). Inputs are videos, single images or image
sequences (`shot.####.exr`, `shot.%04d.png`). Outputs are **half-float EXR depth
(`depth.Z`, PIZ or ZIP16)** plus an optional side-by-side H.264 preview mp4. You can use it from a web UI, a REST API or a Nuke node.

### Requirements

- NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- Docker Compose v2

The image uses PyTorch `cu126` wheels by default. These are the newest builds that
still support Pascal cards (GTX 10xx) and they also run on newer GPUs. On
Turing or newer you can set `TORCH_CUDA=cu128` in `.env`.

### Setup

```bash
cp .env.example .env        # set PROJECTS_DIR, UID/GID
mkdir -p ~/.cache/huggingface "$PROJECTS_DIR"
docker compose up -d --build
curl localhost:8000/health
```

- `PROJECTS_DIR` is mounted at the **same path** inside the container, so paths
  from Nuke work unchanged. Jobs can only read and write under it and `./data`.
- All three models (small/base/large) are baked into the image. The host HF cache
  (`~/.cache/huggingface`) is checked first. Anything missing from both is
  downloaded into the host cache.
- The port is published only on localhost and `DA3_LAN_IP` (set in `.env`) with **no auth**.
  If firewalld is active, open it with
  `sudo firewall-cmd --zone=public --add-port=8000/tcp --permanent && sudo firewall-cmd --reload`.
- The queue is in memory, so queued and running jobs are lost if the container restarts.

### Web UI

Open **http://localhost:8000/ui** (or `http://<machine-ip>:8000/ui` from the LAN;
`/` redirects there). It has these features:
- Upload a video, image or `.zip` sequence, or enter a path on the mounted storage.
- Pick the model and options.
- Watch the live job list and progress, play the preview, download the result zip, or cancel a job.

UI jobs share the API's queue. Gradio's file cache lives in `./data/gradio-cache`
and is cleaned after a day.

### API

| Method | Path | |
|---|---|---|
| `POST` | `/jobs` | JSON job for a path on the mounted storage |
| `POST` | `/jobs/upload` | multipart: `file` (video, image, or `.zip` of a sequence) + same options as form fields |
| `GET` | `/jobs` / `/jobs/{id}` | status, `done`/`total` frames, `queue_position`, `result` |
| `DELETE` | `/jobs/{id}` | cancel (queued or running) |
| `GET` | `/jobs/{id}/result` | zip of the EXRs + preview |
| `GET` | `/jobs/{id}/preview` | preview mp4 |
| `GET` | `/health`, `/models` | GPU / queue / model info |

Interactive docs: `http://localhost:8000/docs`.

```bash
# path job (default output: <input dir>/da3_depth/)
curl -X POST localhost:8000/jobs -H 'Content-Type: application/json' -d '{
  "input": "/home/me/projects/shot010/plate/shot010.####.exr",
  "first": 1001, "last": 1100,
  "encoder": "da3-base", "compression": "piz"
}'

# upload job
curl -F file=@clip.mp4 -F encoder=da3-small localhost:8000/jobs/upload
curl -OJ localhost:8000/jobs/<id>/result
```

Options: `encoder` (`da3-small|da3-base|da3-large`), `first`/`last` (real frame
numbers for sequences; 1-based for videos), `process_res` (default 504),
`batch_size` (frames per inference call; >1 gives a more consistent depth scale
across frames but uses more VRAM), `compression` (`piz|zip16|zip|none`),
`preview`, `fps`, `output_dir`.

EXR inputs are treated as scene-linear and converted to sRGB before inference.
Depth values are DA3's raw (relative) depth, not normalized.

### Nuke

```bash
cp nuke/da3_depth.py nuke/menu.py ~/.nuke/     # or add nuke/ to NUKE_PATH
```

**Nodes → DA3 → DA3 Depth** creates a node. Connect it below a Read, pick a model,
and press **Submit**. The node shows queue and progress status. When the job
finishes, it creates a Read on the EXR depth sequence (`raw` colorspace,
`depth.Z` channel).

---

## Standalone script

### Prerequisites & Installation

Depth Anything V3 works best with **Python 3.10**. Newer versions (e.g. 3.13) commonly cause dependency conflicts.

#### 1. Create a Clean Environment (Recommended)

```bash
conda create -n da3 python=3.10 -y
conda activate da3
````

#### 2. Install PyTorch

Install the PyTorch build matching your CUDA version (example for CUDA 12.1):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 3. Install the Official Depth Anything V3 Library

This script wraps the official ByteDance implementation.

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3
cd Depth-Anything-3
pip install -e .
```

#### 4. Install Additional Dependencies

```bash
pip install opencv-python matplotlib tqdm
```

---

### Usage

Place `da3_video.py` inside the `Depth-Anything-3` directory (or ensure the package is on your `PYTHONPATH`).

#### Basic Usage

```bash
python da3_video.py --video-path input_video.mp4
```

#### Use a Larger, More Accurate Model

```bash
python da3_video.py --video-path input_video.mp4 --encoder da3-large
```

#### Save Only the Depth Map (No Side-by-Side)

```bash
python da3_video.py --video-path input_video.mp4 --pred-only --grayscale
```

---

### Arguments

| Argument       | Description                                                                  | Default           |
| -------------- | ---------------------------------------------------------------------------- | ----------------- |
| `--video-path` | **(Required)** Path to the input video file                                  | N/A               |
| `--outdir`     | Output directory for the rendered video                                      | `./vis_video_da3` |
| `--encoder`    | Model size: `da3-small`, `da3-base`, `da3-large`<br>Large requires more VRAM | `da3-base`        |
| `--pred-only`  | Save only the depth map (no RGB comparison)                                  | `False`           |
| `--grayscale`  | Output depth as grayscale (0–255) instead of heatmap                         | `False`           |

---

### Troubleshooting

#### Warning: `Dependency gsplat is required...`

This warning can be safely ignored. `gsplat` is only needed for 3D Gaussian Splatting, not for video depth inference.

#### Error: Package `depth-anything-3` requires a different Python version

You are likely using Python 3.13 or newer. Switch to a Python 3.10 environment as shown above.

#### CUDA Out of Memory

If you hit VRAM limits, use a smaller model:

```bash
python da3_video.py --video-path video.mp4 --encoder da3-small
```

---

## Credits

This script is an inference wrapper built on top of Depth Anything 3
