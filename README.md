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

## Prerequisites & Installation

Depth Anything V3 works best with **Python 3.10**. Newer versions (e.g. 3.13) commonly cause dependency conflicts.

### 1. Create a Clean Environment (Recommended)

```bash
conda create -n da3 python=3.10 -y
conda activate da3
````

### 2. Install PyTorch

Install the PyTorch build matching your CUDA version (example for CUDA 12.1):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install the Official Depth Anything V3 Library

This script wraps the official ByteDance implementation.

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3
cd Depth-Anything-3
pip install -e .
```

### 4. Install Additional Dependencies

```bash
pip install opencv-python matplotlib tqdm
```

---

## Usage

Place `run_da3_video.py` inside the `Depth-Anything-3` directory (or ensure the package is on your `PYTHONPATH`).

### Basic Usage

```bash
python run_da3_video.py --video-path input_video.mp4
```

### Use a Larger, More Accurate Model

```bash
python run_da3_video.py --video-path input_video.mp4 --encoder da3-large
```

### Save Only the Depth Map (No Side-by-Side)

```bash
python run_da3_video.py --video-path input_video.mp4 --pred-only --grayscale
```

---

## Arguments

| Argument       | Description                                                                  | Default           |
| -------------- | ---------------------------------------------------------------------------- | ----------------- |
| `--video-path` | **(Required)** Path to the input video file                                  | N/A               |
| `--outdir`     | Output directory for the rendered video                                      | `./vis_video_da3` |
| `--encoder`    | Model size: `da3-small`, `da3-base`, `da3-large`<br>Large requires more VRAM | `da3-base`        |
| `--pred-only`  | Save only the depth map (no RGB comparison)                                  | `False`           |
| `--grayscale`  | Output depth as grayscale (0–255) instead of heatmap                         | `False`           |

---

## Troubleshooting

### Warning: `Dependency gsplat is required...`

This warning can be safely ignored. `gsplat` is only needed for 3D Gaussian Splatting, not for video depth inference.

### Error: Package `depth-anything-3` requires a different Python version

You are likely using Python 3.13 or newer. Switch to a Python 3.10 environment as shown above.

### CUDA Out of Memory

If you hit VRAM limits, use a smaller model:

```bash
python run_da3_video.py --video-path video.mp4 --encoder da3-small
```

---

## Credits

This script is an inference wrapper built on top of Depth Anything 3
