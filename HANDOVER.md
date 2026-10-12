# DA3 Video Docker Service: Handover

2026-10-12

## Status at a glance

The Docker GPU service is built, tested on the GTX 1070 and pushed to `main-docker`.
The Nuke node and LAN access are still unverified, and the running container is
pointed at a test folder.

| Item | State |
| --- | --- |
| Repo / branch | `github.com/ddesmond/Depth-Anything-V3-Video`, branch `main-docker`; not merged to `main`, no PR |
| Image | `da3-video:latest`, about 10 GB, torch 2.14.1+cu126, Gradio 6.30, all three models baked |
| Running container | Up on the dev box, but started with `PROJECTS_DIR` set to a temporary test folder in `/tmp`, not real shots |
| `.env` | `PROJECTS_DIR=/home/des/projects`, a placeholder folder that does not exist |
| LAN access | Bound to `127.0.0.1` and `192.168.10.7` only. firewalld `public` zone on `enp8s0` still blocks port 8000, so other LAN machines can't reach it yet |
| Nuke node | Written and syntax-checked only; never run inside Nuke |

## Requirements audit

All but four asks are built and tested. The gaps:
- The Gradio UI was dropped without saying so, then added later.
- The Nuke node hasn't been tested in Nuke.
- The firewall isn't open for LAN use.
- Nothing has been tested on a newer GPU.

| What was asked | Implemented | Verified | Gap |
| --- | --- | --- | --- |
| Run the repo in Docker on a GPU machine; plan compose together | Yes: `Dockerfile`, `docker-compose.yml` | Yes, on the GTX 1070 | None |
| Works on both GPUs (GTX 1070 and newer) | Yes: torch `cu126` (has Pascal kernels), `TORCH_CUDA` build arg | 1070 only | Never run on a 3090 or other Turing+ card |
| Long-running service | Yes: FastAPI app, `restart: unless-stopped` | Yes | None |
| Weights: bake into image and use host folder | Yes: all three baked in `/opt/hf`; host `~/.cache/huggingface` checked first | Baked path yes | Host-cache hit and download-on-miss never exercised (host cache is empty) |
| Load from mounted host dir or push through API | Yes: path jobs and `/jobs/upload` | Yes | None |
| Queue | Yes: in-memory FIFO, one job at a time, model stays loaded | Yes, incl. cancel | Lost on restart (accepted when the queue was kept simple) |
| Push on demand from Nuke via a node | Yes: `nuke/da3_depth.py` | Syntax only | Never run in Nuke |
| API + Web UI | API at first; Gradio was dropped in the plan recap without being called out, then added later | Yes, via `gradio_client` | Not checked in a real browser |
| Nuke on same machine, shared paths | Yes: `PROJECTS_DIR` mounted at the same path | With a test dir | Real `PROJECTS_DIR` not set |
| Output: half-float EXR, PIZ or ZIP16, plus preview | Yes: `depth.Z` half, PIZ default, ZIP16 option; H.264 preview | Yes | None |
| No auth, localhost + LAN (192.168.10.7, not Netbird) | Yes: bound to `127.0.0.1` and `DA3_LAN_IP` | From this host only | firewalld blocks LAN clients; needs `sudo` |
| Precision info for 3090 (info only, keep same models) | Info given; no code change, as agreed | n/a | fp32 / float32 EXR not built, by agreement |

## Open items and known limitations

Four items block real use. The rest are limitations to know about.

**Blocking real use**

- [ ] Set `PROJECTS_DIR` in `.env` to the real shots folder, create it, then `docker compose up -d` (the running container still points at a `/tmp` test folder).
- [ ] Open the LAN port: `sudo firewall-cmd --zone=public --add-port=8000/tcp --permanent && sudo firewall-cmd --reload`, then test `http://192.168.10.7:8000/ui` from another LAN machine.
- [ ] Test the Nuke node in Nuke: install it, submit from a Read, check the auto-created Read loads `depth.Z`.
- [ ] Build and test on the 3090 (cu126 should work there; `cu128` is optional).

**Not built (out of scope or deferred)**

- **fp32 inference and 32-bit float EXR.** DA3 always autocasts to bf16 and output is half float. This was discussed and kept as is.
- **Persistent queue.** Jobs are in memory and lost on container restart.
- **Disk cleanup for `./data/jobs`.** Uploaded inputs and outputs are never deleted. Only Gradio's cache (`./data/gradio-cache`) is pruned after a day.
- **Upload size limits, auth, HTTPS.** None of these exist; the service trusts the LAN by design.
- **Multi-GPU.** There's one worker thread and the compose file reserves one GPU (`count: 1`).
- **OCIO colour management.** EXR inputs are assumed scene-linear and get a plain sRGB curve before inference.
- **Temporal consistency.** Depth is relative per inference call, so scale can drift between frames. `batch_size` > 1 helps within a batch only.
- **Automated tests and CI.** All testing was manual.
- **PR to `main`.** Not opened.

**Licensing**

- `da3-large` (baked in the image) is CC BY-NC 4.0, so non-commercial use only. `da3-small` and `da3-base` are Apache 2.0.
- Upstream deprecated `DA3-LARGE` in favour of `DA3-LARGE-1.1`, which has the same size but retrained weights. The image still uses the original.

## Architecture and file map

One container runs the API, the UI and a single GPU worker thread. Every client
feeds the same in-memory queue.

```
 Gradio UI (/ui) ──┐       ┌──────────── da3 container (1 GPU, :8000) ─────────────┐
 REST API (/jobs) ─┼──────▶│ FastAPI app (server.py, ui.py)                        │
 Nuke node ────────┘       │   checks paths, saves uploads, creates jobs           │──▶ PROJECTS_DIR (same path as host)
                           │          ▼                                            │──▶ ./data (uploads, results, cache)
                           │ FIFO job queue (jobs.py): in memory, one at a time    │
                           │          ▼                                            │◀── model weights: host HF cache,
                           │ GPU worker (processing.py): frames → DA3 → EXR + mp4  │    then baked /opt/hf
                           └───────────────────────────────────────────────────────┘
```

| File | Role |
| --- | --- |
| `Dockerfile` | python:3.10-slim, torch cu126, deps, DA3 at pinned commit, baked weights, runs uvicorn as UID 1000 |
| `docker-compose.yml` | One `da3` service: GPU reservation, port binding, `PROJECTS_DIR` / `./data` / HF cache mounts |
| `.env.example` | Template for `.env` (gitignored) |
| `requirements.txt` | DA3 runtime deps minus xformers/open3d, plus OpenEXR, FastAPI, Gradio |
| `app/server.py` | FastAPI routes, path checks, upload handling, result zip; mounts the UI |
| `app/ui.py` | Gradio UI at `/ui` |
| `app/jobs.py` | `Job` dataclass, `JobQueue` with the worker thread |
| `app/processing.py` | Frame sources (video, sequence, image, EXR), inference loop, EXR and preview writers |
| `app/models.py` | Weight lookup (host cache, then baked), `ModelCache` keeping one model in VRAM, `--bake` CLI |
| `nuke/da3_depth.py`, `nuke/menu.py` | Nuke node and menu entry |
| `da3_video.py` | Original standalone CLI, now exits non-zero on failure |
| `README.md` | Docker, UI, API and Nuke docs plus the original CLI docs |

## How to run and use it

Set the `.env` values, start Compose, then use the UI, the API or the Nuke node.
All three feed the same queue.

**Setup**

1. `cp .env.example .env` and set:
    - `PROJECTS_DIR` to the real shots folder;
    - `UID`/`GID` to the output of `id -u` and `id -g` (currently 1000/1001);
    - `DA3_LAN_IP` to `192.168.10.7`.
2. `mkdir -p ~/.cache/huggingface "$PROJECTS_DIR"`. Docker would otherwise create them as root.
3. `docker compose up -d --build`, then `curl localhost:8000/health`.
4. Open the firewall port (see Open items).

**`.env` settings**

| Variable | Current value | Purpose |
| --- | --- | --- |
| `PROJECTS_DIR` | `/home/des/projects` (placeholder) | Mounted at the same path in the container; the only place path jobs may read or write, besides `./data` |
| `UID` / `GID` | 1000 / 1001 | Container user, so outputs aren't root-owned |
| `DA3_PORT` | 8000 | Published port |
| `DA3_LAN_IP` | 192.168.10.7 | LAN address to publish on, plus `127.0.0.1`. Never the Netbird interface (`wt0`) |
| `HF_CACHE_DIR` | `/home/des/.cache/huggingface` | Host model cache, checked before the baked weights |
| `TORCH_CUDA` | `cu126` | `cu126` runs on Pascal and newer; `cu128`/`cu130` are Turing+ only |
| `BAKE_MODELS` | `da3-small da3-base da3-large` | Models baked at build time |

**Web UI:** `http://localhost:8000/ui`, or `http://192.168.10.7:8000/ui` once the firewall is open. `/` redirects there. You can:
- upload a file or give a server path;
- pick the model and options;
- watch progress and play the preview;
- download the result zip or cancel a job.

**REST API** (interactive docs at `/docs`)

| Method | Path | Does |
| --- | --- | --- |
| `POST` | `/jobs` | JSON job for a path on mounted storage |
| `POST` | `/jobs/upload` | Multipart `file` (video, image or `.zip` sequence) plus option form fields |
| `GET` | `/jobs`, `/jobs/{id}` | Status, `done`/`total`, `queue_position`, `result` |
| `DELETE` | `/jobs/{id}` | Cancel a queued or running job |
| `GET` | `/jobs/{id}/result` | Zip of the EXRs and preview |
| `GET` | `/jobs/{id}/preview` | Preview mp4 |
| `GET` | `/health`, `/models` | GPU, torch, queue, loaded model |

Job options:
- `encoder`: `da3-small`, `da3-base` or `da3-large`.
- `first`/`last`: real frame numbers for sequences, 1-based for videos.
- `process_res`: default 504.
- `batch_size`: default 1.
- `compression`: `piz`, `zip16`, `zip` or `none`.
- `preview`, `fps`.
- `output_dir`: default `<input dir>/da3_depth`.

Each job writes:
- `<name>_depth_<encoder>.####.exr`: half float, channel `depth.Z`, original resolution, input frame numbers kept.
- `<name>_depth_<encoder>_preview.mp4`: H.264, RGB beside colourised depth, at most 720 px tall.

**Nuke:** copy `nuke/da3_depth.py` and `nuke/menu.py` into `~/.nuke`. Nodes → DA3 → DA3 Depth creates a NoOp with server URL, model, resolution, batch, compression, frame range and output dir knobs. When you press Submit, it:
1. finds the upstream Read and posts its path and frame range;
2. polls the job every 2 s;
3. creates a raw Read on the EXR sequence when the job is done.

It uses only the Python standard library.

## What was tested on the GTX 1070

Every API and UI path passed on synthetic test media: a 24-frame 640×360 mp4 and a
six-frame EXR sequence numbered 1001–1006. No real plates were used.

| Test | Result |
| --- | --- |
| torch sees GPU in container | 2.14.1+cu126, sm_61 runs via the sm_60 kernels |
| `da3-small`, video, 24 frames | 2.2 s (about 10 frames/s at `process_res` 504) |
| `da3-base`, EXR sequence 1002–1005 | 4.6 s including model load |
| `da3-large`, `batch_size` 4, 24 frames | 15.7 s including model load; fits in 8 GB |
| EXR output | `depth.Z`, float16, PIZ and ZIP16 confirmed by reading headers back |
| Upload job + `/result` zip | 200, zip holds EXRs + preview |
| Path outside allowed roots (`/etc/passwd`) | Rejected with 400 |
| Missing input | Rejected with 400 |
| Single frame path (`shot010.1001.exr`) | Expanded to `shot010.####.exr` |
| Cancel queued job | Status `cancelled` |
| Gradio UI via `gradio_client` | Upload, path job, refresh, preview and zip all worked |
| Preview codec | H.264 High, yuv420p |
| Port binding | 127.0.0.1 and 192.168.10.7 return 200; Netbird 192.168.40.198 refused |
| Output file ownership | Host user (`des`), not root |

Not tested:
- the Nuke node inside Nuke;
- a real browser session;
- LAN access from another machine;
- a 3090;
- loading models from the host cache, or downloading them;
- real production plates and long sequences.

## Decisions and gotchas

| Decision | Why | Watch out for |
| --- | --- | --- |
| torch `cu126` by default | Last CUDA build of torch with Pascal (GTX 10xx) kernels | Moving to `cu128`+ breaks the 1070 with "no kernel image" |
| DA3 installed with `--no-deps`, its deps listed in `requirements.txt` | Its `xformers` dependency would replace the cu126 torch | DA3 commit pinned to `3d835ec`; bumping it may add imports that need new deps. The build runs an import check |
| `pip` constraints file pins torch during the deps install | Stops any package from swapping torch | — |
| `pip install --upgrade pip` first | The base image's pip fails on a `typing-extensions` name quirk | — |
| Models baked before the other deps | A requirements change doesn't re-download about 2.3 GB | Editing `app/models.py` does invalidate the bake layer |
| Single container, in-memory queue | User asked for a simple queue | Restart loses queued and running jobs |
| Same-path mount of `PROJECTS_DIR` | Nuke paths work unchanged, no translation | Paths outside it are rejected |
| Half-float EXR, `depth.Z` | User's spec | Values are DA3's raw relative depth, not normalised and not metres |
| H.264 preview via `imageio-ffmpeg` | OpenCV's `mp4v` doesn't play in browsers | — |
| Port bound to `127.0.0.1` + `DA3_LAN_IP` | LAN only, never Netbird (`wt0`) | Don't go back to `0.0.0.0` |
| `DA3_LOG_LEVEL=WARN` | DA3 logs three lines per frame otherwise | — |
| Gradio cache in `./data/gradio-cache`, pruned daily | Gradio copies each served file | Large result zips still take double disk space for up to a day |
| `gradio>=5` unpinned (built 6.30) | — | A future Gradio major could break `app/ui.py` |

Other gotchas:

- DA3 autocasts to bf16. On Pascal that is emulated, but it measured fast enough.
- The `gsplat` warning and a `torch.jit.script` deprecation warning at startup are harmless.
- `da3_video.py` (the original CLI) is still in the image: `docker compose run --rm da3 python da3_video.py --video-path …`. It now exits non-zero on failure.
- The host runs firewalld with custom policies (`gateway-lan-to-HOST`, `docker-forwarding`). If opening the port in `public` isn't enough, one of those is the next suspect.

## Recommended next steps

1. Set the real `PROJECTS_DIR`, restart, and delete the test data in `./data/jobs`.
2. Open port 8000 in firewalld and check the UI from a LAN machine.
3. Run the Nuke node on a real plate; fix whatever breaks (Read path forms, frame range, raw colourspace).
4. Run a long real sequence (a few hundred frames, 2K or 4K) to check speed, memory and disk use.
5. Build and test on the 3090.
6. Decide on the licence question for `da3-large` if any of this is paid work.
7. Optional, only if wanted:
    - fp32 inference and float32 EXR;
    - a persistent queue;
    - cleanup of old job data;
    - pinning the Gradio version;
    - a PR to `main`.
