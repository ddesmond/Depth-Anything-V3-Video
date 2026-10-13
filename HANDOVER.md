# DA3 Video Docker Service: Handover

2026-10-13

## Status at a glance

The service runs on **hyper01** (offline machine, RTX 2080) at **http://10.0.0.131:8010/ui**, using the CUDA 12.4 image. It's reachable from hyper01 itself and from your Mac over WireGuard. Higgsfield generation and the Nuke node are built but not yet tested for real.

| Item | State |
| --- | --- |
| Repo / branch | `github.com/ddesmond/Depth-Anything-V3-Video`, branch `main-docker`; not merged to `main`, no PR. The CUDA 12.4 default changes are **not committed yet** |
| Image | `da3-video:latest`, 8.76 GB, torch 2.6.0 + CUDA 12.4, Gradio 6.30, all three models baked |
| hyper01 deployment | Running at `http://10.0.0.131:8010/ui`, GPU reported as RTX 2080. Details below |
| Dev box (r7) | Image built and tested there; the container is removed, nothing running |
| Offline bundle | `dist/da3-offline/` on r7 (5.4 GB, gitignored), copy in `/opt/da3-offline` on hyper01 |
| Nuke node | Written and syntax-checked only; never run inside Nuke |
| Higgsfield tabs | Tested only against a mock API; no key yet, and hyper01 has no internet anyway |

## Deployment on hyper01 (offline)

hyper01 has no internet. It's reached from r7 by jumping through the Mac (`des@192.168.10.5`) over its WireGuard link (`utun5`, `10.10.0.2`): `ssh -J des@192.168.10.5 root@10.0.0.131`, or from the Mac directly.

| Fact (checked on hyper01) | Value |
| --- | --- |
| OS | Rocky Linux 9.5 |
| GPU / driver | NVIDIA GeForce RTX 2080 (sm_75), driver 550.142, supports CUDA 12.4 |
| Docker / Compose | 27.4.1 / v2.32.1 |
| IP | `10.0.0.131` on `ens5f0`; firewalld inactive; iptables INPUT policy ACCEPT |
| Disk | `/` 1.9 TB, about 1.7 TB free |
| Users | `des` is UID 1001 / GID 1003; the container runs as 1000:1001 |

**What's in `/opt/da3-offline`**

| Path | What |
| --- | --- |
| `da3-video-image.tar.gz` + `.sha256` | The CUDA 12.4 image (5.4 GB), checksum verified after the copy, already loaded with `docker load` |
| `docker-compose.yml`, `.env`, `.env.example` | Compose setup; `.env` is configured for hyper01 (below) |
| `nvidia-toolkit-el9/` | NVIDIA container toolkit 1.20.1 RPMs (installed) |
| `projects/`, `data/`, `hf-cache/` | Test projects folder, job data and the HF cache. Owned by 1000:1001 so the container can write |
| `README.md`, `HANDOVER.md`, `OFFLINE.md`, `nuke/` | Docs and the Nuke node |

**`.env` on hyper01:** `PROJECTS_DIR=/opt/da3-offline/projects` (test folder), `DA3_LAN_IP=10.0.0.131`, `DA3_PORT=8010`, `HF_CACHE_DIR=/opt/da3-offline/hf-cache`, `HF_HUB_OFFLINE=1`, no UI password, no Higgsfield key.

**Changes made on hyper01:**
- Installed `nvidia-container-toolkit` 1.20.1 (+ `-base`, `libnvidia-container-tools`, `libnvidia-container1`) from the local RPMs, with all repos disabled.
- `nvidia-ctk runtime configure --runtime=docker` added the `nvidia` runtime to `/etc/docker/daemon.json` (insecure-registries kept). Backup: `/etc/docker/daemon.json.bak-da3`.
- Restarted Docker (no containers were running).
- Loaded `da3-video:latest`, started `da3-offline-da3-1` with `docker compose up -d --no-build`.
- `chown 1000:1001` on `data/`, `projects/`, `hf-cache/` (the first start crashed with `Permission denied: '/data/generate'` because they were created as root).

**Verified:**

| Check | Result |
| --- | --- |
| `/health` | `device: cuda`, `NVIDIA GeForce RTX 2080`, `torch 2.6.0+cu124` |
| UI from hyper01 | `http://localhost:8010/ui` → 200, `http://10.0.0.131:8010/ui` → 200 |
| Listening | `10.0.0.131:8010` and `127.0.0.1:8010` only |
| UI from the Mac (WireGuard, `10.10.0.2`) | 200 |

**Not verified:**
- [ ] Access from another machine on 10.0.x (no login on any; open `http://10.0.0.131:8010/ui` from one).
- [ ] A real depth job on the RTX 2080 (only `/health` and the UI were checked).
- [ ] Real projects folder: `PROJECTS_DIR` points at the test folder `/opt/da3-offline/projects`; Nuke paths only work if the real shot storage is mounted at the same path.
- [ ] Output ownership: files are written as UID 1000:1001, which doesn't match `des` (1001:1003) on hyper01. Changing it needs a rebuild with `UID`/`GID` build args (on a machine with internet).

**Later changes (2026-10-13):**
- Log rotation: `docker-compose.yml` now sets `json-file` logs to 10 MB × 5 files. Before, logs were unlimited (no `log-opts` in hyper01's `daemon.json` either). Applied on hyper01 and confirmed with `docker inspect`.
- Web UI is upload-only: the Depth tab's "Path on server" input was removed. Path jobs remain on the REST API (`POST /jobs`) for the Nuke node. The Generate tab only takes uploads, finished DA3 jobs or https URLs.
- `.dockerignore` now excludes `dist/` (it had pulled the 5.4 GB archive into the image).
- On r7, `docker image prune` removed two old untagged ~300 MB images that weren't from this project.

**Manage it on hyper01:**
```bash
cd /opt/da3-offline
docker compose ps                 # status
docker compose logs -f            # logs
docker compose restart            # restart
docker compose down               # stop and remove the container
```

## CUDA 12.4 change (2026-10-13)

The first image used torch 2.14.1 + CUDA 12.6, chosen from the dev box only (driver 580). hyper01's driver 550 supports CUDA 12.4, so the image was rebuilt with **torch 2.6.0 + CUDA 12.4**, the newest torch PyTorch publishes for cu124. Defaults now say `TORCH_CUDA=cu124`, `TORCH_VERSION=2.6.0` in `Dockerfile`, `docker-compose.yml`, `.env.example` and README. Kernels cover sm_50 to sm_90 (GTX 1070 and RTX 2080 both tested). Rule for later builds: match the CUDA build to the **oldest driver** among the target machines (`nvidia-smi`).

Also added: an `HF_HUB_OFFLINE` switch in `docker-compose.yml`/`.env.example` (set to 1 offline), and `dist/` in `.gitignore`.

Uncommitted: `Dockerfile`, `docker-compose.yml`, `.env.example`, `.gitignore`, `README.md`, `HANDOVER.md`.

## Update: Higgsfield generation (built after the first handover)

The Generate, History and Funds tabs are built and tested end to end against a local mock of the Higgsfield API. They have **not** been run against the real API, because there's no key yet. See the README section "Generate with Higgsfield".

| Asked | Built | Notes |
| --- | --- | --- |
| Submit processed depth videos to Higgsfield from Gradio | Yes | Depth/plate mp4 made from DA3 jobs; all 67 video workflows, forms built from their schemas |
| Follow tasks | Yes | Background worker: uploads, submits within the concurrency limit, polls with backoff, downloads outputs |
| Funds | Partly | No balance endpoint exists, so the tab shows per-job estimates and spend totals plus a link to the console |
| Prompts and job history with all metadata | Yes | SQLite history survives restarts; full record per job |
| Resubmit a 480p job at higher res with same seed and prompt | Yes | Seed only on Wan and Happy Horse models; others are marked as new random results |
| Batch jobs for cheaper rates | Yes, as drafts | No documented batch discount; draft at the lowest tier, then "Upscale picks" |
| Password | Yes | Shared `UI_PASSWORD`; REST API stays open for Nuke |

Open items for this part:
- [ ] Add `HIGGSFIELD_KEY_ID` / `HIGGSFIELD_KEY_SECRET` and `UI_PASSWORD` to `.env`, then run one cheap real job to confirm upload, estimate, submit, poll and download against the live API.
- [ ] Check the form in a real browser. It was tested only through `gradio_client`.
- [ ] Media made from DA3 jobs (`./data/generate/media`) and outputs are never cleaned up.
- [ ] DA3 jobs are in memory, so after a restart old DA3 results can't be added to the media list; history rows keep working because they store file paths.

## Requirements audit

All but four asks are built and tested. The gaps:
- The Gradio UI was dropped without saying so, then added later.
- The Nuke node hasn't been tested in Nuke.
- The firewall isn't open for LAN use.
- Nothing has been tested on a newer GPU.

| What was asked | Implemented | Verified | Gap |
| --- | --- | --- | --- |
| Run the repo in Docker on a GPU machine; plan compose together | Yes: `Dockerfile`, `docker-compose.yml` | Yes, on the GTX 1070 | None |
| Works on both GPUs (GTX 1070 and newer) | Yes: torch 2.6.0 `cu124` (kernels sm_50–sm_90), `TORCH_CUDA` / `TORCH_VERSION` build args | 1070 only | Not yet run on the RTX 2080 (hyper01) or a 3090 |
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
- [ ] Test on hyper01's RTX 2080 (driver 550 / CUDA 12.4) once the NVIDIA container toolkit is installed there, and on the 3090.

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
| `Dockerfile` | python:3.10-slim, torch 2.6.0 cu124, deps, DA3 at pinned commit, baked weights, runs uvicorn as UID 1000 |
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
| `TORCH_CUDA`, `TORCH_VERSION` | `cu124`, `2.6.0` | Needs a driver supporting CUDA 12.4+ (550+); runs on Pascal and newer |
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
| torch sees GPU in container | 2.6.0+cu124, sm_61 runs via the sm_60 kernels (first tested on 2.14.1+cu126, rebuilt for cu124) |
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
| torch 2.6.0 `cu124` by default | hyper01's driver (550) supports CUDA 12.4; 2.6.0 is the newest torch published for cu124 | A newer CUDA build needs a newer driver on every target; `cu128`+ also drops the GTX 1070 |
| DA3 installed with `--no-deps`, its deps listed in `requirements.txt` | Its `xformers` dependency would replace the pinned CUDA torch | DA3 commit pinned to `3d835ec`; bumping it may add imports that need new deps. The build runs an import check |
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
