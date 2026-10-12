"""DA3 weight lookup: host HF cache first, then weights baked into the image,
then download into the host cache."""
import argparse
import logging
import os
import threading

log = logging.getLogger(__name__)

REPOS = {
    "da3-small": "depth-anything/DA3-SMALL",
    "da3-base": "depth-anything/DA3-BASE",
    "da3-large": "depth-anything/DA3-LARGE",
}

HOST_CACHE = os.environ.get("HF_HOST_CACHE", "/hf-host/hub")
BAKED_CACHE = os.environ.get("HF_BAKED_CACHE", "/opt/hf")


def _from_pretrained(repo, **kwargs):
    from depth_anything_3.api import DepthAnything3

    return DepthAnything3.from_pretrained(repo, **kwargs)


def load_model(encoder, device):
    repo = REPOS[encoder]
    for cache in (HOST_CACHE, BAKED_CACHE):
        try:
            model = _from_pretrained(repo, cache_dir=cache, local_files_only=True)
            log.info("loaded %s from %s", repo, cache)
            break
        except Exception:
            continue
    else:
        log.info("downloading %s into %s", repo, HOST_CACHE)
        model = _from_pretrained(repo, cache_dir=HOST_CACHE)
    return model.to(device).eval()


class ModelCache:
    """Keeps a single model resident; switching encoder frees the previous one."""

    def __init__(self, device):
        self.device = device
        self.encoder = None
        self.model = None
        self._lock = threading.Lock()

    def get(self, encoder):
        with self._lock:
            if self.encoder != encoder:
                import torch

                self.model = None
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                self.model = load_model(encoder, self.device)
                self.encoder = encoder
            return self.model


def bake(encoders):
    from huggingface_hub import snapshot_download

    for enc in encoders:
        print(f"baking {REPOS[enc]}", flush=True)
        snapshot_download(REPOS[enc], cache_dir=os.environ.get("HF_HUB_CACHE", BAKED_CACHE))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bake", nargs="*", default=[], choices=sorted(REPOS))
    bake(parser.parse_args().bake)
