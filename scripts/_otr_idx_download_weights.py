"""Download + validate the IndexTTS2 weights (Path B, default char voice).

This is the script named by eng_indextts2.py's fail-closed "weights
incomplete" error. Run it with the ISOLATED index-tts venv python (its
transformers pin ships huggingface_hub); the install wrapper
scripts/_otr_indextts2_install.ps1 does exactly that:

    <ComfyUI-parent>/index-tts/.venv/Scripts/python.exe scripts/_otr_idx_download_weights.py

Downloads IndexTeam/IndexTTS-2 into <index-tts>/checkpoints (env override
OTR_INDEXTTS2_DIR) and FAILS LOUD (exit 1) if any expected artifact is
missing or either large checkpoint has the wrong byte size (truncated or
upstream-replaced download). Re-run safe: snapshot_download resumes.
"""
from __future__ import annotations

import os
import sys

_REPO_ID = "IndexTeam/IndexTTS-2"
_REPO_REVISION = "740dcaff396282ffb241903d150ac011cd4b1ede"

# Repositories IndexTTS2 otherwise pulls during the first render. Warming exact
# revisions here keeps a network stall out of the node's wall-clock budget and
# makes a repeat install auditable.
_RUNTIME_REPOS = (
    ("facebook/w2v-bert-2.0", "da985ba0987f70aaeb84a80f2851cfac8c697a7b"),
    ("amphion/MaskGCT", "265c6cef07625665d0c28d2faafb1415562379dc"),
    ("funasr/campplus", "e4b6ede7ce16997aff4ae69fbca1f0175e2afede"),
    ("nvidia/bigvgan_v2_22khz_80band_256x",
     "633ff708ed5b74903e86ff1298cf4a98e921c513"),
)
_RUNTIME_EXPECTED = {
    "facebook/w2v-bert-2.0": {
        "config.json": 1_874,
        "model.safetensors": 2_322_063_736,
        "preprocessor_config.json": 275,
    },
    "amphion/MaskGCT": {
        "semantic_codec/model.safetensors": 177_183_712,
    },
    "funasr/campplus": {
        "campplus_cn_common.bin": 28_036_335,
    },
    "nvidia/bigvgan_v2_22khz_80band_256x": {
        "bigvgan_generator.pt": 449_228_171,
        "config.json": 1_405,
    },
}

# Byte sizes pinned from hf://models/IndexTeam/IndexTTS-2 (verified
# 2026-07-10; identical to the working install on the nv50 box). If upstream
# ever republishes different weights this validation fails LOUD -- re-derive
# the pins deliberately, never by accident.
_EXPECTED = {
    "config.yaml": 2_882,
    "bpe.model": 475_997,
    "gpt.pth": 3_484_663_079,
    "s2mel.pth": 1_202_198_223,
    "feat1.pt": 57_170,
    "feat2.pt": 374_866,
    "wav2vec2bert_stats.pt": 9_343,
    "qwen0.6bemo4-merge/model.safetensors": 1_192_135_096,
    "qwen0.6bemo4-merge/tokenizer.json": 11_422_654,
    "qwen0.6bemo4-merge/config.json": 727,
}


def _default_model_dir() -> str:
    env = os.environ.get("OTR_INDEXTTS2_DIR")
    if env:
        return env
    source = os.environ.get("OTR_INDEXTTS2_ROOT")
    if source:
        return os.path.join(os.path.abspath(os.path.expanduser(source)),
                            "checkpoints")
    # realpath: junction-safe, same resolution the engine adapter uses.
    here = os.path.realpath(__file__)                 # <repo>/scripts/this.py
    repo_root = os.path.dirname(os.path.dirname(here))
    comfy_root = os.path.dirname(os.path.dirname(repo_root))
    base = comfy_root if os.name == "nt" else os.path.dirname(comfy_root)
    return os.path.join(base, "index-tts", "checkpoints")


def validate_model_dir(model_dir: str) -> list[str]:
    """Return every missing or byte-mismatched pinned artifact."""
    failures = []
    for name, size in _EXPECTED.items():
        path = os.path.join(model_dir, *name.split("/"))
        if not os.path.isfile(path):
            failures.append(f"missing: {name}")
        elif size is not None and os.path.getsize(path) != size:
            failures.append(
                f"size mismatch: {name} ({os.path.getsize(path)} != {size})")
    return failures


def _pin_cache_ref(cache_dir: str, repo_id: str, revision: str) -> None:
    """Make a vendor default-revision lookup resolve the audited snapshot.

    IndexTTS2 calls these repos without a ``revision=`` argument. A commit-only
    snapshot is not enough for offline mode because Hugging Face resolves the
    default through ``refs/main`` first. Publish that tiny ref atomically after
    the pinned snapshot has landed.
    """
    repo_dir = os.path.join(cache_dir, "models--" + repo_id.replace("/", "--"))
    snapshot = os.path.join(repo_dir, "snapshots", revision)
    if not os.path.isdir(snapshot):
        raise RuntimeError("pinned snapshot directory missing: %s" % snapshot)
    for relative, expected_bytes in _RUNTIME_EXPECTED[repo_id].items():
        artifact = os.path.join(snapshot, *relative.split("/"))
        if not os.path.isfile(artifact):
            raise RuntimeError("pinned runtime artifact missing: %s" % artifact)
        actual_bytes = os.path.getsize(artifact)
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                "pinned runtime artifact has %d bytes, expected %d: %s"
                % (actual_bytes, expected_bytes, artifact))
    refs_dir = os.path.join(repo_dir, "refs")
    os.makedirs(refs_dir, exist_ok=True)
    final = os.path.join(refs_dir, "main")
    part = final + ".part"
    try:
        with open(part, "wb") as handle:
            handle.write(revision.encode("ascii"))
        os.replace(part, final)
    finally:
        try:
            if os.path.exists(part):
                os.unlink(part)
        except OSError:
            pass


def main() -> int:
    model_dir = _default_model_dir()
    print(f"[idx-weights] repo:   {_REPO_ID}")
    print(f"[idx-weights] target: {model_dir}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[idx-weights] huggingface_hub is missing -- run this with the "
              "isolated index-tts venv python "
              "(see scripts/_otr_indextts2_install.ps1)")
        return 2

    try:
        snapshot_download(
            _REPO_ID, revision=_REPO_REVISION, local_dir=model_dir)
        cache_dir = os.path.join(model_dir, "hf_cache")
        os.makedirs(cache_dir, exist_ok=True)
        for repo_id, revision in _RUNTIME_REPOS:
            snapshot_download(repo_id, revision=revision, cache_dir=cache_dir)
            _pin_cache_ref(cache_dir, repo_id, revision)
    except Exception as exc:  # noqa: BLE001 - command boundary must return nonzero
        print(f"[idx-weights] FAILED download: {type(exc).__name__}: {exc}")
        return 1

    failures = validate_model_dir(model_dir)

    if failures:
        print("[idx-weights] FAILED validation:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("[idx-weights] OK -- all expected artifacts present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
