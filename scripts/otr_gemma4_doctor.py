r"""otr_gemma4_doctor.py -- prove google/gemma-4-12b-it loads + generates SANE
text on the EXISTING transformers 5.5 stack, TEXT-ONLY, NF4, no sidecar.

WHY: BUG-306. The model's config declares `model_type: gemma4_unified`
(transformers 5.10-dev naming) which 5.5 does not register, so the default
AutoModel load fails. But (proven offline, 2026-06-03):
  * transformers 5.5 `Gemma4TextConfig` ACCEPTS every gemma4_unified text field
    (0 swallowed as kwargs) -- 5.5 already models this text architecture;
  * the checkpoint text tower (`model.language_model.*`, 666 tensors) maps onto
    5.5 `Gemma4ForCausalLM` with 0 unexpected keys / 0 shape issues / 1 tied-head
    artifact (`lm_head.weight`, auto-filled by tie_word_embeddings=True);
  * NF4 (bitsandbytes) is proven on this GPU (mistral-nemo NF4 @ 7.74 GiB).
This checkpoint is a TEXT model with only thin vision/audio adapter tensors (no
full towers), so the multimodal wrapper is the WRONG class -- the text-only
`Gemma4ForCausalLM` is correct. The only adaptation: the checkpoint stores the
text weights under `model.language_model.*`; we remap to `model.*` for the
text-only class. The single thing not provable offline is runtime OUTPUT quality;
this doctor closes that gate.

Does NOT touch the protected venv or the canonical model cache: it writes a small
OVERLAY dir (rewritten text config + a remapped TEXT-ONLY safetensors + hardlinked
tokenizer) under the OS temp dir, and loads from there. The overlay is cached
across runs.

Run (Windows):
  C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\otr_gemma4_doctor.py

Flags: --repo, --max-new-tokens, --prompt, --build-only, --rebuild
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO_DEFAULT = "google/gemma-4-12b-it"
_TEXT_PREFIX = "model.language_model."


def _log(msg: str) -> None:
    print(msg, flush=True)


def find_snapshot(repo: str) -> Path:
    roots = []
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots += [
        Path(r"C:\ComfyUI-Models\huggingface\hub"),
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    folder = "models--" + repo.replace("/", "--")
    for root in roots:
        base = root / folder / "snapshots"
        if base.is_dir():
            snaps = [p for p in base.iterdir() if p.is_dir()]
            if snaps:
                return max(snaps, key=lambda p: p.stat().st_mtime)
    raise SystemExit(f"FATAL: no local snapshot for {repo!r} under {roots}")


def build_overlay(snapshot: Path, rebuild: bool = False) -> Path:
    """Overlay = text-only config + remapped TEXT-ONLY safetensors + hardlinked
    tokenizer. Cached across runs. Never mutates the canonical snapshot."""
    overlay = Path(tempfile.gettempdir()) / "otr_gemma4_textonly_overlay"
    weights = overlay / "model.safetensors"
    if not rebuild and weights.exists() and (overlay / "config.json").exists():
        _log(f"reusing cached overlay: {overlay}")
        return overlay
    if overlay.exists():
        shutil.rmtree(overlay, ignore_errors=True)
    overlay.mkdir(parents=True, exist_ok=True)

    # Text-only config: promote text_config to the top level, model_type
    # gemma4_text -> AutoModelForCausalLM builds Gemma4ForCausalLM directly.
    cfg = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    text_cfg = dict(cfg.get("text_config", {}))
    text_cfg["model_type"] = "gemma4_text"
    text_cfg["architectures"] = ["Gemma4ForCausalLM"]
    text_cfg.setdefault("tie_word_embeddings", cfg.get("tie_word_embeddings", True))
    (overlay / "config.json").write_text(json.dumps(text_cfg, indent=2), encoding="utf-8")

    # Tokenizer + aux files (NOT weights): hardlink the RESOLVED real blob
    # (HF snapshot entries are symlinks to ../../blobs/<hash>).
    for name in os.listdir(snapshot):
        if name == "config.json" or name.endswith(".safetensors"):
            continue
        src = Path(os.path.realpath(snapshot / name))
        if not src.is_file():
            continue
        try:
            os.link(src, overlay / name)
        except OSError:
            shutil.copy2(src, overlay / name)

    # Remapped TEXT-ONLY weights: model.language_model.* -> model.* ; drop the
    # thin vision/audio adapter tensors. ~21 GB write, done once (cached).
    from safetensors.torch import load_file, save_file
    _log("remapping text-only weights (model.language_model.* -> model.*) ...")
    src_w = Path(os.path.realpath(snapshot / "model.safetensors"))
    sd = load_file(str(src_w))
    text_sd = {
        "model." + k[len(_TEXT_PREFIX):]: v
        for k, v in sd.items()
        if k.startswith(_TEXT_PREFIX)
    }
    save_file(text_sd, str(weights), metadata={"format": "pt"})
    _log(f"  wrote {len(text_sd)} text tensors ({len(sd) - len(text_sd)} "
         f"adapters dropped) -> {weights}")
    del sd, text_sd
    return overlay


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument(
        "--prompt",
        default="In two sentences, describe a quiet morning at a lighthouse.",
    )
    ap.add_argument("--build-only", action="store_true",
                    help="CPU-only: build overlay + confirm key match, no GPU load.")
    ap.add_argument("--rebuild", action="store_true",
                    help="Force-rebuild the cached overlay.")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    _log(f"transformers={transformers.__version__} torch={torch.__version__} "
         f"cuda_avail={torch.cuda.is_available()}")

    snapshot = find_snapshot(args.repo)
    _log(f"snapshot: {snapshot}")
    overlay = build_overlay(snapshot, rebuild=args.rebuild)
    _log(f"overlay:  {overlay}")

    if args.build_only:
        import struct as _struct
        from transformers import AutoConfig
        ac = AutoConfig.from_pretrained(overlay, local_files_only=True)
        _log(f"AutoConfig -> {type(ac).__name__} (model_type={ac.model_type})")
        with torch.device("meta"):
            m = AutoModelForCausalLM.from_config(ac)
        _log(f"AutoModelForCausalLM.from_config -> {type(m).__name__}")
        mk = set(m.state_dict().keys())
        with open(overlay / "model.safetensors", "rb") as f:
            n = _struct.unpack("<Q", f.read(8))[0]
            ck = set(json.loads(f.read(n))) - {"__metadata__"}
        missing = sorted(mk - ck)
        unexpected = sorted(ck - mk)
        _log(f"model_keys={len(mk)} overlay_keys={len(ck)} "
             f"MISSING={missing[:6]} UNEXPECTED={unexpected[:6]}")
        ok = (not unexpected) and all(x == "lm_head.weight" for x in missing)
        _log(f"BUILD_ONLY_OK={ok}")
        return 0 if ok else 3

    nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    _log("loading (NF4, text-only)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            overlay,
            quantization_config=nf4,
            dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            local_files_only=True,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        _log(f"LOAD_FAILED: {type(e).__name__}: {e}")
        return 2
    load_s = time.time() - t0
    tok = AutoTokenizer.from_pretrained(overlay, local_files_only=True)

    msgs = [{"role": "user", "content": args.prompt}]
    try:
        inputs = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
    except Exception:
        inputs = tok(args.prompt, return_tensors="pt").input_ids.to(model.device)

    t1 = time.time()
    out = model.generate(inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    gen_s = time.time() - t1
    text = tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
    peak = (torch.cuda.max_memory_allocated() / (1024 ** 3)
            if torch.cuda.is_available() else 0.0)

    _log("\n================ GEMMA4 DOCTOR RESULT ================")
    _log(f"  load OK in {load_s:.1f}s   generate {gen_s:.1f}s "
         f"({args.max_new_tokens} tok)   peak VRAM {peak:.2f} GB (ceiling 14.5)")
    _log(f"  prompt: {args.prompt!r}")
    _log(f"  OUTPUT: {text!r}")
    _log("  JUDGE: coherent English (no repeat/sentinel spam, terminates) "
         "=> PROMOTE gemma-4-12b to a writer option.")
    _log("=====================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
