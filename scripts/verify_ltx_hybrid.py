# verify_ltx_hybrid.py
# Hybrid loader check for LTX-Video on Jeffrey's RTX 5080 / Blackwell stack.
#
# The single-file LTX checkpoint ships with transformer+vae weights only;
# the T5 text encoder lives in a separate sharded repo. This script uses
# the existing local t5xxl_fp16.safetensors in ComfyUI's text_encoders
# folder to build an in-memory T5EncoderModel, then hands it into
# the LTX pipeline constructor alongside the 8.73 GB checkpoint.
#
# Steps (all inside main() to keep module-scope clean):
#   1. Build T5 config from local snapshot text_encoder subfolder
#   2. Instantiate empty T5EncoderModel on CPU at float16
#   3. Load state dict from t5xxl_fp16.safetensors (strict=False)
#   4. Build T5 tokenizer from local snapshot tokenizer subfolder
#   5. Construct LTX pipeline from single file with injected T5 + tokenizer
#
# Exits 0 when the pipe constructs, transformer params are populated, and
# the text_encoder is attached. Everything stays on CPU to keep VRAM free.

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

SINGLE_FILE = Path(
    r"C:\Users\jeffr\Documents\ComfyUI\models\checkpoints\ltx-video-2b-v0.9.safetensors"
)
T5_FILE = Path(
    r"C:\Users\jeffr\Documents\ComfyUI\models\text_encoders\t5xxl_fp16.safetensors"
)
SNAPSHOT_DIR = Path(
    r"C:\Users\jeffr\Documents\ComfyUI\models\huggingface\hub"
    r"\models--Lightricks--LTX-Video\snapshots"
    r"\8984fa25007f376c1a299016d0957a37a2f797bb"
)


def banner(s: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {s}")
    print("=" * 72)


def main() -> int:
    banner("paths")
    print(f"single_file: {SINGLE_FILE}  exists={SINGLE_FILE.exists()}")
    print(f"t5_file:     {T5_FILE}      exists={T5_FILE.exists()}")
    print(f"snapshot:    {SNAPSHOT_DIR} exists={SNAPSHOT_DIR.exists()}")

    for p in (SINGLE_FILE, T5_FILE, SNAPSHOT_DIR):
        if not p.exists():
            print(f"FATAL: missing {p}")
            return 2

    try:
        import torch
        from transformers import T5Config, T5EncoderModel, T5Tokenizer
        from diffusers import LTXImageToVideoPipeline
        from safetensors.torch import load_file
    except Exception as exc:
        print(f"FAIL imports: {exc}")
        traceback.print_exc()
        return 3

    print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}")

    # Step 1 -- build T5 config
    banner("step 1: T5Config.from_pretrained")
    t0 = time.time()
    try:
        t5_cfg = T5Config.from_pretrained(
            str(SNAPSHOT_DIR), subfolder="text_encoder"
        )
        print(f"  ok  d_model={t5_cfg.d_model}  num_layers={t5_cfg.num_layers} "
              f"  vocab={t5_cfg.vocab_size}  ({time.time()-t0:.2f}s)")
    except Exception:
        traceback.print_exc()
        return 4

    # Step 2 -- instantiate empty T5EncoderModel
    banner("step 2: T5EncoderModel(config)")
    t0 = time.time()
    try:
        t5 = T5EncoderModel(t5_cfg).to(torch.float16)
        n_params = sum(p.numel() for p in t5.parameters())
        print(f"  ok  params={n_params:,}  ({time.time()-t0:.2f}s)")
    except Exception:
        traceback.print_exc()
        return 5

    # Step 3 -- load weights from single safetensors
    banner("step 3: load_state_dict from t5xxl_fp16.safetensors")
    t0 = time.time()
    try:
        state = load_file(str(T5_FILE))
        missing, unexpected = t5.load_state_dict(state, strict=False)
        print(f"  ok  state_keys={len(state)}  missing={len(missing)} "
              f"  unexpected={len(unexpected)}  ({time.time()-t0:.2f}s)")
        if missing:
            print(f"  missing sample: {missing[:3]}")
        if unexpected:
            print(f"  unexpected sample: {unexpected[:3]}")
        # A reasonable T5 encoder has ~219 tensors; tolerate up to 2 missing
        # (embed_tokens can tie to shared.weight).
        if len(missing) > 3:
            print("  WARN many missing keys -- T5 may be partially loaded")
    except Exception:
        traceback.print_exc()
        return 6

    # Step 4 -- tokenizer
    banner("step 4: T5Tokenizer.from_pretrained")
    t0 = time.time()
    try:
        tok = T5Tokenizer.from_pretrained(
            str(SNAPSHOT_DIR), subfolder="tokenizer"
        )
        print(f"  ok  vocab_size={tok.vocab_size}  "
              f"({time.time()-t0:.2f}s)")
    except Exception:
        traceback.print_exc()
        return 7

    # Step 5 -- pipeline
    banner("step 5: LTXImageToVideoPipeline.from_single_file")
    t0 = time.time()
    try:
        pipe = LTXImageToVideoPipeline.from_single_file(
            str(SINGLE_FILE),
            text_encoder=t5,
            tokenizer=tok,
            torch_dtype=torch.bfloat16,
        )
        elapsed = time.time() - t0
        print(f"  ok  type={type(pipe).__name__}  ({elapsed:.1f}s)")
        # Sanity-check components
        tx = getattr(pipe, "transformer", None)
        vae = getattr(pipe, "vae", None)
        te = getattr(pipe, "text_encoder", None)
        tx_n = sum(p.numel() for p in tx.parameters()) if tx is not None else 0
        vae_n = sum(p.numel() for p in vae.parameters()) if vae is not None else 0
        te_n = sum(p.numel() for p in te.parameters()) if te is not None else 0
        print(f"  transformer params:  {tx_n:,}")
        print(f"  vae params:          {vae_n:,}")
        print(f"  text_encoder params: {te_n:,}")
        if tx_n < 1e9:
            print("  WARN transformer looks under-populated (expected >=1.8B)")
            return 8
        if te_n < 1e9:
            print("  WARN text_encoder looks under-populated (expected ~4.7B)")
            return 9
    except Exception:
        traceback.print_exc()
        return 10

    banner("RESULT")
    print("  HYBRID LOAD OK")
    print(f"  use: from_single_file({SINGLE_FILE.name}) "
          f"+ text_encoder from {T5_FILE.name} "
          f"+ tokenizer from {SNAPSHOT_DIR.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
