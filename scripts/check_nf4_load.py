"""NF4 load check for Mistral-Nemo against the canonical HF_HOME cache."""
from __future__ import annotations

import os
import sys
import winreg


def _hf_home_from_registry() -> str | None:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            val, _ = winreg.QueryValueEx(key, "HF_HOME")
            return str(val).strip() or None
    except Exception:
        return None


def main() -> int:
    hf_home = _hf_home_from_registry()
    print(f"HF_HOME (HKCU): {hf_home!r}")
    if hf_home:
        os.environ["HF_HOME"] = hf_home
        os.environ["HF_HUB_CACHE"] = hf_home

    snap_root = os.path.join(
        hf_home or r"C:\ComfyUI-Models\huggingface",
        "hub",
        "models--mistralai--Mistral-Nemo-Instruct-2407",
        "snapshots",
    )
    snap_id = next(
        (d for d in os.listdir(snap_root) if os.path.isdir(os.path.join(snap_root, d))),
        None,
    )
    snap_dir = os.path.join(snap_root, snap_id) if snap_id else None
    print(f"Snapshot dir: {snap_dir}")

    sample = os.path.join(snap_dir, "model-00001-of-00005.safetensors")
    print(f"Sample symlink: {sample}")
    try:
        target = os.readlink(sample)
        abs_target = os.path.normpath(os.path.join(os.path.dirname(sample), target))
        print(f"Symlink target: {target}")
        print(f"Absolute target: {abs_target}")
        print(f"Target exists: {os.path.exists(abs_target)}")
        if os.path.exists(abs_target):
            print(f"Target size: {os.path.getsize(abs_target) / 1024**3:.2f} GB")
    except OSError as exc:
        print(f"readlink failed: {exc}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print()
    print("=" * 70)
    print(" Loading Mistral-Nemo with NF4 quantization (DIRECT SNAPSHOT PATH) ")
    print("=" * 70)

    torch.cuda.empty_cache()
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"VRAM before: {free_b/1024**3:.2f} GiB free / {total_b/1024**3:.2f} GiB total")

    qc = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print(f"Loading from: {snap_dir}")
    m = AutoModelForCausalLM.from_pretrained(
        snap_dir,
        quantization_config=qc,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
        attn_implementation="sdpa",
    )

    mem_alloc = torch.cuda.memory_allocated() / 1024**3
    mem_reserv = torch.cuda.memory_reserved() / 1024**3
    print(f"VRAM after load: {mem_alloc:.2f} GiB allocated, {mem_reserv:.2f} GiB reserved")

    nf4_modules = sum(
        1 for _n, mod in m.named_modules() if mod.__class__.__name__ == "Linear4bit"
    )
    linear_total = sum(
        1 for _n, mod in m.named_modules() if "Linear" in mod.__class__.__name__
    )
    print(f"4-bit Linear modules: {nf4_modules} / {linear_total} total")

    tok = AutoTokenizer.from_pretrained(snap_dir)
    inp = tok("Once upon a time", return_tensors="pt").to("cuda")
    print(f"Prefill input_ids shape: {tuple(inp.input_ids.shape)}")
    with torch.no_grad():
        out = m.generate(
            **inp,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"Generated: {text!r}")
    mem_after_gen = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM after generate: {mem_after_gen:.2f} GiB")

    print()
    print("=" * 70)
    print(" VERDICT ")
    print("=" * 70)
    if mem_alloc < 10.0 and nf4_modules > 0:
        print(f"PASS: NF4 working ({mem_alloc:.2f} GiB; expected ~6 GiB)")
        return 0
    elif nf4_modules == 0:
        print(f"FAIL: No Linear4bit -- model loaded at fp16/bf16 ({mem_alloc:.2f} GiB)")
        return 1
    else:
        print(f"AMBIGUOUS: NF4={nf4_modules}, VRAM={mem_alloc:.2f} GiB")
        return 1


if __name__ == "__main__":
    sys.exit(main())
