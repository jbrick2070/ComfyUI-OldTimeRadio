r"""Offline doctor for the official Gemma4Unified Transformers writer lane.

Loads ``google/gemma-4-12b-it`` directly from ``C:\ComfyUI-Models`` with the
same NF4 policy used by OTR, generates one prose sample, then generates one
lm-format-enforcer constrained JSON receipt. No overlay, LoRA, Ollama,
llama.cpp, sidecar, HTTP request, or port is used.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


REPO_DEFAULT = "google/gemma-4-12b-it"


class DoctorReceipt(BaseModel):
    status: Literal["ready"]
    # lm-format-enforcer 0.11.3 mishandles a numeric JSON-Schema `const`.
    # Production SciFi schemas use bounded ints, not numeric Literals, so the
    # doctor mirrors that real surface and validates the prompted value below.
    count: int = Field(ge=0, le=99)


def _log(message: str) -> None:
    print(message, flush=True)


def _generate(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    prefix_allowed_tokens_fn=None,
) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    kwargs = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if prefix_allowed_tokens_fn is not None:
        kwargs.update({
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
            "num_beams": 1,
        })
    with torch.no_grad():
        output = model.generate(**inputs, **kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(
        output[0][prompt_len:], skip_special_tokens=True,
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=REPO_DEFAULT)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--prompt",
        default="In two sentences, describe a quiet morning at a lighthouse.",
    )
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Running as `python scripts/otr_gemma4_doctor.py` puts scripts/, not the
    # repository root, on sys.path.
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import torch
    import transformers
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    from nodes import _otr_hf_env as hf_env
    from nodes._otr_constrained_generate import (
        get_cached_transformers_schema_constraint,
    )
    from nodes._otr_model_loader import _require_transformers_model_support

    _require_transformers_model_support(args.repo, transformers.__version__)
    if not torch.cuda.is_available():
        raise SystemExit("FATAL: this NF4 doctor requires a CUDA GPU")

    hf_home = hf_env.ensure_hf_home()
    snapshot = hf_env.resolve_snapshot_dir(args.repo, hf_home=hf_home)
    if snapshot is None:
        raise SystemExit(
            f"FATAL: no complete local snapshot for {args.repo!r} under "
            f"{Path(hf_home) / 'hub'}"
        )
    _log(f"transformers={transformers.__version__} torch={torch.__version__}")
    _log(f"weighted snapshot: {snapshot}")

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False,
    )
    if not getattr(tokenizer, "chat_template", None):
        template_path = hf_env.resolve_snapshot_file(
            args.repo, "chat_template.jinja", hf_home=hf_home,
        )
        if template_path is None:
            raise SystemExit("FATAL: no local chat_template.jinja")
        tokenizer.chat_template = Path(template_path).read_text(encoding="utf-8")
        _log(f"chat template: {template_path}")

    config = AutoConfig.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False,
    )
    nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        config=config,
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=nf4,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).eval()
    load_seconds = time.monotonic() - started
    allocated_gib = torch.cuda.memory_allocated() / (1024 ** 3)

    prose = _generate(
        model,
        tokenizer,
        [{"role": "user", "content": args.prompt}],
        max_new_tokens=max(8, args.max_new_tokens),
    )
    if len(prose) < 12 or not any(ch.isalpha() for ch in prose):
        raise RuntimeError(f"incoherent prose probe: {prose!r}")

    cache_entry = {"model": model, "tokenizer": tokenizer}
    _schema_parser, prefix_fn = get_cached_transformers_schema_constraint(
        cache_entry, DoctorReceipt,
    )
    raw_json = _generate(
        model,
        tokenizer,
        [{
            "role": "user",
            "content": (
                'Return exactly one JSON object with status "ready" and '
                "count 12. No prose or Markdown."
            ),
        }],
        max_new_tokens=64,
        prefix_allowed_tokens_fn=prefix_fn,
    )
    receipt = DoctorReceipt.model_validate(json.loads(raw_json))
    if receipt.count != 12:
        raise RuntimeError(f"constrained receipt count drift: {receipt.count}")
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    _log("\n================ GEMMA4 HF DOCTOR ================")
    _log(f"model={type(model).__name__} loaded_in_4bit={model.is_loaded_in_4bit}")
    _log(
        f"load={load_seconds:.2f}s allocated={allocated_gib:.3f} GiB "
        f"peak={peak_gib:.3f} GiB"
    )
    _log(f"prose={prose!r}")
    _log(f"constrained_json={raw_json!r} parsed={receipt.model_dump()!r}")
    _log("RESULT=PASS (official Transformers + NF4 + LMFE, fully offline)")
    _log("==================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
