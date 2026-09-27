"""The writer's prompt-keyed sampling seed, shared by every local writer backend.

WHY A LEAF. The transformers writer seeds torch from (OTR_WRITER_SEED, this prompt)
so a controlled comparison can reproduce a script; the Comfy-native Gemma writer
(plan row 0n) must derive the SAME number from the same prompt, or a seeded A/B
between the two backends would compare different seeds. One formula, one owner.

KEYED ON THE PROMPT, NOT A CALL COUNTER (the original reasoning, kept with the
formula): an episode makes many generate calls, and a counter would make every
later seed depend on which passes ran. Hashing the call's own input tokens makes
each seed a function of what that call is asked, order-independent.

Pure and stdlib-only: the callers read the environment and apply the seed.
"""
from __future__ import annotations

import zlib

#: Set to an integer to make the writer's SAMPLING reproducible. Unset (the
#: default, and production) leaves generation unseeded.
WRITER_SEED_ENV = "OTR_WRITER_SEED"


def parse_writer_seed(raw) -> "int | None":
    """The integer base seed, or None when ``raw`` is unset, blank or not an int."""
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def prompt_keyed_seed(base: int, nested_ids) -> int:
    """``(base * 1_000_003 + crc32(str(nested_ids))) & 0x7FFF_FFFF``.

    ``nested_ids`` is the batch-shaped id list exactly as the transformers path
    sees it (``inputs["input_ids"].tolist()``, e.g. ``[[2, 105, ...]]``): the
    crc is over its ``str``, so the nesting is part of the contract."""
    digest = zlib.crc32(str(nested_ids).encode("utf-8"))
    return (int(base) * 1_000_003 + digest) & 0x7FFF_FFFF
