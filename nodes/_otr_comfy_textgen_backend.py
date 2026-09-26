"""A Comfy-native Gemma 4 writer: ComfyUI's own text model writing the script.

WHAT IT IS (plan row 0n; design campaign kibitz-runs/2026-09-26-comfy-gemma-writer).
ComfyUI ships an official ``llm_gemma4_text_gen`` template -- the stock CLIPLoader
loading Comfy-Org's ``gemma4_e2b_it_int8_convrot`` into the core TextGenerate node.
The model then lives inside ComfyUI's own model management, which loads and unloads
it like any other model (operator, 2026-09-26: "unload models after they are used").
Measured the same day: 82-125 tok/s on the RTX 5080 and a 5.1 GB peak on the 8 GB
RTX 4060, against 16-18 tok/s for today's 12B NF4 transformers writer.

BUILT IN SLICES, and this file grows with them. Slice A2 lands the identity and the
one weight the writer needs, so the queue-time preflight can fetch that weight before
the writer runs. Generation (A1) and the ``request_slot`` wiring (A3) follow. Until A3
lands, no dropdown offers :data:`MODEL_ID`, so nothing reaches this module at runtime
except the preflight's lookup, which answers "no weights" for every other id.

IMPORT-LIGHT ON PURPOSE: stdlib only at module scope. The preflight and the catalog
ask it questions while building a queue plan; neither may pay for torch or ComfyUI.
"""
from __future__ import annotations

#: The dropdown id. Virtual (not a Hugging Face repo id): the weight comes from
#: Comfy-Org as one ComfyUI text-encoder file, not a transformers snapshot, and the
#: HF ``google/gemma-4-E2B-it`` row stays a separate, unchanged choice.
MODEL_ID = "comfy_native:gemma4-e2b-it-int8-convrot"

#: The ComfyUI model category the stock CLIPLoader reads, and the file in it.
WEIGHT_CATEGORY = "text_encoders"
WEIGHT_TOKEN = "gemma4_e2b_it_int8_convrot.safetensors"

#: The exact upstream file, read from the Hub API on 2026-09-26 without a token
#: (the repo is ungated). ``_otr_visual_assets._PINNED_SOURCES`` carries the same
#: row; tests/test_comfy_textgen_backend.py holds the two in agreement.
WEIGHT_REPO = "Comfy-Org/gemma-4"
WEIGHT_FILENAME = "text_encoders/gemma4_e2b_it_int8_convrot.safetensors"
WEIGHT_REVISION = "63d0f7c476756b88910170c1df75e2384ea1af31"
WEIGHT_SIZE = 5_199_997_904
WEIGHT_SHA256 = "efeca0fcad2f863e5ed0a75e3af952b72bc963604c1dda6d20aee87a32b17566"

#: The working context this writer is qualified at. The decoder advertises a much
#: larger native window (below); the working cap is what an 8 GB card is asked to
#: hold, and a later probe decides whether it can rise. Kept separate on purpose so
#: neither number is ever read as the other.
WORKING_CONTEXT_CAP = 8192
NATIVE_CONTEXT_CAPACITY = 131072

_WEIGHTS_BY_MODEL = {MODEL_ID: ((WEIGHT_CATEGORY, WEIGHT_TOKEN),)}


def native_writer_weights(model_id) -> tuple:
    """``((category, token), ...)`` a Comfy-native writer id needs, else ``()``.

    Exact match on the normalized id only: a label, a badge or an HF repo id that
    merely resembles this one needs nothing from here."""
    return _WEIGHTS_BY_MODEL.get(str(model_id or "").strip(), ())


def is_native_writer(model_id) -> bool:
    return bool(native_writer_weights(model_id))
