"""Vocabulary for the image-gen platform (C1): the granularity modes the
director offers per role.

Stdlib only, so importing it keeps the cold-import invariant (V-12: no torch /
transformers / diffusers at import).
"""
from __future__ import annotations


#: Image generation granularity. ``per_object`` REUSES one image per
#: character/prop/announcer (maps to mesh-once-per-character for 3D); ``per_beat``
#: generates a FRESH image per beat. Any role on the 3D Model Renderer is
#: hard-locked to ``per_object`` (per_beat -> mesh-rebuild-per-beat; BANNED).
GRANULARITY_MODES: tuple = ("per_object", "per_beat")

# The enforced input-token vocabulary is
# ``nodes/_otr_shared/role_compat.INPUT_TOKENS`` ({text_prompt, init_image,
# audio_ref, base_clip_ref}), checked at role_compat.py's
# ``required_set <= INPUT_TOKENS`` gate. One vocabulary, one owner -- import
# from ``role_compat`` if you need the token set; do not keep a subset here.
