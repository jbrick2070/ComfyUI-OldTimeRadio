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

# (lean-mean 2026-08-22) ``IMAGE_INPUT_TOKENS`` was here and is DELETED. It
# advertised itself as "shared vocabulary with role_compat" and was neither:
# nothing in the repo read it, and the vocabulary it claimed to share was a
# 2-token subset of the 4-token frozenset that is actually enforced --
# ``nodes/_otr_shared/role_compat.INPUT_TOKENS`` ({text_prompt, init_image,
# audio_ref, base_clip_ref}), checked at role_compat.py's
# ``required_set <= INPUT_TOKENS`` gate.
#
# So it was worse than dead: it was a DECOY. An author adding an image engine
# would have consulted it, seen two legal tokens, and had no way to learn that
# the real gate accepts four and lives in another module. One vocabulary, one
# owner -- import from ``role_compat`` if you need the token set.
