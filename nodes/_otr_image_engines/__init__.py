"""OTR image-engine namespace (the model-agnostic image-gen adapter registry).

Subproject C, one level UP from the video platform: every image generator
(Flux gen-1, and later Z-Image / Klein / any "+ Add Custom Model") is a
pluggable adapter the user selects PER ROLE -- "Flux" is just gen 1, never a
hardcoded default. Swap the image-gen and nothing downstream changes (A's video
init-image / portrait + B's portrait->mesh resolve by ledger, not by engine).

A THIRD parallel namespace beside ``nodes/_otr_audio_engines`` (frozen) and
``nodes/_otr_video_engines`` -- same proven pattern (AS-4) via the dep-free
:mod:`nodes._otr_shared.engine_registry_base`, its OWN registry dict, zero
cross-pollution. The image protocol is a REDUCED ``prompt -> image`` set (no
``canonicalize``), mirroring the SHIPPED ``AudioEngine(Protocol)`` core.

Import-time is cold-import-clean (invariant V-12): importing this package +
:mod:`registry` + :mod:`schemas` + any adapter descriptor pulls in NO heavy lib
(torch / transformers / diffusers). Adapters lazy-import their frameworks inside
``load`` / ``render_image``, never at module scope. The Flux gen-1 adapter is
imported here (guarded) so it self-registers on package import while staying
cold-import clean.

---------------------------------------------------------------------------
HOW TO ADD YOUR OWN IMAGE ENGINE (same shape as the other three namespaces):
---------------------------------------------------------------------------

1. Copy an adapter close to your model's shape -- ``z_image_turbo.py`` for a
   local diffusion checkpoint, ``eng_cloud_image.py``'s ``_CloudImageBase``
   for a partner-API row -- to ``<yourname>.py``, rename the class and its
   ``name`` field, decorate with ``@register`` from :mod:`registry`. There is
   no validated-subset filter: registering IS joining the dropdown (C4).

2. DECLARE, never inherit-by-silence (docs/IMAGE_GEN_PREFLIGHT.md, Gate IG1):
     name              -- the dropdown id, permanent once shipped.
     engine_version    -- str. Part of the still cache key
                          (role, object_id, prompt_hash, seed, engine_id,
                          engine_version). The dispatcher falls back to "1"
                          for a silent engine, which means you could never
                          invalidate your own cached stills; declare it and
                          bump it whenever output should stop being reused.
     commercial_clean  -- a real bool. flux_gen1 is False (BFL
                          non-commercial); the Apache locals are True.
     roles             -- all three ("announcer_visual", "music_visual",
                          "character_video"): the dropdown offers every
                          engine in every slot, so serving fewer would fail
                          at render after the episode is written and voiced.
     required_inputs   -- ("text_prompt",), the reduced prompt->image
                          contract every still composer writes against.

3. Implement the lifecycle: ``assert_usable`` (raise a NAMED error when your
   weights/key are absent -- never let render discover it), ``prepare``,
   ``render_image`` (lazy-import torch/comfy INSIDE it), ``teardown``.

4. Add a CAPABILITIES row in :mod:`registry` (one row per engine and vice
   versa -- tests/test_capability_profiles.py holds the bijection).

5. Run the preflight: docs/IMAGE_GEN_PREFLIGHT.md, enforced by
   tests/test_image_gen_preflight_matrix.py, which sweeps the LIVE registry --
   your engine is covered the moment it registers, no test edits needed.
   If your provider can REFUSE a prompt, read Gate IG4 first: a refusal can
   arrive as a normal SUCCESS with a valid PNG (measured on Ideogram 4,
   2026-08-21), so classify it in the adapter rather than trusting status.
"""
from __future__ import annotations

# Register the gen-1 Flux adapter on package import so the image registry is
# non-empty (Flux selectable per role). The adapter is cold-import clean (it
# lazy-imports torch/comfy only inside load/render), so this import pulls in
# nothing heavy. Guarded so a future packaging quirk never breaks the namespace.
try:  # pragma: no cover - trivial guard
    from . import flux_gen1 as _flux_gen1  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# C2 (build-or-NO-GO, default-OFF): the Z-Image-Turbo adapter -- a 2nd image
# engine proving the model-agnostic layer holds >=2 engines. Cold-import clean.
# IN-PROCESS since 2026-06-18 (see z_image_turbo.py header: "No sidecar") --
# its registry row is requires_sidecar: False; do NOT "fix" it back to a
# sidecar from this comment (the old cu128-sidecar wording misled). Separate
# guard so a quirk in one adapter never blocks the other from registering.
try:  # pragma: no cover - trivial guard
    from . import z_image_turbo as _z_image_turbo  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# Lumina-Image 2.0 (native lightweight/Apache-2.0), a model-agnostic image peer
# from the C2 dep/license matrix: registers identically, greyed until its weights
# exist. Its own guard so a quirk never blocks the others -- and proves the
# "+ Add Custom Model" open-set story: the registry grows by dropping in an
# adapter, no other edits.
# (HiDream-I1 was UNREGISTERED 2026-06-29 (C3): a NotImplementedError dark
# scaffold is no longer imported/selectable. Chroma1-HD was DROPPED 2026-06-18:
# a de-restricted/uncensored FLUX finetune OTR will not ship a path to.)
try:  # pragma: no cover - trivial guard
    from . import lumina_image as _lumina_image  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# FLUX.2 Klein (FLUX-family, verify -- plus BUG-070 SageAttention at render time),
# a model-agnostic image peer registered HONESTLY (commercial_clean per its real
# license terms); greyed until its weights exist; own guard.
# (SD 3.5 Large was UNREGISTERED 2026-06-29 (C3): a NotImplementedError dark
# scaffold is no longer imported/selectable.)
try:  # pragma: no cover - trivial guard
    from . import flux2_klein as _flux2_klein  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# Cloud partner STILLS adapters (S1 stills lane, 2026-07-03): flux_pro
# / nano_banana_2 / seedream_2 -- the model-agnostic layer's first CLOUD image
# engines. Register unconditionally with EMPTY default_roles (selectable, never
# automatic; the dropdown pick is the enable). Cold-import clean (the bridge /
# canonicalizer / PIL import lazily inside render_image); own guard so a quirk
# never blocks the local adapters above.
try:  # pragma: no cover - trivial guard
    from . import eng_cloud_image as _eng_cloud_image  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# Direct Google Gemini/Nano Banana BYO API stills adapter. Not a Partner node and
# not local GPU; registers as an external image engine with native=False.
try:  # pragma: no cover - trivial guard
    from . import eng_google_image as _eng_google_image  # noqa: F401
except Exception:  # noqa: BLE001
    pass
