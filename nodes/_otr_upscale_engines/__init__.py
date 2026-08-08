"""OTR upscale-engine namespace (the model-agnostic post-render upscale/enhance
registry).

A parallel namespace to ``nodes/_otr_audio_engines``, ``nodes/_otr_video_engines``
and ``nodes/_otr_image_engines`` -- same proven pattern (AS-4), its OWN registry
dict, zero cross-pollution with the other three.

Import-time is cold-import clean (invariant V-12): importing this package +
:mod:`registry` + any adapter descriptor pulls in NO heavy lib (torch / spandrel).
Adapters lazy-import their frameworks inside ``load`` / ``upscale_frames``, never
at module scope, so registering them here stays cold-import clean.

---------------------------------------------------------------------------
HOW TO ADD YOUR OWN UPSCALE ENGINE (same shape as the other three namespaces):
---------------------------------------------------------------------------

1. Copy ``eng_off.py`` to ``eng_<yourname>.py`` and rename the class + ``name``
   field. Adapters are plain classes decorated with ``@register`` from
   :mod:`registry` -- inheritance is never required; the shipped ``Protocol``
   ducks-types on the required members.

2. Implement three lifecycle methods:

     def load(self, device: torch.device) -> None:
         # MUST set self.device. Import torch / your model framework LAZILY
         # in here (never at module scope), so a missing dep never breaks
         # namespace import.

     def unload(self) -> None:
         # MUST clear self.device and release the model weights so ComfyUI's
         # VRAM baseline returns near-zero after the composite finishes.

     def upscale_frames(self, frames: torch.Tensor) -> torch.Tensor:
         # frames: BHWC float32 in [0, 1] on self.device.
         # returns: BHWC float32 in [0, 1] on self.device, upscaled by
         # self.intrinsic_scale in each spatial dim (2x, 3x, 4x, ...).

3. Add your engine's CAPABILITIES row in :mod:`registry` -- ``device_backends``
   subset of ``("cuda", "cpu", "mps")``, ``requires_vendor`` None or one of
   ``"nvidia" | "amd" | "apple"``, ``model_requirements`` naming the checkpoint
   asset ids you need. The registry-consistency invariant forbids a CAPABILITIES
   row without a registered engine (and vice-versa) -- ``audit_engine_roster``
   surfaces any drift on boot.

4. If your engine needs a model file, drop it under ``models/upscale_models/``
   (ComfyUI's canonical dir). Add a preflight downloader in
   ``scripts/ensure_upscale_models.py`` (URL + SHA-256 + license note) if you
   want it auto-fetched. The shipped ``spandrel_esrgan`` adapter is the
   reference.

5. Register the engine's dropdown label in the SilentComposite widget: the
   dropdown auto-populates from ``all_engine_names()``, so no widget code
   changes are required. A profile that picks your engine simply names it in
   ``upscale_stage.engine``.

6. Add an ``@register`` import line here in this ``__init__.py`` inside the
   guarded try/except block (mirrors the video and audio patterns). A packaging
   quirk in your module cannot break the namespace import -- ROSTER AUDIT
   below will LOG the missing engine but never raise.

---------------------------------------------------------------------------
DEVICE / VENDOR COVERAGE (as of the 2026-08-08 first ship):
---------------------------------------------------------------------------

* NVIDIA CUDA -- proven on the 5080 during the item-8 ship gate.
* AMD ROCm -- supported by design: torch's ROCm build reports as ``"cuda"``,
  so an AMD user with ROCm-torch installed sets ``device: "cuda"`` in their
  profile and the same adapter code runs. NOT independently proven on this
  Windows/NVIDIA box, but no code changes are needed.
* Apple MPS -- DELIBERATELY DEFERRED. ``"mps"`` is NOT in either shipped
  adapter's ``device_backends``. Adding MPS is a 2-line change (append
  ``"mps"`` to the row + accept the token in ``_resolve.resolve_device``)
  plus a live Mac integration receipt. Without the receipt, the HONEST-SWITCH
  LAW forbids advertising the capability.
"""

# eng_off is the mandatory pass-through / HONEST-SWITCH default. Never guarded
# with try/except because failure to register `off` breaks the shipped design
# (no dropdown default, no byte-identity anchor). If this import raises, the
# roster audit below will surface it, but production should FAIL LOUD sooner
# than that -- the adapter has no dependencies beyond stdlib + the registry.
from . import eng_off as _eng_off  # noqa: F401

# spandrel_esrgan is the first real (non-off) engine. Guarded so a missing
# spandrel install cannot break the namespace import -- the adapter itself
# raises EngineUnusable(MALFORMED_CONFIG) with a pip-install pointer at
# load() time when spandrel is absent.
try:  # pragma: no cover - trivial guard
    from . import eng_spandrel_esrgan as _eng_spandrel_esrgan  # noqa: F401
except Exception:  # noqa: BLE001
    pass


# ---------------------------------------------------------------------------
# ROSTER AUDIT -- runs LAST, after every guarded adapter import above.
# ---------------------------------------------------------------------------
# Mirrors ``nodes/_otr_video_engines/__init__.py:221-261``. Position matters
# and is the whole point: this must sit at the BOTTOM of this module, not
# inside registry.py -- run from inside registry.py it would report every
# not-yet-imported adapter as missing.
#
# LOG, never raise: a box with one broken adapter must still render with
# the other. CI enforcement is a test
# (``tests/test_upscale_engines_registry.py``) which is the right place
# for a hard gate.
try:  # pragma: no cover - trivial guard
    from . import registry as _registry_for_audit

    _roster = _registry_for_audit.audit_engine_roster()
    if _roster["missing"]:
        import logging as _logging

        _logging.getLogger("OTR").error(
            "[OTR upscale] ROSTER AUDIT: %d declared engine(s) FAILED TO "
            "REGISTER: %s -- their adapter import raised and was swallowed "
            "by the guards above, so they are silently absent from the "
            "SilentComposite dropdown. This is a real break, not a warning.",
            len(_roster["missing"]), ", ".join(_roster["missing"]))
    if _roster["unexpected"]:
        import logging as _logging

        _logging.getLogger("OTR").warning(
            "[OTR upscale] ROSTER AUDIT: %d engine(s) registered without "
            "a CAPABILITIES row: %s -- 'registry IS the menu' requires a "
            "row per registered engine.",
            len(_roster["unexpected"]), ", ".join(_roster["unexpected"]))
except Exception:  # noqa: BLE001 -- the audit must never break the import
    pass
