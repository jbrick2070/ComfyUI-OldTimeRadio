# Bark clean-break (sprint 1a) -- decision brief + ready-to-land design

**Date:** 2026-06-03 - **Branch:** v2.0-alpha - **HEAD after this work:** `bf10ef7` (pushed, suite 3727/12/0)

## TL;DR
I did the **safe, headless half of 1a** and stopped before the consequential half.
Done + pushed: the pure Bark per-line inference (`_clean_text_for_bark`,
`_chunk_text_for_bark`, `_generate_single_line`) is relocated byte-exactly into
`_otr_bark_lib.py`, re-exported from `batch_bark_generator.py`, so `eng_bark` can
become delegation-free. Zero behavior change.

I did **not** flip `bark` to `per_line` or delete `batch_bark_generator.py`,
because that step is blocked on **two operator-only gates** and **one design
decision**. All three are spelled out below with ready-to-apply code.

## Why I stopped (the blast radius is wider than "extract + delete")
`batch_bark_generator.py` is not a thin wrapper. Beyond raw inference it carries:

1. **Freeze-halt safety gate (BUG-276/300).** Enforced ONLY inside
   `BatchBarkGenerator.generate_batch`. No v2 node re-homes it -- I checked
   `cast_lock.py`, `scene_sequencer.py`, `_otr_freeze_cascade.py` (the cascade
   only *stamps* `meta.freeze_verdict`; the bark node *enforces* it). Today the
   v2 lane still runs this gate via batch delegation. Deleting the node **drops a
   tested production safety gate**. Pinned by `tests/test_bark_freeze_halt_bypass.py`.
2. **Per-line ledger timing write-back (BUG-096 `dur_s`/`start_s`).** This one is
   **redundant and safe to drop** -- `scene_sequencer.py:768-903` already writes
   authoritative `start_s`/`dur_s` from the *actual assembled audio* (more
   accurate than Bark's pre-assembly per-clip durations). Pinned by
   `tests/test_bark_ledger.py`, which would be deleted with the node.

Plus: flipping `bark` to `per_line` is **not byte-identical** to the batch path
(the batch path groups by preset + length-sorts; per-line runs in script order).
The clean-break directive already says the new reference is `baseline_v2`
captured from the new engines -- which is an **operator GPU capture I cannot do
headless**.

## What needs YOU (gates)
### Gate A -- decide where the freeze-halt gate lives in the v2 lane
Options:
- **(Recommended) Move it into `OTR_CastLock`.** CastLock is the "single ledger
  authority," runs first in the audio chain (CastLock -> CharacterVoices ->
  Announcer -> Theme), and already reads the ledger. Gating there protects the
  whole audio chain in one place, not just bark.
- Put it in the shared `OTRVoiceNodeBase.generate` (gates each voice node).
- Drop it (rely on CastLock's cast contract + the per-line `v2/*` preset check).
  I do **not** recommend silently dropping a safety gate.

The per-node `bypass_freeze_halt` **widget** cannot move as-is (v2 nodes forbid
extra widgets, plan E.4). It becomes an **env var** `OTR_BYPASS_FREEZE_HALT=1`
(matches the OTR "every knob is an env var" convention; `OTR_BARK_HALT_ON_QUALITY_BLOCK`
already follows it). Proposed code in the appendix.

### Gate B -- capture `baseline_v2` for bark on the GPU
Flipping bark to per-line changes the live audio path. Per the clean-break
directive, the new reference is a render-twice `baseline_v2` from the new engine.
That capture is yours (RTX 5080). Until it exists, the bark byte-identity tests
become headless **contract** pins (shape/SR/fail-closed/preset-routing), not
byte pins -- I'll wire them that way, but you own the byte reference.

## What does NOT need a decision (already designed, ready to land)
### Voice-preset routing (solved)
`_render_per_line` calls `generate_voice(prepared, ref_clip_path, None, seed)` and
never forwards `voice_preset`. Bark needs the preset, not a clip. Fix without
touching chatterbox/indextts2: add a `voice_ref_field` attribute per adapter and
have the dispatch read the right cast field into the existing ref slot. Code in
the appendix.

### eng_bark per-line body (ready)
Self-contained, sources inference from `_otr_bark_lib`, preserves the
`[clears throat]` first-line anti-hallucination guard by tracking
`self._presets_started` across the per-line loop (same first-line set as the
grouped batch path, since it keys on first occurrence of each preset). Code in
the appendix.

## Remaining 1a steps once Gate A + B are settled
1. Apply the `voice_ref_field` dispatch patch (appendix A).
2. Land the `eng_bark` per-line body (appendix B); flip `interface` to `per_line`.
3. Re-home the freeze-halt per Gate A (appendix C).
4. Delete `batch_bark_generator.py` + remove `OTR_BatchBarkGenerator` from
   `__init__.py` registration, `_otr_legacy_manifest.LEGACY_AUDIO_NODES`, and
   `config/legacy_invocation_manifest.json` (bark entry only -- kokoro/musicgen/
   audiogen stay; they still use batch delegation).
5. Repoint `story_orchestrator._bark_test_presets` (already on `_otr_bark_lib`).
6. Convert/retire the bark tests:
   - `test_batch_character_voices.py`: 4 bark-batch tests -> per-line contract.
   - `test_bark_ledger.py`: retire (feature redundant; scene_sequencer covers timings).
   - `test_bark_freeze_halt_bypass.py`: repoint to the freeze-halt's new home.
   - `test_legacy_audio_seeding.py`: drop the bark parametrization (3 nodes remain).
   - `test_core.py`: `_clean_text_for_bark` import -> `_otr_bark_lib` (it currently
     resolves via the re-export; after deletion, repoint).
   - `test_audio_byte_identical.py` FIXED_SEEDS: drop the `OTR_BatchBarkGenerator` key.
7. Add the guard test: `BatchBarkGenerator` / `batch_bark_generator` must not reappear.
8. Full suite green; capture `baseline_v2` (Gate B); commit + push.

## Verify-at-build flags I hit
- `assert_model_available(char_bark_v1)` with empty `model_path` must be lenient
  (Bark loads from HF cache, not a `model_path` file) -- confirm it does not raise
  `MISSING_MODEL` on the per-line bark path. (chatterbox/indextts2 dodge this via
  the zero-line short-circuit in tests.)
- `_render_per_line` builds the `ResolvedVoiceRequest` with `channels=1` and packs
  `mono_safe` -- matches Bark's mono `[1,1,T]`. Good.

---

## Appendix A -- dispatch patch (`_otr_voice_node_common.py`, `_render_per_line`)
Replace the ref-path extraction + call:
```python
        ref_field = getattr(adapter, "voice_ref_field", "voice_ref_path")
        # bark routes its discrete preset through the same positional ref slot;
        # cloning engines (chatterbox/indextts2) keep the clip path.
        ...
        for occ, ln in enumerate(lines):
            ...
            cast = _OTRLC.cast_lookup(led, char_id)
            voice_ref_id = cast.get("voice_ref_id")
            voice_preset = cast.get("voice_preset")
            if ref_field == "voice_ref_path":
                voice_ref = cast.get("voice_ref_path") or cast.get("ref_path")
            else:
                voice_ref = cast.get(ref_field)
            ...
            with deterministic_inference(engine_seed, warn_only=True):
                audio = adapter.generate_voice(prepared, voice_ref, None, engine_seed)
```
Non-breaking: chatterbox/indextts2 omit `voice_ref_field` -> default
`"voice_ref_path"` -> identical behavior.

## Appendix B -- `eng_bark.py` (self-contained per_line)
```python
"""Bark character-voice adapter -- self-contained per_line (clean-break 1a).

Sources inference from _otr_bark_lib (relocated, delegation-free); no construction
of the heavy batch node. interface == "per_line". Library imports are lazy so
importing the registry package stays light (C-5).
"""
from __future__ import annotations

from .registry import register


@register
class BarkEngine:
    name = "bark"
    roles = ("char_voice",)
    default_roles = ("char_voice",)     # internal default until promotion (I)
    commercial_clean = False            # Suno Bark terms not confirmed commercial
    requires_flag = None
    interface = "per_line"
    sample_rate = 24000
    supports_external_generator = False  # Bark.generate binds no external Generator
    voice_ref_field = "voice_preset"     # dispatch routes cast.voice_preset to the ref slot

    def __init__(self):
        self._loaded = False
        self._presets_started = set()    # first-line anti-hallucination guard tracking

    def load(self):
        if self._loaded:
            return
        from .._otr_bark_lib import _load_bark
        _load_bark("suno/bark")
        self._loaded = True

    def unload(self):
        self._loaded = False
        self._presets_started = set()    # reset so the next episode re-guards first lines
        try:
            from .._otr_bark_lib import _unload_bark
            _unload_bark()
        except Exception:
            pass

    def prepare_text(self, text, delivery_vector=None):
        from .._otr_bark_lib import _clean_text_for_bark
        return _clean_text_for_bark(text)

    def generate_voice(self, text, voice_preset, delivery_vector, seed):
        """One character line -> mono AUDIO {"waveform":[1,1,T], "sample_rate"}.

        voice_preset (e.g. "v2/en_speaker_3") arrives via voice_ref_field. Runs
        inside the caller's deterministic_inference wrap; Bark binds no external
        Generator. Preserves the [clears throat] first-line guard per preset.
        """
        import numpy as np
        import torch
        from .._otr_bark_lib import _generate_single_line, _load_bark
        from .registry import EngineUnusable, EngineUsabilityReason

        if not voice_preset or not str(voice_preset).startswith("v2/"):
            raise EngineUnusable(
                self.name, "char_voice", EngineUsabilityReason.MALFORMED_CONFIG,
                f"bark requires a v2/* voice_preset; got {voice_preset!r}",
            )
        model, processor = _load_bark("suno/bark")
        self._loaded = True
        is_first = voice_preset not in self._presets_started
        self._presets_started.add(voice_preset)
        audio_np, sr = _generate_single_line(
            text, voice_preset, model, processor,
            temperature=0.7, is_first_line=is_first,
        )
        wav = torch.from_numpy(np.asarray(audio_np, dtype=np.float32)).reshape(1, 1, -1)
        return {"waveform": wav, "sample_rate": int(sr)}
```
Note: `temperature=0.7` is the `char_bark_v1` profile default. If you want the
per-line path to honor `profile.default_params['text_temp']`, thread it through
`generate_voice` (the dispatch already has `profile.default_params`); say the word
and I'll wire the param instead of hardcoding.

## Appendix C -- freeze-halt re-home (recommended: `OTR_CastLock`)
Add near the top of CastLock's execute, after it loads the ledger meta:
```python
        verdict = (meta or {}).get("freeze_verdict")
        if verdict == "needs_full_rerun":
            block_class = (meta or {}).get("freeze_block_class")
            bypass = os.environ.get("OTR_BYPASS_FREEZE_HALT", "0") == "1"
            strict_quality = os.environ.get("OTR_BARK_HALT_ON_QUALITY_BLOCK", "0") == "1"
            if bypass:
                log.warning("[CastLock] FREEZE HALT BYPASSED (OTR_BYPASS_FREEZE_HALT=1); "
                            "rendering a flagged ledger. BUG-LOCAL-276.")
            elif block_class == "quality" and not strict_quality:
                log.warning("[CastLock] freeze_verdict=needs_full_rerun, block_class=quality "
                            "-- renderable; proceeding. BUG-LOCAL-300.")
            else:
                raise ValueError(
                    "OTR_CastLock: freeze cascade stamped freeze_verdict="
                    "'needs_full_rerun' (structural). Refusing to cast/render. "
                    "Re-run the writer phase. Set OTR_BYPASS_FREEZE_HALT=1 only for "
                    "sprint-time smoke iteration. BUG-LOCAL-276.")
```
Then `test_bark_freeze_halt_bypass.py` repoints its source pins to `cast_lock.py`
and swaps the widget assertions for the env-var contract.
