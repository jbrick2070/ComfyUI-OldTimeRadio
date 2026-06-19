<!-- requested_model: ~anthropic/claude-opus-latest | resolved_model: anthropic/claude-4.8-opus-20260528 -->

VERDICT: no — the "promote chatterbox" recommendation is a no-op without bank/engine-profile wiring the plan never specifies, and the "kokoro-as-char_voice" option as designed breaks the per-character uniqueness invariant.

MUST-FIX BEFORE BUILD:

1. [Starting recommendation: "promote chatterbox" + Open Q4] Promoting chatterbox to a castable char engine has an unstated, blocking dependency. In `cast_lock.py::_resolve_char_engine`, auto_registry only considers engines where `eng in engines_with_refs = {e.engine for e in bank_entries}`. The operator's curated refs live at `refs/indextts2/vz_*.wav` and are (per the path) namespaced/registered to indextts2. Unless bank entries tagged `engine="chatterbox"` exist AND a `voice_bank` profile lists chatterbox in `allowed_voice_banks`, chatterbox is never selected and the promotion silently does nothing. FIX: add an explicit step to register the vz_* WAVs under the chatterbox engine (or make WAV-clone engines engine-agnostic in the bank) and define the engine profile + allowed_voice_banks that selects it. verify: contents of `_otr_voice_bank` bank entries and `_otr_engine_profiles` profiles.

2. [Starting recommendation: chatterbox selection] There is no specified mechanism for the operator to choose chatterbox *over* indextts2. `_resolve_char_engine` returns the *first* `legacy_first_engines("char_voice")` that has refs and whose profile allows the chosen `voice_bank`; with indextts2 as the promoted default it will win. The CastLock widgets are fixed to `voice_bank`/`cast_voice_policy`/`delivery_profile`/`allow_voice_reuse` (per E.4 docstring) — none selects an engine. FIX: state the concrete selector (a dedicated voice_bank whose profile excludes indextts2, or reorder legacy_first_engines), otherwise "secondary engine" is unreachable.

3. [Open Q2 / kokoro-as-char_voice] The kokoro engine as written cannot satisfy "per-character unique, deterministic." `eng_kokoro.generate_voice` contains a LOCAL-ONLY guard that, when a `voice_ref` has no local `.pt`, silently swaps to `self._episode_voice` — which `begin_episode` always sets from `ANNOUNCER_VOICE_POOL` (4 British announcer voices). For char_voice this collapses every unresolved character onto one shared announcer voice, violating both the uniqueness invariant and role separation. Also `begin_episode` has no char pool at all. FIX (if kokoro char_voice is kept): make the char path fail-closed on a missing preset instead of falling back, and add a separate seeded per-character gendered char pool distinct from the announcer pool. (Recommend cutting instead — see CUT 1.)

SHOULD-FIX:

1. [OTR context: render interface bullet; candidate table indextts2 row] The interface bullet claims engines declare a single `voice_ref_field`, but the code disagrees: `eng_indextts2.py` declares `voice_ref_kind = "wav_path"` (no `voice_ref_field`), while `eng_kokoro.py` declares `voice_ref_field = "voice_ref_id"`. The table's "indextts2 (`voice_ref_id`)" is inaccurate; indextts2's ref is a wav path resolved by `_resolve_ref`. Reconcile the attribute name across engines before any dispatcher reads it uniformly.

2. [Candidate table: kokoro "in-venv (CPU-capable)"] Contradicts the code: `eng_kokoro.load()` hardcodes `KPipeline(..., device="cuda", ...)`. As written kokoro cannot run on CPU. Drop the CPU-capable claim or parametrize device.

3. [Starting recommendation: "Demote bark to last-resort fallback only"] Bark is structurally load-bearing: `eng_indextts2.py` sets `missing_ref_fallback = "bark"` (PD1: episode always renders). The plan must explicitly keep bark wired as the no-ref fallback, not just "last resort." Confirm chatterbox/kokoro also declare a valid `missing_ref_fallback`. verify: chatterbox/dia adapters (not shown).

4. [Invariants: frozen audio spine] Sample rates differ across engines (kokoro 24000, indextts2 22050; bark unknown). Mixing char (22050) and announcer (24000) on one timeline requires resampling. verify: that the v2 audio chain resamples to a common rate; if not, this is a quality/length defect.

5. [Candidate table: chatterbox + dia rows] All chatterbox/dia claims (MIT, ~2-3 GB, consumes delivery_vector, "wired; re-smoke needed") are unverifiable — no adapter excerpts provided. verify: chatterbox/dia adapters actually implement `generate_voice(... delivery_vector ...)` and consume it before treating chatterbox as "expressive."

OPTIONAL / NICE-TO-HAVE:
- `current_emo_alpha` env re-anchor from the P0 audition matrix (indextts2) — fine to defer.
- `tempfile.mktemp` in `eng_indextts2.generate_voice` is deprecated/racy; switch to `mkstemp` — cosmetic.

CUT THESE (over-engineering):

1. [Starting recommendation: "Add kokoro as a low-VRAM char_voice option" + Open Q2] Cut kokoro-as-char_voice for this round; keep kokoro announcer-only as it is today. Justification: chatterbox (MIT, ~2-3 GB, zero-shot clone, consumes the delivery vector) already covers the 8 GB tier and commercial-clean cloning, so kokoro's only unique value is sub-1 GB preset determinism — marginal. Against that, it is the largest net-new, fragile change set (new seeded char assigner + gendered char pool + `roles += char_voice` + modeling presets in a WAV-ref bank + fixing the uniqueness-collapsing fallback in MUST-FIX 3). It is safe to cut because nothing else depends on it and the cast already renders via indextts2/chatterbox + bark fallback.

2. [Starting recommendation: "Defer Qwen3-TTS"] Already deferred — keep it cut. The rationale stands (emotion knob doesn't apply to the clone path; 7 GB isolated standup with Blackwell torch risk; no marginal value over existing cloners).

[ASSUMPTION] vz_* refs are registered to engine="indextts2" only — inferred from the `refs/indextts2/` path and the operator note; verify against `_otr_voice_bank` entries.
[ASSUMPTION] indextts2 is first in `legacy_first_engines("char_voice")` given its 2026-06-04 promotion to default; verify against `_otr_engine_profiles`.