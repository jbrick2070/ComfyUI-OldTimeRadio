# OTR Cast & SFX Contract Gates

**Last updated:** 2026-05-12 (voice-path-cleanbreak Sprint 5)
**Owners:** writer (cast-lock) + freeze cascade + voice / sfx consumers

This file is the canonical reference for OTR's defense-in-depth
invariants: where each gate runs, what it checks, what it does on
failure, and the naming convention for downstream code.

---

## Gate 1 — Writer cast-lock exit

**Where:** `nodes/_otr_casting.py::_assert_voice_preset_invariant`,
called at the end of `lock_cast()` right after
`_assert_unique_bark_voices`.

**Checks:** Every non-ANNOUNCER cast row carries a non-empty
`voice_preset` starting with `v2/`.

**On violation:** Raises `CastingFailedError`. The writer's run
aborts before the ledger is saved.

**Why this gate exists:** Earliest possible enforcement point.
Catches cast-lock contract violations at source, before they
propagate into the ledger or downstream consumers.

**Tests:** `tests/test_writer_cast_lock_voice_preset.py` (7 cases).

---

## Gate 2 — FreezeCascade Phase 0 / Phase 10 (G6 + G7)

### G6: cast voice_preset invariant

**Where:** `nodes/_otr_ledger_freeze.py::_check_per_cast_invariants`,
called every time `run_gap_audit()` runs.

**Checks:** Same shape as Gate 1 (non-ANNOUNCER cast rows have
non-empty `v2/*` voice_preset). Phase 0 logs the violation as a
critical error; Phase 10 hard-fails.

**On violation:** Phase 0 returns the gap-audit report with G6
errors stamped on `meta.gap_audit_pre`. Phase 10 raises
`FreezeAssertionError` and the freeze is rejected.

**Why this gate exists:** Mid-pipeline catch. If a future refactor
bypasses Gate 1 (e.g. ledger constructed by a non-writer path), G6
catches the violation before any audio renders.

**Tests:** `tests/test_freeze_cascade_g6.py` (8 cases).

### G7: SFX dur_s bounds (voice-path-cleanbreak Sprint 3)

**Where:** `nodes/_otr_ledger_freeze.py::_check_g7_sfx_dur_invariant`,
called from the same `run_gap_audit()` walk as G6.

**Checks:** Every SFX line whose `dur_s` is set to a numeric value
must fall in `[SFX_DUR_MIN_S=0.25, SFX_DUR_MAX_S=12.0]`. Lines
without `dur_s` (back-compat) and non-sfx lines (whose `dur_s` is
post-render data) are skipped.

**On violation:** Same Phase 0 / Phase 10 contract as G6 — Phase 10
raises `FreezeAssertionError`.

**Why this gate exists:** Catches outline-prompt drift if a future
writer change emits a per-cue `dur_s` outside AudioGen's practical
generation window.

**Tests:** `tests/test_per_cue_sfx_dur.py::test_g7_*` (6 cases).

---

## Gate 3 — Voice consumer hard-raise

Per-consumer last line of defense. By the time execution reaches
Gate 3, Gates 1 and 2 should have already caught any cast contract
violation. Gate 3 is the belt-and-suspenders layer that catches
upstream-bypass scenarios (developer wires consumer directly off
a non-frozen ledger, etc).

### Gate 3A — BatchBarkGenerator (unconditional)

**Where:** `nodes/batch_bark_generator.py`, inside the `iter_lines`
walk where each character line resolves its preset.

**Checks:** `_OTRLC.voice_preset(led, line)` returns a non-empty
string starting with `v2/`. Fires on every character line, every
run.

**On violation:** Raises `ValueError` with a contract-violation
message identifying the character + line_id + char_id + the bad
preset value. The Bark batch aborts before any audio renders.

**Why this is unconditional:** Every BatchBark dispatch reads the
preset, so the gate cost is zero (one extra check per line). Catches
the violation at the moment of consumption.

**Tests:** `tests/test_bark_cast_contract.py` (6 cases).

### Gate 3B — SceneSequencer inline-Bark fallback (conditional)

**Where:** `nodes/scene_sequencer.py`, inside the dialogue branch's
inline-Bark fallback path (the `else` clause that fires only when
both pre-rendered audio buses have no clip for the current line).

**Checks:** Same as Gate 3A — non-empty `v2/*` cast.voice_preset.

**On violation:** Raises `ValueError`. The Sequencer aborts.

**Why this is conditional:** SceneSequencer's normal flow uses
pre-rendered Bark / Kokoro audio buses; the inline-Bark fallback
only fires when those are exhausted (a soak edge case). Eager
validation at the iter-lines level would also fire on announcer
rows whose Kokoro-namespace presets (`bm_*` / `bf_*`) are
legitimately not `v2/*`. Localizing the gate to the inline-Bark
branch keeps the announcer + pre-rendered-Bark paths from
incorrectly tripping on Kokoro ids.

**Tests:** Implicit coverage via the integration ledger tests in
`tests/test_sequencer_ledger.py`; the explicit Gate-3B unit test is
deferred to a follow-up since the inline-Bark fallback path requires
mocking the Bark model load.

### Naming history

The voice-path-cleanbreak QA round-robin (Q1) noted asymmetric
coverage between the Bark gate and the Sequencer gate. Sprint 5
locked the naming as `Gate 3A` (eager, in Bark) and `Gate 3B`
(lazy, in Sequencer's inline fallback only) so future code review
catches the difference at a glance.

---

## Cast row reference shape (post-Sprint 2)

For reference, the shape Gates 1 + 2 + 3 are validating:

```jsonc
{
  "char_id":               "c01",          // unique within episode
  "name":                  "LEMMY",        // uppercase character name
  "gender":                "male",         // male / female / unspecified
  "voice_preset":          "v2/en_speaker_8",   // Bark preset; ANNOUNCER excepted
  "tts_model":             "bark",         // bark | kokoro (announcer row)
  "voice_params":          null,           // future: per-character TTS knobs
  "character_description": "Worn space-trucker, baritone, gravelly..."
}
```

ANNOUNCER row gets `voice_preset` of `bm_*` or `bf_*` (Kokoro
namespace) and is intentionally excluded from the v2/* check.

---

## Adding a new invariant

1. Pick the next G-letter (currently G1–G7 are taken; next is G8).
2. Add the checker function to `nodes/_otr_ledger_freeze.py` next
   to `_check_g7_sfx_dur_invariant`.
3. Wire into `run_gap_audit()` after the existing G-checks.
4. Decide: Phase 0 ERROR (becomes Phase 10 hard-fail) or Phase 0
   WARNING (advisory only)?
5. Mirror in writer-side gate (Gate 1) if catching at source matters.
6. Mirror in consumer hard-raise (Gate 3-style) if defense-in-depth
   matters.
7. Add tests under `tests/test_freeze_cascade_g<N>.py` (use g6 +
   g7 as templates).
8. Update this file with the new gate + its location + tests.
