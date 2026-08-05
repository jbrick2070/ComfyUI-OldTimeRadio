# Driver anchor -- meta.episode_seed is never stamped, so every episode casts from one frozen constant

**Driver:** Claude (Opus 5), Cowork, sole judge. **Date:** 2026-08-05.
**Repo:** ComfyUI-OldTimeRadio @ `v2.0-alpha`, HEAD `f5a5d174`.
**Panel:** Codex `gpt-5.6-sol` high + Antigravity `Gemini 3.6 Flash (High)`.
**Profile:** `.kibitz/comfyui.local.md`.

## The defect, measured on published episodes

Overnight 2026-08-04/05 the continuity queue shipped: the kokoro announcer was
unpinned from `bm_george` and `_ladder_pick` stopped accepting a candidate tier of
one. Both fixes verify green in unit tests and in the ledgers.

**The listener would still hear the same voices every night.** Across the 14
episodes rendered after the fixes:

- announcer: `bf_emma` on **14 of 14**
- character voices: **5 distinct across 18 cast rows**, `vz_caro_davy` 13 times

Cause: every voice draw is seeded on `episode_seed`, and `meta.episode_seed` is
**absent** on these lanes. `coerce_int_seed(None)` folds to the constant
`5362114964413277558`, so every episode draws from the same seed.

Measured on the real modules:

| slot shape | today (constant seed) | with a per-episode seed |
|---|---|---|
| `c02 female warm` | ONE voice, every episode | **20 distinct** over 30 episodes |
| `c03 male sharp` | ONE voice, every episode | **13 distinct** over 30 episodes |
| kokoro announcer | `bf_emma` always | all four of the curated pool |

The fixes are correct. The seed feeding them is frozen.

## The seed already exists, one key away

A published ledger (`signal_lost_lute_strings_fools_tongue_20260805_021040`):

```
meta.episode_seed              -> None
meta.cast_seed                 -> None            (not a top-level key)
meta.cast_contract.cast_seed   -> 2142006639      source: "OS entropy"
```

So a genuine per-episode seed is minted and stamped -- into
`meta["cast_contract"]` at `nodes/OTR_LedgerScriptWriter.py:4087` -- while every
consumer reads `meta["episode_seed"]`, which nothing on this lane writes.

## Why the existing stamp does not fire

`nodes/OTR_LedgerScriptWriter.py:6055` does stamp it:

```python
if meta.get("episode_seed") is None:
    _episode_seed, _episode_seed_source = _resolve_cast_rng_seed()
    meta["episode_seed"] = int(_episode_seed)
```

but it sits inside `if delivery_mode_for_meta(meta) == CONTENT_OWNED:` at `:6041`.
shakespeare / public_domain / scifi_news are NOT content-owned, so the branch is
skipped. **Grepped the whole night's server log: the "content-owned episode seed"
line appears 0 times.**

The comment at `:6025-6030` states the intended contract: *"That receipt has one
owner per lane family -- the seeded cast picker upstream for legacy lanes, the
content-owned block just below for lanes that never run it."* The legacy-lane half
of that contract is not implemented: the seeded cast picker stamps
`cast_contract.cast_seed`, not `meta.episode_seed`.

## Every consumer of the frozen value (grepped, non-test)

| site | what it seeds |
|---|---|
| `nodes/cast_lock.py:502` | `coerce_int_seed(meta.get("episode_seed"))` -> character voice picks AND the announcer draw |
| `nodes/_otr_voice_node_common.py:427` | render-time reference resolution |
| `nodes/otr_credits_roll.py:314` | the durable seed receipt on the credits roll |
| `nodes/stable_audio_theme.py:265` | `music_rng_seed_v1` -- the music bed |
| `nodes/_otr_audio_engines/eng_kokoro.py:93` | kokoro's own announcer fallback seed |

So this is not only a voice defect: the **music bed** is drawing from the same
frozen constant, and the credits carry a seed receipt that is identical on every
episode and therefore reproduces nothing.

## What I want the panel to answer

**Q1 -- where does the stamp belong?** Two candidates:
(a) beside the `cast_contract` stamp at `:4087`, using the same entropy the
picker already minted, so the legacy-lane half of the stated contract is finally
implemented; or (b) lift the `CONTENT_OWNED` gate at `:6041` so the tail stamps it
for every lane. Which one preserves the documented ownership rule, and does (b)
risk minting a seed for a lane whose picker already minted one (two owners)?

**Q2 -- is `cast_contract.cast_seed` the right value to promote?** The comment at
`:6048-6054` warns emphatically that `cast_seed` is NOT a generic episode seed --
it is a replay claim, and CastLock replays the picker whenever it sees it
("num_characters must be 1-6, got 0"). Does stamping the same NUMBER into
`meta.episode_seed` create any risk of it being read back as a replay claim, or
are the two keys genuinely independent once the number is copied?

**Q3 -- what breaks when this value stops being constant?** It is currently a
de-facto constant, so anything that silently depends on that is about to change.
Name every test or invariant that pins a voice id, a music seed, or a credits
receipt without setting `episode_seed` explicitly. In particular: does
`tests/golden/cast_pool_baseline.json` or any determinism/C7 test rely on it?

**Q4 -- is a frozen seed load-bearing anywhere on purpose?** Bake-off and A/B
harnesses may WANT a fixed seed across arms. If so, the fix must leave an explicit
way to pin it rather than removing the ability.

**Q5 -- blast radius on the music bed.** `stable_audio_theme.py:265` derives
`music_rng_seed_v1` from this. Today every episode gets the same music seed. Is
that currently masked by something else varying, or has every episode been drawing
the same music parameters too?

## Rules for this panel

- Ground every claim in `file:line` from the REAL repo. A claim I cannot check is
  discarded, not weighed.
- The FIX is the subject, not the diagnosis -- the diagnosis is measured above and
  is not up for re-derivation.
- Do not propose story/prose changes. Story quality is CLOSED.
- The driver is the sole judge; the panel proposes and every surviving claim is
  re-checked against the Windows files before it is folded in.
