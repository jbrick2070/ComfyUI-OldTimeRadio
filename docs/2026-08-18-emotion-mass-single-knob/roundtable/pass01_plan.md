# Plan v2 -- collapse the emotion blend onto ONE knob at 0.560

Revised after R1. Supersedes `pass00_plan.md`; R1's grounding is in
`pass01_judgment.md`.

## 1. The decision

Operator, on `otr/episodes/lemmy_emotion_ladder_logodds_2026-08-18/` -- a ladder
that pinned alpha at 1.0 and varied ONLY the effective-mass ceiling:

> "IF I WERE A KID I'D LIKE MORE BUT AS AN ADULT ARM0P560 IS PERFECT."

and of the uncapped arm: *"its not a real emtion ist coimputer emoption"*.

Effective mass **0.560**, speaker embedding retained **0.440**.

## 2. The mechanism (read, and measured)

`emotion_payload` (`nodes/_otr_audio_engines/eng_indextts2.py:340`) applies
alpha FIRST, the ceiling SECOND:

1. `_apply_vendor_alpha` -- mirrors the vendor's pre-scaling, truncating each
   weight to 4 decimals, **and only when `scale != 1.0`**.
2. `mass = sum(applied)`; if `mass > cap`, rescale the RAW vector by `cap/mass`
   flooring to 3 decimals, re-serialize through JSON, re-measure, then a bounded
   0.001-per-step shave.

Effective mass is therefore **approximately** `min(alpha * sum(raw), cap)` and
lands at or just under it. It is NOT that expression exactly: the emotional line
measures **0.5590**, not 0.5600, and the ladder's cap-0.4 rung measured 0.398.
Anything asserting this value reads `emotion_payload()["effective_mass"]`.

The vendor spends the mass against the speaker:
`emovec = emovec_mat + (1 - sum(w)) * emovec`. Mass 0.56 leaves 0.44 of him;
mass 1.0 leaves none, which is the "computer emotion" he heard.

## 3. Why the ceiling alone is not enough

Alpha binds first, so with alpha left at 0.4 the ceiling never binds:

| line | raw sum | alpha 0.4 | vs cap 0.56 | effective |
|---|---|---|---|---|
| neutral (`calm=1.0`) | 1.000 | 0.400 | under | **0.400** |
| emotional (measured) | 0.934 | 0.374 | under | **0.374** |

Measured under the proposed 1.0/0.56 the same three audition lines land at
**0.5600 / 0.5590 / 0.5600**, speaker retained 0.440 -- the approved rung.

## 4. Saturation evidence, stated at its real strength

Across **57 character lines sampled from the 6 most recent episode ledgers**:
raw sums 0.733 .. 1.333, and **0 of those 57** fall below 0.56. Old effective
mass ran 0.293 .. 0.400 with 46 of 57 (81%) already pinned at the old ceiling;
new is 0.560 on all 57.

This is a sample, not a law. Below-cap vectors remain valid behaviour and get a
test rather than being designed away.

**Two honest consequences.** Total mass becomes uniform -- the shipped build
already flattened 81% of lines, this takes the rest. And 56 of those 57 lines
are dominated by `calm`, so the audible change on ordinary dialogue is *less of
the speaker, more of the generic calm prototype* -- the same direction the
voice-identity fix moved away from. The operator judged 0.560 on the EMOTIONAL
line only. The re-render therefore renders neutral lines too, so he hears the
case that dominates production before it locks in.

## 5. The change

**5.1 Adapter** (`nodes/_otr_audio_engines/eng_indextts2.py`)
* `EMO_ALPHA_DEFAULT` 0.4 -> **1.0**
* `EFFECTIVE_EMOTION_MASS_CAP` 0.4 -> **0.56**
* `EMOTION_MASS_CAP_DISABLED` unchanged at 8.0 -- still the control arm.
* Rewrite the docstrings that justify the two-knob split: `current_emo_alpha`
  (the 1.0 -> 0.4 history), `current_emo_mass_cap` (delete the 2x2-degeneracy
  rationale -- that experiment is concluded), `emotion_payload` (delete "alpha
  is the taste knob above it"), and the constant comments. Alpha is now a
  compatibility/diagnostic override that is inert **on the default path only**.

**5.2 Profile** (`config/audio_engine_profiles.yaml`, `char_indextts2_v1`)
* `default_params.emo_alpha` 0.4 -> **1.0**
* **add `emo_mass_cap: 0.56`** so the profile declares the knob that actually
  governs. Safe: nothing sha256-pins this file.
* `engine_impl_version` 2 -> **3**, matching the convention the seed change set.

**5.3 Audition instrument** (`scripts/otr_lemmy_production_audition.py`)
* The shipped arm **clears** `OTR_INDEXTTS2_EMO_ALPHA` / `_EMO_MASS_CAP` rather
  than restating them, so it can never certify a build the constants do not
  describe.
* **Three arms**, to unconfound the two variables:
  * `shipped` -- character seed, adapter defaults
  * `ceiling_control` -- character seed, cap 8 (isolates the ceiling)
  * `pre_fix_control` -- per-line seed, alpha 1.0, cap 8 (the historical arm)
* Refuse any **non-empty** output dir OR key dir, not just a present
  `MANIFEST.json` -- WAVs and `KEY.json` are evidence too.
* Manifest records the runtime that made it: resolved alpha, cap, live engine
  fingerprint, reference id + sha256, per-line effective mass and seed.

**5.4 Acceptance harness** (`scripts/otr_voice_identity_2x2.ps1`)
* Fix arms track the shipped defaults instead of hardcoding 0.4/0.4; the
  documented invocation becomes `--expect-alpha 1.0 --expect-mass-cap 0.56`.

**5.5 Tests**
* The 8 stale route tests (DONE): demotion covered by a synthetic stale record
  whose rotted fingerprint is the REAL withdrawn `b965453f355661a3` read from
  `superseded_native_routes`; "preserved unedited" retargets to the superseded
  copy; route id v1 -> v2; `audited_on` 2026-08-18.
* **New** behaviour tests: default alpha 1.0, default cap 0.56, profile and
  adapter agree, a production-derived vector binds at the cap, a below-cap
  vector passes through unchanged, both env overrides still work, and the cache
  key moves when either resolved value moves.

## 6. Sequencing -- the part that bites

`live_engine_impl_version` sha256s the **whole file**, so comments and
docstrings move the fingerprint exactly as constants do.

1. Land EVERY edit to `eng_indextts2.py` -- constants AND prose. Freeze it.
2. Land the profile, audition, harness and test edits.
3. Compute `live_engine_impl_version("indextts2")`.
4. Render the audition into a **new** directory.
5. Hash the manifest; write the qualification record citing (3) and (4).
6. Run the suite.

Doing (5) before (1) writes a record that demotes itself. Exactly one route is
fingerprint-bound (`approved_native_routes` has a single entry, Lemmy's), so
"re-qualify everything affected" means re-qualify him.

## 7. Not in scope

No seed-policy change, no delivery-table change, no vendor-call change. No
workflow JSON change: `emo_alpha` and `emo_mass_cap` are not widgets --
`indextts2` appears in `workflows/otr_canonical.json` only as the selected
engine value on nodes 80 and 81. No content filtering, no word-count gate, no
story-quality work, and nothing that reduces what reaches `otr/obs/`.
