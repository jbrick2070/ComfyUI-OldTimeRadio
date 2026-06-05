# Question -- 2026-05-30

# Design consult: make OTR character NAME coherent with GENDER + VOICE

## Product
"Old-Time Radio" (OTR): a local ComfyUI pipeline that auto-generates radio-drama
episodes. A cast of characters is generated per episode; each character gets a
NAME, a GENDER (male/female/other), and a TTS VOICE. Then an LLM writes dialogue
and TTS (Bark/Kokoro) speaks each line in that character's voice.

## The bug to fix
Generated casts are INCOHERENT: a male-coded name lands on a female-gendered slot
with a female voice, and vice-versa. Real example from a run:
- "MALIK HIBBERT"  -> char_gender=female -> female voice
- "PHYLLIS OKAFOR" -> char_gender=male   -> male voice
The VOICE correctly follows the assigned gender; it's the NAME that doesn't match.

## Exact current code flow (ground truth, traced from live code)
File `nodes/_otr_casting.py`, function `lock_cast()`:
1. NAME is rolled FIRST from a gender-BLIND pool: `config/cast_pools.py` has one
   flat `FIRST_NAMES` list (~110 names) mixing male/female/ambiguous, no gender
   tags. `pick_first_last(rng)` = `rng.choice(FIRST_NAMES)` + `rng.choice(LAST_NAMES)`.
2. GENDER is decided SEPARATELY: `_plan_gender_distribution()` does largest-remainder
   allocation of weights (male 0.40, female 0.40, other 0.20), then `rng.shuffle(genders)`,
   then binds gender to slots POSITIONALLY: `for i, slot in enumerate(open_slots): gender = genders[i]`.
   Nothing reads the name when choosing gender.
3. VOICE follows gender correctly: `python_assign_voice_preset()` filters a voice
   pool by the slot's gender (`VOICE_PROFILES` tuples carry an explicit gender field;
   Bark voices are binary male/female; "other" draws from the full pool).
4. The casting LLM call only writes a prose `character_description` -- it does NOT
   choose name, gender, or voice. The name is rolled in Python BEFORE the LLM runs.

=> Incoherence originates at the positional zip of independent RNG draws (name vs
gender). There is NO name->gender signal anywhere in the codebase today.

## Hard constraints a fix MUST respect
- Reproducibility: all draws use one `cast_rng = random.Random(cast_seed)`; a fixed
  `OTR_CAST_SEED` env must keep output byte-identical (C7 audio gate). Keep new draws on cast_rng.
- Ensemble balance: the global 40/40/20 male/female/other quota is an intentional invariant.
- "other" gender is intentional and must still map to a usable voice.
- No new `model_id` widget may be added (only the writer node exposes model slots);
  the casting LLM call is already tagged "creative".
- Voice uniqueness per episode is asserted post-cast (no duplicate Bark voices).
- 100% local/offline product; LLM calls go through the existing local writer model.
- Jeffrey may WANT some non-stereotypical pairings -- the fix should fix ACCIDENTAL
  incoherence, not rigidly enforce stereotypes. Ideally there's a knob.

## The question
Propose 2-4 concrete approaches to make NAME coherent with GENDER (and thus VOICE),
with trade-offs, then recommend one + the minimal safe implementation. Consider at least:

(A) Code-side name->gender tagging: tag FIRST_NAMES with gender (male/female/unisex),
    then after the gender plan is fixed, draw a name from the matching-gender sublist
    (unisex usable for any/"other"). Pure Python, deterministic, no extra LLM calls.

(B) Jeffrey's creative idea -- LLM-driven coherent cast: the LLM (or a deterministic
    step) first picks GENDER, then picks the VOICE, then generates a character NAME that
    matches that gender/voice, possibly across MULTIPLE LLM rounds (e.g. one pass per
    character, or a coherence/repair pass that re-rolls names that mismatch their gender).
    How to keep this reproducible under OTR_CAST_SEED, fast, and within the 40/40/20 quota?

(C) Hybrid: keep the Python 40/40/20 gender plan + voice mapping, but replace the
    gender-blind name roll with either (A) tagged pools OR a small LLM "name this
    {gender} character" call, plus a cheap validator/repair that flags name<->gender
    mismatch and re-rolls.

For each: reproducibility impact, added latency/LLM calls, how it preserves the quota
and "other"/non-stereotype flexibility, failure modes, and test strategy. Recommend the
approach with the best coherence-per-complexity and describe the smallest first
implementation (and where it plugs into `_otr_casting.py`).
