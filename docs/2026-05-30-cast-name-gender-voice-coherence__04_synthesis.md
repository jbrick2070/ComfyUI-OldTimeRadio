# Synthesis -- 2026-05-30

**Question:** # Design consult: make OTR character NAME coherent with GENDER + VOICE

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

---

## ChatGPT (gpt-5.5)

## Short recommendation

Use **Approach A / deterministic tagged name pools** as the first fix.

It gives the best coherence-per-complexity:

- No new model/widget.
- No extra LLM calls.
- Preserves the existing 40/40/20 gender quota.
- Keeps voice assignment exactly gender-driven.
- Fully local/offline.
- Easy to test.
- Best chance of preserving C7-style repeatability because all randomness stays in `cast_rng`.

I would **not** make the LLM responsible for names in the first implementation. It adds latency, parsing/repair complexity, and reproducibility risk for a problem that can be solved in plain Python.

I do not have exact line numbers from your repo snapshot, so I’ll reference the known functions/locations from your trace: `nodes/_otr_casting.py::lock_cast()`, `config/cast_pools.py::FIRST_NAMES`, and `pick_first_last(rng)`.

---

# Approach A — Code-side first-name gender tagging

## Design

Replace the flat gender-blind `FIRST_NAMES` list in `config/cast_pools.py` with tagged buckets:

```python
FIRST_NAMES_BY_GENDER = {
    "male": [
        "Malik",
        "Arthur",
        "Walter",
        ...
    ],
    "female": [
        "Phyllis",
        "Eleanor",
        "Mabel",
        ...
    ],
    "unisex": [
        "Alex",
        "Casey",
        "Morgan",
        ...
    ],
}
```

Then replace the current gender-blind:

```python
pick_first_last(rng)
```

with something like:

```python
pick_first_last_for_gender(rng, gender)
```

where:

```python
def pick_first_last_for_gender(rng, gender, *, other_policy="unisex", cross_gender_rate=0.0):
    ...
```

Suggested default behavior:

| Slot gender | First-name pool |
|---|---|
| `male` | `male + unisex` |
| `female` | `female + unisex` |
| `other` | `unisex`, fallback to all if empty |

Last names remain gender-neutral:

```python
last = rng.choice(LAST_NAMES)
```

## Where it plugs in

Currently, per your trace, `lock_cast()` does:

1. Roll name first from flat pool.
2. Plan/shuffle genders.
3. Bind genders positionally.
4. Assign voice by gender.

Change that to:

1. Plan/shuffle genders first.
2. For each slot, bind gender.
3. Roll first name from the matching gender-aware pool.
4. Assign voice by gender.

Conceptually, in `nodes/_otr_casting.py::lock_cast()`:

```python
genders = _plan_gender_distribution(...)
cast_rng.shuffle(genders)

for i, slot in enumerate(open_slots):
    gender = genders[i]

    first, last = pick_first_last_for_gender(
        cast_rng,
        gender,
        other_policy=name_other_policy,
        cross_gender_rate=name_cross_gender_rate,
    )

    name = f"{first} {last}"

    voice = python_assign_voice_preset(..., gender=gender, rng=cast_rng)
    ...
```

The exact edit point is the current area where `pick_first_last(rng)` happens before the `_plan_gender_distribution()` / `rng.shuffle(genders)` block. I would move name generation into the existing loop that already has the final slot gender.

## Reproducibility impact

Good.

All draws remain on `cast_rng = random.Random(cast_seed)`. No global `random`, no LLM nondeterminism, no model sampling.

Important caveat: this will not preserve byte-identical output compared to old builds for the same seed, because the cast content changes and the RNG call order likely changes. But after this change lands, the same code/config/seed should reproduce identically between runs.

If you need to minimize RNG-order disruption, you can be more careful, but I would not over-optimize that unless you have a formal “old version seed compatibility” requirement. The important C7 property is usually: same version + same inputs + same seed = same bytes.

## Ensemble quota

Preserved exactly.

The quota is still produced by `_plan_gender_distribution()` and `rng.shuffle(genders)`. Name choice no longer influences quota.

## “Other” gender

Handle explicitly.

Recommended default:

```python
other_policy = "unisex"
```

So `other` slots draw from names like `Alex`, `Morgan`, `Riley`, `Jordan`, etc.

Optional policies:

```python
other_policy = "unisex"  # default
other_policy = "all"     # any first name can belong to an other-gender character
```

This keeps `other` intentional without forcing a fake binary name classification.

## Non-stereotypical pairings knob

Add an optional config/env knob, but default it to strict coherence.

For example:

```python
OTR_NAME_CROSS_GENDER_RATE=0.0
```

Meaning:

- `0.0`: male slots get male/unisex names; female slots get female/unisex names.
- `0.05`: 5% of binary-gender slots intentionally draw from the broader pool.
- `1.0`: effectively return to gender-free names.

Implementation detail: if `cross_gender_rate` is `0.0`, do not even call `rng.random()`. That avoids adding an extra RNG draw in the default path.

Example:

```python
if cross_gender_rate > 0.0 and gender in ("male", "female"):
    if rng.random() < cross_gender_rate:
        pool = all_first_names
    else:
        pool = matched_pool
else:
    pool = matched_pool
```

## Failure modes

- Some names are culturally ambiguous or vary by region.
- The first pass of tagging may be imperfect.
- If the pool is small, names may repeat more often.
- `other` coherence is inherently fuzzier than binary name matching.

These are acceptable and fixable over time.

## Test strategy

Add tests around the pool function, not the whole audio pipeline first.

Minimum tests:

1. **Determinism**
   - Given seed `123`, generate a cast twice.
   - Assert identical cast JSON: names, genders, voices, descriptions if included.

2. **Quota preservation**
   - For several cast sizes, assert `_plan_gender_distribution()` still returns the expected 40/40/20 largest-remainder counts.

3. **Name/gender coherence**
   - For `gender == "male"`, first name is in `male + unisex`.
   - For `gender == "female"`, first name is in `female + unisex`.
   - For `gender == "other"` with default policy, first name is in `unisex`.

4. **Voice still follows gender**
   - Existing voice profile gender filtering still applies.
   - Existing no-duplicate Bark voice assertion still passes.

5. **C7 gate**
   - Run the same episode twice with same code/config/`OTR_CAST_SEED`.
   - Byte-compare the final audio.

---

# Approach B — LLM-driven coherent cast

## Design

Let the local writer model generate the cast more holistically.

Safer version:

1. Python computes the exact 40/40/20 gender plan.
2. Python assigns each slot a gender.
3. Python assigns a voice by gender.
4. LLM receives a structured prompt:

   ```text
   Generate a character name and short description for this slot.

   Gender: female
   Voice profile: kokoro_af_heart
   Genre: old-time radio mystery
   Constraints:
   - Name should plausibly fit the gender.
   - Return JSON only.
   ```

5. Parse JSON and use the returned name.

Riskier version:

- LLM generates the entire cast, including genders, names, descriptions, maybe voices.
- Then Python validates/repairs quota and voices.

I would avoid the riskier version because the quota is an invariant and Python already does it well.

## Reproducibility impact

Weak to medium.

Even with a fixed prompt and seed, local LLM inference may not be byte-identical across runs depending on backend, GPU kernels, sampling, quantization, thread scheduling, and decode settings.

If you do this, you would need:

- Fixed local model.
- Fixed prompt.
- Fixed decoding parameters.
- Ideally deterministic greedy decoding or temperature `0`.
- A per-character LLM seed derived from `cast_rng`.

Example:

```python
name_seed = cast_rng.randrange(2**32)
```

Then pass that seed into the existing local writer call, if the API supports it.

But I would still treat this as less reliable than Python pools for C7.

## Added latency / calls

Potentially significant.

Options:

1. **One LLM call for whole cast**
   - Faster.
   - More brittle JSON.
   - If one name is bad, repair complexity increases.

2. **One LLM call per character**
   - Better local control.
   - Slower.
   - More opportunities for nondeterministic variation.

3. **LLM repair pass**
   - Initial Python or LLM cast.
   - Validator flags bad names.
   - LLM repairs only flagged slots.
   - Still adds calls and nondeterminism.

## Quota preservation

Preserve quota by keeping the Python gender plan authoritative.

Do not ask the LLM to invent the gender distribution. Give it the fixed slots.

## “Other” gender

Prompting can help:

```text
Gender: other/nonbinary.
Choose a name that could plausibly fit a nonbinary character in a 1940s-style radio drama.
```

But this is also culturally subjective and model-dependent.

## Non-stereotypical flexibility

Good creatively.

You can prompt:

```text
Most names should conventionally fit the gender, but occasional nontraditional pairings are acceptable if they seem intentional.
```

However, this is hard to make deterministic and hard to test.

## Failure modes

- Invalid JSON.
- Names with titles, nicknames, punctuation, or multi-part formatting.
- Model ignores gender.
- Model creates duplicate names.
- Model returns culturally odd names for the OTR setting.
- Reproducibility drift.
- Extra local latency.

## Test strategy

Same as Approach A, plus:

- JSON schema validation.
- Retry budget test.
- Deterministic LLM call test on your exact workstation.
- Snapshot tests for fixed prompts/seeds.
- Validator test for obvious mismatches.

## Verdict

This is creatively attractive but operationally overkill for the stated bug. I would not make it the first fix.

---

# Approach C — Hybrid Python quota + deterministic name pools or LLM name repair

## Design

Keep the current good pieces:

- Python gender quota.
- Python voice assignment.
- Existing casting LLM only for prose description.

Then replace only the name source.

There are two useful sub-variants.

---

## C1 — Python gender plan + tagged name pools + validator

This is basically Approach A with a validator layered on top.

Flow:

1. Python computes gender plan.
2. Python rolls matching name from tagged pools.
3. Validator checks:

   ```python
   is_name_allowed_for_gender(first, gender)
   ```

4. If invalid, repair by re-rolling from the matching pool.

In practice, if your picker is correct, the validator should never fire. But it is useful as a guardrail while the pools evolve.

## Reproducibility impact

Very good.

Repair uses `cast_rng`.

If the validator triggers, it consumes extra RNG, but still deterministically.

## Added latency

None.

## Quota preservation

Preserved.

## “Other” handling

Same as Approach A.

## Non-stereotypical flexibility

Can expose a policy:

```python
name_gender_policy = "strict"      # default
name_gender_policy = "soft"        # small cross-gender chance
name_gender_policy = "free"        # old behavior
```

Possible env vars:

```text
OTR_NAME_GENDER_POLICY=strict
OTR_NAME_CROSS_GENDER_RATE=0.0
OTR_OTHER_NAME_POLICY=unisex
```

No new ComfyUI widget required.

## Failure modes

- Validator quality depends on name tags.
- “Soft” mode means some apparent mismatches are allowed by design.
- If you repair too aggressively, you may erase intentionally unusual pairings.

## Test strategy

Same as Approach A.

---

## C2 — Python gender plan + small LLM “name this character” call + deterministic validator/repair

Flow:

1. Python computes gender plan.
2. Python assigns voice.
3. For each character, ask LLM for only the name.
4. Validate name against tagged known-name table.
5. If validator rejects, either:
   - re-prompt LLM, or
   - fall back to deterministic tagged pool.

This gives Jeffrey’s creative idea some room while retaining a safe fallback.

## Reproducibility impact

Medium to weak because of extra LLM calls.

Can be improved with:

- One fixed local model.
- Fixed prompt.
- Fixed seed derived from `cast_rng`.
- Greedy/temperature-zero decode.
- Deterministic fallback on parse/validation failure.

Still not as good as pure Python.

## Added latency

Moderate to high.

One LLM call per character is expensive relative to a random name choice. One whole-cast call is faster but harder to validate and repair.

## Quota preservation

Preserved if Python remains authoritative for gender.

## “Other” handling

Potentially better creatively than fixed pools, but less predictable.

## Failure modes

- Nondeterministic output.
- Invalid formatting.
- Names outside validator’s known list get rejected even if reasonable.
- Extra local compute.
- More moving parts in a C7-sensitive pipeline.

## Verdict

Useful later as an optional “creative naming” mode, but I would not use it as the default fix.

---

# Approach D — Keep existing name rolls, then assign genders to best-match names

This is another pure-Python option worth mentioning.

## Design

Instead of changing name generation, keep the existing flat `FIRST_NAMES` roll. Then use name tags to assign the preplanned gender list to slots in a way that maximizes coherence.

Example:

- Rolled names:

  ```text
  Malik Hibbert
  Phyllis Okafor
  Alex Mercer
  ```

- Gender quota produces:

  ```text
  female, male, other
  ```

Instead of zipping positionally, do a deterministic matching step:

- Malik gets male.
- Phyllis gets female.
- Alex gets other.

The quota is preserved because the same number of male/female/other slots exists. You are just assigning those genders to the already-rolled names more intelligently.

## Reproducibility impact

Good if the matching algorithm is deterministic and uses only `cast_rng` for tie-breaking.

## Added latency

None.

## Quota preservation

Preserved exactly.

## “Other” handling

Unisex names can be preferentially matched to `other`.

## Non-stereotypical flexibility

Could add a mismatch allowance:

```python
OTR_NAME_GENDER_POLICY=soft
```

However, this approach is slightly more complex than Approach A because you need a stable matching/scoring algorithm.

## Failure modes

- If the flat pool rolls too many male-coded names but the quota requires many female slots, some mismatches remain.
- Requires name tags anyway.
- Harder to reason about than drawing from the right pool in the first place.

## Verdict

Interesting, but not the smallest clean fix. If you are going to tag names, it is simpler to draw from the correct tagged pool after gender is known.

---

# Recommended implementation

## Use Approach A with a tiny validator from C1

The minimal safe implementation:

1. Add gender-tagged first-name buckets in `config/cast_pools.py`.
2. Keep `FIRST_NAMES` as a compatibility flattened list if anything else imports it.
3. Add `pick_first_last_for_gender(rng, gender, ...)`.
4. In `nodes/_otr_casting.py::lock_cast()`, move name generation to after the gender plan is known.
5. Keep voice assignment unchanged.
6. Add tests for determinism, quota, and name/gender coherence.

---

## Example `config/cast_pools.py` change

Current shape, per your trace:

```python
FIRST_NAMES = [
    "Malik",
    "Phyllis",
    ...
]

LAST_NAMES = [
    "Hibbert",
    "Okafor",
    ...
]

def pick_first_last(rng):
    return rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
```

Recommended shape:

```python
FIRST_NAMES_BY_GENDER = {
    "male": [
        "Malik",
        "Arthur",
        "Walter",
        "Edgar",
        # ...
    ],
    "female": [
        "Phyllis",
        "Eleanor",
        "Mabel",
        "Clara",
        # ...
    ],
    "unisex": [
        "Alex",
        "Casey",
        "Jordan",
        "Morgan",
        "Riley",
        # ...
    ],
}

# Compatibility for any existing import sites.
FIRST_NAMES = (
    FIRST_NAMES_BY_GENDER["male"]
    + FIRST_NAMES_BY_GENDER["female"]
    + FIRST_NAMES_BY_GENDER["unisex"]
)

def _all_first_names():
    return FIRST_NAMES

def first_name_pool_for_gender(gender, *, other_policy="unisex"):
    male = FIRST_NAMES_BY_GENDER["male"]
    female = FIRST_NAMES_BY_GENDER["female"]
    unisex = FIRST_NAMES_BY_GENDER["unisex"]

    if gender == "male":
        return male + unisex

    if gender == "female":
        return female + unisex

    if gender == "other":
        if other_policy == "all":
            return male + female + unisex
        return unisex or male + female

    return male + female + unisex

def pick_first_last_for_gender(
    rng,
    gender,
    *,
    other_policy="unisex",
    cross_gender_rate=0.0,
):
    matched_pool = first_name_pool_for_gender(
        gender,
        other_policy=other_policy,
    )

    if cross_gender_rate > 0.0 and gender in ("male", "female"):
        if rng.random() < cross_gender_rate:
            pool = _all_first_names()
        else:
            pool = matched_pool
    else:
        pool = matched_pool

    first = rng.choice(pool)
    last = rng.choice(LAST_NAMES)
    return first, last

def pick_first_last(rng):
    """
    Backward-compatible gender-blind picker.
    Prefer pick_first_last_for_gender() for new casting code.
    """
    return rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
```

I would keep `pick_first_last(rng)` around initially to avoid breaking unrelated imports.

---

## Example `_otr_casting.py::lock_cast()` change

Before, conceptually:

```python
for slot in open_slots:
    first, last = pick_first_last(cast_rng)
    slot["name"] = f"{first} {last}"

genders = _plan_gender_distribution(...)
cast_rng.shuffle(genders)

for i, slot in enumerate(open_slots):
    gender = genders[i]
    slot["char_gender"] = gender
    slot["voice"] = python_assign_voice_preset(..., gender=gender)
```

After:

```python
genders = _plan_gender_distribution(...)
cast_rng.shuffle(genders)

for i, slot in enumerate(open_slots):
    gender = genders[i]
    slot["char_gender"] = gender

    first, last = pick_first_last_for_gender(
        cast_rng,
        gender,
        other_policy=other_name_policy,
        cross_gender_rate=name_cross_gender_rate,
    )
    slot["name"] = f"{first} {last}"

    slot["voice"] = python_assign_voice_preset(
        ...,
        gender=gender,
        rng=cast_rng,
    )
```

I would source the knobs from environment/config, not widgets:

```python
name_cross_gender_rate = float(os.getenv("OTR_NAME_CROSS_GENDER_RATE", "0.0"))
other_name_policy = os.getenv("OTR_OTHER_NAME_POLICY", "unisex")
```

Validate them once:

```python
name_cross_gender_rate = max(0.0, min(1.0, name_cross_gender_rate))

if other_name_policy not in ("unisex", "all"):
    other_name_policy = "unisex"
```

No new `model_id` widget is needed.

---

# Important C7 note

This change will intentionally alter generated casts for existing seeds. That means old audio for the same seed will not remain byte-identical across the code change.

But the corrected system should satisfy the important reproducibility rule going forward:

> same code + same config + same local models + same `OTR_CAST_SEED` = same generated cast and same audio bytes.

If historical seed compatibility is mandatory, say so before implementing; that would require a more careful compatibility mode. But for the smallest correct fix, I would accept the one-time seed-output change.

---

# Final recommendation

Implement **gender-tagged first-name pools plus gender-aware picking**.

Do not involve the LLM in name selection yet.

Use this policy:

```text
male slot   -> male + unisex names
female slot -> female + unisex names
other slot  -> unisex names by default
```

Add optional env knobs:

```text
OTR_NAME_CROSS_GENDER_RATE=0.0
OTR_OTHER_NAME_POLICY=unisex
```

This directly fixes the root cause: the current positional zip of independent name and gender draws. It preserves the 40/40/20 ensemble invariant, keeps voice behavior unchanged, remains fully local, and is the least risky path for deterministic audio generation.

---

## Gemini (gemini-3.1-pro-preview-customtools)

1. **PARTIALLY AGREE:** Approach A’s core idea (pure-Python tagged pools) is the right direction for your VRAM constraints, but ChatGPT’s implementation reorders the RNG calls, which will needlessly nuke your historical reproducibility for *all* past episodes.

2. **FACTUAL ERRORS in ChatGPT's answer:**
   * **RNG Sequence Destruction:** ChatGPT’s code moves `pick_first_last_for_gender` to happen *after* `cast_rng.shuffle(genders)`. Because both functions consume from the same `cast_rng` Mersenne Twister, swapping their execution order completely changes the random numbers they receive. This breaks your C7 byte-identical gate for **100% of historical seeds**, even the ones that didn't have the incoherence bug.
   * **Variable RNG Consumption:** ChatGPT suggests implementing the cross-gender knob with `if rng.random() < cross_gender_rate:`. This introduces a *variable* number of RNG calls per character. If this triggers, all subsequent random draws in the pipeline (including voice assignment and LLM seeds) will be permanently desynced.
   * **Dismissal of Approach D:** ChatGPT dismisses Approach D (post-roll alignment) as "harder to reason about." In a strict byte-identical pipeline, Approach D is actually the *only* mathematically sound way to fix the bug while preserving the exact RNG state sequence.

3. **IMPORTANT OMISSIONS:**
   * **The 14.5 GB VRAM Ceiling:** ChatGPT completely ignored your hardware context. You are on a 16GB RTX 5080 Laptop. ComfyUI overhead + the Writer LLM + Bark/Kokoro TTS models will easily push you right up to that 14.5 GB ceiling. Approaches B and C2 (adding extra LLM calls for naming/repair) aren't just "slower"—they carry a massive risk of OOM (Out of Memory) crashes or forcing brutal swap-to-RAM slowdowns.
   * **Voice Assignment RNG:** The trace shows `python_assign_voice_preset(..., rng=cast_rng)`. If we change the order in which genders are evaluated, we change the order in which voices are drawn, which also breaks the RNG chain.

4. **MY SHORT RECOMMENDATION:**
   * **Use a refined Approach D (Post-Roll Alignment).** This is the smallest change with the largest payoff, costs zero VRAM, and protects your RNG chain.
   * **Tag the pools:** In `config/cast_pools.py`, group `FIRST_NAMES` into a dictionary by gender, but keep the flat `FIRST_NAMES` list exactly as it is for the initial draw.
   * **Do not touch the RNG order:** In `nodes/_otr_casting.py::lock_cast()`, let the code roll the names and shuffle the genders exactly as it does today.
   * **Insert a deterministic alignment step:** Right after the genders are shuffled (and *before* voice assignment), write a 10-line Python function that looks at the rolled names and the rolled genders, and swaps the slot assignments to match them up (e.g., Male name gets paired with the Male gender slot).
   * **Why this wins:** If a historical seed randomly rolled a perfectly coherent cast, the alignment step does nothing, and your C7 audio output remains 100% byte-identical to previous runs. You only alter the output for seeds that were actually broken.

5. **UNCERTAINTIES to verify:**
   * **Voice Assignment RNG Consumption:** I need to know exactly how `python_assign_voice_preset` uses `cast_rng`. If it consumes a different number of RNG calls for a "male" voice vs a "female" voice, then swapping the slot genders during alignment will still desync the RNG chain for the LLM later in the pipeline. If this is true, you must capture the RNG state (`cast_rng.getstate()`) before the casting block and restore/advance it predictably.
   * **Strictness of C7:** I am assuming you want to preserve historical seeds if they were already coherent. If you *only* care about C7 going forward (i.e., you don't mind invalidating all past episode seeds), then ChatGPT's Approach A is slightly simpler to code. Please confirm your backward-compatibility requirements.

---

## NVIDIA (nvidia/llama-3.3-nemotron-super-49b-v1.5)

### Analysis of Previous Answers & Recommendations

---

#### **1. Areas of Agreement Between ChatGPT and Gemini**
- **Approach A (Tagged Pools) is Valid**: Both agree that tagging names by gender and drawing from gender-specific pools is a clean, deterministic solution.
- **Approach B/C2 (LLM Involvement) is Risky**: Both caution against using the LLM for name generation due to latency, VRAM pressure, and reproducibility risks.
- **Approach D (Post-Roll Alignment) is Feasible**: Gemini explicitly endorses it; ChatGPT dismisses it but acknowledges its theoretical validity.

**My Take**:  
I concur with both on the validity of Approach A and the risks of LLM involvement. However, Gemini’s critique of ChatGPT’s RNG reordering is **critical** for C7 compliance. If backward compatibility (same seed → same output) is required, Approach A as described by ChatGPT breaks this. Gemini’s Approach D preserves the RNG chain and is safer for historical seeds.

---

#### **2. Key Disagreements & Third Path**
- **ChatGPT vs. Gemini on RNG Order**:  
  - **ChatGPT** moves name generation *after* gender shuffling, altering the RNG sequence and breaking historical reproducibility.  
  - **Gemini** argues for Approach D: keep the original RNG order (name → gender → voice) but align them post-hoc.  

**My Recommendation**:  
**Approach D (Post-Roll Alignment)** is the minimal viable fix if backward compatibility is required. If not, Approach A with **RNG state capture/restore** (see below) is acceptable.  

**Third Path (Hybrid D+A)**:  
- Use Gemini’s Approach D for backward compatibility.  
- If historical seed compatibility is **not** required, use ChatGPT’s Approach A but **capture the RNG state** before name generation to isolate its impact on downstream steps (e.g., voice assignment, LLM seeds).

---

#### **3. Factual Errors in Previous Answers**
- **ChatGPT**:  
  - Claims Approach A preserves reproducibility, but reordering RNG calls breaks it for historical seeds.  
  - Suggests `rng.random()` for cross-gender rate, which introduces variable RNG consumption (bad for determinism).  
- **Gemini**:  
  - Correctly identifies ChatGPT’s RNG flaws but overstates the complexity of Approach D.  

---

#### **4. Important Omissions**
1. **VRAM Constraints**:  
   - Gemini notes the 14.5 GB ceiling but doesn’t quantify the risk of LLM calls.  
   - **Critical**: Bark/Kokoro TTS + ComfyUI + LLM (e.g., LLaMA2-70B) can easily exceed 14.5 GB with multiple LLM calls (Approach B/C2).  
2. **Voice Assignment RNG**:  
   - Both answers assume `python_assign_voice_preset` uses a fixed number of RNG calls. If it varies by gender/voice, even Approach D could desync the RNG chain.  
3. **C7 Backward Compatibility**:  
   - Not explicitly clarified by the user. If required, Approach D is mandatory. If not, Approach A is simpler.  

---

#### **5. Prioritized Recommendation**
1. **If Backward Compatibility (C7) is Required**:  
   - **Use Gemini’s Approach D (Post-Roll Alignment)**:  
     - Tag `FIRST_NAMES` by gender in `config/cast_pools.py`.  
     - Keep the original code order: roll names → shuffle genders → align via deterministic matching.  
     - Example alignment logic in `lock_cast()`:  
       ```python
       # After rolling names and shuffling genders:
       named_slots = sorted(open_slots, key=lambda s: name_gender_score(s["name"], "male"))
       gender_slots = sorted(open_slots, key=lambda s: s["char_gender"] == "male")
       for i, slot in enumerate(open_slots):
           slot["char_gender"] = gender_slots[i]["char_gender"]
           slot["voice"] = python_assign_voice_preset(..., gender=slot["char_gender"], rng=cast_rng)
       ```
       This reassigns genders to slots to match names without altering RNG order.  

2. **If Backward Compatibility is Not Required**:  
   - **Use Approach A with RNG Isolation**:  
     - Tag `FIRST_NAMES` and draw names *after* gender assignment.  
     - Capture the RNG state before name generation to minimize downstream impact:  
       ```python
       rng_state = cast_rng.getstate()
       genders = _plan_gender_distribution(...)
       cast_rng.shuffle(genders)
       cast_rng.setstate(rng_state)  # Reset for name generation
       for slot in open_slots:
           first, last = pick_first_last_for_gender(cast_rng, slot["char_gender"])
       ```

3. **Avoid LLM-Based Approaches (B/C2)**:  
   - High VRAM risk (LLM + TTS on 16GB GPU).  
   - Reproducibility challenges (model nondeterminism, parsing edge cases).  

4. **Add Config Knobs for Flexibility**:  
   - `OTR_OTHER_NAME_POLICY=unisex` or `all`.  
   - `OTR_NAME_CROSS_GENDER_RATE=0.0` (default strict).  

---

#### **6. Uncertainties Requiring Verification**
1. **Voice Assignment RNG**: Does `python_assign_voice_preset` consume a fixed or variable number of RNG calls? If variable, Approach D may still desync the chain.  
2. **C7 Backward Compatibility**: Must historical seeds produce identical audio post-fix, or is forward compatibility sufficient?  
3. **Name-Gender Scoring**: How to deterministically align names with genders in Approach D (e.g., simple tag matching vs. ML-based scoring)?  

---

### **Final Recommendation**
**Implement Gemini’s Approach D (Post-Roll Alignment)** if backward compatibility (C7) is required. If not, use Approach A with RNG state isolation.  

**Minimal Implementation Steps**:  
1. **Tag Names**:  
   - In `config/cast_pools.py`, create `FIRST_NAMES_BY_GENDER` (male/female/unisex).  
   - Keep the original `pick_first_last(rng)` for backward compatibility.  

2. **Align Names and Genders**:  
   - In `nodes/_otr_casting.py::lock_cast()`, after rolling names and shuffling genders:  
     ```python
     # Align names with genders deterministically
     male_names = [slot for slot in open_slots if is_male_name(slot["name"])]
     female_names = [slot for slot in open_slots if is_female_name(slot["name"])]
     other_names = [slot for slot in open_slots if is_unisex_name(slot["name"])]
     
     # Reassign genders to match names while preserving quota
     final_genders = []
     for slot in open_slots:
         if slot in male_names and "male" in genders:
             final_genders.append("male")
             genders.remove("male")
         elif slot in female_names and "female" in genders:
             final_genders.append("female")
             genders.remove("female")
         else:
             final_genders.append(genders.pop())  # Fallback to quota
     for i, slot in enumerate(open_slots):
         slot["char_gender"] = final_genders[i]
     ```

3. **Preserve Voice Assignment**:  
   - No changes needed to `python_assign_voice_preset` if gender is correctly aligned.  

This approach fixes incoherence without breaking historical seeds or exceeding VRAM limits.

---

## To decide (Claude / human)

- [ ] All three agree:
- [ ] Two-vs-one splits:
- [ ] Facts to verify:
- [ ] Final grounded recommendation:
