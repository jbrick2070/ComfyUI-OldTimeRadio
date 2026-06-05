# Round A -- ChatGPT (gpt-5.5) elapsed=115.8s

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
