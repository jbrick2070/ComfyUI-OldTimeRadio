# Round A -- ChatGPT (gpt-5.5) elapsed=103.2s

I can’t fetch GitHub from this interface, so I’m reviewing by symbol/file from your description, not verified line numbers.

## Top issues I’d actually change

### 1. `nodes/_otr_casting.py:cast_one_character()` — loop is basically correct, but tighten edge semantics

Your condition:

```python
attempt_idx == max_attempts - 1 and last_raw
```

does mean: “only on the final attempt, and only if there is a prior truthy raw response.”

For `max_attempts = 3`:

- attempt `0`: normal
- attempt `1`: normal
- attempt `2`: repair, if `last_raw` truthy

So the control flow is right.

Two edge tweaks I’d make:

```python
if max_attempts < 1:
    raise ValueError("max_attempts must be >= 1")

use_repair = (
    attempt_idx == max_attempts - 1
    and attempt_idx > 0
    and last_raw is not None
)
```

Why:

- `max_attempts=1` is not inherently invalid. It just means “single-shot, no repair.”
- `last_raw` should probably be checked with `is not None`, not truthiness. An empty raw response is still a prior raw response and may be useful to mention in the repair path or at least should not accidentally alter branch semantics.
- `max_attempts=0` is the real invalid value.

I would **not** reject `< 2` unless you want to guarantee that repair mode is always possible. As an API, `max_attempts=1` is a reasonable “fast fail” mode.

One more important repair-loop edge: if the LLM returns a huge malformed response, the repair prompt can blow your token budget. I’d truncate `last_raw` before embedding it in the repair prompt.

Example:

```python
repair_raw = last_raw[:1200]
```

or similar.

---

### 2. `nodes/_otr_casting.py:lock_cast()` — no off-by-one, but fail-fast would be cleaner

Checking voice availability immediately before each per-slot LLM call avoids the classic off-by-one. If you check before consuming one voice, you won’t index an empty pool.

But the current check likely fails late if the invariant ever changes.

Better pattern:

```python
remaining_slots = num_open_slots - len(open_cast_rows)
if len(available_voices) < remaining_slots:
    raise CastingFailedError(...)
```

Do this once before the open-slot loop, and optionally keep the per-iteration assert/check too.

Today, with:

- pool size 9
- or 8 if LEMMY reserves one
- `num_characters <= 6`

you should not exhaust. But adding an entry gate makes the contract obvious and saves wasted LLM calls if future constraints reduce the pool.

So: **no off-by-one bug**, but I’d add a fail-fast capacity check.

---

### 3. Determinism: avoid module/global `random`

This is the one thing I’d flag hard because of your byte-identical requirement.

If `config/cast_pools.py` or `nodes/_otr_casting.py` uses bare:

```python
random.choice(...)
random.shuffle(...)
random.sample(...)
```

then the cast contract is not reproducible unless the global RNG is seeded consistently before every run and nothing else consumes it.

Prefer passing an explicit RNG:

```python
def pick_first_last(..., rng: random.Random) -> str:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
```

Same for:

- announcer preset selection
- open voice pool shuffling
- name picking
- any 50/50 gender or voice selection

This is not architectural; it’s implementation hygiene for C7. Use a local `random.Random(seed)` derived from the episode/job seed and pass it through. Avoid global RNG state.

Also avoid nondeterministic ordering from sets when building prompts. If any `set` is converted to a list for prompt-visible output, sort it first.

---

## Answers to your numbered questions

### 1. `cast_one_character()` repair branch

Yes, the branch fires only on the last attempt and only when there is a prior truthy `last_raw`.

For `max_attempts=1`, it performs one normal attempt and no repair. That’s sane.

I’d validate:

```python
max_attempts >= 1
```

not necessarily `>= 2`.

But change `and last_raw` to `and last_raw is not None`.

---

### 2. `lock_cast()` voice exhaustion

No obvious off-by-one if the check happens before each slot consumes a voice.

But I’d add a preflight check anyway:

```python
if len(available_voices) < open_slots_needed:
    raise CastingFailedError(...)
```

This is clearer, fails before spending LLM calls, and protects future changes.

Keep the per-loop check as a defensive assertion if desired.

---

### 3. `_extract_json_block()`

The known weakness of the “first `{` to last `}`” approach is that it can produce confusing garbage when the model emits:

```text
Here is one example:
{"foo": 1}

Final answer:
{"bar": 2}
```

A naïve extractor returns:

```json
{"foo": 1}

Final answer:
{"bar": 2}
```

Then `json.loads()` raises an “extra data” or syntax-ish error.

A brace-balance precheck is not hugely valuable unless it is quote-aware. Braces inside strings make simple balancing misleading.

I’d probably leave the caller’s `try/except` as-is, but improve the error message to say something like:

```text
Could not parse a single JSON object from model response
```

If you want a modest robustness improvement, use `json.JSONDecoder().raw_decode()` from candidate `{` positions and accept the first complete object with only whitespace after it. But that may be more complexity than this module needs.

---

### 4. `_format_prior_entry()`

#### 4a. Trailing punctuation

Current:

```python
.rstrip(',.; ')
```

is okay, but I’d include `!?` and maybe `:` because it costs nothing:

```python
.rstrip(",.;:!? ")
```

This is harmless and avoids:

```text
"hard-bitten detective!..."
```

#### 4b. Prior-cast context growth

I think your token estimate is high. Sixty **characters** is not sixty tokens. Four prior entries at 60 chars each is probably closer to 50–80 tokens total, plus names/voices, not 240 tokens.

With `num_characters <= 6`, I would not add last-N truncation yet.

If the cap grows later, use a simple char-budgeted formatter rather than arbitrary last-N:

```python
max_prior_chars = 400
```

and stop adding entries once the rendered string would exceed that.

---

### 5. `config/cast_pools.py:pick_first_last()`

Fifty retries is fine. It’s overkill statistically, but not harmful.

What I would not love is silently accepting a collision if name uniqueness is meant to be a contract.

Options:

1. If uniqueness is best-effort: current behavior is okay.
2. If uniqueness is required: after 50 misses, pick deterministically from the remaining Cartesian product or raise.

Given your pool size, the fallback should essentially never happen at current cast sizes. I would not add logging unless you already have a debug logger. Logs can become determinism-adjacent clutter and aren’t needed here.

A better test than logging: monkeypatch the RNG/name pools to force retry exhaustion and assert the intended fallback behavior.

---

### 6. `config/cast_pools.py:open_voice_pool()`

Dropping role-shaped tags like `"officer"`, `"pilot"`, `"android"` sounds correct.

Voice choice should communicate vocal/timbre qualities, not accidentally bias the character concept. Role tags create feedback loops:

```text
voice: officer
```

may cause the LLM to make the character an officer even when the story slot wanted something else.

So yes, keep only vocal/adjectival tags.

The only caveat: some “role-shaped” tags can also imply vocal texture. `"android"` might imply synthetic/flat/metallic delivery. But I’d rather encode that as explicit vocal tags:

```python
["synthetic", "flat", "precise"]
```

rather than keep `"android"`.

---

### 7. Imports in `nodes/_otr_casting.py`

The relative-then-absolute import fallback is a normal, acceptable pattern for ComfyUI custom nodes plus pytest harnesses.

I would avoid `importlib.util.spec_from_file_location()` here. Too much complexity.

The main issue with:

```python
try:
    from ..config.cast_pools import ...
except (ImportError, ValueError):
    from config.cast_pools import ...
```

is that it can accidentally mask a real `ImportError` raised inside `cast_pools.py`.

If `cast_pools.py` is dependency-free, risk is low.

Cleaner version:

```python
try:
    from ..config.cast_pools import ...
except ImportError as exc:
    from config.cast_pools import ...
```

I’m not convinced you need `ValueError`; modern Python relative import failures are usually `ImportError`.

If you have observed `ValueError` under ComfyUI, keep it. Otherwise, drop it.

---

### 8. Announcer 50/50 test, 200 trials, 80–120 band

Your math is right: mean 100, std dev about 7.07, so 80–120 is roughly ±2.8 sigma.

That gives a real but small flake probability. For a unit test, I’d rather make it deterministic than statistical.

Best options:

1. Use a fixed local RNG seed and assert the exact expected count.
2. Inject a fake RNG that alternates male/female.
3. If you keep it statistical, loosen to maybe `75–125`.

Given your determinism requirement, I’d prefer option 1 or 2.

---

### 9. `CastingResponse` schema: `character_description` 10–200 chars

That range is reasonable.

I would **not** forbid leading articles like `"A "` or `"The "`. They’re natural in one-line visual/personality descriptions, and banning them will cause unnecessary retries for little gain.

More useful validations:

- strip whitespace before length check
- forbid newlines
- maybe forbid tabs/control characters
- maybe require at least one alphabetic character

Example intent:

```python
description = description.strip()
if "\n" in description:
    raise ValueError("description must be one line")
```

Also consider `extra="forbid"` on the Pydantic model if you want the cast contract to be strict.

---

### 10. `CastingFailedError`

Raw responses are sufficient for parse/schema debugging, but not always sufficient for contract debugging.

I would add optional snapshots:

- `available_voices`
- `prior_cast`
- maybe `character_slot` / `slot_index`

But keep them as copied, compact data, not mutable references.

Example:

```python
self.available_voices = tuple(available_voices)
self.prior_cast = tuple(prior_cast)
```

I would not necessarily dump all of that into `str(error)` by default. Store it on the exception for debug/test inspection.

This would help reproduce failures caused by:

- duplicate name rejection
- duplicate voice rejection
- bad prior-cast conditioning
- voice pool exhaustion

The raw LLM response alone won’t explain those.

---

## Other technical flags worth considering

### A. Add `extra="forbid"` to `CastingResponse`

If not already present, I’d make the schema strict.

LLMs often emit:

```json
{
  "character_name": "...",
  "character_description": "...",
  "voice_id": "...",
  "reasoning": "..."
}
```

If you silently accept extra keys, the contract can drift without tests noticing.

---

### B. Ensure length validation happens after stripping

If using Pydantic, make sure:

```text
"          Bob        "
```

does not pass length because of padding.

Same for descriptions.

---

### C. Test `max_attempts=1` and `max_attempts=0`

I’d add two small tests:

- `max_attempts=1` performs one normal prompt, no repair.
- `max_attempts=0` raises `ValueError`.

This locks down the loop semantics.

---

### D. Test voice-pool insufficiency directly

Monkeypatch `open_voice_pool()` to return one or two voices and ask for more slots. Assert:

- failure happens before LLM calls, if you add fail-fast
- or at least fails cleanly with a useful error

This catches the future off-by-one/regression case better than today’s normal pool tests.

---

### E. Test multi-object JSON extraction

Add one test for:

```text
Here is an example:
{"bad": true}

Final:
{"character_name": "...", ...}
```

Even if you choose not to support it, the test documents current behavior.

---

## Bottom line

I’d make these changes before wiring into the writer:

1. Use explicit deterministic RNG plumbing instead of global `random`.
2. Add `lock_cast()` preflight capacity check.
3. Tighten `cast_one_character()` attempt validation and use `last_raw is not None`.
4. Truncate raw failed response inside repair prompts.
5. Prefer deterministic announcer balance tests over statistical tolerance.

Everything else is mostly polish.
