# Synthesis -- 2026-05-10

**Question:** WIRING-ONLY CODE REVIEW. Just shipped commit 3e1509e to v2.0-alpha
of https://github.com/jbrick2070/ComfyUI-OldTimeRadio. This commit
wires the cast contract module (drafted across 7 prior commits)
into OTR_LedgerScriptWriter.

ARCHITECTURE IS SETTLED -- do NOT relitigate the control-plane vs
prose-plane split, the lean-prompts strategy, the per-character LLM
call shape, the ANNOUNCER=Kokoro / Bark=characters TTS namespace
split, or the LLM-agnostic constraint. Those are locked.

What I want you to flag: STRUCTURAL WIRING DEFECTS only. The kind
of thing that would only show up on the first real run against the
local Mistral-Nemo if I missed it now.

THE NEW run() FLOW (excerpt from nodes/OTR_LedgerScriptWriter.py):

```python
def run(self, episode_title="", target_words=350, num_characters=2,
        seed=0, model_id=DEFAULT_MODEL_ID, ..., perfect_run_spacesaver=False):

    # A. Resolve widget inputs (RSS news_seed fetch happens here).
    resolved = _resolve_inputs(...)

    # B. Late imports + LLM cache + generate_fn.
    import random as _random
    from . import _otr_outline as _OTRO
    from . import _otr_casting as _OTRCAST
    from . import production_ledger as _PL
    ...
    cache_entry = _OTRML.load_llm(...)
    generate_fn = _build_truncating_generate_fn(...)

    # D.1 LEDGER UP FRONT (moved from old position F).
    led = _PL.new_ledger(episode_id=None)
    episode_id = led.episode_id
    audio_dir = Path(led.out_dir)
    episode_root = audio_dir.parent
    meta = led.data.setdefault("meta", {})
    meta["cast_status"] = "building"
    meta["requested_num_characters"] = resolved["num_characters"]
    meta["episode_seed"] = int(seed)

    # D.2 Style auto-resolve.
    if resolved["style_pending"]:
        generated_style = _generate_style_via_llm(
            generate_fn, resolved["news_seed"],
            temperature=resolved["temperature"],
        )
        resolved["style"] = generated_style

    # D.3 LOCK THE CAST. seed=0 means "fresh random.Random()" inside
    # _otr_casting.lock_cast (non-deterministic). Non-zero seeds it.
    # LEMMY's 11% roll is INDEPENDENT (SystemRandom) and never
    # affected by `seed`.
    cast_rng = _random.Random(int(seed)) if int(seed) != 0 else None
    cast_rows, cast_meta = _OTRCAST.lock_cast(
        generate_fn,
        num_characters=resolved["num_characters"],
        news_seed=resolved["news_seed"],
        style=resolved["style"],
        rng=cast_rng,
    )
    led.set_cast(cast_rows)
    meta["cast_status"]           = "locked"
    meta["cast_locked"]           = True
    meta["cast_contract_version"] = "cast-v1"
    meta["cast_contract"] = {
        "lemmy_hit":              cast_meta["lemmy_hit"],
        "casting_attempts":       cast_meta["casting_attempts"],
        "num_characters_request": cast_meta["num_characters_request"],
        "num_characters_locked":  cast_meta["num_characters_locked"],
    }

    # Build the name->char_id index the per-beat composer needs.
    # Excludes ANNOUNCER (announcer-role beats hardcode "announcer"
    # cid downstream, not the c01 cast row's char_id).
    char_id_by_name: dict[str, str] = {
        row["name"]: row["char_id"]
        for row in cast_rows
        if row["name"] != "ANNOUNCER"
    }
    character_cast: tuple[str, ...] = tuple(char_id_by_name.keys())

    # D.5 OUTLINE consumes the locked cast.
    outline_req = _OTRO.OutlineRequest(
        news_seed=resolved["news_seed"],
        style=resolved["style"],
        character_cast=character_cast,
        target_words=resolved["target_words"],
    )
    outline = _OTRO.generate_outline(generate_fn, outline_req)

    # ... rest of pipeline unchanged: word-budget check, episode_canon,
    # per-beat composer that reads cast from led.data["cast"] and uses
    # char_id_by_name, title regen, ledger save.
```

THE NEW OutlineRequest SHAPE (excerpt from nodes/_otr_outline.py):

```python
@dataclass(frozen=True)
class OutlineRequest:
    news_seed: str
    style: str
    target_words: int
    character_cast: tuple[str, ...] = ()  # 1-6 ALL-CAPS, locked, ANNOUNCER excluded

    def __post_init__(self) -> None:
        n = len(self.character_cast)
        if not (1 <= n <= 6):
            raise ValueError(...)
        if self.target_words < 5:
            raise ValueError(...)
        for name in self.character_cast:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(...)
            if name != name.upper():
                raise ValueError(...)

    @property
    def cast_size(self) -> int:
        return len(self.character_cast)  # back-compat


def _build_user_prompt(req: OutlineRequest) -> str:
    cast_line = ", ".join(req.character_cast)
    return (
        f"Plan a science-fiction audio drama outline.\n\n"
        f"Science story (the factual seed): {req.news_seed}\n"
        f"Style: {req.style}\n"
        f"Cast (already chosen -- use exactly these names in "
        f"character-role beats): {cast_line}\n"
        f"Target total dialogue length: ~{req.target_words} words "
        f"(sum of per-beat target_words should land near this number).\n\n"
        f"Build a dramatic outline that extrapolates from the science story "
        f"in the chosen style. Echo the cast list verbatim in the JSON "
        f"\"cast\" field. Return only the JSON outline."
    )


# in generate_outline, after pydantic validates the Outline:
locked_set = set(req.character_cast)
outline_set = set(outline.cast)
if outline_set != locked_set:
    extra = outline_set - locked_set
    missing = locked_set - outline_set
    err_msg = (
        "CastContractError: outline.cast drifted from locked "
        f"character_cast. extra (invented): {sorted(extra)!r}, "
        f"missing (dropped): {sorted(missing)!r}. Expected "
        f"exactly: {sorted(locked_set)!r}"
    )
    attempts.append((last_raw, err_msg))
    continue  # standard reroll/repair
```

QUESTIONS -- structural wiring only, not architecture:

1. Is there a real order-of-operations bug in this flow? Specifically:
   does new_ledger() running BEFORE style auto-resolve create any
   issue I'd hit on a real run? The ledger creation just allocates
   an episode_id and out_dir; nothing structural changes after that.
   But style auto-resolve also makes an LLM call -- is there any
   ledger-mutating side effect from _generate_style_via_llm I'd
   need to worry about?

2. Below code in OTR_LedgerScriptWriter.py: an `outline.cast` field
   still exists on the Outline pydantic model (the LLM echoes the
   locked names there). Downstream uses of outline.cast: are there
   any other consumers in the codebase that read this field besides
   the writer? If so, do they need updating? I removed the writer's
   reliance on outline.cast (it now uses char_id_by_name from the
   locked rows directly) but I'm worried other modules may grep
   outline.cast expecting the LLM-generated names.

3. The ANNOUNCER exclusion when building character_cast:
       row["name"] != "ANNOUNCER"
   Is a string-literal compare. The locked cast row format is
   guaranteed by config.cast_pools.pick_announcer to use the exact
   string "ANNOUNCER". Any edge case where this comparison could
   miss (lowercase, trailing whitespace, etc.)? Should I belt-and-
   braces with `row["name"].strip().upper() != "ANNOUNCER"`?

4. The `seed=0 means non-deterministic` convention. Today the
   widget default is 0. If the user never touches the seed widget,
   they get fresh random.Random() each run -- so episodes are
   non-reproducible by default. Two structural concerns:
   (a) Is `seed=0` a safe sentinel? Python's random.Random(0) is a
       valid seed that gives a specific deterministic sequence. So
       a user who DELIBERATELY sets seed=0 gets non-determinism,
       which is the opposite of intent. Should I use `seed=-1` or
       some other sentinel, or accept the convention as documented?
   (b) The meta stamp `meta["episode_seed"] = int(seed)` always
       records what the widget said -- so seed=0 in the ledger
       means "use process RNG", not "the seed was 0." Is that
       acceptable telemetry or do I need a separate `seed_mode`
       field?

5. The LEMMY 11% roll uses SystemRandom inside config.cast_pools.
   It is NOT affected by `seed`. So even with seed=42, two runs of
   the same widget config can differ on whether LEMMY hits. Is
   that documented well enough in the seed widget tooltip and in
   the lock_cast docstring? Will a user reading the ledger see
   episode_seed=42 + lemmy_hit varying and be confused?

6. The cast_contract meta dict embeds `casting_attempts` as a list
   of ints (one per open slot, telemetry-only). Is dumping that
   into the meta safe across all consumers that read led.data["meta"]?
   Specifically, the legacy `gen_params_initial` stamp also lives
   in meta later in the same run() flow -- does this commit cause
   any meta-key collision I might not have seen?

7. The line "meta = led.data.setdefault("meta", {})" appears in
   my new D.1 block AND in the legacy section K stamp later in
   run() (gen_params_initial). Is this safe? setdefault returns
   the existing dict if present, so the legacy stamp adds to the
   same dict my D.1 / D.4 stamps already populated. Or is there a
   shadow-binding issue where the second setdefault rebinds a
   local name and the legacy code writes to a different dict?

8. config.cast_pools.lemmy_row() / pick_announcer() / lock_cast()
   stamp `voice_params=None` on every row. production_ledger.set_cast()
   persists it. Downstream consumers (Bark renderer, Kokoro renderer,
   per-line composer) do NOT yet read voice_params. Is there a real
   ledger-schema-version issue here, or is voice_params=None safely
   ignored by every existing consumer (since it's a brand-new field)?

9. The post-pydantic cast-drift check in generate_outline (set
   equality between outline.cast and req.character_cast) -- is set
   equality the right semantic? What if the LLM returns the right
   names but with duplicates? E.g. character_cast=("ALICE", "BOB")
   and outline.cast=["ALICE", "ALICE", "BOB"]. Set equality would
   pass; len() inequality wouldn't. Should I tighten to
   `Counter(outline.cast) == Counter(req.character_cast)` or
   `sorted(outline.cast) == sorted(req.character_cast)`?

10. Anything else structural you'd flag.

What you have: just this document. No need to fetch the repo. Focus
strictly on wiring defects -- ordering bugs, off-by-ones, missed
consumers, edge cases. Skip nits.

---

## ChatGPT (gpt-5.5)

I can’t verify actual repo-wide consumers or cite real line numbers from the GitHub commit because I’m only using the excerpts you provided. So for questions like “does any other module grep/use `outline.cast`?” I can only flag the structural risk and the exact searches I’d run.

## Short verdict

No obvious fatal ordering bug from moving `new_ledger()` before style auto-resolve, **assuming** `_generate_style_via_llm()` is pure with respect to the ledger.

The bigger structural risks are:

1. **Seed semantics are currently misleading / non-reproducible**, especially because LEMMY uses `SystemRandom`.
2. **`outline.cast` set-equality is too weak**; duplicates can pass.
3. **Legacy callers of `OutlineRequest` may now break**, because `character_cast` has a default but `__post_init__` rejects the default.
4. **Potential off-by-one / meaning mismatch around `num_characters`**: confirm it means non-announcer character count, not total cast rows.
5. **`char_id_by_name` silently collapses duplicate names**; add an assertion after locking the cast.

---

## 1. `new_ledger()` before style auto-resolve

Structurally safe **if** `_generate_style_via_llm()` only calls `generate_fn` and does not touch global/current ledger state.

The only real behavioral change is failure mode:

```python
led = _PL.new_ledger(...)
...
if resolved["style_pending"]:
    generated_style = _generate_style_via_llm(...)
```

If the style LLM call fails, you may now have an allocated `episode_id` / `out_dir` before the run aborts. That is not a logical wiring defect unless you require “no artifact directory on failed preflight.”

Things to check:

- `_generate_style_via_llm()` should not import/use `production_ledger`.
- It should not assume a ledger has not yet been created.
- It should not mutate `resolved` except via the returned style you assign.

Given your description, I would not block on this. Worst case is orphaned/empty episode directories on early LLM failure.

---

## 2. Other consumers of `outline.cast`

Cannot confirm without repo grep.

Structurally, this is the exact thing I’d check:

```bash
grep -R "outline\.cast" nodes/
grep -R "\.cast" nodes/_otr_*.py nodes/OTR_LedgerScriptWriter.py
grep -R "OutlineRequest" nodes/ tests/
```

Two separate risks:

### A. Remaining runtime consumers of `outline.cast`

If downstream code still uses:

```python
for name in outline.cast:
    ...
```

or uses `outline.cast` to build speaker maps, that code may still be relying on LLM-returned order/content. Since you now have locked cast rows, all speaker-to-char-id mappings should derive from `cast_rows` / ledger cast, not from `outline.cast`.

Keeping `outline.cast` as a validated echo is fine, but it should be treated as **checksum/contract validation only**, not source of truth.

### B. Legacy `OutlineRequest` callers

This is more likely to bite on first run/tests.

Your new dataclass has:

```python
character_cast: tuple[str, ...] = ()
```

but `__post_init__` rejects zero names:

```python
if not (1 <= n <= 6):
    raise ValueError(...)
```

So the default is not actually valid. Any old caller doing something like:

```python
OutlineRequest(
    news_seed=news_seed,
    style=style,
    target_words=target_words,
)
```

will now fail immediately.

That may be intentional, but it is not “back-compat” in init shape. The `cast_size` property only preserves read compatibility after construction.

If `OTR_LedgerScriptWriter` is the only caller, fine. If tests or other nodes instantiate it directly, update them.

---

## 3. ANNOUNCER exclusion

Current code:

```python
char_id_by_name = {
    row["name"]: row["char_id"]
    for row in cast_rows
    if row["name"] != "ANNOUNCER"
}
```

If `pick_announcer()` guarantees the exact string `"ANNOUNCER"`, this is structurally fine.

That said, the belt-and-braces version is cheap and safer:

```python
char_id_by_name = {
    row["name"].strip(): row["char_id"]
    for row in cast_rows
    if str(row.get("name", "")).strip().upper() != "ANNOUNCER"
}
```

But I would go one step better if rows have role/type metadata:

```python
if row.get("role") != "announcer"
```

or:

```python
if row.get("char_id") != "announcer"
```

depending on your schema.

Given your note that the ANNOUNCER cast row may have a char id like `c01` while downstream hardcodes announcer lines as `"announcer"`, name-based exclusion is acceptable, but I would still normalize. The risk is low, the cost is tiny.

Also, add an assertion immediately after building the map:

```python
if not char_id_by_name:
    raise RuntimeError("Cast lock produced no non-announcer characters.")
```

---

## 4. `seed=0` as nondeterministic sentinel

### 4a. Is `seed=0` safe?

Structurally, no. It is confusing because `random.Random(0)` is a perfectly valid deterministic seed.

This line:

```python
cast_rng = _random.Random(int(seed)) if int(seed) != 0 else None
```

means a user cannot intentionally request deterministic seed zero.

If you want a nondeterministic sentinel, `-1` is clearer:

```python
if int(seed) < 0:
    cast_rng = None
else:
    cast_rng = _random.Random(int(seed))
```

But even that is not enough if reproducibility matters.

Best low-impact pattern:

```python
requested_seed = int(seed)

if requested_seed < 0:
    effective_seed = _random.SystemRandom().randrange(0, 2**63)
    seed_mode = "randomized"
else:
    effective_seed = requested_seed
    seed_mode = "fixed"

cast_rng = _random.Random(effective_seed)

meta["episode_seed_requested"] = requested_seed
meta["episode_seed_effective"] = effective_seed
meta["seed_mode"] = seed_mode
```

That preserves “fresh by default” while making the run reproducible from the ledger.

### 4b. Is `meta["episode_seed"] = 0` acceptable telemetry?

I would not leave it as the only seed field.

Right now:

```python
meta["episode_seed"] = int(seed)
```

does not tell the truth about replayability. `0` means “nondeterministic mode,” not “seed 0.”

At minimum, add:

```python
meta["seed_mode"] = "randomized" if int(seed) == 0 else "fixed"
```

Better:

```python
meta["episode_seed_requested"] = int(seed)
meta["episode_seed_effective"] = effective_seed_or_none
```

If rule C7 / byte-identical reruns matter, you need an effective seed.

---

## 5. LEMMY `SystemRandom` independent of seed

This is the largest structural determinism problem.

If `seed=42` can produce different `lemmy_hit` values across runs, then `seed=42` does **not** fully determine the episode. A user seeing:

```json
"episode_seed": 42,
"lemmy_hit": true
```

in one ledger and:

```json
"episode_seed": 42,
"lemmy_hit": false
```

in another will reasonably think the seed is broken.

Documentation/tooltips are not enough if reproducibility is part of the product contract.

Smallest robust fix: pass the cast RNG into the LEMMY roll too.

For example, instead of `config.cast_pools` internally doing:

```python
SystemRandom().random() < 0.11
```

have `lock_cast()` pass its RNG:

```python
lemmy_hit = cast_pools.roll_lemmy(rng)
```

where:

```python
def roll_lemmy(rng):
    return rng.random() < 0.11
```

If you intentionally want LEMMY to be independent of user seed, then ledger telemetry should say that explicitly:

```python
meta["cast_contract"]["lemmy_rng"] = "system"
meta["cast_contract"]["lemmy_seeded_by_episode_seed"] = False
```

But from a wiring/reproducibility standpoint, I would make LEMMY deterministic under the effective cast seed.

---

## 6. `casting_attempts` list in `meta`

Structurally safe as JSON:

```python
"casting_attempts": [1, 2, 1]
```

A list of native Python ints is fine for normal ledger serialization.

No obvious collision if it is nested here:

```python
meta["cast_contract"] = {
    ...
    "casting_attempts": cast_meta["casting_attempts"],
}
```

That should not collide with later:

```python
meta["gen_params_initial"] = ...
```

unless legacy consumers assume every `meta` value is scalar/string. That would be brittle, but possible. Based on your description, no obvious defect.

I would just ensure `casting_attempts` contains plain `int`, not NumPy ints or custom types.

---

## 7. Repeated `meta = led.data.setdefault("meta", {})`

This is safe.

This:

```python
meta = led.data.setdefault("meta", {})
```

returns the existing dict if present. Rebinding the local name later does not create a shadow-copy. Both local variables point to the same dict unless some intervening code does:

```python
led.data["meta"] = {}
```

or deep-copies/replaces `led.data`.

So this pattern is fine:

```python
meta = led.data.setdefault("meta", {})
meta["cast_status"] = "locked"

...

meta = led.data.setdefault("meta", {})
meta["gen_params_initial"] = ...
```

No shadow-binding issue.

---

## 8. `voice_params=None` in cast rows

`None` serializes safely as JSON `null`.

If downstream consumers truly ignore unknown/new fields, this is fine.

The real structural risks would be:

1. A strict cast-row schema that rejects unknown keys.
2. Code that does:

   ```python
   SomeCastRow(**row)
   ```

   where `SomeCastRow` does not accept `voice_params`.

3. Code that assumes all values are strings and does:

   ```python
   row["voice_params"].items()
   ```

4. Code that passes the full row as `**kwargs` to a renderer that does not accept `voice_params`.

If none of those exist, no ledger schema version bump is strictly required. Since you already have:

```python
meta["cast_contract_version"] = "cast-v1"
```

that is probably enough for this feature.

But if you have an explicit overall ledger schema version elsewhere, adding a new persisted field is technically a schema evolution. Not necessarily a breaking one.

---

## 9. `outline.cast` set equality

Set equality is too weak.

This would pass incorrectly:

```python
req.character_cast = ("ALICE", "BOB")
outline.cast = ["ALICE", "ALICE", "BOB"]
```

because:

```python
set(outline.cast) == {"ALICE", "BOB"}
```

Yes, tighten it.

If order does not matter:

```python
from collections import Counter

if Counter(outline.cast) != Counter(req.character_cast):
    ...
```

If order should echo the prompt exactly, use tuple equality:

```python
if tuple(outline.cast) != tuple(req.character_cast):
    ...
```

Given your prompt says:

> Echo the cast list verbatim in the JSON `"cast"` field.

I would use exact tuple equality unless you intentionally want to tolerate reordering.

Also add duplicate validation to `OutlineRequest`:

```python
if len(set(self.character_cast)) != len(self.character_cast):
    raise ValueError("character_cast contains duplicate names")
```

And probably after pydantic validation:

```python
if len(set(outline.cast)) != len(outline.cast):
    ...
```

The current set check can hide malformed LLM output.

---

## 10. Other structural flags

### A. Confirm `num_characters` means non-announcer count

This is the biggest possible off-by-one.

Your `OutlineRequest` requires:

```python
1 <= len(character_cast) <= 6
```

and `character_cast` excludes ANNOUNCER.

So make sure:

```python
resolved["num_characters"]
```

means “number of Bark/drama characters excluding ANNOUNCER.”

If the widget means “total cast size including announcer,” then a request of `1` would produce only ANNOUNCER, then `character_cast=()`, then `OutlineRequest` throws.

Add a post-lock assertion:

```python
non_announcer_count = len(char_id_by_name)

if non_announcer_count != resolved["num_characters"]:
    raise RuntimeError(
        f"Cast lock count mismatch: requested {resolved['num_characters']} "
        f"non-announcer characters, got {non_announcer_count}."
    )
```

Adjust if `num_characters_locked` has different semantics.

### B. `char_id_by_name` silently collapses duplicate names

This dict:

```python
char_id_by_name = {
    row["name"]: row["char_id"]
    for row in cast_rows
    if row["name"] != "ANNOUNCER"
}
```

will silently collapse duplicates.

Example:

```python
[
    {"name": "ALICE", "char_id": "c02"},
    {"name": "ALICE", "char_id": "c03"},
]
```

becomes:

```python
{"ALICE": "c03"}
```

Add an explicit uniqueness check:

```python
names = [
    row["name"].strip()
    for row in cast_rows
    if row["name"].strip().upper() != "ANNOUNCER"
]

if len(names) != len(set(names)):
    raise RuntimeError(f"Duplicate cast names after lock: {names!r}")
```

This is worth doing even if `lock_cast()` already guarantees uniqueness. It catches contract drift immediately.

### C. `OutlineRequest` default `character_cast=()` is misleading

Because `__post_init__` rejects it, this is not a meaningful default:

```python
character_cast: tuple[str, ...] = ()
```

I would remove the default:

```python
character_cast: tuple[str, ...]
```

But because dataclass non-default fields must precede default fields, the clean shape is probably:

```python
@dataclass(frozen=True)
class OutlineRequest:
    news_seed: str
    style: str
    character_cast: tuple[str, ...]
    target_words: int
```

You are already using keyword construction in the writer, so this should be safe for that path. It will intentionally break stale positional callers loudly.

If you want to minimize churn, leave it but know it is a trap for old callers.

### D. ANNOUNCER char-id mismatch deserves one audit

You wrote:

> announcer-role beats hardcode `"announcer"` cid downstream, not the `c01` cast row’s `char_id`

That can be fine, but audit any code that does:

```python
cast_by_char_id[line.char_id]
```

for every line, including announcer lines.

If announcer dialogue uses `char_id="announcer"` but the ledger cast row says `char_id="c01"`, generic lookup code may fail. If Kokoro rendering is already special-cased by namespace/role, fine.

This is not a new architecture objection; it is just a wiring audit point.

### E. If style generation fails after ledger creation, cast status may remain `"building"`

If you save failure ledgers somewhere, note that:

```python
meta["cast_status"] = "building"
```

is stamped before style auto-resolve and before cast lock.

If `_generate_style_via_llm()` raises and some outer handler saves the ledger, the failure artifact will say cast is building even though casting never started. Not fatal, but if you care:

```python
meta["cast_status"] = "pending_style"
```

then later:

```python
meta["cast_status"] = "building"
```

immediately before `lock_cast()`.

Not required unless you persist failed ledgers.

---

## Recommended minimum patch set

If you want the smallest high-value wiring fixes before first real Mistral-Nemo run:

1. **Make seed telemetry unambiguous.**

   Add:

   ```python
   meta["episode_seed_requested"] = int(seed)
   meta["seed_mode"] = "randomized" if int(seed) == 0 else "fixed"
   ```

   Better: generate and store `episode_seed_effective`.

2. **Stop using unseeded `SystemRandom` for LEMMY if reproducibility matters.**

   Route it through the same effective cast RNG.

3. **Tighten cast drift check.**

   Replace set equality with either `Counter` equality or exact tuple equality.

4. **Add post-lock cast sanity assertions.**

   Check:
   - at least one non-announcer character
   - no duplicate non-announcer names
   - locked non-announcer count matches requested count
   - all non-announcer names are clean uppercase strings

5. **Grep for `OutlineRequest` and `outline.cast` callers.**

   Any caller constructing `OutlineRequest` without `character_cast` will now fail. Any downstream consumer using `outline.cast` as source of truth should switch to locked cast rows.

---

## Gemini (gemini-3.1-pro-preview-customtools)

### 1. Verdict
I PARTIALLY AGREE; ChatGPT correctly identified the severe Rule C7 (reproducibility) violations regarding LEMMY and the `seed=0` sentinel, but it completely missed ComfyUI's specific frontend/backend seed semantics and introduced a Python type-comparison trap that will cause an infinite retry loop.

### 2. Factual Errors in ChatGPT
*   **ComfyUI Seed Semantics:** ChatGPT suggests generating an `effective_seed` in the backend using `SystemRandom()` if `seed < 0`. This fundamentally misunderstands ComfyUI. ComfyUI's frontend handles "randomize" by generating a random 64-bit integer in JS and sending it to the Python backend. The backend should *never* generate its own random seeds; it should blindly trust the incoming `seed` integer to ensure the ComfyUI workflow graph and UI remain the source of truth for reproducibility.
*   **Type Mismatch in Equality Suggestion:** ChatGPT suggested `tuple(outline.cast) == tuple(req.character_cast)` or exact equality. If `outline.cast` is a list (standard JSON/Pydantic array parsing) and `req.character_cast` is a tuple, `outline.cast == req.character_cast` evaluates to `False` even if the contents match. You must explicitly cast to compare them (e.g., `outline.cast == list(req.character_cast)`).

### 3. Important Omissions
*   **The ComfyUI `seed=0` Anti-Pattern:** Your code `cast_rng = _random.Random(int(seed)) if int(seed) != 0 else None` actively fights ComfyUI. If a user sets the ComfyUI seed widget to "fixed" and types `0`, your backend treats it as "unseeded/random", breaking Rule C7. 
*   **LLM Seeding for Style Generation:** You pass `temperature=resolved["temperature"]` to `_generate_style_via_llm` (Step D.2), but you don't explicitly pass the `seed`. If your local Mistral-Nemo inference engine isn't receiving the episode seed for this call, the style output will drift, violating Rule C7.
*   **`OutlineRequest` Default Trap:** ChatGPT caught this, but missed *why* it's so dangerous. `character_cast: tuple[str, ...] = ()` followed by `if not (1 <= n <= 6): raise ValueError` in `__post_init__` means `OutlineRequest()` will crash on instantiation. Dataclasses evaluate defaults *before* `__post_init__`. Any legacy tests doing `OutlineRequest(news_seed="x", style="y", target_words=350)` will instantly fail.

### 4. Short Recommendation
*   **Fix Rule C7 (Determinism):** In `nodes/OTR_LedgerScriptWriter.py`, delete the `if int(seed) != 0` logic. Always use `cast_rng = _random.Random(int(seed))`. Pass `cast_rng` into `config.cast_pools` so LEMMY's 11% roll uses the deterministic sequence, not `SystemRandom`.
*   **Fix Cast Drift Check:** In `nodes/_otr_outline.py`, change the set equality to `if outline.cast != list(req.character_cast):`. This enforces exact order, exact count, and handles the list vs. tuple type difference without triggering false-positive retry loops.
*   **Fix OutlineRequest Default:** Remove the `= ()` default from `character_cast` in `OutlineRequest` to force explicit passing and prevent instantiation crashes in legacy code/tests.
*   **Add Uniqueness Assertion:** In `OTR_LedgerScriptWriter.py`, add `assert len(char_id_by_name) == len(cast_rows) - 1` (assuming 1 announcer) after building the dict to catch silent duplicate name collapsing.
*   **Verify LLM Seeding:** Ensure `generate_fn` and `_generate_style_via_llm` are actually passing the `seed` parameter to the local LLM backend.

### 5. Uncertainties
*   I am uncertain how `resolved["num_characters"]` is defined in your UI. If the user selects "2", does that mean 2 actors *plus* the announcer, or 2 actors *including* the announcer? If the latter, and they select "1", `OutlineRequest` will receive 0 characters, crashing `__post_init__`.
*   I am uncertain if your `pick_announcer()` guarantees the `char_id` is `"c01"` or `"announcer"`. If the announcer's `char_id` is `"c01"`, but downstream beats hardcode `"announcer"`, the downstream composer might crash looking up `"announcer"` in a ledger that only has `"c01"`.

---

## NVIDIA ()



---

## To decide (Claude / human)

- [ ] All three agree:
- [ ] Two-vs-one splits:
- [ ] Facts to verify:
- [ ] Final grounded recommendation:
