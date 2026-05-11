# Round A -- ChatGPT (gpt-5.5) elapsed=124.3s

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
