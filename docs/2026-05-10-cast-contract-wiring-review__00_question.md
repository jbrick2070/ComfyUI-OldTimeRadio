# Question -- 2026-05-10

WIRING-ONLY CODE REVIEW. Just shipped commit 3e1509e to v2.0-alpha
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
