# Wiring review — OTR Phases 0-3 (cast-gated reviewer + episode budget + progressive ledger)

You are an experienced Python / ComfyUI engineer reviewing a chunk of just-landed code on the v2.0-alpha branch of ComfyUI-OldTimeRadio. The branch's authoritative design is at https://github.com/jbrick2070/ComfyUI-OldTimeRadio (branch `v2.0-alpha`); HEAD is `2177a0c` plus the uncommitted Phase 0-3 work below.

**Find bugs and improvement opportunities in the wiring between phases.** Be concrete: file + line range + what to do about it. Severity-rate each finding. Skip the in-line code review (we already did that); focus on cross-phase wiring, race conditions, sequencing hazards, footguns, and back-compat traps.

---

## What just shipped (the five phases)

- **Phase 0** — name-roster gate in `nodes/_otr_line_composer.py` (detect + flag phantoms, no reroll; `LineResult(text, compose_flags)` replaces bare-string return).
- **Phase 1** — composer prompt enrichment (static-first layout: STYLE → EPISODE CONTEXT → OUTLINE spine → ALLOWED NAMES → CHARACTER → RECENT DIALOGUE → WRITE LINE) and N=5 sliding window (was 3).
- **Phase 2A** — new `nodes/_otr_episode_budget.py` (acts 1–7), `act_count` widget on the writer, 8 outline validators in `nodes/_otr_outline.py::validate_outline_against_budget`, `Beat.arc_phase` Optional field, beats cap 24→32, JS extension `web/js/otr_act_count_widget.js`, `WEB_DIRECTORY = "./web"` in `__init__.py`.
- **Phase 2B** — progressive ledger writes (`init_lines_from_outline` + per-iteration `update_line_text` + `led.save()` after every line) in `nodes/production_ledger.py`.
- **Phase 3** — `nodes/_otr_ledger_reviewer.py` (three-pass cast-gated reviewer: Pass 1 + Pass 3 share one `audit_cast_contract` function, deterministic Levenshtein repair between, Script Doctor Pass 2, Step 2.5 deterministic phantom-skip fallback) + `nodes/OTR_LedgerScriptReviewer.py` node + `tests/test_phase3_ledger_reviewer.py`.

The synthesis MD that drove this is `04_synthesis (1).md` (uploaded 2026-05-11). The original brief is at `docs/2026-05-10-script-writing-architecture__00_question.md`.

---

## Concrete wiring concerns to scrutinize

Weight your response toward these. Replace any item with something worse you've spotted.

### 1. `compose_line` return-type change

Old: `compose_line(...) -> str`. New: `compose_line(...) -> LineResult` (frozen dataclass: `.text` + `.compose_flags`).

Grep inside the repo shows only `nodes/OTR_LedgerScriptWriter.py` and the in-module self-test consume `compose_line`. The writer has been updated to consume the dataclass. Hidden call-sites anywhere else in the repo (tests/, scripts/, sister packs)? Any back-compat shim that expected bare-string would silently break.

### 2. `act_count` widget appended at END of optional INPUT_TYPES

`act_count` (default 0 = auto-derive) was appended at the END of optional INPUT_TYPES so legacy `widgets_values` arrays (17 entries, no act_count) preserve positional mapping. New default 0 auto-derives via `default_act_count(target_words)`.

ComfyUI maps `widgets_values` POSITIONALLY at workflow-load time onto the widget list returned by `INPUT_TYPES`. Confirm: appending at end is back-compat-safe; inserting mid-list shifts every subsequent widget's value. `run()` signature has `act_count=0` as the LAST kwarg; ComfyUI invokes node functions with kwargs (not positional) so run-signature order shouldn't matter — but confirm. Anything else about `widgets_values` ordering I missed?

### 3. EpisodeBudget threading via `OutlineRequest.budget: object` (lazy duck-type)

To avoid module-load coupling, `_otr_outline.OutlineRequest.budget` is typed `object` and `_get_budget(req)` does a duck-type check (returns `None` if budget lacks `arc_phases / per_phase_words / per_phase_beats`).

```python
def _get_budget(req):
    b = getattr(req, "budget", None)
    if b is None:
        return None
    if (hasattr(b, "arc_phases") and hasattr(b, "per_phase_words")
            and hasattr(b, "per_phase_beats")):
        return b
    return None
```

Failure mode I'm worried about: a half-built dict that quacks like an EpisodeBudget but isn't one. `_get_budget` lets it through, validators index `arc_phases[k]`, garbage validators fire, outline rejected for the wrong reason. Should this be tightened to `isinstance(b, EpisodeBudget)` even at the cost of import coupling? Cleaner pattern?

### 4. arc_phase reroll signal — does the outline LLM actually fix it on attempt 2/3?

When budget is non-None, the outline prompt includes "Every voiced beat MUST carry an `arc_phase` field set to one of: setup, complication, resolution." The pydantic `Beat.arc_phase: Optional[str] = None` (so LLM may omit it). Then `validate_outline_against_budget` rejects with structured strings like `Beat b003 has arc_phase=''; not in budget arc_phases=['setup', 'complication', 'resolution']`.

The existing `_REPAIR_PROMPT_TEMPLATE` feeds the error verbatim into attempt 3 (the repair call). Does a 12B Mistral-Nemo actually reliably patch this on the repair attempt? Or do we need to make `arc_phase` REQUIRED in the pydantic schema when budget is non-None?

Trade-off: making `arc_phase` required-in-pydantic gives a sharper "missing field" error, but conditional-requirement (required only when budget is set) is awkward in pydantic without a discriminator.

### 5. Progressive ledger writes — race conditions vs `_otr_ledger.save_ledger_safe`

The writer now calls `led.save()` after every line (~15-25× per episode instead of once). Each save is `_merge_with_disk(dict(self.data), path)` → write to `path + ".tmp"` → `os.replace(tmp, path)`. Atomic per save.

Two specific worries:

(a) There's no other writer touching this file during a single ComfyUI run — so the read-modify-write cycle is safe. Confirm.

(b) After the compose loop, `_OTRNW.override_announcer_close(led.data["lines"], nc_brief)` mutates `led.data["lines"]` in place, then `_OTRL.patch_line_text(led.data, ...)` recomputes counts, then a FINAL `led.save()`. Then the reviewer (downstream node) may rewrite the same announcer line via Pass 2 Script Doctor. Which precedence is right? Right now: `news_close_brief` lands first (in writer), then reviewer's doctor sees the news-stamped text and MAY rewrite over it. Is that the intended precedence?

### 6. Reviewer `copy.deepcopy(ledger_data)` and `led.data.update(candidate)` semantics

`review_ledger`:
1. `original_snapshot = copy.deepcopy(led.data)`
2. `candidate = copy.deepcopy(led.data)`
3. On failure path: `led.data.update(original_snapshot)` to restore.
4. On success: `led.data.update(candidate)` to commit.

`dict.update()` is shallow. If `candidate` adds new top-level keys (e.g., `meta.reviewer_disposition`), those land in `led.data`. But if `candidate` REMOVED keys, the original keys persist in `led.data` — leak risk?

Better to clear-then-update or rebind: `led.data.clear(); led.data.update(candidate)` (preserves the `self.data` reference). Or `led.data = candidate` but the existing `led.save()` path holds `self.data` by reference — would `get_ledger()` consumers see the new dict or stale?

### 7. Reviewer skip behavior on TTS

Reviewer's Step 2.5 sets `line.skip = True` + `line.tts_skip_reason = "phantom_titled_name:..."` for any titled phantom that survives Pass 1's auto-remap and the Script Doctor. The synthesis (§11.4) says `OTR_BatchBarkGenerator` honors `lines[].skip`. Has anyone verified that Bark actually checks this field today? If not, a phantom-skipped line will still get TTS'd and the "Dr. Patel" voice will still hit the audio.

Should I add a `skip`-checking test fixture in the Bark batch test suite, or defer until soak surfaces the failure?

### 8. ComfyUI `WEB_DIRECTORY` auto-serve assumption

`WEB_DIRECTORY = "./web"` in `__init__.py`; JS at `web/js/otr_act_count_widget.js`. ComfyUI's documented behavior: auto-serve anything under WEB_DIRECTORY at `/extensions/<custom-node>/` on server start.

The JS uses `import { app } from "../../scripts/app.js"`. Is that path correct relative to where ComfyUI mounts the extension? Common pitfalls (path differences across ComfyUI Desktop versions, CORS, browser cache holding the stale UI) to check before the next ComfyUI launch?

### 9. Reviewer LLM cost on wall-clock

Three extra Mistral-Nemo calls per episode: Pass 1 audit (~2k-tok prompt), Pass 2 doctor (~3.5k-tok prompt), Pass 3 audit (~2k-tok prompt). NO KV-cache reuse (deferred per synthesis §8 path b).

Order-of-magnitude estimate of added wall-clock per episode on RTX 5080 / Mistral-Nemo 12B? Back-of-envelope: ~1.5-2.5 minutes added. Worth flagging soak baseline if off. Architecturally cheaper alternative I should consider — e.g., running Pass 1 + Pass 3 with a smaller distilled model since they're pure JSON-schema-validation tasks?

### 10. Static-first prompt ordering for future KV cache

Composer prompt puts everything stable (style + canon + outline_spine + allowed_roster) BEFORE everything variable (character_voice_card + recent_dialogue + WRITE LINE). Once KV-cache reuse lands in `_otr_model_loader.make_generate_fn`, the cached prefix covers everything up to CHARACTER.

Per-call CHARACTER changes between speakers in alternating-cast scenes — so CHARACTER position is correct (just past the cache boundary). `allowed_roster` is sorted alphabetically inside `_build_user_prompt` for byte-stable prefix.

`outline_spine` renders ALL beats. If outline LLM produces 20 beats, spine is ~20 lines, ~200 tokens. Whole composer prompt ~700-800 tokens. Acceptable on 5080 even without KV. Open question: any tighter prompt structure I missed that would let me hit ≤500 tokens per call without trimming the spine?

---

## What to return

For each bug or improvement, give me:

1. **File + line range** (best effort).
2. **What's wrong / what would be better.**
3. **Severity:** critical (corrupts ledger / crashes / silent data loss), high (wrong behavior under common scenarios), medium (footgun for future maintainers), low (cosmetic).
4. **Fix sketch:** one-paragraph description of the change.

Keep the writeup tight. If you spot an entire category of wiring concern not in the 1–10 list above, raise it under a "G — gaps" section.

---

## Relevant code excerpts

### `nodes/_otr_line_composer.py` — LineResult + LineRequest

```python
@dataclass(frozen=True)
class LineRequest:
    speaker: str
    intent: str
    mood: str
    target_words: int
    canon_header: str
    last_lines: list[tuple[str, str]]
    allowed_roster: frozenset[str] = field(default_factory=frozenset)
    style_descriptor: str = ""
    outline_spine: str = ""
    character_voice_card: str = ""
    arc_phase: str = ""

@dataclass(frozen=True)
class LineResult:
    text: str
    compose_flags: tuple[str, ...] = ()

def build_allowed_roster(cast_rows, key_terms=(), *, include_announcer=True):
    roster = set()
    if include_announcer:
        roster.add("ANNOUNCER")
    for row in cast_rows or ():
        name = ""
        if isinstance(row, dict):
            name = str(row.get("name") or "").strip()
        elif isinstance(row, (list, tuple)) and row:
            name = str(row[0] or "").strip()
        elif isinstance(row, str):
            name = row.strip()
        if name:
            roster.add(name.upper())
    for term in key_terms or ():
        term_s = str(term or "").strip()
        if term_s:
            roster.add(term_s.upper())
    return frozenset(roster)
```

### `nodes/_otr_episode_budget.py` — EpisodeBudget + compute

```python
def compute_episode_budget(target_words, act_count, include_act_breaks, num_characters):
    if target_words < 30: raise InvalidEpisodeBudgetError(...)
    if not (1 <= act_count <= 7): raise ...
    if num_characters < 1: raise ...
    dflt = default_act_count(target_words)
    max_allowed = max_act_count(target_words)
    if act_count < dflt: raise ...
    if act_count > max_allowed: raise ...
    cfg = ACT_COUNT_CONFIG[act_count]
    return EpisodeBudget(
        act_count=act_count,
        arc_phases=cfg["arc_phases"],
        per_phase_words=tuple(round(target_words * f) for f in cfg["act_word_fractions"]),
        per_phase_beats=tuple(cfg["voiced_beats_per_act"]),
        words_per_beat_range=tuple(cfg["words_per_beat_range"]),
        music_inter_count=(act_count - 1) if include_act_breaks else 0,
        announcer_beats=2,
        cast_size=num_characters,
        target_words=target_words,
    )
```

### `nodes/_otr_outline.py` — validator hook inside generate_outline

```python
# After pydantic + cast-membership check, before returning:
budget_violation = validate_outline_against_budget(outline, req)
if budget_violation is not None:
    err_msg = f"OutlineBudgetViolation: {budget_violation}"
    attempts.append((last_raw, err_msg))
    continue
```

### `nodes/OTR_LedgerScriptWriter.py` — progressive ledger writes (Phase 2B)

```python
# After outline validates, before the composer loop:
led.init_lines_from_outline(outline, char_id_by_name)
led.save()

# Inside the per-beat loop, after compose_line returns LineResult:
led.update_line_text(beat.beat_id, cleaned)
_OTRL.patch_line_fields(
    led.data, beat.beat_id,
    {"char_id": cid, "traits": traits, "compose_flags": list(beat_compose_flags)},
)
led.save()

# After the loop, news-wiring overlay operates on led.data["lines"]:
overridden = _OTRNW.override_announcer_close(led.data["lines"], nc_brief)
if overridden is not None:
    _OTRL.patch_line_text(led.data, overridden["line_id"], overridden["text"])

# Final aggregate + save:
meta["compose_flag_summary"] = _OTRLC.aggregate_compose_flags(led.data)
led.save()
```

### `nodes/_otr_ledger_reviewer.py` — review_ledger flow

```python
def review_ledger(generate_fn, led):
    ledger_data = led.data
    meta = ledger_data.setdefault("meta", {})
    if meta.get("skip_reviewer"):
        ...  # G9 bypass
    cast_rows = list(ledger_data.get("cast") or [])
    cast_roster_upper = {row.get("name", "").upper()
                        for row in cast_rows if row.get("name")} | {"ANNOUNCER"}
    voiced_beats = sum(
        1 for ln in ledger_data.get("lines", []) or []
        if (ln.get("speaker_role") or "") in ("character", "announcer")
    )
    edit_cap = compute_edit_cap(voiced_beats)
    original_snapshot = copy.deepcopy(ledger_data)
    candidate = copy.deepcopy(ledger_data)
    pre_audit = audit_cast_contract(generate_fn, candidate, label="pre")
    # speaker_unknown -> cast_unrecoverable (restore + return)
    repairs_applied = apply_deterministic_cast_repairs(candidate, pre_audit, cast_rows)
    doctor_report = run_script_doctor(generate_fn, candidate, cast_rows, edit_cap)
    # needs_full_rerun / too_many_edits -> restore + return
    edits_applied = apply_doctor_edits(candidate, doctor_report, edit_cap=edit_cap)
    phantom_skip_count = apply_phantom_skip_fallback(candidate, cast_roster_upper)
    post_audit = audit_cast_contract(generate_fn, candidate, label="post")
    final_phantoms = _final_phantom_check(candidate, cast_roster_upper)
    if not (post_audit.pass_clean and not post_audit.violations and not final_phantoms):
        led.data.update(original_snapshot)   # shallow update; concern #6
        ...
        return disp
    led.data.update(candidate)
    ...
```

End of brief. Please be specific and concrete; vague critique is less useful than a precise "in `_otr_ledger_reviewer.py::review_ledger`, the `led.data.update(candidate)` pattern leaves leftover keys when candidate removed them — replace with `led.data.clear(); led.data.update(candidate)`."
