# Round A -- ChatGPT (gpt-5.5) elapsed=129.6s

## Short recommendation

Use **two separate creative passes**:

1. **Intro pass**: run when filling the first announcer beat.
2. **Outro pass**: run after the character script is complete, immediately before final ledger validation/export.

Do **not** keep the current unconditional `override_announcer_close` behavior. Replace it with a corrected deterministic fallback path: if the creative outro pass fails validation, stamp a deterministic close derived from `news_close_brief`.

This gives you the biggest payoff with the least architectural disruption: the ledger shape stays unchanged, downstream audio sees the same `lines[]` structure, and you avoid touching low-level VRAM/model-loading behavior.

One caveat: if “byte-identical to baseline” means “identical to outputs produced by the current broken system,” then any text-generation change violates that. If C7 means “byte-identical between repeated runs with the same inputs/config,” this design can comply, assuming your existing LLM/TTS stack is already deterministic.

I cannot honestly cite exact line numbers from the code map alone, so I’ll cite the relevant files and symbols/anchors rather than inventing line numbers.

---

## 1. One combined bookend pass or two separate passes?

Use **two separate passes**.

### Why not one combined pass?

A single “bookend” call that emits both intro and outro sounds attractive because it can enforce tonal coherence, but it creates ordering problems.

The outro wants:

- the finished script,
- the final character arc,
- `news_close_brief`.

The intro wants:

- `script_brief`,
- maybe title/series/tone,
- no dependence on generated dialogue.

If you run the combined call at the end, then you overwrite the intro after the character lines were already composed. If `compose_line` in `nodes/_otr_line_composer.py` uses previous ledger context, the characters may have been written against a generic/placeholder intro and then that intro changes underneath them. That is a subtle coherence hazard.

If you run the combined call at the beginning, the outro cannot see the finished script.

So the clean split is:

- **Intro**: early, before subsequent dialogue depends on it.
- **Outro**: late, after the episode exists.

### How to recover open/close coherence

You do not need one call for coherence. Feed the generated intro text into the outro prompt:

```text
Opening announcer line:
"{intro_text}"

Now write the closing announcer line using news_close_brief and lightly echo the tone, without repeating the opening.
```

That gives you tonal continuity without the ordering hazard.

### Relevant files

- `nodes/_otr_outline.py`  
  Existing deterministic stamping of first and last beat as `speaker_role="announcer"` should remain unchanged.

- `nodes/_otr_line_composer.py`  
  Add `compose_announcer_intro()` and `compose_announcer_outro()` near `compose_line()` / `_build_user_prompt()` so they reuse the existing LLM call machinery and creative-model routing.

- Writer node containing `OTR_LedgerScriptWriter`  
  Change the per-beat loop dispatch so first announcer slot uses `compose_announcer_intro()` and the last announcer slot is filled by the post-loop outro pass.

---

## 2. Should the outro run inside the per-beat loop or post-loop?

Keep it as a **post-loop pass**.

The outro needs the finished script plus `news_close_brief`. Conceptually, it is not just another beat-level line. It is a closing narration pass over the completed episode.

Recommended ordering:

1. Outline already provides first and last announcer slots via `nodes/_otr_outline.py`.
2. In the writer loop:
   - If this is the first announcer beat: call `compose_announcer_intro()`.
   - If this is the final announcer beat: either skip normal composition and leave a deterministic placeholder, or fill with fallback text temporarily.
   - Compose all character lines normally via `compose_line()` in `nodes/_otr_line_composer.py`.
3. After loop:
   - Call `compose_announcer_outro()` using:
     - `news_close_brief`,
     - finished ledger text excluding the outro placeholder,
     - generated intro text if available.
   - Validate result.
   - If invalid, use deterministic fallback.

This also lets you retire the broken “compose generic line, then override” pattern. You no longer need to waste an LLM call composing a generic final announcer line that you intend to replace.

If minimal change is more important than avoiding wasted work, you can leave the old generic final-line composition in place and overwrite it post-loop. But architecturally, skipping the generic final `compose_line()` call is cleaner.

---

## 3. Deterministic fallback design

You want fallbacks that are:

- deterministic,
- local,
- one line,
- no missing bookend,
- no new ledger fields,
- safe for TTS,
- not dependent on another LLM call.

I would use plain string construction from already-deterministic inputs.

### Intro fallback

Input: `script_brief`.

Example:

```python
def fallback_announcer_intro(script_brief: str) -> str:
    brief = clean_one_line(script_brief, max_chars=220)
    if brief:
        return f"Good evening. This is SIGNAL LOST. Tonight, {brief} Listen closely."
    return "Good evening. This is SIGNAL LOST. Tonight, a signal breaks through the static. Listen closely."
```

This is simple, deterministic, and gives the episode a frame even if the intro LLM pass fails.

### Outro fallback

Input: `news_close_brief`.

I would keep the existing news-derived close as the grounding source, but wrap it lightly in announcer voice:

```python
def fallback_announcer_outro(news_close_brief: str) -> str:
    close = clean_one_line(news_close_brief, max_chars=260)
    if close:
        return f"This has been SIGNAL LOST. {close} Good night."
    return "This has been SIGNAL LOST. The report ends, but the signal remains. Good night."
```

If you strongly prefer preserving the old intended behavior, make the fallback exactly the cleaned `news_close_brief` when present:

```python
if close:
    return close
```

But I prefer the wrapped version because it preserves the radio-drama frame instead of dropping into a purely journalistic close.

### Clean one-line helper

Use one shared deterministic sanitizer for both LLM outputs and fallbacks:

```python
def clean_one_line(value: str, max_chars: int) -> str:
    if not value:
        return ""

    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = " ".join(text.split())
    text = text.strip().strip('"').strip("'").strip()

    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:-")
        if text and text[-1] not in ".!?":
            text += "."

    return text
```

Keep this deterministic. Do not include timestamps, random IDs, retry counts, or environment-dependent data.

### Detecting LLM failure or drift

Make the announcer passes return strict JSON, not free text.

Prompt contract:

```json
{"text": "one announcer line here"}
```

Validation should reject:

- invalid JSON,
- missing `text`,
- non-string `text`,
- empty text,
- multiple lines,
- markdown/code fences,
- speaker labels like `ANNOUNCER:`,
- stage directions like `[music]`, `(static)`, `SFX:`,
- excessive length,
- obvious meta text,
- multiple dialogue lines,
- wrong language if the project assumes English.

Example validator:

```python
ANNOUNCER_BAD_PREFIXES = (
    "ANNOUNCER:",
    "HOST:",
    "NARRATOR:",
    "SFX:",
    "MUSIC:",
)

def validate_announcer_line(text: str, *, min_chars=20, max_chars=320) -> tuple[bool, str]:
    text = clean_one_line(text, max_chars=max_chars)

    if not text:
        return False, ""

    upper = text.upper()
    if any(upper.startswith(prefix) for prefix in ANNOUNCER_BAD_PREFIXES):
        return False, ""

    if "[" in text or "]" in text:
        return False, ""

    if "{" in text or "}" in text:
        return False, ""

    if len(text) < min_chars or len(text) > max_chars:
        return False, ""

    if "\n" in text:
        return False, ""

    return True, text
```

For drift detection, do not try to solve semantic correctness with heavy logic. Use cheap structural checks and source anchoring:

For outro, require that either:

- it contains at least one meaningful token from `news_close_brief`, or
- it stays under a very tight length and does not introduce unsupported specifics.

But be careful: token-overlap checks can reject good paraphrases. I would keep this lightweight.

---

## 4. Risk in retiring `override_announcer_close`

Do not keep it as an unconditional overlay.

But do keep its intended function as a **fallback**.

The current behavior you described is:

- final announcer line is composed generically,
- `override_announcer_close` is supposed to overwrite it with `news_close_brief`,
- that overwrite is currently broken by a key-name contract bug.

The right change is:

1. Remove/disable the unconditional “always stamp close” behavior.
2. Add `compose_announcer_outro()` as the primary path.
3. Keep a corrected `news_close_brief`-derived deterministic fallback.

So the old helper can be retired as a named behavior, but the concept should survive as something like:

```python
fallback_announcer_outro(news_close_brief)
```

or:

```python
news_close_to_announcer_fallback(news_close_brief)
```

Do not leave the old broken helper in place with ambiguous key contracts. That creates future confusion.

### Important key-contract fix

Wherever `override_announcer_close` currently reads the news interpreter result, centralize the lookup. For example:

```python
def get_news_close_brief(news_payload: dict) -> str:
    return (
        news_payload.get("news_close_brief")
        or news_payload.get("close_brief")
        or news_payload.get("closing_brief")
        or ""
    )
```

But do not silently support too many aliases forever. During migration, aliases are fine. Long-term, enforce one canonical key: `news_close_brief`.

Relevant location: writer node containing `OTR_LedgerScriptWriter`, near the existing `override_announcer_close` call.

---

## 5. Missing risks / ordering hazards / simpler approach

### A. Existing `polish_line` may undo your dedicated announcer pass

You mentioned `nodes/_otr_line_composer.py` has optional `polish_line` with an announcer-specific system prompt.

Decide explicitly:

- Either dedicated announcer passes are final and should not be polished.
- Or the polish pass runs inside `compose_announcer_intro()` / `compose_announcer_outro()` and is part of their deterministic contract.

I recommend: **do not run a separate polish pass after the dedicated announcer pass**. It adds another LLM call, another failure point, and another source of nondeterminism.

If polish currently happens globally after `compose_line()`, make sure the new announcer bookend text is not later rewritten accidentally.

### B. The final announcer slot should not be composed generically anymore

If the writer loop currently treats every beat uniformly and calls `compose_line()` for the final announcer beat, you have two options:

#### Cleaner

Skip generic composition for the final announcer beat.

```python
if is_final_announcer_beat:
    line["text"] = fallback_announcer_outro(news_close_brief)
    continue
```

Then overwrite with `compose_announcer_outro()` post-loop if valid.

#### Smaller diff

Let current generic composition happen, then overwrite post-loop.

This is less clean and wastes time, but it minimizes control-flow changes. If you want smallest patch, this is acceptable.

### C. First and last beat could theoretically be the same row

If the outline ever creates a one-beat episode, the first and last announcer slot are identical. Then the row cannot be both intro and outro unless you allow a combined line.

Add a guard:

```python
if first_announcer_idx == last_announcer_idx:
    line["text"] = fallback_single_bookend(script_brief, news_close_brief)
```

Or assert that valid episodes must have at least two beats.

### D. Do not add a new model widget

Per your rule, both new functions should be tagged/routed as `creative`.

No new widget. No new model selection.

Implementation-wise, `compose_announcer_intro()` and `compose_announcer_outro()` should call the same underlying creative LLM invocation path used by `compose_line()` in `nodes/_otr_line_composer.py`.

### E. VRAM impact should be negligible

On the RTX 5080 Laptop / 16 GB VRAM with a 14.5 GB ceiling, the proposed change should not materially affect VRAM if:

- the same creative model remains loaded,
- no second model is loaded for repair/validation,
- you do not route fallback through the technical model,
- you do not add parallel LLM execution.

This is additional inference time, not additional resident model memory.

### F. Determinism needs more than fallback

The fallback path is deterministic, but the successful LLM path also needs deterministic settings.

Use whatever your project already uses for C7, but for these passes specifically prefer:

- fixed seed,
- temperature `0` or existing deterministic creative setting,
- no sampling if supported,
- stable prompt text,
- stable JSON schema,
- stable whitespace normalization,
- no timestamps,
- no environment-dependent fields.

If your local LLM backend uses nondeterministic GPU kernels, strict byte identity between runs can still be fragile. Since the existing project already has LLM calls, this is not a new class of risk, but two more creative calls increase the surface area. If C7 is absolutely strict, ledger caching keyed by input/prompt/model/settings is the robust solution. That is not a VRAM optimization; it is a determinism guard.

### G. Audio byte identity

Changing announcer text will change generated audio bytes compared to old outputs. The safe interpretation is:

- same input + same config + same code version = same ledger text = same audio bytes.

If you need old episodes to remain byte-identical, gate the new behavior behind a version/config flag and default it off for old workflows.

Example:

```python
enable_dedicated_announcer_bookends: bool = False
```

But if the owner wants the improvement now and accepts new deterministic outputs, no new widget is necessary; use an internal version bump instead.

---

## Suggested implementation shape

### In `nodes/_otr_line_composer.py`

Add two functions beside `compose_line()`:

```python
def compose_announcer_intro(
    *,
    creative_model,
    script_brief: str,
    series_title: str = "SIGNAL LOST",
    deterministic_fallback: bool = True,
) -> str:
    ...
```

```python
def compose_announcer_outro(
    *,
    creative_model,
    news_close_brief: str,
    finished_script_text: str,
    intro_text: str = "",
    series_title: str = "SIGNAL LOST",
    deterministic_fallback: bool = True,
) -> str:
    ...
```

Both should:

1. Build announcer-specific prompt.
2. Call the existing creative LLM route.
3. Parse strict JSON.
4. Validate one-line text.
5. Return validated text or deterministic fallback.

### Intro prompt shape

```text
You are the radio announcer for SIGNAL LOST.

Write exactly one opening announcer line for an old-time radio drama.

Use this premise brief:
"{script_brief}"

Requirements:
- one spoken line only
- no speaker label
- no stage directions
- no sound effects
- no markdown
- 20 to 220 characters
- frame the story with mystery, restraint, and period radio tone
- do not invent proper names not present in the brief

Return only JSON:
{"text": "..."}
```

### Outro prompt shape

```text
You are the radio announcer for SIGNAL LOST.

Write exactly one closing announcer line for an old-time radio drama.

Opening announcer line:
"{intro_text}"

News close brief:
"{news_close_brief}"

Finished episode dialogue:
"{finished_script_text}"

Requirements:
- one spoken line only
- no speaker label
- no stage directions
- no sound effects
- no markdown
- 30 to 260 characters
- close the episode in announcer voice
- preserve the factual meaning of the news close brief
- lightly echo the opening tone without repeating it

Return only JSON:
{"text": "..."}
```

Keep `finished_script_text` bounded. You do not need to feed the whole episode if it is long. A compact deterministic excerpt is enough:

- intro line,
- first few character lines,
- last few character lines before outro,
- maybe episode title/premise.

This avoids prompt bloat without any VRAM tinkering.

---

## Final decision matrix

| Question | Recommendation |
|---|---|
| Combined or separate bookend pass? | **Separate intro and outro passes.** Better ordering, simpler failure isolation. |
| Outro inside loop or post-loop? | **Post-loop.** It needs the finished script and `news_close_brief`. |
| Fallback design? | Deterministic string functions from `script_brief` and `news_close_brief`; validate LLM JSON strictly. |
| Retire `override_announcer_close`? | Retire as unconditional overlay; preserve corrected news-close stamping as fallback. |
| Simpler approach? | If you want the smallest possible patch: fix the `news_close_brief` key bug first, then add only `compose_announcer_intro()`. But the cleaner target is two creative announcer passes. |

My preferred minimal-but-correct patch:

1. Add `compose_announcer_intro()` and `compose_announcer_outro()` in `nodes/_otr_line_composer.py`.
2. In `OTR_LedgerScriptWriter`, dispatch the first announcer slot to intro composition.
3. Skip or placeholder the final announcer slot during the normal loop.
4. After the loop, call `compose_announcer_outro()` and write into the existing final `ledger.lines[]` row.
5. Replace `override_announcer_close` with a corrected deterministic fallback using `news_close_brief`.
6. Do not change ledger row shape, model widgets, or downstream audio interfaces.
