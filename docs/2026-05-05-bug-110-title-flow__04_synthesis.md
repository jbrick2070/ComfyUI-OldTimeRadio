# Synthesis -- 2026-05-06

**Question:** # Round-robin consult — title flow end-to-end fix (BUG-LOCAL-110)

## Context

OTR is a ComfyUI sidecar pipeline that generates 1940s-style sci-fi radio dramas with AI-rendered video over a vintage broadcast aesthetic. The script writer is `OTR_LLMScriptWriter` (Mistral-Nemo NF4); the visual stack is HuMo (character lipsync) + LTX (announcer/music/sfx) + composite + RTXUpscale + procgen blend. Today the procgen blend produces the broadcast deliverable at `output/otr/obs/<episode_id>_procgen_blended.mp4`.

The "episode title" is currently broken end-to-end. We have a resolved title computed inside the script writer (BUG-LOCAL-035 fallback chain — user widget → LLM-emitted "TITLE:" → derived from environment → timestamp fallback), but the resolved title doesn't make it to:

1. The ledger (`title: None` in every recent ledger we've inspected).
2. The on-disk filename (built from the news-headline slug, not the resolved title).
3. The episode_id used for folder naming.

The current filenames look like `signal_lost_scientists_connect_time_crystal_to_real__20260505_222015_procgen_blended.mp4` — note the `signal_lost_` hardcoded prefix, the news-headline-derived slug instead of an LLM-generated evocative title, and a doubled underscore `..._real__20260505_...` from punctuation slugification.

This is on `v2.0-alpha` of OTR. Branch policy: only Jeffrey merges to `main`; we ship sprints on `v2.0-alpha`.

## Proposed fix scope (3 layers, ~50-80 LOC across 2 files + 1 test update)

### Layer 1 — strip trailing underscore in slug, fix the doubled `_`

`nodes/video_engine.py:1482-1484` today:

```python
safe_title = "".join(c if c.isalnum() or c in "_ " else "" for c in episode_title)
safe_title = safe_title.strip().replace(" ", "_").lower()[:40]
out_path = os.path.join(out_dir, f"signal_lost_{safe_title}_{ts}.mp4")
```

Patch: add `.rstrip("_")` after the `[:40]` so a title that ends in punctuation (which becomes `_` after slugification) doesn't produce `..._real__<ts>...`. Plus collapse runs of `_` in case the source already had them.

```python
safe_title = "".join(c if c.isalnum() or c in "_ " else "" for c in episode_title)
safe_title = safe_title.strip().replace(" ", "_").lower()[:40]
import re as _re
safe_title = _re.sub(r"_+", "_", safe_title).strip("_")  # collapse runs + strip ends
out_path = os.path.join(out_dir, f"signal_lost_{safe_title}_{ts}.mp4")
```

### Layer 2 — stamp resolved title to the ledger

`nodes/story_orchestrator.py` already computes `_resolved_title` at line ~6545 via the BUG-035 fallback chain. Right before the final `led.save()` in the same try-block where `meta.ltx_style_brief` got added today, also stamp:

```python
led.data["title"] = _resolved_title
led.data.setdefault("meta", {})["title_source"] = _title_source
```

Top-level `title` so a JSON path of `ledger.title` reads it out cleanly. `meta.title_source` for forensics ("user", "llm", "derived", or "timestamp_fallback").

### Layer 3 — `video_engine.py` reads ledger.title instead of inbound `episode_title`

Today `video_engine.py:1482` uses the `episode_title` parameter passed in by the workflow link, which is often empty (user widget unfilled) or stale. After Layer 2 lands, `video_engine.py` can prefer `ledger.title` over the inbound parameter:

```python
# Resolve canonical title for filename construction. Layer 2 of BUG-110:
# story_orchestrator stamps led.data["title"] with the BUG-035 resolved title.
# Prefer that over the inbound episode_title parameter (which is often the
# user's empty widget value or a stale workflow link).
try:
    from .production_ledger import get_ledger
    _led = get_ledger()
    _ledger_title = (_led.data.get("title") or "").strip()
    if _ledger_title:
        episode_title = _ledger_title
except Exception:
    pass  # fall through to whatever the caller passed in

# ... rest of safe_title slugification unchanged ...
```

Net result: a workflow with empty `episode_title` widget now produces filenames like `signal_lost_<llm-evocative-title>_<ts>.mp4` instead of `signal_lost_<news-headline-slug>_<ts>.mp4`.

### What we're explicitly NOT doing in this sprint

- Not changing the `signal_lost_` hardcoded prefix (that's a separate v2.1 cleanup item to make `show_name` configurable; Roadmap line 644).
- Not touching `LLMDirector`'s hardcoded `"episode_title": "Signal Lost"` JSON field (separate prompt-template work).
- Not adding a new `episode_title` socket to `OTR_SignalLostVideo` (Roadmap line 374 v2.1 cleanup; the read-from-ledger path in Layer 3 sidesteps the need).

## Questions for you

1. **Layer 3 read-from-ledger pattern**: is reading `ledger.title` directly inside `video_engine.py` (via `from .production_ledger import get_ledger`) the right move, or does it create a hidden dependency we'll regret? Alternative is adding a node-output socket on `OTR_LLMScriptWriter` that downstream nodes wire — but that requires a workflow JSON edit which has back-compat tax.

2. **Layer 2 ledger schema**: storing `ledger.title` at the top level (alongside `episode_id`, `commit`, `total_episode_dur_s`) vs `ledger.meta.title`. Top-level feels right because it's a first-class identifier; meta is for forensics. Any reason to prefer meta?

3. **Filename collision risk**: the on-disk filename today is `<prefix>_<slug>_<timestamp>.mp4`. Timestamp ensures uniqueness. After Layer 3, the slug source changes from news-headline to LLM-evocative-title, but the timestamp still appears. So no collision risk added — agree?

4. **Layer 1 slug regex**: `re.sub(r"_+", "_", safe_title).strip("_")` collapses runs and strips ends. Any edge case I'm missing? (Empty string post-strip is the obvious one — should I substitute a fallback like `"untitled"` then?)

5. **Order-of-operations**: Layer 2 must land before Layer 3 takes effect on a fresh run. Within a single commit that's fine, but is there a cross-process race where the ledger save in Layer 2 hasn't flushed before video_engine reads in Layer 3? `Ledger.save()` writes synchronously to disk; `get_ledger()` returns the in-memory singleton. Both Layer 2 write and Layer 3 read happen in the same Python process within the same workflow run. So no race — agree?

6. **Anything we should add to the sprint scope** that's also "easy" within the same theme — or anything in the proposed scope you'd cut?

Goal of this consult: ship a clean Layer 1+2+3 patch tonight on `v2.0-alpha` with no surprises. Ranking on: API correctness, schema choice, race conditions, edge cases, and whether the read-from-ledger pattern in Layer 3 is sound.

Be concrete, code-review-style. Disagreements with each other are fine; we'll synthesize.

---

## ChatGPT (gpt-5.5)

Short version: I’d ship Layer 1 + Layer 2 + Layer 3 tonight, with two tweaks:

1. Add an empty-slug fallback.
2. Be explicit that this patch fixes ledger title + final deliverable filename, but probably **does not fix `episode_id` / folder naming** unless there is additional code not shown.

The `video_engine.py` read-from-ledger pattern is acceptable for `v2.0-alpha` as a pragmatic bridge, but I would treat it as a documented transitional dependency and eventually replace it with an explicit socket / run-context object in `v2.1`.

---

## 1. Layer 3: reading `ledger.title` from `video_engine.py`

I would accept this for tonight.

Given the constraints:

- single local workstation,
- no workflow JSON churn desired,
- resolved title already lives in the ledger/run context,
- adding sockets has back-compat/workflow tax,

this is the smallest change with the biggest payoff.

That said, it **is** a hidden dependency. The main risks are:

### Risk A — stale/global ledger state

If `get_ledger()` returns a process-global singleton, then `video_engine.py` is trusting that the singleton belongs to the current workflow run.

On a single ComfyUI queue, this is probably fine. But it becomes fragile if:

- two OTR jobs run concurrently,
- the video node is manually re-run out of sequence,
- a previous failed run left the singleton populated,
- a future refactor allows multiple prompt workers.

If `video_engine.py` has access to `episode_id`, I’d prefer this guard:

```python
_ledger_episode_id = str(_led.data.get("episode_id") or "").strip()
if _ledger_title and (not episode_id or _ledger_episode_id == episode_id):
    episode_title = _ledger_title
```

If `video_engine.py` does **not** have an `episode_id` in scope, I would still proceed, but add a comment that this is intentionally relying on the current OTR single-run ledger context.

### Risk B — graph invisibility

ComfyUI cannot “see” that the filename depends on `ledger.title`. As a result, the node’s side-effect filename can change without a visible input socket changing.

For this case, that is acceptable because the title is only affecting the output path/slug, not audio generation or model outputs. But architecturally, the clean version is still a socket or a structured run context.

### Recommended Layer 3 patch shape

I would avoid a totally silent `except Exception: pass`. If ledger import/read fails, fallback is correct, but a debug print is useful.

In `nodes/video_engine.py` around the supplied `1482-1484` area:

```python
# BUG-LOCAL-110: Prefer the canonical resolved title stamped by
# story_orchestrator into the production ledger. The inbound episode_title
# socket is often the empty user widget value or stale workflow text.
try:
    from .production_ledger import get_ledger

    _led = get_ledger()
    _ledger_title = str(_led.data.get("title") or "").strip()
    if _ledger_title:
        episode_title = _ledger_title
except Exception as exc:
    print(f"[OTR][BUG-LOCAL-110] Could not read ledger.title for filename; "
          f"falling back to inbound episode_title: {exc}")
```

If you want zero log noise, keep the silent fallback, but I’d personally log it.

Verdict: **yes, use the ledger for `v2.0-alpha`; document it as a bridge.**

---

## 2. Layer 2 ledger schema: top-level `ledger.title` vs `ledger.meta.title`

Top-level `ledger.title` is the right choice.

An episode title is not merely diagnostic metadata. It is a first-class episode identity field alongside things like:

- `episode_id`
- `commit`
- `total_episode_dur_s`

So I agree with:

```python
led.data["title"] = _resolved_title
led.data.setdefault("meta", {})["title_source"] = _title_source
```

I would slightly harden it:

```python
_resolved_title = str(_resolved_title or "").strip()
_title_source = str(_title_source or "unknown").strip() or "unknown"

if _resolved_title:
    led.data["title"] = _resolved_title
    led.data.setdefault("meta", {})["title_source"] = _title_source
```

But if BUG-035 guarantees `_resolved_title` is always non-empty due to timestamp fallback, unconditional assignment is fine.

Place this near the existing final ledger write in `nodes/story_orchestrator.py` around the line you cited, approximately `~6545`, inside the same try-block where `meta.ltx_style_brief` is currently added.

Verdict: **top-level `title` is correct; `meta.title_source` is correct for forensics.**

---

## 3. Filename collision risk

Agree: this change does not materially increase collision risk.

Current shape:

```text
signal_lost_<headline-slug>_<timestamp>.mp4
```

Proposed shape:

```text
signal_lost_<resolved-title-slug>_<timestamp>.mp4
```

The slug source changes, but timestamp remains the uniqueness component.

Caveat: if the timestamp only has second precision, two runs with the same resolved title in the same second could collide. But that risk already exists today with the headline slug.

So for BUG-LOCAL-110: **no new collision class is introduced.**

If you wanted to make this more robust later, use millisecond precision or include a short episode-id suffix. I would not add that tonight unless you already have a standard timestamp helper.

---

## 4. Layer 1 slug regex and edge cases

Your proposed slug cleanup is directionally correct:

```python
safe_title = _re.sub(r"_+", "_", safe_title).strip("_")
```

But I would add a fallback:

```python
safe_title = safe_title or "untitled"
```

Otherwise an all-punctuation/all-space title could produce:

```text
signal_lost__20260505_222015.mp4
```

### Important note

Based on the code you showed:

```python
safe_title = "".join(c if c.isalnum() or c in "_ " else "" for c in episode_title)
safe_title = safe_title.strip().replace(" ", "_").lower()[:40]
```

punctuation is currently **removed**, not converted to `_`.

So a title ending in `?` should become:

```text
real?
=> real
```

not:

```text
real_
```

The doubled underscore is more likely coming from one of these cases:

- existing trailing underscore in source text,
- multiple spaces converted to multiple underscores,
- truncation landing on an underscore,
- inbound title/headline already containing underscores,
- source already had `real_` before timestamp assembly.

Your cleanup still fixes the symptom, but I would not describe the old behavior as “punctuation becomes `_`” unless there is other slug code elsewhere.

### Better order of operations

I prefer collapsing before and after truncation:

```python
import re as _re

safe_title = "".join(c if c.isalnum() or c in "_ " else "" for c in str(episode_title or ""))
safe_title = safe_title.strip().replace(" ", "_").lower()
safe_title = _re.sub(r"_+", "_", safe_title).strip("_")
safe_title = safe_title[:40].strip("_")
safe_title = safe_title or "untitled"
```

Why strip after truncation too?

Because this can happen:

```text
"the_call_from_the_void_and_the_great_"[:40]
```

and the 40th character may be `_`.

If you want minimal diff, your version plus fallback is sufficient:

```python
safe_title = "".join(c if c.isalnum() or c in "_ " else "" for c in str(episode_title or ""))
safe_title = safe_title.strip().replace(" ", "_").lower()[:40]
import re as _re
safe_title = _re.sub(r"_+", "_", safe_title).strip("_")
safe_title = safe_title or "untitled"
out_path = os.path.join(out_dir, f"signal_lost_{safe_title}_{ts}.mp4")
```

Verdict: **add empty fallback; strip after truncation; collapse underscores.**

---

## 5. Order-of-operations / race conditions

Mostly agree: no meaningful disk flush race if both nodes use the same in-process ledger singleton.

The relevant thing is not `Ledger.save()` flushing to disk. The relevant thing is that Layer 2 mutates:

```python
led.data["title"] = _resolved_title
```

before Layer 3 reads:

```python
_led.data.get("title")
```

If this is the same Python process and same singleton, `save()` is not required for the read to see the mutation.

The real ordering requirement is graph execution:

- `story_orchestrator.py` must execute before `video_engine.py`.
- There must already be some actual ComfyUI dependency edge between those nodes.

If the video engine already consumes script/audio/scene outputs from the writer/orchestrator path, you are fine.

If the only dependency were the hidden ledger read, then ComfyUI would not know about it. But from your description, the workflow already has normal upstream/downstream links, so I would not worry.

Verdict: **no disk race; only ensure graph order already exists.**

---

## 6. Scope additions / cuts

### I would add

#### A. Empty slug fallback

This is tiny and prevents ugly paths.

```python
safe_title = safe_title or "untitled"
```

#### B. One explicit comment in `video_engine.py`

Make the hidden dependency intentional, not mysterious:

```python
# Transitional v2.0-alpha bridge. Long-term v2.1 should pass this via
# an explicit title socket or run-context object.
```

#### C. A lightweight test update

At minimum, test slug cleanup behavior with something like:

```python
"The Signal__From   Beyond!!!"
```

Expected slug:

```text
the_signal_from_beyond
```

Also test an empty/all-punctuation title:

```python
"!!!"
```

Expected:

```text
untitled
```

#### D. Log line or ledger evidence

When stamping title in `story_orchestrator.py`, optionally print:

```python
print(f"[OTR][BUG-LOCAL-110] Resolved episode title: {_resolved_title!r} source={_title_source}")
```

Not required, but useful tonight while validating.

### I would not add

- No new socket tonight.
- No workflow JSON edit tonight.
- No configurable `signal_lost_` prefix tonight.
- No LLMDirector title template work tonight.
- No episode-id folder refactor unless the sprint acceptance explicitly requires it.

---

## Important gap: this does not obviously fix `episode_id`

Your context says the title is broken end-to-end and does not make it to:

1. ledger,
2. on-disk filename,
3. `episode_id` used for folder naming.

The proposed three layers fix:

- ledger title: yes,
- final deliverable filename: yes,
- doubled underscore slug issue: yes,

but they do **not obviously fix `episode_id` / folder naming**.

Unless `episode_id` is generated later from `ledger.title` somewhere else, this patch will probably still leave folders named from the old headline-derived id.

I would be careful with the sprint claim:

- “BUG-LOCAL-110 partial end-to-end title propagation: ledger + deliverable filename”
- not “episode_id naming fixed”

Fixing `episode_id` can be more invasive because folder paths may already be created before the LLM title is resolved. Renaming episode folders after generation is riskier and could break downstream path assumptions.

My recommendation: **do not attempt the episode_id/folder renaming tonight** unless you can prove episode_id is computed after `_resolved_title` and before any path materialization. Otherwise punt that to a separate cleanup.

---

## Recommended final patch

### `nodes/story_orchestrator.py` around `~6545`

Inside the same final ledger-save try-block:

```python
# BUG-LOCAL-110: Stamp canonical resolved title into the ledger.
# _resolved_title and _title_source come from the BUG-LOCAL-035 fallback chain:
# user widget -> LLM TITLE -> derived -> timestamp fallback.
try:
    _resolved_title = str(_resolved_title or "").strip()
    _title_source = str(_title_source or "unknown").strip() or "unknown"

    if _resolved_title:
        led.data["title"] = _resolved_title
        led.data.setdefault("meta", {})["title_source"] = _title_source
except Exception as exc:
    print(f"[OTR][BUG-LOCAL-110] Failed to stamp ledger title: {exc}")
```

If you are already inside a broader `try`, do not nest unnecessarily; just assign.

### `nodes/video_engine.py` around `1482-1484`

```python
# BUG-LOCAL-110: Prefer canonical resolved title from the production ledger.
# The inbound episode_title parameter is often an empty user widget value or
# stale workflow text. Transitional v2.0-alpha bridge; v2.1 should pass this
# via an explicit socket or run-context object.
try:
    from .production_ledger import get_ledger

    _led = get_ledger()
    _ledger_title = str(_led.data.get("title") or "").strip()
    if _ledger_title:
        episode_title = _ledger_title
except Exception as exc:
    print(f"[OTR][BUG-LOCAL-110] Could not read ledger.title; "
          f"falling back to inbound episode_title: {exc}")

safe_title = "".join(
    c if c.isalnum() or c in "_ " else ""
    for c in str(episode_title or "")
)
safe_title = safe_title.strip().replace(" ", "_").lower()
safe_title = _re.sub(r"_+", "_", safe_title).strip("_")
safe_title = safe_title[:40].strip("_")
safe_title = safe_title or "untitled"

out_path = os.path.join(out_dir, f"signal_lost_{safe_title}_{ts}.mp4")
```

With either:

```python
import re as _re
```

near the top of the file, preferred, or locally near the slug code if you want minimal blast radius.

---

## Direct answers

1. **Layer 3 read-from-ledger?**  
   Yes for `v2.0-alpha`. It is a hidden dependency, but acceptable as a transitional bridge. Add a comment. Long-term, prefer explicit socket/run context.

2. **Top-level ledger schema?**  
   Yes. `ledger.title` top-level is correct. `meta.title_source` is correct for provenance.

3. **Collision risk?**  
   No new practical risk. Timestamp remains the uniqueness component. Same-second same-title collisions already exist in principle.

4. **Slug regex edge cases?**  
   Add `safe_title = safe_title or "untitled"`. Also strip after truncation. Note: shown code removes punctuation; it does not convert punctuation to underscores.

5. **Race/order?**  
   No disk flush race in the same Python process/singleton. Ensure the graph already forces writer/orchestrator before video. The hidden ledger read itself does not create a ComfyUI dependency.

6. **Add/cut?**  
   Add fallback + test + comment. Do not add sockets, prefix config, LLMDirector changes, or episode-id folder renaming tonight unless that code path is already trivial and proven.

Final recommendation: **ship Layer 1+2+3 with the small hardening above, but explicitly leave `episode_id`/folder naming as not fixed unless verified separately.**

---

## Gemini (gemini-3.1-pro-preview-customtools)

**1. Agreement Statement**
PARTIALLY AGREE; ChatGPT's string manipulation logic is sound, but it severely underestimates the ComfyUI execution graph risks (specifically node caching) and cross-run state leakage inherent in the Layer 3 singleton read.

**2. Factual Errors in ChatGPT's Answer**
*   **Origin of the double underscore:** ChatGPT states, "punctuation is currently removed... The doubled underscore is more likely coming from... existing trailing underscore". This is incorrect. If a title is `"Signal Lost - The Crystal!"`, the generator `c.isalnum() or c in "_ "` strips the `-` and `!`, leaving `"Signal Lost  The Crystal"` (note the double space). The subsequent `.replace(" ", "_")` turns that double space into the double underscore. 
*   **ComfyUI Graph Execution / Race Conditions:** ChatGPT claims, "If the video engine already consumes script/audio/scene outputs... you are fine." This ignores ComfyUI's aggressive node caching. If the inputs to `story_orchestrator.py` haven't changed between queue runs, ComfyUI will *skip* executing it. If your `production_ledger` singleton is re-initialized per run, `video_engine.py` will read an empty title because the orchestrator never ran to populate it.

**3. Important Omissions**
*   **Rule C7 (Byte-identical audio):** ChatGPT failed to explicitly clear your C7 constraint. I will clear it: This patch is 100% safe for Rule C7. You are strictly mutating downstream filename strings and ledger metadata. No RNG seeds, VRAM allocations, or audio tensors are touched.
*   **Singleton State Leakage:** Because ComfyUI keeps the Python process alive, if Run A succeeds (Title A), and Run B fails upstream of the orchestrator but somehow triggers the video node (e.g., via a bypassed/cached graph), `video_engine.py` might read Title A for Run B's output. 
*   **Ledger Persistence:** If `get_ledger()` loads state from disk based on an `episode_id` passed through the graph, the caching issue is mitigated. If it's purely an in-memory dict, you are highly vulnerable to ComfyUI caching bugs.

**4. Own Short Recommendation**
*   **Layer 1 (Slug Cleanup):** Adopt ChatGPT's refined regex and truncation order. It safely handles the space-to-underscore collapse.
    ```python
    safe_title = "".join(c if c.isalnum() or c in "_ " else "" for c in str(episode_title or ""))
    safe_title = _re.sub(r"\s+", " ", safe_title).strip() # Collapse spaces BEFORE underscore conversion
    safe_title = safe_title.replace(" ", "_").lower()[:40]
    safe_title = _re.sub(r"_+", "_", safe_title).strip("_") or "untitled"
    ```
*   **Layer 2 (Ledger):** Stamp `led.data["title"] = _resolved_title` at the top level in `nodes/story_orchestrator.py` (~line 6545). Top-level is architecturally correct for a primary identity field.
*   **Layer 3 (Video Engine):** Implement the hidden read for tonight's `v2.0-alpha` sprint, but **guard it with an episode ID check** to prevent cross-run leakage. 
    ```python
    _ledger_title = ""
    try:
        from .production_ledger import get_ledger
        _led = get_ledger()
        # Only trust the ledger if it belongs to the current run (assuming you have episode_id in scope)
        if _led.data.get("episode_id") == current_episode_id: 
            _ledger_title = str(_led.data.get("title") or "").strip()
    except Exception:
        pass
    episode_title = _ledger_title or episode_title
    ```
*   **Scope Cut:** Do *not* attempt to fix the `episode_id` folder naming tonight. If the folder is created before the LLM resolves the title, fixing it requires a lifecycle refactor that threatens your VRAM/disk I/O assumptions.

**5. Uncertainties to Verify**
*   **ComfyUI Caching Behavior:** I need to know if `get_ledger()` is a purely in-memory singleton that wipes on a new ComfyUI prompt, or if it hydrates from disk. If it wipes, and `story_orchestrator` gets cached, Layer 3 will fail silently and fall back to the old behavior.
*   **`episode_id` Availability:** Does `video_engine.py:1482` have access to the current run's `episode_id`? If not, you cannot safely validate that the singleton belongs to the current run, elevating the risk of cross-run state leakage.

---

## NVIDIA (nvidia/llama-3.3-nemotron-super-49b-v1.5)

### Senior Systems Architect Review: Breaking the Tie on OTR Title Fix (BUG-LOCAL-110)

---

### **1. Agreement Where ChatGPT and Gemini Concur**
- **Layer 3 Read-from-Ledger Pattern (Q1):**  
  **Concur.** Both agree it’s acceptable for `v2.0-alpha` as a transitional bridge. Gemini’s concern about ComfyUI caching is valid but manageable.  
  **Flaw Missed:** Neither addressed whether `episode_id` is available in `video_engine.py` to guard against cross-run leakage. If unavailable, the singleton read risks stale data in cached graph executions.

- **Layer 2 Ledger Schema (Q2):**  
  **Concur.** Top-level `ledger.title` is correct. Both agree `meta.title_source` is appropriate for forensics.

- **Filename Collision Risk (Q3):**  
  **Concur.** Timestamp ensures uniqueness. No new risk introduced.

---

### **2. Disagreements: Where to Side?**
#### **Q1: Layer 3 Read-from-Ledger Pattern**
- **ChatGPT:** Accepts hidden dependency as transitional.  
- **Gemini:** Warns of ComfyUI caching risks (orchestrator skipped → stale ledger).  
- **Verdict:** Side with **Gemini**.  
  - If `episode_id` is available in `video_engine.py`, guard the read:  
    ```python
    if _led.data.get("episode_id") == current_episode_id:
        episode_title = _ledger_title
    ```
  - If `episode_id` is unavailable, **document the risk** and ensure the ComfyUI graph explicitly forces `story_orchestrator.py` to run before `video_engine.py` (e.g., via a dummy input dependency).

#### **Q4: Layer 1 Slug Regex**
- **ChatGPT:** Proposes `re.sub(r"_+", "_", ...)` after truncation.  
- **Gemini:** Fixes space-to-underscore collapse by collapsing spaces *first*.  
- **Verdict:** Side with **Gemini**.  
  - **Corrected Slug Workflow:**  
    ```python
    safe_title = "".join(c if c.isalnum() or c in "_ " else "" for c in str(episode_title or ""))
    safe_title = re.sub(r"\s+", " ", safe_title).strip()  # Collapse spaces first
    safe_title = safe_title.replace(" ", "_").lower()[:40]
    safe_title = re.sub(r"_+", "_", safe_title).strip("_") or "untitled"
    ```

---

### **3. Factual Errors in Previous Answers**
- **ChatGPT:**  
  - Incorrectly claimed punctuation becomes `_` in slugs. Actual code removes punctuation, leaving spaces that become `_`.  
  - Underestimated ComfyUI caching risks (see Gemini’s correction).

- **Gemini:**  
  - Correctly identified the space-to-underscore issue but assumed `episode_id` is available in `video_engine.py` without verification.

---

### **4. Important Omissions by Both**
1. **ComfyUI Graph Execution Guarantees:**  
   - Neither confirmed whether `story_orchestrator.py` is guaranteed to run before `video_engine.py` in all cases (e.g., cached nodes, manual reruns).  
   - **Risk:** If the orchestrator is skipped, `ledger.title` may be stale/missing.

2. **Rule C7 (Byte-Identical Audio):**  
   - ChatGPT addressed it; Gemini did not. Both agree the fix is safe, but Gemini should have explicitly confirmed.

3. **`episode_id` Availability in `video_engine.py`:**  
   - Critical for Gemini’s proposed guard but unverified.

---

### **5. Prioritized Recommendations**
1. **Slug Cleanup (Layer 1):**  
   Implement Gemini’s space-collapse-first approach with `untitled` fallback.  
   ```python
   safe_title = re.sub(r"\s+", " ", safe_title).strip()  # First
   safe_title = safe_title.replace(" ", "_").lower()[:40]
   safe_title = re.sub(r"_+", "_", safe_title).strip("_") or "untitled"
   ```

2. **Ledger Title Stamping (Layer 2):**  
   Stamp `ledger.title` top-level and `meta.title_source` in `story_orchestrator.py` (~line 6545). Harden with string checks.

3. **Guarded Ledger Read (Layer 3):**  
   - If `episode_id` is available in `video_engine.py`, validate ledger ownership:  
     ```python
     if _led.data.get("episode_id") == current_episode_id:
         episode_title = _ledger_title
     ```
   - If not, **add a dummy dependency** (e.g., pass a dummy file from orchestrator to video engine) to force execution order.

4. **Testing:**  
   Add tests for:  
   - `"The Signal__From Beyond!!!" → "the_signal_from_beyond"`  
   - `"!!!" → "untitled"`  
   - `"Hello  World" → "hello_world"` (double space → single underscore).

5. **Documentation:**  
   - Comment in `video_engine.py` about the transitional dependency.  
   - Log warnings if ledger read fails or fallback is used.

---

### **6. Uncertainties Requiring Verification**
1. **ComfyUI Caching Behavior:**  
   - Does skipping `story_orchestrator.py` due to cached inputs leave `ledger.title` stale?  
   - How is the `production_ledger` singleton managed (in-memory vs. disk-backed)?

2. **`episode_id` Availability:**  
   - Is `current_episode_id` accessible in `video_engine.py`? If not, the guard cannot be implemented.

3. **Graph Execution Order:**  
   - Is there an explicit ComfyUI edge forcing `story_orchestrator.py` → `video_engine.py`?

---

### **Final Verdict**
Ship **Layer 1 + Layer 2 + Layer 3** tonight with:  
- Gemini’s slug processing order and fallback.  
- Guarded ledger read (if `episode_id` is available) or dummy dependency to enforce execution order.  
- Tests and logging.  
**Do not fix `episode_id` folder naming** unless trivial. Document risks and plan for v2.1 refactor.

---

## To decide (Claude / human)

- [ ] All three agree:
- [ ] Two-vs-one splits:
- [ ] Facts to verify:
- [ ] Final grounded recommendation:
