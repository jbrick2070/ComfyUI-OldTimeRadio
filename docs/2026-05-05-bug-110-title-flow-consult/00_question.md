# Round-robin consult — title flow end-to-end fix (BUG-LOCAL-110)

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
