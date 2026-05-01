# Round A -- ChatGPT (gpt-5.4) elapsed=83.6s

Here’s the blunt read: **BUG-121 looks directionally right, BUG-123 is the right tactical fix for your setup, BUG-124 was a classic “type-label != runtime contract” trap, and the two lingering symptoms almost certainly have different root causes.** One is probably **timing data not making it into the ledger / not being honored in `humo_concat` composite mode**; the other is probably **radio-bookend render/stamp observability being too weak to tell whether render failed, stamp was lost, or episode identity drifted**.

I’ll go question by question.

---

# Q1: BUG-121 sanity check — layered radio-still resolver

## Short answer

**Yes, the pattern is correct.** For a ComfyUI pipeline where:
- node outputs are permissive strings,
- ledger fields may be absent or stale,
- downstream consumers need to survive partial state,

a resolver that tries:
1. `ledger.radio_bookend_path`
2. `ledger.meta.radio_bookend_path`
3. deterministic filesystem reconstruction from `episode_id`

with **existence checks at each layer** and **graceful `None`** is exactly the kind of defense-in-depth I’d want.

That is the **smallest change with the biggest payoff**.

---

## Why this is the right pattern

You’ve got three different “truth sources” for an asset:

- **Stamped canonical path in ledger top-level**
- **Stamped path in nested metadata**
- **Deterministic convention on disk**

That’s a good hierarchy because it goes from:
- explicit runtime state,
- to legacy/alternate state,
- to reconstructable convention.

For ComfyUI custom nodes, this is especially appropriate because graph wiring and node execution order often create situations where:
- a file exists but ledger wasn’t updated,
- ledger was updated but later overwritten,
- or a consumer receives only enough information to reconstruct.

So yes: **this is a proper asset resolver, not a hack.**

---

## Failure modes you should still guard against

You named the right ones. I’d add a few.

## 1. `episode_id` type / normalization / path traversal

If you’re reconstructing:

```python
radio_bookend_{episode_id}.png
```

then `episode_id` should be treated as **data**, not path text.

### Risks
- non-string values (`int`, `None`, list, bool)
- path separators (`../`, `\`, `/`)
- reserved Windows chars (`<>:"/\|?*`)
- trailing spaces/dots on Windows
- absurdly long names

### Recommendation
Before using `episode_id`, normalize it to a **safe filename token**.

Minimal defensive pattern:
- cast to `str`
- strip whitespace
- reject empty / `"None"` / `"[]"` / `"{}"`
- replace path separators and reserved chars with `_`
- optionally allow only `[A-Za-z0-9._-]`

If sanitization changes the value, log it once at debug/warn level.

### Why this matters
Not because I think you’re under attack, but because malformed IDs in local pipelines happen all the time from upstream serialization bugs.

---

## 2. `otr_stills_dir()` failure

Yes, this is a real edge case.

If `otr_stills_dir()` can:
- throw,
- return `None`,
- return a non-existent path,
- or depend on cwd in a brittle way,

then your fallback layer can still fail noisily.

### Recommendation
Wrap the deterministic-path branch in its own `try/except`, and log:
- resolved stills dir
- episode_id used
- final candidate path

If resolution fails, return `None` rather than exploding.

That preserves the “defensive fallback” intent.

---

## 3. File exists but is 0-byte / corrupt

Also real.

An existence check alone only proves “pathname exists,” not “usable image.”

### Tradeoff
I would **not** do a full `Image.open()` verification inside the resolver unless this path is hit rarely, because:
- it changes the resolver from “cheap path lookup” to “I/O + decode validation”
- it may create extra file handle churn
- it moves image validation responsibility upstream of where the image is actually consumed

### Smallest useful improvement
Add a **size > 0** check before returning the path.

That catches the worst silent-corruption case almost for free.

If you want one more notch of safety:
- optionally `Image.open(...).verify()` only in debug mode or behind a flag.

For your “smallest change, biggest payoff” rule, I’d stop at:
- exists
- is file
- size > 0

---

## 4. Stale ledger path pointing to old episode asset

This is the one I’d watch most closely.

If `ledger.radio_bookend_path` exists on disk but belongs to a previous episode/run, your current resolver will happily accept it.

That may be okay if the ledger is authoritative. But given your symptom history, it’s worth considering whether the resolver should prefer:
- stamped path if it exists, **unless**
- its basename clearly disagrees with current `episode_id`

### Recommendation
At minimum, log:
- which layer won (`top-level`, `meta`, `reconstructed`)
- candidate basename
- current `episode_id`

That gives you post-mortem visibility without changing behavior.

I would **not** add aggressive basename validation unless you’ve seen stale-path reuse in practice.

---

## 5. Ambiguity when both top-level and meta paths exist but differ

If both exist and point to different files, your current precedence rule chooses top-level. That’s fine, but it’s worth logging a warning when they disagree.

That kind of disagreement is a strong signal that ledger writes are racing or overwriting.

---

## Verdict on BUG-121

**Keep it.**  
I’d make only these small additions:

1. sanitize `episode_id` before filename reconstruction  
2. wrap `otr_stills_dir()` resolution in `try/except`  
3. require `is_file()` and `stat().st_size > 0`  
4. log which fallback layer succeeded

That’s enough.

---

# Q2: BUG-123 sanity check — end-of-run VRAM cleanup in `VideoComposite`

## Short answer

For your current architecture, **yes, this is the right tactical fix**.

Would a queue-completion hook be architecturally cleaner? **Yes.**  
Would I recommend building that instead right now? **No.**

Because:
- you have a single-developer local system,
- `VideoComposite` is the last node in every OTR workflow,
- the failure is real and reproducible,
- and the fix is already shipped and tested.

So this is the right **pragmatic containment point**.

---

## Is a consumer node the right place to call `unload_all_models()`?

### Architecturally
Not ideal. A consumer node generally shouldn’t own global process cleanup.

### Practically
In your system, **it’s acceptable** because:
- `nodes/video_composite.py::execute()` is the terminal node by design
- you explicitly want cleanup at end-of-run, not mid-graph
- ComfyUI’s default residency behavior is the actual problem
- you do not want to chase lower-level memory optimization

So this is one of those “wrong layer, right move” fixes.

### Caveat
This assumption holds only if `VideoComposite` is truly present in **every** production OTR workflow and is **last**.

If someone later creates a variant workflow that:
- omits `VideoComposite`, or
- branches after it,

then your cleanup guarantee disappears or becomes dangerous.

### Smallest safeguard
Add a loud comment/docstring in `video_composite.py` near the finally block:

- this node is intentionally the end-of-run cleanup boundary
- do not place downstream model-using nodes after it
- if workflow topology changes, move cleanup to a queue-level hook

That’s cheap and prevents future self-foot-shooting.

---

## Is `cuda.synchronize()` necessary?

## Short answer
**Not strictly always necessary, but reasonable here.**

`torch.cuda.empty_cache()` does **not** guarantee synchronization of all outstanding kernels in the way people often assume. If there are pending GPU ops, synchronization can help ensure memory is actually reclaimable before the next run.

Given your symptom:
- Run 1 completes,
- Run 2 starts,
- Mistral-Nemo prefill OOMs with effectively no free memory,

a sync in the fallback path is defensible.

### My recommendation
Keep it, but only in the **fallback cleanup path**, exactly as you described.

Why:
- it’s end-of-run only, so latency cost is irrelevant
- it improves determinism of cleanup timing
- it avoids subtle “cache emptied before kernels retired” confusion

So: **yes, keep `cuda.synchronize()`**.

---

## Are there other ComfyUI caches to clear?

Possibly, but this is where I want to be candid: **without your exact ComfyUI version and local custom-node stack, I can’t guarantee a complete list.**

That said:

## What `unload_all_models()` likely covers
Usually the big-ticket resident model objects managed through ComfyUI model management:
- UNet / diffusion models
- text encoders / CLIP
- VAE
- other registered model wrappers

## What it may not fully cover
Depending on implementation and custom nodes:
- ad hoc global caches in custom nodes
- image embedding caches (IP-Adapter style)
- clip vision encoder instances not registered through model management
- tokenizer/model objects held in module globals
- ffmpeg subprocesses holding resources indirectly
- Python references in closures/singletons preventing GC

### For your specific stack
You mentioned:
- HuMo
- WanTE
- WanVAE
- Whisper
- Mistral-Nemo

The biggest risk is not “ComfyUI forgot one official cache,” but **custom nodes or helper modules holding references outside model_management**.

### Smallest high-payoff observability
After cleanup, log:
- `torch.cuda.memory_allocated()`
- `torch.cuda.memory_reserved()`
- maybe `torch.cuda.mem_get_info()` if available on your stack

Log it:
- before cleanup
- after `unload_all_models()`
- after `soft_empty_cache(force=True)`
- after torch fallback

That will tell you whether the residual memory is:
- allocator cache,
- live references,
- or something else.

This is more valuable than guessing at every possible cache.

---

## Verdict on BUG-123

**Keep it.** It’s the right tactical fix.

Small additions I’d recommend:
1. document that `VideoComposite` is the intentional end-of-run cleanup boundary
2. keep `cuda.synchronize()` in fallback
3. add staged VRAM telemetry around each cleanup step
4. if residual VRAM remains high, then inspect custom-node globals before adding more blanket cache-clearing calls

---

# Q3: Real root causes of the two open symptoms

This is the most important section.

---

## Symptom 1: No scene-boundary cuts in composited mp4

> “Radar base layer plays straight through with HuMo clips overlaid violently — no scene-boundary cuts in the composite.”

## Most likely root cause classes

Given your description, I think there are **two main possibilities**, in this order:

### A. `VideoComposite` in `audio_source=humo_concat` mode does not actually use scene timing the way you think
or
### B. `ledger.lines[].start_s / dur_s` were absent, zero, or invalid at composite time

The fact that BUG-124 was irrelevant strongly suggests the issue is **inside timing semantics**, not path routing.

---

## Why I think `humo_concat` mode is suspicious

You explicitly said:

- `VideoComposite` uses `ledger.lines[]` and `start_s` / `dur_s` to know when to overlay each HuMo clip
- `master_mix_per_clip_mux` uses pillarbox + concat-demuxer + final mux
- **Run 1 was generated with `audio_source=humo_concat`**
- `per_clip_mux` had failed strict C7 and fell through

That is a giant clue.

If `humo_concat` mode was a fallback path, it may:
- preserve one continuous base video,
- then overlay HuMo clips by absolute time,
- but **not perform hard scene cuts** between base segments.

In other words, the symptom may not be “timing broken,” but “this mode was never designed to cut the radar base at scene boundaries.”

### What would confirm this
Read `nodes/video_composite.py` and identify:
- branch on `audio_source`
- branch on `master_mix_per_clip_mux` vs `humo_concat`
- where concat lists / trim filters / overlay enable expressions are built
- whether scene boundaries are only used in one branch

I don’t have that source here, so I can’t cite exact lines. But based on your description, this is the first place I’d inspect.

---

## Alternative: ledger timing fields missing/zero

You already identified the other likely cause:
- if `start_s` / `dur_s` are missing or all `0.0`, then overlays won’t align meaningfully, and “violent overlay” behavior follows.

Given BUG-LOCAL-106 added authoritative `dialogue_positions` writeback into `ledger.lines[]`, there are a few sub-failure modes:

### Possible sub-failures
1. `dialogue_positions` were computed but never written back to the ledger file consumed by `VideoComposite`
2. they were written to an in-memory ledger, but later overwritten by another node
3. line IDs mismatched, so writeback didn’t land on the intended entries
4. values were written as strings / nulls / zeros
5. `VideoComposite` reads a different ledger than the one updated by the audio/timing stage

This class of bug is very plausible in your pipeline.

---

## My current best guess

### If I had to bet:
**Primary suspect: `humo_concat` composite path behavior, not graph wiring.**

Why:
- symptom is specifically about **composite behavior**
- run used a **fallback audio/composite mode**
- your own note says `per_clip_mux` failed strict C7 and the run fell through
- “base layer plays straight through” sounds like a mode-level behavior mismatch more than a missing manifest

### Secondary suspect:
**timing fields in ledger were missing/invalid at composite time**

These are not mutually exclusive. You could have both:
- `humo_concat` doesn’t cut hard by design
- and timing metadata is also weak, making overlays worse

---

## What to inspect on the next run

You need one high-value diagnostic dump before `VideoComposite` starts rendering:

### Add a single structured log block in `VideoComposite.execute()`:
For each line/clip actually considered for composition, log:
- `line_id`
- `scene_id`
- `speaker_role`
- `start_s`
- `dur_s`
- source clip path
- clip exists?
- selected composite mode / `audio_source`
- whether this line contributes:
  - overlay only
  - hard cut segment
  - skipped

And log run-level:
- ledger path loaded
- number of lines
- count of lines with valid `start_s`
- count with `dur_s > 0`
- count of announcer lines
- count of humo clips found

This will tell you in one run whether:
- timing is absent
- mode ignores cuts
- clips are being treated as overlays only

That’s the smallest observability change with the biggest payoff.

---

## Symptom 2: Radio bookend FLUX render not happening / not stamping ledger

> Log: `[BatchHumoRender] line l001 speaker_role=announcer wanted radio still but it's missing`

You listed four hypotheses. I think all are plausible, but they are **not equally likely**.

---

## My ranking of hypotheses

### Most likely: (a) render attempted but failed, warning-only path hid the cause
You cited `batch_flux_render.py` docstring around line 455-467 saying rendering “ALWAYS attempts under the new default,” and that lines 481-502 catch exceptions and only warn.

That is a classic silent-degradation pattern:
- render should happen
- exception occurs
- warning emitted
- pipeline continues
- downstream sees missing asset

If the warning is generic and not tied to episode_id/output path, it’s easy to miss.

This is my top suspect.

---

### Second most likely: (b) render succeeded, ledger stamp lost/overwritten later
Also very plausible in a ledger-heavy pipeline.

Typical pattern:
1. node A loads ledger
2. node B loads same ledger
3. node A writes radio_bookend_path
4. node B writes older in-memory ledger later
5. stamp disappears

If you have multiple nodes mutating the same JSON file without merge discipline, this happens.

Given your own discovery that consumer code has lots of fallback logic, I’d absolutely keep this on the table.

---

### Third most likely: (c) episode_id drift between FLUX render and HuMo render
Possible, but I’d rank it below the first two unless you’ve seen episode renaming or timestamp-based IDs mutate mid-run.

If `episode_id` is derived from title/timestamp and recomputed in multiple places instead of carried as a stable identifier, then yes, this can happen.

But I wouldn’t assume it without evidence.

---

### Less likely: something else entirely
Possible examples:
- `open_close` not actually propagated to the render gate used by `BatchFluxRender`
- announcer line classification mismatch
- radio prompt dynamic mode producing empty/invalid prompt and skipping render
- output path collision causing file to be written elsewhere

These are real possibilities, but I’d investigate after the top two.

---

## What observability would prove which hypothesis is right?

You need **correlation IDs and before/after ledger evidence**.

## In `batch_flux_render.py`
At the exact radio-bookend render attempt site (you cited docstring around 455-467; exception block around 481-502), log:

- `episode_id`
- ledger path
- `open_close`
- `radio_bookend_prompt` raw value
- whether dynamic mode engaged
- resolved final prompt length/hash
- intended output path
- render start
- render success/failure
- file exists after render?
- file size
- stamp target fields written:
  - `ledger.radio_bookend_path`
  - `ledger.meta.radio_bookend_path`
- ledger save success

If exception occurs, log:
- exception type
- exception message
- output path attempted
- episode_id

That alone will separate:
- attempted+failed
- attempted+succeeded+not stamped
- attempted+stamped+later lost

## In `batch_humo_render.py`
At `_resolve_radio_still_path`, log:
- ledger path loaded
- episode_id seen
- top-level path candidate
- meta path candidate
- deterministic candidate
- which one won
- if none won, why each failed:
  - absent
  - path invalid
  - file missing
  - zero-byte

This makes BUG-121 far more useful.

## Add one ledger-write fingerprint
Whenever a node writes the ledger, log:
- ledger path
- episode_id
- whether `radio_bookend_path` present after write
- maybe a short hash of the serialized ledger

This is the cheapest way to catch “stamp got overwritten later.”

---

## My best current diagnosis for symptom 2

If I had to choose one:
**FLUX radio-bookend render likely attempted and failed inside a warning-only `try/except`, or succeeded but the stamp was later overwritten.**

I cannot distinguish those two from the information given.  
What would unblock certainty is:
- the exact warning text from `BatchFluxRender`
- the ledger before/after `BatchFluxRender`
- whether the expected `output/otr/stills/radio_bookend_<episode_id>.png` existed after that node completed

---

# Q4: BUG-LOCAL-125 — populate `scene_manifest_json` or deprecate it?

## Recommendation: deprecate, don’t populate

For v2.0-beta, **do not spend time populating `scene_manifest_json`** unless you already have a concrete downstream consumer that needs it.

Right now it is:
- present in the interface
- always `"[]"`
- not useful
- and actively dangerous because it looks legitimate

That makes it a **footgun**, not a feature.

---

## Why I recommend deprecation over fixing

### Cost to populate correctly is higher than it looks
To make `scene_manifest_json` real, you’d need to define:
- exact schema
- timing source of truth
- whether entries are scene-level, shot-level, beat-level, or line-level
- how it stays consistent with ledger-on-disk
- whether it is authoritative or advisory

That’s not just “append to a list.” It’s a contract.

### Value appears near zero
You already said:
- no apparent production consumer
- ledger JSON on disk already contains the useful data

So this would be duplicate state with synchronization risk.

### Current state is worse than absent
Because it emits `"[]"`, it invites exactly the kind of false-positive “unused but wireable” fix that caused BUG-124.

---

## Smallest high-payoff action

In `scene_sequencer.py`, where `manifest = []` is initialized around line 689 and serialized around line 890:

### Do one of these:
1. **Best option:** remove it from `RETURN_NAMES` / outputs entirely if safe
2. **Second-best:** keep the slot for compatibility but rename it loudly:
   - `scene_manifest_json_DEPRECATED`
   - and emit a warning if accessed
3. **Third-best:** keep the name but emit `""` or a sentinel plus warning, not `"[]"`

I prefer option 2 if you need graph compatibility.

### Why not option 3?
Because empty string is also a permissive ComfyUI STRING and can still be misused. A deprecation label is clearer.

---

## Verdict
For release-blocker calculus: **deprecate, don’t implement.**

---

# Q5: Preventing the BUG-124 false-positive pattern

## Short answer

For a one-developer alpha-stage project, the most cost-effective gate is:

## **(c) Mandatory “read the consumer code” rule before applying any graph wiring fix**
with a tiny checklist.

Not a full shape checker. Not a synthetic dry-run harness first.

Those can come later if needed, but right now the highest ROI is process discipline.

---

## Why (c) beats (a) and (b) right now

### (a) Pre-commit shape check against INPUT_TYPES + actual output implementation
This sounds good, but in your case it still would have missed the real issue unless it became much smarter than “type tags.”

Why?
- ComfyUI STRING is permissive
- the source output “type” was not the real contract
- the consumer explicitly accepted `.mp4` path strings in a fallback chain
- the “bad” route was semantically valid at runtime

To catch that, the checker would need to understand:
- source node runtime value shapes
- consumer fallback semantics
- file-on-disk conventions

That’s a lot of complexity for limited payoff.

### (b) 30-second dry-run with synthetic inputs
Useful, but more expensive to build and maintain than it sounds, especially for:
- file-path-based nodes
- ffmpeg behavior
- ledger mutation
- local asset assumptions

Also, synthetic inputs often fail to reproduce the exact runtime semantics that matter.

### (c) Read the consumer code
This would have caught BUG-124 immediately:
- `_load_ledger_with_path` in `batch_humo_render.py` lines 1694-1756 already documented the `.mp4` fallback behavior
- `scene_manifest_json` was visibly a stub in `scene_sequencer.py` around lines 689 and 890

This is the cheapest gate that directly addresses the failure mode you hit.

---

## My recommended lightweight rule

Before shipping any workflow wiring change, require this mini-checklist:

### Wiring-change checklist
1. **Read the consumer input handling code**
   - not just `INPUT_TYPES`, but actual parsing/dispatch logic
2. **Read the producer output implementation**
   - not just output label/type
3. **Verify one real runtime example**
   - what value actually flows through this edge in production?
4. **If replacing an existing wire, explain why the old wire worked or didn’t**
   - in one sentence in commit message / PR note

That’s it.

This is lightweight enough that you’ll actually do it.

---

## One small automation I *would* add

Not a full shape checker. Just a **workflow linter for known footguns**.

Examples:
- flag any connection into `scene_manifest_json` because it is deprecated/stubbed
- flag outputs known to be placeholders
- flag nodes whose output labels are semantically misleading
- flag any edge changed without a short annotation in commit message

This is cheap because it’s **project-specific**, not a general type system.

For your codebase, that’s much more realistic than building a generic static analyzer.

---

# Concrete recommendations summary

## Keep BUG-121, with 4 small hardening tweaks
In `nodes/batch_humo_render.py::_resolve_radio_still_path`:
- sanitize `episode_id`
- guard `otr_stills_dir()` resolution
- require `is_file()` and `size > 0`
- log which fallback layer succeeded

## Keep BUG-123 as-is conceptually
In `nodes/video_composite.py::execute()`:
- retain `try/finally`
- retain `unload_all_models()` and `soft_empty_cache(force=True)`
- retain `cuda.synchronize()` in fallback
- add staged VRAM telemetry
- document that this node is the workflow cleanup boundary

## For Symptom 1
Most likely root cause:
- `humo_concat` mode not enforcing scene-boundary cuts, and/or
- ledger timing fields missing/invalid at composite time

### Next-run observability:
Add a pre-render timing dump in `VideoComposite`:
- mode
- ledger path
- per-line `start_s`, `dur_s`, clip path, contribution type

## For Symptom 2
Most likely root cause:
- radio-bookend render failed inside warning-only exception handling, or
- stamp was later overwritten

### Next-run observability:
In `batch_flux_render.py` around the radio-bookend attempt:
- log episode_id, output path, prompt mode, render start/success/failure, file exists/size, stamp success
In ledger writers:
- log whether `radio_bookend_path` is present after each save

## For BUG-125
**Deprecate/remove `scene_manifest_json`; do not implement it now.**

## For preventing future BUG-124s
Adopt:
- **mandatory “read producer + consumer code” rule**
- plus a tiny project-specific linter for known stub outputs / misleading slots

---

# Where I’m uncertain

I want to be explicit about the parts I can’t prove from the info provided:

1. **I cannot confirm whether `humo_concat` mode inherently avoids hard scene cuts** without reading `nodes/video_composite.py`.
2. **I cannot distinguish render-failed vs stamp-overwritten for the radio bookend** without logs from `batch_flux_render.py` and ledger snapshots.
3. **I cannot say whether additional ComfyUI caches matter in your environment** without seeing your exact custom-node globals and cleanup telemetry.

If you want, I can do a tighter second pass if you paste:
- `nodes/video_composite.py::execute()` and the `audio_source` branch,
- `batch_flux_render.py` around lines 455-502,
- and `batch_humo_render.py::_resolve_radio_still_path` plus `_load_ledger_with_path`.
