# Round A -- ChatGPT (gpt-5.5) elapsed=108.6s

## Executive verdict

**BUG-LOCAL-027:** The shipped fix is structurally sound for the observed failure. The regex + total-collapse hard gate directly address the real root cause: the preservation gate was blind to `[N] CHARACTER:` dialogue and therefore accepted an all-scaffolding revision. I would not change the per-character floor right now. The only notable residual risk is that the parser still assumes fairly strict all-caps speaker labels.

**BUG-LOCAL-028:** The core direction is sound: writers now target the per-episode workspace, and readers were updated to look there. The **radio bookend writer fix is low-risk**. The new `OTR_SaveToEpisodeWorkspace` node is conceptually correct for ComfyUI. The main thing I do **not** fully buy is the statement that the wildcard reader glob plus mtime filter makes cross-episode pollution “mathematically impossible.” It makes it unlikely in normal serial runs, but not impossible if episodes overlap, reruns happen close together, or the singleton/ledger timestamp assumptions wobble.

No VRAM concerns here. No low-level optimization implications.

---

# 1. BUG-LOCAL-027 review

## 1.1 Regex change

Given old location:

```python
nodes/story_orchestrator.py:6916
```

Old:

```python
r'^\s*\*{0,2}([A-Z][A-Z0-9_ ]+?)\*{0,2}\s*(?:\([^)]*\))?\s*:'
```

New:

```python
r'^\s*(?:\[\d+\]\s+)?\*{0,2}([A-Z][A-Z0-9_ ]+?)\*{0,2}\s*(?:\([^)]*\))?\s*:'
```

### Verdict

**Correct for the observed writer output.**

It now matches:

```text
[12] FLETCHER WELLS: We have to move.
[3] **FLETCHER WELLS**: We have to move.
FLETCHER WELLS: We have to move.
**FLETCHER WELLS**: We have to move.
```

That directly fixes the parser blindness.

### Edge cases where it could fail

It will still fail on variants like:

```text
[12]FLETCHER WELLS: text        # no space after bracket
12. FLETCHER WELLS: text        # numeric prefix but not [N]
- [12] FLETCHER WELLS: text     # markdown bullet before [N]
[012] FLETCHER-WELLS: text      # hyphen in speaker name
[12] DR. VOSS: text             # period in speaker name
[12] O'NEILL: text              # apostrophe in speaker name
```

But if the writer’s canonical format is indeed:

```text
[N] CHARACTER NAME: dialogue
```

then the regex is appropriate.

### False-positive risk

The false-positive risk is not from the new `[N]` prefix itself. The bigger risk remains the original pattern’s broad “all caps plus colon” match.

For example, depending on `_struct_exclude`, these could still be counted as characters:

```text
FADE OUT: ...
CONTROL ROOM: ...
TITLE CARD: ...
END CREDITS: ...
```

The new first-word exclusion at `nodes/story_orchestrator.py:6924` helps:

```python
first_word = char_name.split()[0] if char_name else ""
if char_name not in _struct_exclude and first_word not in _struct_exclude:
```

That is a good tightening. It catches things like:

```text
ACT 2:
MUSIC STING:
SCENE THREE:
```

provided `ACT`, `MUSIC`, `SCENE`, etc. are in `_struct_exclude`.

### Bottom line

The regex change is sound and targeted. I would not call it bulletproof, but it is correct for the documented failure.

---

## 1.2 Total-collapse gate

New logic:

```python
draft_total = sum(draft_char_counts.values())
revised_total = sum(revised_char_counts.values())

if draft_total >= 3:
    min_revised = max(1, _math.ceil(draft_total * 0.5))
    if revised_total < min_revised:
        ...
        return draft_text
```

### Verdict

**Sound. This is the most important part of the BUG-027 fix.**

The previous per-character gate could no-op when the parser returned `{}`. The new aggregate check catches the exact failure mode:

```text
draft:   18 character lines
revised: 0 character lines
```

Result:

```text
min_revised = ceil(18 * 0.5) = 9
0 < 9 → reject revision
```

Good.

### Is 0.5 the right threshold?

For this bug, yes.

This gate should not be trying to judge artistic quality. Its purpose is to reject catastrophic structural collapse. A 50% retention threshold is a reasonable hard safety floor.

For example:

```text
draft_total = 4
revised_total = 2
min_revised = ceil(4 * 0.5) = 2
2 < 2 → false, accept
```

That is the right behavior for a collapse guard. If the gate rejected exactly 50%, it would become more of an aggressive rewrite limiter rather than a catastrophic-failure detector.

So to your specific question:

> draft has 4 lines, revision has 2 — that’s exactly at threshold, should it be > or >=?

Current behavior is right:

```python
if revised_total < min_revised:
```

not:

```python
if revised_total <= min_revised:
```

Accepting exactly 50% is appropriate.

### Short draft skip

```python
if draft_total >= 3:
```

Also reasonable. For very short fragments, a hard ratio can be noisy. If a draft has only 1–2 character lines, other checks should handle it or the revision should be judged by downstream script validity.

---

## 1.3 Prompt addition

The added prompt language is strong:

```text
ABSOLUTE REQUIREMENT — DIALOGUE MUST SURVIVE THE REVISION:
...
Every CHARACTER speaker present in the draft MUST appear in the revision.
...
```

### `.format()` / BUG-026 risk

Based on the provided text, I see no literal unescaped `{}` braces in the added clause. Your new test for unescaped braces is also the right kind of regression guard.

So: **safe with respect to the BUG-026 `.format()` footgun**, assuming the actual committed string matches the summary.

### Over-correction risk

Yes, there is some over-correction risk, but I think it is acceptable for this product.

The line:

```text
Every CHARACTER speaker present in the draft MUST appear in the revision.
```

could prevent a legitimate revision from deleting a one-line throwaway character. But for this radio-drama generator, that is probably the correct bias. Losing a voice is more damaging than preserving one extra line.

Given the observed production failure was “zero dialogue,” I would rather have the model over-preserve dialogue than under-preserve it.

### One caveat

The secondary contributor remains:

```text
revision pass uses caller temperature, possibly 0.95
```

The prompt and gate mitigate this, but they do not remove the stochastic pressure. The system can still produce a weird revision that preserves dialogue counts while degrading content. That is not the same bug, but it is the remaining quality-risk path.

---

## 1.4 Should `min_line_count_per_character` change?

I would **not** tighten or loosen it right now.

The new total-collapse gate and the existing per-character gate serve different purposes:

- **Total gate:** catches “the whole dialogue layer disappeared.”
- **Per-character gate:** catches “one speaker got mostly deleted.”
- **Prompt clause:** reduces the chance the bad revision is produced in the first place.

Changing the per-character floor immediately after adding the total gate would make it harder to interpret soak results. Keep it stable for now.

If future soaks show excessive false rejections, then revisit. But based on this fix set, I would leave the default at `2`.

---

# 2. BUG-LOCAL-028 review

## 2.1 `OTR_SaveToEpisodeWorkspace` ComfyUI API

Summary:

```python
class SaveToEpisodeWorkspace:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "role_kind": (["stills", "portraits"], {"default": "stills"}),
            "filename_pattern": ("STRING", {"default": "full_env"}),
        }, "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"

    def save(...):
        ...
        return {"ui": {"images": [...]}}
```

### Verdict

**Structurally correct for a ComfyUI output/save node.**

Key points are right:

```python
RETURN_TYPES = ()
OUTPUT_NODE = True
FUNCTION = "save"
```

That should cause ComfyUI to execute it as a terminal save node even though it has no normal graph output.

The hidden inputs are also conventional:

```python
"hidden": {
    "prompt": "PROMPT",
    "extra_pnginfo": "EXTRA_PNGINFO",
}
```

### ComfyUI-specific gotchas

A few things to watch, not necessarily bugs:

1. **Server restart likely required after adding the node.**  
   New class mappings in `__init__.py` are not always cleanly picked up by hot reload. For production soak, restart ComfyUI after adding `nodes/otr_save_to_episode_workspace.py`.

2. **Workflow node type must match `NODE_CLASS_MAPPINGS`.**  
   The workflow JSON node id 25 needs `class_type` matching the registered mapping key, e.g. `OTR_SaveToEpisodeWorkspace`, not merely the Python class name unless they are the same.

3. **UI image payload format matters if you rely on preview display.**  
   ComfyUI usually expects UI image entries shaped somewhat like:

   ```python
   {
       "filename": "...",
       "subfolder": "...",
       "type": "output"
   }
   ```

   If the image saves but the UI preview is odd, this is where I would look. It would not necessarily break the pipeline if readers use filesystem paths.

4. **Tensor-to-PIL conversion must be deterministic and robust.**  
   The summary does not show `_tensor_to_pil`. It should clamp and convert consistently. This matters for visual determinism, though not directly for C7 audio byte identity.

5. **Filename pattern sanitization.**  
   Since this is local and workflow-controlled, low concern. But if `filename_pattern` can contain path separators, it could escape the intended directory. Probably not a practical issue here.

---

## 2.2 Legacy fallback when no singleton is active

Summary:

```python
episode_id = _resolve_episode_id()
target_dir = _resolve_target_dir(role_kind, episode_id)
```

Fallback:

```text
no singleton → legacy dirs
```

### Verdict

**Acceptable as a compatibility fallback, but not truly safe for production episode isolation.**

This fallback is useful because it avoids hard failures in ad-hoc/manual ComfyUI runs. However, it can absolutely reintroduce the class of “wrong place / wrong episode” behavior if the singleton is missing unexpectedly.

So I would phrase the risk like this:

- For interactive/manual/debug runs: safe enough.
- For production batch episode generation: the fallback should be considered degraded mode.
- It should be noisy in logs so you can tell when an episode failed to resolve.

You said the node “never raises — save failure logs and silently skips.” The “never raises” part is fine for Comfy graph survivability. But if singleton resolution fails, I would want that to be visible in logs because otherwise soak results can look superficially green while assets went to legacy storage.

No new fix demanded here, but this is a residual observability risk.

---

## 2.3 Per-episode wildcard glob in `batch_humo_render.py`

New pattern:

```python
"otr/episodes/*/stills/full_env_*.png"
```

with freshness filter:

```python
fresh_floor = ledger_mtime - 60s
```

### Verdict

**The direction is correct, but the “mathematically impossible” claim is too strong.**

The wildcard creates a nonzero cross-episode leak risk.

The mtime filter helps, but it does not prove episode correctness. It assumes:

1. only one episode is being generated at a time;
2. no other episode has fresh files inside the mtime window;
3. ledger mtime is a reliable episode boundary;
4. reruns do not happen close together;
5. filesystem timestamps behave as expected on Windows;
6. no concurrent Comfy queue or batch process mutates the singleton.

In normal solo-workstation serial use, those assumptions are probably true most of the time. But they are not mathematical guarantees.

### Specific leak case

Episode A writes:

```text
output/otr/episodes/A/stills/full_env_00001_.png
```

Episode B starts soon afterward. If B’s Humo reader scans:

```text
output/otr/episodes/*/stills/full_env_*.png
```

and A’s file is newer than:

```text
ledger_mtime(B) - 60s
```

then A’s still can pass the freshness filter. Whether it is selected depends on sorting/selection logic not shown in the summary.

That is probably low probability in your current serial pipeline, but not impossible.

### Bottom line

The wildcard reader update is better than looking only in flat legacy dirs. But if another follow-up happens, this is the area I would expect it to touch first: replacing wildcard discovery with explicit `episode_id`-scoped lookup where possible.

---

## 2.4 Per-episode counter starting at 1

### Verdict

**Correct.**

Per-episode counters are preferable here.

Old behavior:

```text
output/otr/stills/full_env_00213_.png
```

New behavior:

```text
output/otr/episodes/<ep>/stills/full_env_00001_.png
```

That is cleaner, more deterministic, easier to debug, and avoids global counter coupling.

I do not see a reason for global monotonic numbering unless some downstream node depends on “latest global still” behavior. Based on your description, the downstream nodes should be episode-scoped, so per-episode numbering is the right model.

### Minor determinism caveat

If you rerun the same episode into a non-cleaned episode directory, then `next_idx` will continue from the existing files. That can produce:

```text
first run:  full_env_00001_.png
second run: full_env_00002_.png
```

instead of overwriting. That is usually safer operationally, but if any reader picks “latest” or “first glob result” ambiguously, stale files can matter.

For C7, this is not an audio-byte-identity issue unless visual asset selection feeds back into audio, which it should not.

---

# 3. What may have been missed

## 3.1 Other legacy path read/write sites

You checked:

- `video_composite.py`
- `batch_humo_render.py`
- `batch_ltx_render.py`

That covers the obvious consumers you named.

I would still search the full repo for these strings:

```text
otr/stills
otr_stills
_legacy_stills
full_env_
radio_bookend_
filename_prefix
SaveImage
otr_portraits
portraits_dir
stills_dir
otr_stills_dir()
otr_portraits_dir()
```

Particularly important:

```python
otr_stills_dir()
```

called with no `episode_id`.

Given `nodes/_otr_paths.py:208-218` falls back to `_legacy_stills/` when no episode id is supplied, any no-arg call is suspicious unless it is intentionally legacy/debug.

## 3.2 Workflow JSON still containing old `SaveImage`

You updated node id 25 in:

```text
workflows/otr_scifi_16gb_full.json
```

Good.

But check for other workflow files or variants. If there are older or alternate workflows, they may still contain:

```json
"class_type": "SaveImage"
```

with:

```json
"filename_prefix": "otr/stills/full_env"
```

That would recreate the bug only when those workflows are used.

## 3.3 ComfyUI hot-reload / `sys.modules` caching

This is a real operational risk after adding a new node file and changing `__init__.py`.

Recommended operational stance:

- fully stop ComfyUI;
- start it clean;
- verify startup logs show the new node registered;
- load the workflow fresh;
- confirm node id 25 resolves as `OTR_SaveToEpisodeWorkspace`.

If ComfyUI was hot-reloaded in place, you could get confusing states:

- workflow references a class not registered yet;
- old module object still loaded;
- UI cache shows stale node definitions;
- `NODE_CLASS_MAPPINGS` does not include the new class until restart.

This is not a design flaw in the fix, just a ComfyUI deployment gotcha.

## 3.4 Race conditions in `in_flight_ledger_path()`

Low risk for a single serial queue. Higher risk if anything can overlap.

Potential race scenario:

1. Episode A starts.
2. Singleton points to A’s ledger.
3. Episode B starts before A’s save node executes, or a batch node mutates the singleton.
4. Save node resolves singleton at runtime and writes A’s images into B’s episode folder.

If the pipeline truly executes one episode at a time on one Comfy queue, this is probably fine.

But the singleton model is inherently global state. It is less robust than explicit episode id propagation through node inputs or workflow metadata.

Again: not necessarily a required fix now, but it is a residual architectural risk.

## 3.5 BUG-027 interaction with determinism / C7

The prompt change may cause the LLM revision output to differ compared to prior commits. That is expected.

But if C7 means “same inputs, same run, byte-identical final audio,” then the LLM path must already be deterministically seeded and run under stable settings. The BUG-027 changes do not add new randomness, but the critique pass still operating at high temperature is worth remembering.

The gate can also cause branch variation if the model output is nondeterministic:

```text
run 1: revision accepted
run 2: revision rejected, draft used
```

That would absolutely change downstream Bark input and final audio. If your LLM generation is already deterministic despite temperature, fine. If not, BUG-027 did not create that risk, but it makes the accept/reject branch more visible.

---

# 4. Follow-up-fix probability estimates

These are my subjective estimates for “will need a follow-up fix in the next two weeks of soak.”

| Fix element | Follow-up probability | Reasoning |
|---|---:|---|
| BUG-027 Part 1: regex for `[N] CHARACTER:` | **15%** | Correct for observed format. Residual risk is speaker-label variants: bullets, no space after bracket, hyphens, apostrophes, periods, or unexpected structural headers. |
| BUG-027 Part 2: total-collapse gate | **10%** | Strong fix for the exact failure. Threshold is sane. Only likely follow-up would be threshold tuning or short-draft behavior, not core logic. |
| BUG-027 Part 3: prompt addition | **12%** | Safe with respect to braces. May need wording adjustment if it over-preserves minor characters or if high-temp revisions still produce malformed dialogue. |
| BUG-028 Site 1: `batch_flux_render.py` radio bookend writer using `otr_stills_dir(episode_id)` | **8%** | Very direct fix. Main risk is whether `episode_id` resolution at line 768/772 is always populated before line 833. If yes, this is solid. |
| BUG-028 Sites 2–4: new save node plus Humo/LTX read alignment | **30%** | Correct direction, but larger surface area: new Comfy node registration, workflow class type, singleton fallback, UI return shape, wildcard episode glob, mtime freshness assumptions, and hot-reload behavior. Most likely follow-up area. |

If I had to pick the single most likely future issue, it is:

```text
batch_humo_render wildcard episode glob + freshness filter chooses a wrong fresh still
```

or:

```text
new node not active because ComfyUI was not cleanly restarted / workflow class_type mismatch
```

---

# 5. Risk from skipping pre-implementation round-robin

Yes, some blind spots are more likely because consultation happened after implementation.

A pre-implementation review probably would have challenged these points earlier:

1. **Wildcard episode glob is not a proof of episode isolation.**  
   The phrase “mathematically impossible” would likely have been questioned.

2. **Legacy fallback can mask production routing failures.**  
   It preserves compatibility, but if singleton resolution fails, assets silently go to legacy dirs again.

3. **ComfyUI deployment semantics.**  
   Adding a new node file plus workflow JSON change often needs a clean restart and mapping verification.

4. **Strict speaker-name parser assumptions.**  
   The regex fix handles `[N]`, but not all plausible all-caps names.

5. **High-temperature critique pass remains a contributor.**  
   The gate prevents catastrophic silence, but does not fully solve stochastic rewrite quality.

That said, the shipped fixes are not reckless. They are targeted and backed by a decent test expansion: 23 new tests, 155 cumulative regression tests, and Bug Bible green except known skip/xfail. The skipped round-robin mostly increases the chance of residual edge-case cleanup, not the chance that the main fixes are wrong.

---

## Final recommendation

Do not roll back.

For BUG-027, the fix is good.

For BUG-028, the per-episode write migration is good, but I would watch the next few soaks specifically for:

```text
legacy fallback warnings
missing per-episode stills/
missing per-episode portraits/
Humo binding to a still from a different episode
ComfyUI node registration/load errors
old SaveImage nodes in alternate workflows
```

If the next failure appears, I would expect it to be around the BUG-028 read-discovery path, not the BUG-027 dialogue preservation gate.
