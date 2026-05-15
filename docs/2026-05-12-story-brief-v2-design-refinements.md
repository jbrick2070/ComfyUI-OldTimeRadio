# `meta.story_brief` v2 — design refinements

Synthesized from three reviewer passes against the Cowork R1 research paper. The brief is part of every visual prompt and the MusicGen mood layer. Every refinement below either tightens that integration or fits it into the 16 GB VRAM envelope of the RTX 5080 Laptop hardware target.

This is the refined design surface. The go-forward build plan follows separately.

---

## 1. Brief scope

The brief describes the **scene**, not the **action**. Environment, lighting, props, period markers (only when the script proves them), atmosphere, and the human presence as part of the scene's composition. No dialogue verbs ("speaking", "arguing"), no named characters (use compositional roles — "one figure", "two seated"), no plot actions ("interrogating" is action; "single-room interrogation under a swinging bulb" is composition).

The brief reflects what the script *shows*, not what the slug *suggests*.

---

## 2. Reflection-pass input builder (capped)

Do not feed the full assembled script to the local LLM. On longer episodes this randomly degrades. The reflection pass receives a structured, length-capped input:

| Field | Source | Cap |
|---|---|---|
| Title | `meta.episode_title` or `meta.title` | one line |
| Style slug | `meta.style` | one slug |
| Cast | `cast[].name` + `cast[].character_description` | one line per character |
| Scene headers | extracted from `lines[]` (any line with `speaker_role == "scene"` or section markers) | all |
| Opening | first 12–20 lines from `lines[]` | hard cap |
| Closing | last 6–10 lines from `lines[]` | hard cap |
| Non-dialogue rows | `lines[]` filtered to `speaker_role in {"sfx", "music", "env", "scene"}` | all |

A single helper `_build_reflection_input(led) -> str` assembles this from the ledger. The helper is the only piece that knows ledger shape; the reflection prompt sees a fixed-shape string. Total input stays under ~1500 tokens regardless of episode length.

---

## 3. Reflection-pass LLM contract

### 3.1 Strict JSON output

Local models under 16 GB VRAM frequently ignore "output one sentence" instructions. The prompt requests a JSON object; Python validates and extracts.

```json
{
  "story_brief":      "single-room interrogation under a swinging bare bulb; rain-streaked window; a detective and a suspect across a steel table; sweat and cigarette smoke; harsh top-down shadows",
  "setting_terms":    ["single-room interrogation", "steel table", "rain-streaked window"],
  "lighting_terms":   ["swinging bare bulb", "harsh top-down shadows"],
  "atmosphere_terms": ["sweat", "cigarette smoke", "tense"]
}
```

Three sidecar term arrays let the central helpers (§5) extract narrowed fragments per-consumer without re-parsing the prose.

### 3.2 LLM settings

For local models, compliance matters more than creativity:

```text
temperature:     0.2–0.4
top_p:           0.8–0.9
max_new_tokens:  ~120  (covers JSON object + prose)
```

The legacy `_generate_ltx_style_brief` used temperature 0.7 — too creative for compliance-critical output.

### 3.3 No invented period or location

Add an explicit rule to the prompt:

> Do not include a date, decade, century, city, country, or historical period unless the script explicitly names one. If the script implies a period without naming it, use atmosphere terms ("smoke-filtered", "incandescent glow", "polished brass") instead of dated terms ("1940s", "Victorian").

This kills the "1947 LA grime" hallucination class.

### 3.4 Post-generation validation gate

After the LLM call, validate. Reject if any of:

- contains a named character (matched against `cast[].name` list)
- contains dialogue verbs (`speaking`, `arguing`, `watching`, `whispering`, etc.)
- contains plot-action verbs (`interrogates`, `escapes`, `discovers`, etc.)
- contains a date, decade, century, city, country, or historical period not present in the source ledger
- exceeds 300 chars
- contains more than one sentence
- contains quotation marks or Markdown
- fails JSON parse

### 3.5 Repair pass before failure

If validation rejects, run ONE repair pass at a higher temperature with an explicit critical-failure prefix:

```text
CRITICAL: You previously failed validation because: {rejection_reasons_list}.

Rewrite this visual brief to obey the schema. Remove named characters,
dialogue verbs, plot actions, unsupported dates or locations, extra
sentences, quotation marks, and Markdown. Return only the JSON object.

Failed brief: {failed_output}
Rejection reasons: {validation_errors}
```

The repair pass runs at `repair_temperature = min(reflection_temperature + 0.15, 0.55)`. Effective range 0.35-0.55. The clamp keeps repair temperature inside the declared safe range even if a future operator sets `reflection_temperature` above 0.4.

The temperature bump together with the explicit `CRITICAL:` prefix break the deterministic-retry failure loop characteristic of low-temperature local-model JSON output: at temp 0.2-0.4 a literal repeat of the same prompt produces nearly identical output, so the second pass would just fail the same way. The bump introduces enough variance to escape the local minimum; the prefix re-orients the model toward the actual rejection reason rather than re-generating the failed shape.

Validate the repair output. If it also fails, fall through to the empty-string-with-status path (§4.1).

> **C0b amendment (2026-05-15):** prior text specified a single repair attempt at the same temperature with no critical prefix. Round-robin-2 finding R-06 surfaced the deterministic-retry-loop class. Sprint C E-18 / RR-B5 added the upper-bound clamp at 0.55. See `SPRINT.md` §1.3 R-06 and §1.2 E-18 for full disposition.

### 3.6 Reflection-pass timeout contract (BUG-LOCAL-228)

The current `_run_with_timeout` in `story_orchestrator.py` is non-blocking by design: `executor.shutdown(wait=False)` lets the orphan worker drain naturally rather than blocking the calling thread. The reflection pass MUST follow the same contract.

Required behavior on `_LLMTimeout`:

- **Do not block.** No `future.cancel()`, no `torch.cuda.synchronize()`, no busy-wait for the orphan worker to drain.
- **Invalidate cache dict references** via `_otr_loader_mod.invalidate_cache_no_gpu_teardown()` so the next `request_slot` forces a fresh load when GPU work eventually completes.
- **Raise `_LLMTimeoutWorkflowPause`** to halt the ComfyUI queue; the operator decides whether to retry or abandon. The orphan worker drains naturally in the background; the next request_slot finds an invalidated cache and reloads cleanly.

This is the BUG-LOCAL-228 contract, shipped in S31 B4 (`a4fe67a`). See `nodes/story_orchestrator.py:336-372` for the canonical implementation. A blocking sync barrier — once attempted — proved unsafe under stalled GPU forward passes that cannot be terminated: `future.cancel()` returns immediately but the kernel continues executing, and a subsequent `cuda.synchronize()` blocks the main thread indefinitely waiting for that uncancellable kernel. Detail in §11.4.

> **C0b amendment (2026-05-15):** prior text specified "Reflection pass MUST clear before the workflow advances. No drain in the background. On `_LLMTimeout`: cancel the future, force GPU sync, empty cache, verify VRAM headroom before yielding." That is the exact anti-pattern BUG-LOCAL-228 was filed against. Sprint C L-3 amends the spec to the BUG-LOCAL-228 contract. See `SPRINT.md` §1.1 L-3 and §A.8 for evidence.

---

## 4. Storage schema (provenance + status)

The reflection pass stamps:

```json
"meta": {
  "story_brief":               "...",
  "story_brief_status":        "ok",
  "story_brief_error":         null,
  "story_brief_model":         "google/gemma-4-E4B-it",
  "story_brief_prompt_version": "v1",
  "story_brief_source":        "post_script_reflection",
  "story_brief_char_count":    234,
  "story_brief_terms": {
    "setting":    ["..."],
    "lighting":   ["..."],
    "atmosphere": ["..."]
  }
}
```

### 4.1 Failure mode: empty string with explicit status

When the reflection pass fails (timeout, both validation passes rejected, JSON malformed), stamp:

```json
"story_brief":        "",
"story_brief_status": "failed",
"story_brief_error":  "timeout_or_invalid_output"
```

Empty `story_brief` lets consumers fall through. The non-null status field makes the failure visible to debugging without forcing the writer to abort the episode on a 5-second flavor-text failure.

This resolves Cowork's open question 6.3 — not raw empty-string (silent fallback violates standing directive #1), not raise (wrong cost-benefit to abort a whole episode on flavor-text failure).

---

## 5. Central consumer helpers

Consumers do not parse the brief prose. They call helpers:

```python
def get_story_brief_full(meta: dict) -> str:
    """Full brief prose, empty string if absent or failed."""

def get_story_brief_ltx(meta: dict, max_chars: int = 90) -> str:
    """Brief fragment safe for LTX motion prompts.
    Trimmed at sentence/clause boundary; never mid-word."""

def get_story_brief_lighting(meta: dict) -> str:
    """lighting + atmosphere terms joined. For portrait builders
    that need scene match without prop/setting noise."""

def get_story_brief_music_mood(meta: dict) -> list[str]:
    """Mood keywords extracted from atmosphere_terms.
    Intersected with MusicGen's known mood vocabulary."""

def get_story_brief_status(meta: dict) -> str:
    """'ok', 'failed', or 'absent' for older ledgers."""
```

One helper per consumer shape. The alternative is N slightly different bad implementations across N consumer files.

---

## 6. Per-consumer integration shapes

Every text-prompt visual or music consumer reads the brief through the helpers in §5. Integration is unconditional per the R1 standing directive — no exceptions, no indirect paths.

| Consumer | Helper used | Placement |
|---|---|---|
| FLUX env render | `get_story_brief_full` | Between env description and style_suffix tail |
| FLUX radio bookend | `get_story_brief_full` | Replaces the weak `scenes[0].env` / `episode_id` tiers in the existing tier chain |
| FLUX portraits | `get_story_brief_lighting` | Append after appearance description; lighting + atmosphere only |
| LTX motion clips | `get_story_brief_ltx(max_chars=90)` | After the motion-centric role template; motion verbs lead, brief fragment follows |
| HuMo lip-sync | `get_story_brief_lighting` | Append before `_DEFAULT_POS_SUFFIX`; suffix's HuMo-specific lighting tuning preserved |
| MusicGen theme | `get_story_brief_music_mood` | Mood keywords merged into `_mood_suffix` alongside `script_brief` keywords |
| `otr_video_plan` composite | `get_story_brief_full` | Replaces the always-empty `scene_visual` slot in `compose_shot_prompt` |
| HUD card / treatment text | `get_story_brief_full` | Decorative — display only |

### 6.1 LTX budget — 220–240 total, 80–100 brief fragment

The R1 paper started at 300 chars total. First reviewer pass tightened to 240–260. Second reviewer pass argued for 200–250 on independent VRAM grounds (longer text embeddings inflate VRAM during the LTX cross-attention phase). The conservative intersection is the starting point:

- Total LTX prompt cap: **220–240 chars**
- Brief fragment cap (via `get_story_brief_ltx(max_chars=90)`): **80–100 chars**
- Motion verbs MUST come first in the composed prompt
- Drop the brief fragment if including it would push motion verbs past char 140

Dual-purpose: prompt-dilution fix (BUG-LOCAL-112) AND VRAM micro-optimization. If soak-tests show motion preserved AND VRAM headroom comfortable, loosen in a follow-up commit. Don't start loose.

### 6.2 Portrait integration is narrowed, not abstained

Every-consumer mandate stands. The full prose brief would add props, rooms, and settings to a head-and-shoulders prompt and invite clutter. The narrowed `get_story_brief_lighting` helper threads the needle — portraits inherit lighting and atmosphere context (matching the bookends and HuMo clips visually) without inheriting setting noise.

### 6.3 MusicGen never sees full prose

Keep MusicGen deterministic. The helper extracts mood keywords from `story_brief_terms.atmosphere`, intersects with MusicGen's known mood vocabulary, returns a list. Same shape as the existing `_mood_suffix(script_brief)` keyword scan. Two mood signals merged: pre-write news mood + post-write story mood. The full brief prose never enters a MusicGen prompt.

---

## 7. Hard caps (char-count based)

Word count doesn't matter for LTX or prompt assembly; char count does. The brief contract:

- Exactly 1 sentence
- 180–260 chars preferred, 300 chars hard max
- No quotation marks, no Markdown
- No dates / decades / centuries / cities / countries / periods unless explicitly named in the script
- No named characters
- No dialogue verbs
- No plot actions

Validation gate in §3.4 enforces all of these.

---

## 8. Pre-flight sequencing

Three cleanbreaks must land before the `story_brief` build sprint, in this order:

### 8.1 Era literals (must land first)

- `visual/batch_flux_portrait_render.py:107` — `style_anchor` default `"1940s noir radio drama style"` replaced with era-neutral text
- `nodes/otr_video_plan.py:79` — `_DEFAULT_STYLE_TAIL` drops `"1980s broadcast aesthetic"`, keeps cinematic-grammar parts

If these don't clean first, brief testing is polluted. Visual drift in a soak-test could come from the brief or from a hardcoded decade fighting it.

### 8.2 Genre deletion (lands second)

Per Cowork R1 §1.4. Add one back-compat safety test before deleting: confirm older ledgers without `meta.visual_plan.genre` still render HUD and treatment text without crashing. The three video_engine fall-throughs at lines 711, 836, 1075 should already handle this, but confirm before pulling the rug.

### 8.3 VRAM envelope tightenings (lands third — see §11)

The §11 hardware items either close concurrently with the `story_brief` build or land just before it. They are infrastructure for the reflection pass to run safely.

### 8.4 `story_brief` build sprint (lands fourth)

Only after 8.1, 8.2, and 8.3 close.

---

## 9. Test discipline — three ugly ledgers required

Normal-case fixtures will pass anything. The reflection pass earns its keep on adversarial cases:

| Test ledger | Pressure-test |
|---|---|
| Noir slug + space colony script | Does the brief follow the actual script (Mars/colony/oxygen) or hallucinate noir tropes from the slug? |
| Detective script with no clear setting | Does the brief invent a setting (forbidden by §3.3) or correctly produce a sparse atmosphere-only brief? |
| Long script (15+ minute) with three distinct locations | Does the brief pick a dominant scene or smear them all together? Does input-builder cap (§2) cause information loss? |

These three fixtures are also the soak-test set for §6.1's LTX budget tuning and §11 VRAM monitoring.

---

## 10. Naming — `meta.story_brief` retained

Reviewer suggested `meta.visual_story_brief` to make scope explicit. Rejected because MusicGen consumes the brief too (mood keywords), so "visual" would be misleading.

Resolution: keep `meta.story_brief`. Scope defined as **"A scene-composition summary of the episode as written — environment, lighting, props, period (when proven), atmosphere, and human presence as compositional element. Consumed by visual prompt builders (FLUX, LTX, HuMo) and as a mood-keyword source for MusicGen."**

---

## 11. VRAM envelope discipline (16 GB ceiling, RTX 5080 Laptop)

Five hardware-grounded findings. All locked. The build sprint inherits them as infrastructure prerequisites.

### 11.1 Default model: `google/gemma-4-E4B-it`

**Change in:** `otr_scifi_16gb_full.json` Node 1 `OTR_LedgerScriptWriter` default, `story_orchestrator.py` default model_id constant, `_MODEL_CONTEXT_CAPS` mapping.

Replace `mistralai/Mistral-Nemo-Instruct-2407` default with `google/gemma-4-E4B-it`. The 12B Nemo eats ~7.5 GB in NF4 and leaves no room for Bark / MusicGen / FLUX / LTX / HuMo co-residency without swapping to shared system memory. The 4B Gemma architecture gives audio and video stages enough breathing room to load without forcing a full LLM unload between every stage.

The reflection pass adds a second LLM invocation per episode. Cutting per-invocation footprint from 7.5 GB to ~3 GB makes the second call free in VRAM terms; with Nemo it would require an unload-reload cycle just to fit.

### 11.2 Flagship VRAM threshold: 15.0 → 14.5

**Change in:** `story_orchestrator.py` line 580 and any other site gating on `total_vram >= 15.0`.

Current threshold was tuned for desktop cards reporting a clean 16 GB. Laptop GPUs report 14.7–14.9 GB to PyTorch after OS and hardware reservations. The strict `device_map={"": 0}` allocation lock never triggers on the 5080 Laptop, which lets `bitsandbytes` silently fragment the model across system RAM. Drop to `>= 14.5`.

### 11.3 Gemma 4-E4B context cap: 16384 → 8192

**Change in:** `_MODEL_CONTEXT_CAPS["google/gemma-4-E4B-it"]` in `story_orchestrator.py`.

16K context with unquantized KV caching consumes an additional 2–3 GB dynamically as the prompt fills. The §2 capped input builder produces ~1500-token reflection inputs; 8K is comfortable headroom (5× typical fill) without paying the 16K KV-cache tax. Match Nemo's and Qwen's caps for predictable VRAM budgets across models.

### 11.4 Reflection-pass timeout contract (BUG-LOCAL-228)

**Change in:** none — `_run_with_timeout` already follows the correct contract as of S31 B4 (`a4fe67a`).

The current behavior on `_LLMTimeout` IS the correct behavior:

```python
# nodes/story_orchestrator.py:336-372 (abbreviated)
except FuturesTimeout:
    # orphan worker still running GPU forward pass; cannot kill
    # invalidate cache dict references WITHOUT touching GPU
    _otr_loader_mod.invalidate_cache_no_gpu_teardown()
    # raise workflow-pause subclass so ComfyUI halts the queue
    raise _LLMTimeoutWorkflowPause(...)
finally:
    # Do not wait for the orphaned worker -- let it drain in the background.
    executor.shutdown(wait=False)
```

The reflection pass inherits this contract by reusing `_run_with_timeout` on its `technical_fn` calls (writer composition → reflection → repair). Three potential timeout points, all governed by the same non-blocking pattern. VRAM is reclaimed when the orphan worker finishes its forward pass and Python garbage-collects the cache entry that was invalidated above. The next workflow run starts from a clean cache.

A blocking hard sync barrier was once specified here and was an anti-pattern: GPU forward passes cannot be terminated mid-kernel. `future.cancel()` returns immediately but the kernel continues executing; a subsequent `torch.cuda.synchronize()` blocks the main thread indefinitely waiting for that uncancellable kernel; ComfyUI's queue stalls; the operator sees a frozen UI with no way to recover. The non-blocking contract above lets the queue advance to a recoverable state (the workflow-pause exception) while the GPU work drains naturally in the background.

Non-negotiable on the 16 GB envelope. The reflection pass cannot ship with a blocking barrier.

> **C0b amendment (2026-05-15):** prior text specified a blocking hard-sync-barrier with `future.cancel()` + `torch.cuda.synchronize()` + `torch.cuda.empty_cache()` + a VRAM-headroom verification check. That implementation was prototyped, was the root cause of indefinite UI freezes, and was reverted. Sprint C L-3 amends the spec to match the shipped BUG-LOCAL-228 contract. See `SPRINT.md` §1.1 L-3 and §A.8.

> **Forensic footnote (BUG-LOCAL-228):** filed against the original `future.cancel()` + `cuda.synchronize()` implementation. Symptom: indefinite UI freeze on LLM timeout; `cuda.synchronize()` would never return because the orphan kernel was uninterruptible. Fix: S31 B4 commit `a4fe67a` switched to `executor.shutdown(wait=False)` + `invalidate_cache_no_gpu_teardown()` + `_LLMTimeoutWorkflowPause` raise. Implementation lives at `nodes/story_orchestrator.py:336-372`.

### 11.5 LTX budget reflects VRAM pressure (cross-reference §6.1)

The 220–240 char total / 80–100 char brief-fragment caps in §6.1 are dual-purpose: prompt-dilution prevention (BUG-LOCAL-112) AND VRAM micro-optimization (smaller text embeddings → less cross-attention VRAM during LTX text encoding). The two reviewer streams converged on this budget independently.

### 11.6 Verification

After §11.1–§11.4 land, run the §9 three ugly ledgers with VRAM monitoring active. Capture peak VRAM during:

- Writer composition pass (baseline)
- Reflection pass (new)
- Reflection pass + repair pass + LTX render (worst case)

Expected ceiling: ≤14.5 GB at any point. If any phase exceeds, the failure is concrete (KV cache, model overlap, orphan thread, prompt embedding) and the fix lands per-phase, not as broad VRAM tightening.

---

## 12. Round-robin questions closed

The following Cowork R1 open questions are resolved:

- **6.1 word-count window** → §7 (char-count caps, word count discarded)
- **6.3 failure mode** → §4.1 (empty-string with explicit status)
- **6.4 slug-vs-brief conflict** → no conflict; brief follows script, §3.3 prevents slug-hallucination
- **6.5 token budget** → §2 (capped input builder makes this deterministic)
- **6.6 retire `meta.ltx_style_brief`** → confirmed retire; `meta.story_brief` canonical
- **6.7 LTX prompt-length budget** → §6.1 (220–240 total, 80–100 brief fragment)

Still open for one focused round-robin before build:

- **6.2 reflection pass position** — inside `OTR_LedgerScriptWriter.execute()` vs new `OTR_StoryBriefReflection` node. The §11.4 BUG-LOCAL-228 timeout contract applies either way; this question is about where the call site lives. (Resolved by Sprint C Q1 lock to K.5.5 — see `SPRINT.md` §3.)

---

End of design refinement.
