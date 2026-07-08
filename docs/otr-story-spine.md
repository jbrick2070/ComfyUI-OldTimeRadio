# OTR Story Spine -- Plan + Execution Runbook (Model-Agnostic)

**Target:** a Cowork / agentic coding session (repo access, Desktop Commander, git, subagents) executes each stream below; round-robin sessions can run streams without coordinating.
**Branch base:** `v2.0-alpha`. **Priority:** good story first, clean ledger second, audio/render after that.
**No model names anywhere.** Every rule holds whatever model is bound to a slot on a given run.
**Loop, per stream:** `REVIEW -> CODE -> WIRE -> REGRESS -> COMMIT`. Repeat until all five streams are green.

> **Status:** REVIEWED against the live tree 2026-05-31 (commit `4f85010`, branch `v2.0-alpha`). Section 0 lists every place the original plan drifted from the real code and what the corrected contract is. The stream sections below are rewritten to the corrected contracts. **Read Section 0 first** -- it is the difference between this plan and code that would have broken "audio is king."

---

# PART 0 -- REVIEW FINDINGS (live-tree drift corrections)

The original spine was written without the live code in front of it. The skeleton (4 stages, 6 invariants, 5 streams) is sound and is kept. These nine contract corrections are load-bearing; coding to the old contract would have produced wrong or audio-unsafe changes.

| # | Original plan said | Live tree (file:line) | Corrected contract |
|---|---|---|---|
| D1 | Stream B: "the cap is `min(cap, target_words * 4)` ... pull the multiplier to x2 ... this is the 829-word overshoot source." | `_otr_line_composer.py:1606` -- `min(int(max_new_tokens_cap), max(40, int(req.target_words) * 4))` is the **`max_new_tokens` TOKEN budget**, not a word cap. The **word** ceiling is `_word_bands()` (`:1348`): `word_cap = max(15, int(target_words * _MAX_OVERSIZE_RATIO))`, `_MAX_OVERSIZE_RATIO = 3.0`, per line. `min_words = 1`, `max_words = target_words * 1.7`. | **Do NOT touch the `*4` token cap** -- pulling it to x2 truncates lines mid-sentence (Bark/timing/"audio is king" hazard). The length levers are the per-line `word_cap` (3.0x) and `max_words` (1.7x drift ceiling). See D2. |
| D2 | Stream B tightening is "free," no downside. | `_word_bands` docstring (`:1351-1402`): the floor was **removed** (`min_words=1`) because the documented failure mode was **UNDERSHOOT** -- mistral-nemo emits 4-15 word lines and 4 operator soaks (BUG-LOCAL-279) tripped the floor and exhausted the reroll loop. "Overshoot was never the failure mode" for the local model. | The 350->829 overshoot is a **premium-model** phenomenon (Opus), not local. Tightening per-line ceilings re-opens the BUG-279 undershoot-exhaustion risk for the local slot. Stream B = **rider (safe) + a CONSERVATIVE word_cap tightening that keeps `min_words=1`**, regressed hard. Hard total-length authority belongs to **Stream D's editor**, not a per-line cap. |
| D3 | Stream A targets `OutlineSchema` and "3 fields ride the same `structured_call` / retry-repair ladder. No new LLM call." | `_otr_outline.py:176` class is **`Outline`** (not `OutlineSchema`); `Beat` at `:84`; `generate_outline` at `:1453` is **Path C** -- a tree of small calls: Stage 1 macro (`_MacroShape`: title/premise/setting/time_of_day, `:1550`), Stage 2 per-phase speakers (`_PhaseSkeleton`), Stage 3 per-beat intent+mood (`_BeatFleshout`), then a **Python combiner** assembles the final `Outline`. There is **no single `structured_call` that emits `Outline`.** | `central_tension` -> add to **`_MacroShape`** (Stage 1, rides the existing `structured_call`). `turning_point` / `button` reference `beat_index`, which does not exist until the combiner builds beats -> **stamp them deterministically in the Python combiner** (turn = first voiced beat of the climactic phase; button = last voiced beat; `what_changes`/`payoff` drawn from the climax beat intent / macro). Still no new LLM call -- but not the mechanism the old plan described. |
| D4 | C/D editor & critic are "broadcast outputs / env bindings." | The writer broadcasts exactly **one** model-id OUTPUT socket: `technical_model` (RETURN index 4, `OTR_LedgerScriptWriter.py:1865-1869`). There is **no `creative_writing_model` output socket.** | C and D are **in-process helper calls inside `run()`**, mirroring `run_story_brief_reflection` (called at `:3737`), **not** new nodes and **not** broadcast outputs. They read the model id from the in-process `resolved[...]` dict: `resolved["creative_writing_model"]` for creative passes, `resolved["technical_model"]` for technical/critic passes. |
| D5 | "WIRE = register into the workflow JSON as broadcast outputs (CONVERGENCE)." | Reflection adds no node surface and no JSON entry. `_otr_constrained_generate.py:20`: "PD3 (workflow JSON): N/A; this module adds no node surface." | For A/B/C/D/E **the workflow JSON (`otr_canonical.json`, 29 nodes) needs NO change** -- none add a node surface or widget. The only file Wave 2 edits is `OTR_LedgerScriptWriter.py`. PD3 is satisfied by "no node surface," not by a JSON edit. (A JSON edit is needed ONLY if we ever add a `creative_writing_model` output socket -- not required by this plan.) |
| D6 | Critic must run "on a different slot than the writer." | Reflection runs on **technical** (`:3740`, `technical_model_id=resolved["technical_model"]`); the writer's creative passes run on `creative_writing_model`. | Route the critic to **`resolved["technical_model"]`** -- it is already the "other slot" whenever the two widgets differ. No new socket, satisfies the not-self-judge rule for free. |
| D7 | Stream D Guard 3: "scene plan MAY be empty ... confirm scene-plan availability." | `OTR_LedgerScriptWriter.py:3711` -- `meta["visual_plan"] = {"characters": {name: {"portrait_prompt": ...}}, "scenes": [], "style": ...}`. `scenes` is **always `[]`** (writer emits no scene blocking, comment `:3695`). No `genre` key. | Guard-3 noun baseline = **the original beat's own prose** (always) **union** the cast `portrait_prompt`s (available in `characters`). The scene plan is empty -- do not depend on it. |
| D8 | "Move the post-script LLM unload (writer ~L3744)." | The unload is at **`:3754`** (`:3744` is the comment header): `meta["writer_llm_unload"] = _otr_writer_vram.unload_writer_llm_after_script()` -- a **full `unload_llm()` teardown** (model->cpu, del, gc, empty_cache, ipc_collect, synchronize; clears `LLM_CACHE`), gated by env `OTR_WRITER_UNLOAD_AFTER_SCRIPT` (default on). It is NOT `_flush_vram_keep_llm`. | Every LLM pass (reflection `:3737`, new critic C, editor D, the repair) MUST be inserted **before `:3754`** or it pays a cold reload. The deterministic scrub E (no LLM) goes **after `:3754`**. "Move the unload after Stage 3.5" = relocate the `:3754` call to sit after C+D+repair, before E. |
| D9 | Implied: regression runs uniformly. | The Bug Bible regression (`comfyui-custom-node-survival-guide/tests/bug_bible_regression.py`, 26 tests) is pure pattern/AST/JSON -- runs **headless in 0.02s, no GPU**. The audio byte-identical test (`tests/test_audio_byte_identical.py`) renders with fixed audio-generator seeds and skips if fixtures are missing. | **Headless gates (any session, venv python):** Bug Bible + core + unit + module self-tests + forbidden-sweep. **GPU-host gates (Jeffrey's machine only):** the live audio byte-identical clean-run render and the scored story-quality A/B vs the 2026-05-31 baseline. Streams A/B change the *writer* (script), not the audio path, so they cannot be value-proven headless -- the scored A/B is the proof, and it is a GPU-host step. |

**Authoritative symbol map (verified live):**

| Symbol | Location |
|---|---|
| `Outline` (final schema) / `Beat` | `nodes/_otr_outline.py:176` / `:84` |
| `generate_outline` (Path C) | `nodes/_otr_outline.py:1453` |
| `_MacroShape` / `_PhaseSkeleton` / `_BeatFleshout` | `nodes/_otr_outline.py:936 / 955 / 976` |
| `structured_call` | `nodes/_otr_structured_call.py:311` |
| `compose_line` / `compose_line_draft` / `LineRequest` | `nodes/_otr_line_composer.py:1735 / 1527 / 553` |
| `_word_bands` (word_cap/min/max) / `_MAX_OVERSIZE_RATIO` | `nodes/_otr_line_composer.py:1348 / 75` |
| `_build_user_prompt` tail ("Speak now.") | `nodes/_otr_line_composer.py:~1234` |
| `run_story_brief_reflection` (def / call site) | `nodes/_otr_story_brief.py:861` / `OTR_LedgerScriptWriter.py:3737` |
| `meta["visual_plan"]` stamp | `OTR_LedgerScriptWriter.py:3711` |
| post-script LLM unload (full teardown) | `OTR_LedgerScriptWriter.py:3754` |
| cast guards | `nodes/_otr_casting.py` (`_assert_no_structural_tokens_in_cast`, `_assert_voice_preset_invariant`, `_assert_unique_bark_voices`) |
| dormant-module convention to mirror | `nodes/_otr_constrained_generate.py:1-27` |
| canonical workflow JSON | `workflows/otr_canonical.json` (29 nodes; writer = node id 1) |

---

# PART I -- THE PLAN (what to build)

## 1. The decision

Adopt the 4-stage spine:

```
Narrative Draft -> Radio Editor -> Creative QA -> [one editor repair] -> Ledger Scrub -> TTS
```

...implemented under the agnostic invariants in Sec 2 that make it safe in the *real* pipeline (a speaker-bound ledger, not free text). The invariants are the actual work; the stage names are just the skeleton.

### The one reconciliation that matters

The clean 4-stage idea treats the script as prose an editor reshapes. In this codebase the script is a ledger of speaker-bound beats (`char_id -> voice_preset -> beat`) produced by per-beat `compose_line`. A free-text editor pass would have to re-parse its output back into that ledger, re-running cast binding, the leak filter, the SFW guard, and the speaker patch -- reopening F3/BUG-295, the F1 attribution collapse, and SFX loss.

**Fix:** every LLM stage operates *on the ledger at beat granularity* and treats the ledger as read-only. Editors and repairs emit an edit list against fixed slots; they never re-attribute lines, merge speakers, or return free prose. This is what lets the editor own length without owning the deferred bugs -- and, deeper, without desyncing the **FLUX -> HuMo -> LTX -> Bark** cascade that renders off the same ledger: a beat's segment count, scene, and visual nouns are render contracts, not just text.

---

## 2. Agnostic invariants (hard rules for every stream)

1. **Enforce in schemas, not prompts.** Arc, length, and cleanliness are enforced at the pydantic/grammar/validation layer via `structured_call`, because that is the only enforcement that behaves identically across models. Prompts *describe*; schemas *enforce*.
2. **Categorical, never numeric.** Critic output is yes/no flags + beat indices + short reasons. No 1-5 axis scores -- absolute numeric calibration is the capability that varies most between models.
3. **The ledger is read-only to every LLM pass -- for render-safety, not just audio.** The ledger feeds the FLUX -> HuMo -> LTX -> Bark cascade; an edit that changes a beat's bindings, scene, segment count, or visual nouns can desync stills from motion and break timing. Every LLM pass returns *edit instructions + replacement prose only*; Python maps them back into the original slots. Allowed actions are the two-tier, three-guard set in Stream D. Never re-attribute, never free-prose-rewrite, never introduce a new visual noun.
4. **Deterministic where possible.** The final scrub is pure Python wiring existing guards -- zero model dependency, maximally agnostic.
5. **One repair, no loops.** A single `REPAIR_ONCE` cycle total across Creative QA and Scrub. Once consumed, downstream failures fail-closed; they do not re-trigger.
6. **One universal prompt rider, no per-model branches.** Drop the per-model-class rider/temperature idea (P2/P3) -- it bakes in assumptions about which model is in the slot.
7. **(NEW, from D4/D5) In-process, not new nodes.** The editor and critic are functions called inside the writer's `run()`, fed the model id from the in-process `resolved[...]` dict, exactly like `run_story_brief_reflection`. They add no widget, no output socket, no workflow-JSON entry. The Prime Directive ("no node but the writer exposes a model_id widget") is satisfied because they are not nodes.

---

## 3. Work streams

Ordered by leverage and dependency. **A + B are free wins and land first.** C, D, E follow. A/B/E are independent of each other; D depends on C and B.

### Stream A -- Outline arc gate (Stage 0 of the draft) -- *highest leverage, smallest*

**Goal:** force a real arc at the validation layer so a weak model cannot draft a flat story. This is where "is it a good story" actually lives, and it is nearly free.

**File:** `nodes/_otr_outline.py`. Touch points: `_MacroShape` (`:936`), the final `Outline` schema (`:176`), and the Python combiner inside `generate_outline` (`:1453`, beats assembled around `:1722`+).

**Schema additions (corrected per D3 -- Path C, not one call):**
```python
# On _MacroShape (Stage 1 -- rides the existing macro structured_call):
central_tension: str                 # the dramatic question, one sentence

# On Outline (final schema -- stamped by the Python combiner, NOT the LLM):
turning_point: TurnRef               # {beat_index: int, what_changes: str}
button: ButtonRef                    # {beat_index: int, payoff: str}
```
**Where each comes from:**
- `central_tension` is a macro property -> add to `_MacroShape`'s schema + macro prompt; it rides the Stage-1 `structured_call` retry/repair ladder already there (`:1550`). This is the only LLM-touching change.
- `turning_point` / `button` are **stamped deterministically in the combiner** once beats exist: `turning_point.beat_index` = first voiced beat of the climactic arc phase; `button.beat_index` = last voiced beat; `what_changes` / `payoff` = derived from that beat's intent or the macro `central_tension`. No new LLM call (invariant intact via a different, real mechanism).

**Validators (on `Outline`, structural -- repaired by the existing ladder, do NOT fail-closed on borderline):**
- `turning_point.beat_index` and `button.beat_index` are valid indices into `beats`.
- `button.beat_index` is at/near the final beat.
- `turning_point.beat_index < button.beat_index`.
Because the combiner stamps these from real beat positions, they pass by construction; the validator is a guard against future regressions, not a live failure path.

**Gate:** an outline whose arc fields are missing/incoherent is repaired (or re-stamped), not composed. Self-test in `_otr_outline.py`'s `__main__`: a typical budget yields a structurally valid arc on the first try.

---

### Stream B -- Length at the source (Stage 1) -- *free rider; CONSERVATIVE on the cap (per D1/D2)*

**Goal:** draft near-length and on-premise so the Radio Editor is a light touch. **The original cap premise was wrong (D1).** Split into a safe part and a careful part.

**File:** `nodes/_otr_line_composer.py` (`_build_user_prompt` tail `~:1234`; `_word_bands` `:1348`).

**B1 -- Universal rider (SAFE, ships first).** One block at the WRITE LINE tail (just above "Speak now.", after the Sprint-3 anti-decorative block `:1225-1233`), applied to every model:
> Ground every line in the provided news facts and this scene's premise; do not invent people, places, or objects the news does not imply. Keep each line spoken-length -- one breath, about 20-30 words, concrete, no nested clauses.
`LineRequest` already carries `theme` (from `meta.news.script_brief`), `allowed_people`, `allowed_things` -- the grounding has material, and the phantom/cast gates enforce entity discipline post-hoc. This block is byte-additive to the prompt; legacy callers with the Sprint-2 optional fields empty still render it (it is unconditional), so capture a fresh story A/B, not a byte-identical-prompt assumption.

**B2 -- Length lever (CAREFUL, gated on regression).** Do **not** change the `*4` token cap (D1). The real per-line word ceiling is `word_cap = target_words * _MAX_OVERSIZE_RATIO` (`_MAX_OVERSIZE_RATIO = 3.0`). Options, least-risky first:
- *Default this sprint:* **leave `_word_bands` unchanged**; let the rider + Stream D own length. The per-line 3x ceiling barely constrains the EPISODE total anyway (829 total = many lines each under their per-line cap), so the editor is the real authority.
- *If a tighter source draft is wanted:* lower `_MAX_OVERSIZE_RATIO` modestly (e.g. 3.0 -> 2.5) **while keeping `min_words = 1`** (never restore a floor -- that is the BUG-279 undershoot trap). Regress the composer unit suite + a local-model length smoke before trusting it.

**Gate:** rider present; composer unit tests green; forbidden-sweep clean; on the GPU host, a long-prone (premium-model) run lands materially closer to 350 than the baseline, byte-identical AUDIO path unaffected (writer-only change), SFW pass. The total-length proof is the Stream D gate; B just stops the bleeding at the source.

---

### Stream C -- Creative QA critic (Stage 3) -- *the one new always-on round*

**Goal:** a read-only taste gate that also absorbs the leak/attribution/SFW checks.

**File:** new `nodes/_otr_creative_qa.py`, mirroring `run_story_brief_reflection` (`_otr_story_brief.py:861`): same post-script call pattern, slot plumbing, fail-loud sentinel. **In-process, not a node (D4).** Receives the critic model id as a plain `str` argument from `resolved["technical_model"]` (D6).

**Schema (`CreativeQAVerdict`, via `structured_call` -- fail-closed):**
```python
has_turn: bool
turn_beat_index: Optional[int]
ending_earned: bool
ending_note: str                 # <= ~15 words
grounded_in_premise: bool
voices_distinct: bool
weakest_beat_index: Optional[int]
weakest_problem: str             # <= ~12 words
overlong_line_indices: list[int]
cast_name_leak_indices: list[int]
sfw_ok: bool
verdict: Literal["PASS", "REPAIR_ONCE", "FAIL"]
```
**Verdict rule (pass-biased -- P4/P5 lesson: brittle gates kill good runs):**
- `PASS` -- clean, or only cosmetic issues.
- `REPAIR_ONCE` -- recoverable: missing turn, unearned ending, weak beat, overlong lines, a leak, or an SFW slip, all fixable by editing named beats.
- `FAIL` -- unrecoverable (e.g. wholly off-premise). Reserve for the genuinely irredeemable.

**Agnostic notes:** all criteria live in the prompt, so the pass assumes only "follows instructions, emits JSON" -- which the ladder guarantees. Critic slot = `technical_model` (the "other slot" per D6).

**Gate:** verdict deterministic from the flags; the pass never rewrites by itself; one `REPAIR_ONCE` max; `__main__` self-test prints `SELF-TEST PASS: N/N`. **Commit dormant** (authored, not yet wired -- mirror `_otr_constrained_generate.py:16-27`).

---

### Stream D -- Radio Editor + the single repair (Stage 2 + 3.5) -- *largest piece*

**Goal:** own length and radio pacing as **ledger-native per-beat edits** that are also **render-safe**. The editor proposes edit instructions + replacement prose only; **Python maps prose back into the original ledger slots.**

**File:** new `nodes/_otr_radio_editor.py`. **In-process, not a node (D4).** The LLM never writes ledger keys; it returns an edit list applied deterministically. Recompose reuses `compose_line` (`_otr_line_composer.py:1735`) on the creative slot (`resolved["creative_writing_model"]`).

**Action set -- two tiers by render impact:**

```text
Tier 1  (segment-set STABLE -- only one line's text/audio changes):
  KEEP
  SHORTEN_LINE
  CLEAN_PUNCTUATION
  RECOMPOSE_BEAT_SAME_INTENT     # the one rewrite action; tightly fenced (below)

Tier 2  (segment-set CHANGING -- adds/removes a dialogue unit -> forces re-alignment):
  CUT_LINE
  REMOVE_REDUNDANT_BEAT
  SPLIT_LINE
  MERGE_SHORT_LINES
```

This split is the load-bearing render-safety distinction. **Tier 1** leaves the beat/slot/scene set byte-identical; downstream re-renders only that one line's audio and re-times that single segment. **Tier 2** changes the *count* of dialogue units, so Python must (a) re-index the affected beats with clean IDs and (b) **flag the affected span for cascade re-alignment** (`needs_render_realign`). Tier 2 operates on the *list of beats*; it may never reassign a surviving beat's bindings.

**Schema (`RadioEditPlan`, via `structured_call`):**
```python
class BeatEdit(BaseModel):
    beat_index: int
    action: Literal[
        "KEEP", "SHORTEN_LINE", "CLEAN_PUNCTUATION", "RECOMPOSE_BEAT_SAME_INTENT",
        "CUT_LINE", "REMOVE_REDUNDANT_BEAT", "SPLIT_LINE", "MERGE_SHORT_LINES",
    ]
    new_line: Optional[str] = None          # replacement prose for SHORTEN/RECOMPOSE/SPLIT/MERGE
    merge_with_index: Optional[int] = None   # MERGE_SHORT_LINES only (adjacent same-speaker beat)

class RadioEditPlan(BaseModel):
    edits: list[BeatEdit]
    projected_word_total: int
```

**Guard 1 -- Structural Lock (deterministic Python).** Reject any plan that would alter `beat_id`, `speaker_id`, `dialogue_slot_id`, `character_id`, `scene_id`, `voice_preset`, `visual_plan`, or ledger structure other than the explicit Tier-2 add/remove-with-reindex. No line is ever reassigned to a different speaker.

**Guard 2 -- Text Length (deterministic Python).** Every edited/recomposed line obeys a hard ceiling: `max_words_per_line ~= 35` (one breath) **and** a fixed `max_chars_per_line` cap. `SPLIT_LINE` is the remedy when a line exceeds the cap. Protects Bark timing, HuMo mouth motion, LTX temporal consistency.

**Guard 3 -- Visual Noun (deterministic Python, synonym-tolerant) -- per D7.** Before accepting a `RECOMPOSE_BEAT_SAME_INTENT`, diff the concrete-noun set of `new_line` against the allowed set = **(the original beat's prose) union (the cast `portrait_prompt`s in `meta.visual_plan.characters`)**. The `scenes` list is **always empty** in the current writer, so do not rely on it. Reject any **new** physical prop, location, weather, costume, machine, creature, or scenery; allow morphological/synonym variants. On a new noun, **REPAIR (re-run the recompose with the offending noun named), do not hard-FAIL** (P4/P5).

**RECOMPOSE_BEAT_SAME_INTENT -- enforceable vs. semantic locks.**
- *Structural locks* (same beat/speaker/slot/cast/scene/voice/visual_plan) and *no-new-visual-noun* are **deterministically enforced** by Guards 1 + 3.
- *Semantic locks* (same story function, same emotional turn, same payoff direction, no new plot facts) **cannot be regex-checked.** Enforced by the recompose **prompt** + the **Creative QA re-verification** (the critic re-confirms the turn still lands after a recompose; same single repair cycle, nothing extra consumed).

**`post_validator` (deterministic gate over a plan):** every `beat_index` valid and unique; Guard 1 + 2 + 3 pass; Tier-2 removals never drop the `turning_point` or `button` beat (Stream A arc fields); `projected_word_total` within **350 +/- 20%**, else the ladder repairs once toward range.

**Apply step (pure Python):** Tier-1 edits replace line text in place (segment set untouched). Tier-2 edits add/remove dialogue units, re-index cleanly, and stamp the affected span as `needs_render_realign`. SFX/music beats and all bindings are preserved by construction -- the editor never sees or writes them.

**Repair loop (Stage 3.5):** when Creative QA returns `REPAIR_ONCE`, scope the editor to the flagged beats only (`weakest_beat_index`, `overlong_line_indices`, `cast_name_leak_indices`). Tier 1/2 for compression/cleanup; `RECOMPOSE_BEAT_SAME_INTENT` for a genuinely flat beat, under the fence. One cycle, then stop.

**Default slot:** `editor_model = resolved["creative_writing_model"]` unless separately bound. On a near-length draft (Stream B did its job) most beats come back `KEEP`; recompose is the rare emergency action.

**Gate:** force an 800-word draft and assert the editor lands within 350 +/- 20% with **every ledger key byte-identical except explicit Tier-2 reindex**, turn/button beats intact, no new visual noun on any recompose, every line under the word+char caps, `needs_render_realign` stamped wherever the segment count changed; leak/SFW/attribution clear; one repair max. Self-test prints `SELF-TEST PASS: N/N`. **Commit dormant.**

---

### Stream E -- Ledger Scrub (Stage 4) -- *deterministic, no LLM*

**Goal:** one mechanical gate wiring the guards you already trust. No model = perfectly agnostic. **Runs AFTER the LLM unload (D8)** -- it needs no resident model.

**File:** new `nodes/_otr_ledger_scrub.py` aggregating existing guards; no new judgment logic.

**Wire in (call existing code, do not reimplement):**
- JSON/schema validity (ledger round-trips);
- the BUG-279/F3 cast-name leak filter (`_otr_line_composer.py`);
- the SFW/profanity validator (add `damn(ed)` per P8 while here);
- `_assert_no_structural_tokens_in_cast`, `_assert_voice_preset_invariant`, `_assert_unique_bark_voices` (`_otr_casting.py`);
- no empty/duplicate dialogue lines; whitespace/quote/punctuation normalization; TTS-safe text; no stage directions inside spoken fields;
- word count and per-line length within bounds.

**Scope rule:** the scrub **normalizes** (whitespace, punctuation, JSON-safety) freely. It does **not** improve the story, rewrite dull lines, or change the ending. If a *mechanical* defect needs regeneration rather than normalization (empty line, leaked cast name mid-phrase), it raises `NEEDS_EDITOR_REPAIR`, which consumes the single repair **only if not already used** -- otherwise fail-closed.

**Gate:** forbidden-sweep clean; force a leaked cast name and assert scrub either normalizes it or triggers the one repair; the SFW `damn(ed)` add regresses standalone. Self-test prints `SELF-TEST PASS: N/N`. **Commit dormant** (the SFW wordlist add can ship live + regressed on its own).

---

### Housekeeping (folds into C/D)

- **Move the post-script LLM unload (D8).** It fires at `OTR_LedgerScriptWriter.py:3754` (full `unload_llm()` teardown), right after `run_story_brief_reflection` (`:3737`). Move it to **after Stage 3.5** so the critic and the repair both have a model resident. The scrub (E) stays after the unload.
- Keep `story_brief_reflection` or let Creative QA supersede it; do not run both on the same slot back-to-back.

---

## 4. Model surface (in-process resolved ids, NOT new widgets/outputs -- corrected per D4/D6)

Per the Prime Directive and D4: editor/critic are **in-process helper calls** fed by the writer's `resolved[...]` dict, never new per-node widgets or output sockets. Two physical widgets already exist; logical roles map onto them in-process:

```
resolved["creative_writing_model"]   # draft, recompose, quality rewrite, editor (Stream D)
resolved["technical_model"]          # structured JSON; critic (Stream C, the "other slot" per D6); reflection
# scrub (Stream E) is pure Python -- no model.
```

There is no `editor_model` / `critic_model` widget and none is added. The only broadcast OUTPUT socket on the writer is `technical_model` (unchanged); consumers like `OTR_LedgerFreezeCascade` keep reading it. **No workflow-JSON change for this plan (D5).**

---

# PART II -- EXECUTION (how to ship it)

## 5. The execution loop (one stream)

1. **REVIEW.** (DONE for all streams -- see Part 0.) Re-confirm the contract against the cited file:line before editing; files may drift further. If anything contradicts Part 0, report back, do not code blind.
2. **CODE.** Write to the corrected contract. Pure module, no GPU/network at import. Add a `__main__` self-test (`SELF-TEST PASS: N/N`, mirror the writer). New modules land **dormant** until wired (mirror `_otr_constrained_generate.py:1-27`). UTF-8, no BOM. Never the word "dummy" -> use "placeholder"/"test".
3. **WIRE.** **In-process call inside `run()`, no new node/widget/output, no workflow-JSON edit (D4/D5).** Mirror `run_story_brief_reflection`'s call-site for C; add D's Stage-2 + REPAIR_ONCE loop; relocate the `:3754` unload to after Stage 3.5 (D8). Tag every LLM call `# LLM slot: creative|technical` with a one-line reason; pass the id from `resolved[...]`.
4. **REGRESS.** Headless (venv python): Bug Bible + core + the touched unit suite + forbidden-sweep + the module self-test. GPU-host (Jeffrey): audio byte-identical clean run + scored A/B vs the 2026-05-31 baseline. No ship on a content opinion alone.
5. **COMMIT.** One atomic commit per stream. git via Desktop Commander **cmd** (no-spaces paths need no quotes) + `.git\COMMIT_EDITMSG` `-F`; one push attempt; **verify `local HEAD == origin HEAD`**. Red headless regression -> do not commit (Sec 8).

## 6. Orchestration: parallel authoring, serial wiring

Subagents help only when they touch **disjoint files**. Map ownership before dispatching.

| Stream | Owns (edit/create) | Reads/calls (no edit) | Collision class |
|--------|--------------------|-----------------------|-----------------|
| A -- outline arc | `_otr_outline.py` (`_MacroShape` + `Outline` + combiner + validators) | -- | disjoint |
| B -- rider (+ optional cap) | `_otr_line_composer.py` **internals only** (`_build_user_prompt` string; optional `_MAX_OVERSIZE_RATIO`; no signature change) | -- | disjoint at signature |
| C -- creative QA | **new** `_otr_creative_qa.py` | `_otr_story_brief.py` (mirror call), `_otr_structured_call.structured_call` | disjoint (new file) |
| D -- radio editor + repair | **new** `_otr_radio_editor.py` | `_otr_line_composer.compose_line` (recompose), `structured_call` | disjoint (new file) |
| E -- ledger scrub | **new** `_otr_ledger_scrub.py` + SFW wordlist (`damn(ed)`) | leak filter + cast guards (`_otr_line_composer`, `_otr_casting`) | disjoint (new file + 1-line add) |
| **WIRE (C, D, unload-move into run())** | `OTR_LedgerScriptWriter.py` | (no JSON) | **CONVERGENCE -- serialize** |

**Consequence:** authoring parallelizes across subagents; the wiring that lands C, D, and the unload-move into the writer is **one thread, one at a time**. Parallel edits to the writer = conflicts + a broken byte-identity gate a subagent can't validate in isolation.

## 7. Wave plan

### Step 0 -- land B1 first (main thread, ~minutes)
B1 (the rider) is a tiny internal edit to `_build_user_prompt`, no signature change. Land + headless-regress + commit first so every downstream author builds against the final composer. B2 (cap) is deferred/optional per D2.

### Wave 1 -- parallel authoring (4 subagents, disjoint files)
Dispatch simultaneously; each runs `REVIEW -> CODE -> self-test -> report diff`. **No WIRE.**
- **Subagent A:** `_otr_outline.py` arc fields (`central_tension` on `_MacroShape`; `turning_point`/`button` stamped in the combiner) + `Outline` index validators + self-test. Self-contained -> headless-regress + commit on report.
- **Subagent C:** new `_otr_creative_qa.py` + `CreativeQAVerdict` (observational, categorical) + self-test. Commit **dormant**.
- **Subagent D:** new `_otr_radio_editor.py` + `RadioEditPlan` (two-tier, ledger read-only, `RECOMPOSE` fenced) + Guards 1/2/3 + deterministic apply (Tier-2 stamps `needs_render_realign`) + self-test. Commit **dormant**.
- **Subagent E:** new `_otr_ledger_scrub.py` aggregating existing guards + the `damn(ed)` SFW add + self-test. SFW add regresses standalone; scrub module commits **dormant**.

### Wave 2 -- serial wiring into the writer (main thread, REGRESS + COMMIT each)
Order = dependency + risk (low-risk independents first; D last, needs C + B). **All in-process; no workflow-JSON edit (D5).**
1. **A** integration check (fields ride the Path-C combiner) -> headless-regress -> commit.
2. **E** as the deterministic final gate, inserted **after** the `:3754` unload, before TTS -> headless-regress -> commit.
3. **C** post-script call-site (mirror `run_story_brief_reflection`, `resolved["technical_model"]`) **+ relocate the `:3754` unload after it** -> headless-regress -> commit.
4. **D** as Stage 2 + the single REPAIR_ONCE loop between critic <-> editor (`resolved["creative_writing_model"]`); **unload now sits after Stage 3.5** -> headless-regress -> commit.
5. **GPU-host gate (Jeffrey):** one clean episode render -> audio byte-identical + scored A/B vs baseline before any of A-D is called "done."

## 8. Where regression runs (corrected per D9)

- **Headless, any session (venv python `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`):** the module's `__main__` self-test; Bug Bible regression (`comfyui-custom-node-survival-guide/tests/bug_bible_regression.py`, no GPU); core + touched unit suites; the forbidden-sweep helper (`tests/_s28_forbidden_sweep.py`). Subagents validate units only.
- **GPU host, Jeffrey's machine only:** the live audio byte-identical clean-run render (`tests/test_audio_byte_identical.py` with `OTR_CAST_SEED`/`OTR_STYLE_SEED` + fixed generator seeds) and the scored story-quality A/B vs the 2026-05-31 baseline (score from the cascade ledger, 7 axes x 0-5). A/B is the only proof of B's length win and any quality gain; it cannot run headless.
- Red headless regression -> **do not commit.** Keep subagent output uncommitted; fix on main or re-dispatch with the failure attached.

## 9. Subagent brief template (so each runs without coordination)

> **Context:** this plan -- Part 0 (full) + Part I (full) + your stream section (Sec 3) + the Agnostic Invariants (Sec 2) + Standing Constraints (Sec 11).
> **Task:** author `<file>` to the corrected contract + a `__main__` self-test. **Author only -- do NOT wire into the writer or workflow JSON. Do NOT edit any file outside `<owned file>`.**
> **Gate:** your self-test passes (`SELF-TEST PASS: N/N`); no GPU/network at import; UTF-8 no BOM; never the word "dummy" (use "placeholder"/"test").
> **Report:** the diff + self-test output. Do not commit.

Keeping each subagent to one owned file is what prevents the convergence collision in Sec 6.

---

# PART III -- GATES

## 10. Acceptance / exit criteria (a reviewer can grade)

On the real tree, with all five streams committed:
- Bug Bible + core green (headless); **audio byte-identical** on a clean run (GPU host).
- Arc present by construction: every shipped outline carries a turn and a payoff.
- Actual word count **350 +/- 20%**; no absurdly long spoken line.
- Zero SFW slips, zero speaker-attribution errors, zero cast-name leaks in dialogue.
- Zero fail-closed aborts on a *valid* news story.
- Zero render/ledger crashes.
- **Render contract intact:** no edit changes `scene_id`/`voice_preset`/bindings; every Tier-2 segment-count change stamps `needs_render_realign`; no recompose introduces a new visual noun; every line within the word + char caps.
- **No more than one repair cycle**, ever.
- No model-specific branch anywhere in the spine.

## 11. Standing constraints (every stream, every subagent)

Audio byte-identical to the blessed baseline (GPU-host gate); 14.5 GB VRAM ceiling (insert LLM passes BEFORE the `:3754` unload, never co-resident with the cascade); **no node surface / no workflow-JSON change** for in-process passes (D4/D5); SFW + non-violent; never the word "dummy" (use "placeholder"/"test"); every LLM call tagged creative/technical and fed from `resolved[...]`; all new structured-JSON passes go through `structured_call` (fail-closed, no new parser/retry); run the Bug Bible regression after every code change; git via Desktop Commander **cmd**, one push attempt, verify HEAD.

## 12. Suggested order / ship sequence

1. **A + B1** together -- free; near-length drafts with a real arc. Measure against the 2026-05-31 baseline (GPU host) before anything else.
2. **C** -- the taste gate. Cheapest new round; tells you whether A+B alone already clear the rubric.
3. **D** -- editor + the single repair. Largest surface; lands once C produces trustworthy notes.
4. **E** -- scrub. Independent; can land any time.

If **A + B1 alone** clear the rubric on the baseline, ship that as a release and add C/D as a second wave -- the leanest path to "good story first."
