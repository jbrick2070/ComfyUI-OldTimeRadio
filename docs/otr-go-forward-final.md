# OTR Story Spine — Final Go-Forward Plan

**Target:** an agentic coding session (repo, Desktop Commander, git, subagents) executes the sprints below.
**Branch base:** `v2.0-alpha`. **Priority:** good story first, clean ledger second, render after that.
**Loop, per sprint:** `REVIEW -> CODE -> WIRE -> REGRESS -> COMMIT`, until all sprints are green.

This supersedes the earlier plan + critic-revisions docs.

---

## 0. Verdict on the criticism (what was taken, what was refined)

**Accepted** — the criticism is right and corrects my earlier "cut the critic" stance:
- **Keep the Radio Editor.** It is a length/pacing normalizer, not the grader I was skeptical of. Remove it and the deterministic scrub becomes the only length control — and the scrub can flag a long line but cannot edit one.
- **Critic = narrow defect router, not a grader.** It only answers "is there a severe defect that makes this episode bad?" No vanity scoring.
- **QA always runs — fully model-agnostic (revised per your call; I should not have accepted the skip).** I'd taken the criticism's `critic_model != creative_writing_model` auto-skip; it's a footgun (the default single-model setup would silently get no QA) and requiring two models is itself a non-agnostic dependency. QA now runs on any config. The self-grading weakness is handled by **task design** (Sprint 1), not by disabling the gate. A different technical model gives stronger recall and is **recommended, not required**.
- **Micro-repair only, on flagged beats, fenced.** No whole-ledger rewrite.
- **Scrub stays mechanical, fail-closed, last.**

**Refined** (the part of the skepticism that survives):
- The editor runs on **objective triggers** (over word-band, or any line over the spoken cap), not unconditionally — don't pay for it on an already-clean draft.
- The router must split **locally-fixable** defects (-> `MICRO_REPAIR_NEEDED`) from **structural** ones like a dead ending (-> `REJECT`). Never request a micro-repair for a defect a one-line edit can't fix.
- **REJECT aborts at the writer boundary**, not inside the spine (preserves the spine's never-raises / PD1 contract).
- Once a second model is bound, **measure** the micro-repair / reject rate to confirm QA earns its per-run call.

The key insight that changed my mind: keeping the editor introduces a pass that can **chop dialogue**, which creates a real need for a QA check on the editor's output — protection A+B+scrub cannot give.

---

## 1. The pipeline

```text
Writer            A: arc-gated outline (turning_point + button); B: length-capped compose
  |
Radio Editor      runs only if draft is out of spec (over band OR any overlong line)
  |               subtractive + pacing; beat-local; binding-preserving
Targeted Story QA always runs (model-agnostic)
  |               defect router -> PASS / MICRO_REPAIR_NEEDED / REJECT
  |                 REJECT -> abort the run at the writer boundary
Micro-repair      only on MICRO_REPAIR_NEEDED; flagged beats only; fenced actions; ONE cycle
  |
Ledger Scrub      deterministic, mechanical, fail-closed, LAST
  |
Render            FLUX -> HuMo -> LTX -> Bark
```

The editor makes the story better; the critic only decides whether a tiny repair is needed or the episode is unshippable; the scrub only protects the ledger.

---

## 2. Hard rules (every sprint)

1. **QA always runs — never auto-skips on model config (model-agnostic).** A judge on the writer's own model is a weaker second opinion — lower recall on the writer's own blind spots — but the fix is task design, not skipping. Same-model QA degrades gracefully because it judges **concrete defects, not quality** (a model spots a *missing* turn or a chopped line in its own output far better than it can be objective about overall quality), runs on a **cold context** with a **skeptical bar** (Sprint 1). A different technical model is recommended for stronger recall but never required.
2. **The ledger is read-only to every LLM pass** — for render-safety, not just audio. It feeds FLUX -> HuMo -> LTX -> Bark; a changed binding, scene, segment count, or visual noun desyncs stills from motion. LLM passes emit edit instructions + replacement prose only; Python maps them into fixed slots.
3. **Beat-local, binding-preserving.** No pass may change `beat_id`, `speaker_id`, `dialogue_slot_id`, `scene_id`, `voice_preset`, `visual_plan`, or reassign a line to a different speaker. Forbidden actions are rejected at the schema: `RECOMPOSE_WHOLE_LEDGER`, `REWRITE_SCENE`, `CHANGE_PLOT`, `ADD_OBJECTS`, `ADD_LOCATION`, `CHANGE_SPEAKER`.
4. **At most one bounded micro-repair cycle.** No loops, no ping-pong.
5. **Scrub is deterministic and last**, fail-closed; it never rewrites the story.
6. **Enforce in schemas, not prompts**; categorical (yes/no), via `structured_call`, fail-closed. No per-model prompt branches.

---

## 3. Current state vs the delta

- **A (outline arc fields):** implemented — outline exposes `turning_point` / `button`. Verify only.
- **B (rider + per-line cap):** verify the universal one-breath rider and the per-line cap (`min(cap, target_words*2)`) are in `compose_line`; add if missing.
- **E (scrub):** implemented, deterministic, fail-closed. Verify it never rewrites story.
- **Critic:** implemented as a `PASS/REPAIR_ONCE/FAIL` grader with leak/SFW fields -> **convert to the defect router** (Sprint 1).
- **Editor:** implemented as repair-only on a broad ledger -> **add the conditional length pass and narrow to fenced micro-repair** (Sprint 2).
- **REJECT abort:** not implemented -> add (Sprint 3).

---

## 4. Sprints

### Sprint 0 — Verify A / B / E
Confirm the arc fields (A), the compose rider + x2 cap (B; add if absent), and the scrub (E) are present and behave. No new design; this is the baseline gate. **Gate:** A/B/E confirmed on the real tree; if B was missing, byte-identical audio still green after adding it.

### Sprint 1 — Targeted Story QA router (`_otr_creative_qa.py`)
**Goal:** convert the grader into a severe-defect router.
**Verdict schema (`StoryQAVerdict`, via `structured_call`):**
```python
verdict: Literal["PASS", "MICRO_REPAIR_NEEDED", "REJECT"]
flagged_beats: list[int]      # only for MICRO_REPAIR_NEEDED
reason: str                   # <= ~15 words
# evidence flags (reasons behind the verdict, not a score):
dead_ending: bool; broken_turn: bool; flat_contrast: bool
unclear_grounding: bool; chopped_dialogue: bool; pacing_failure: bool
```
**Router logic:** locally-fixable defect (chopped dialogue, a flabby/overlong beat) -> `MICRO_REPAIR_NEEDED` + `flagged_beats`. Structural defect a one-line edit can't fix (dead ending, broken logical turn, unclear premise grounding) -> `REJECT`. Clean -> `PASS`.
**Drop** every check the scrub already owns (JSON, speaker IDs, SFW/profanity, cast-name leaks, empty lines, schema) — those stay deterministic.
**Always runs (model-agnostic):** no model-class skip. To keep same-model QA honest, the call (a) sees only the final script on a **cold context** (no generation history to rationalize from), (b) uses a **skeptical / adversarial** framing ("assume the writer was careless; find why it fails"), and (c) keeps a **high REJECT bar** (only severe, concrete defects). A model is far better at spotting a *concrete* missing turn or chopped line in its own output than at judging overall quality — which is exactly why the router flags defects, not quality. A different technical model gives stronger recall and is recommended, not required.
**Gate:** emits only the three verdicts; never rewrites; runs on any model config; flagged beats are valid indices.

### Sprint 2 — Radio Editor: conditional length pass + fenced micro-repair (`_otr_radio_editor.py`)
**Goal:** one module, two entries, all edits applied deterministically by Python from an LLM-proposed plan.
- **`normalize_length(led)`** — runs only when total > 350 +/-20% OR any line > spoken cap. Two tiers by render impact:
  - Tier 1 (segment-set STABLE): `KEEP`, `SHORTEN_LINE`, `CLEAN_PUNCTUATION`.
  - Tier 2 (segment-set CHANGING -> re-index + stamp `needs_render_realign`): `CUT_LINE`, `REMOVE_REDUNDANT_BEAT`, `SPLIT_LINE`, `MERGE_SHORT_LINES`.
  - Lands total in band and every line under the spoken cap.
- **`micro_repair(led, flagged_beats)`** — beat-local, only the flagged beats, ONE cycle. Actions: `SHORTEN_LINE`, `SPLIT_LINE`, `CLEAN_PUNCTUATION`, `SMOOTH_DIALOGUE`, `CLARIFY_TURN`.
- **Three deterministic guards on every emitted `new_line`:** Structural Lock (rule 3); Text Length (<= ~35 words / one breath + char cap); Visual Noun (no new physical prop/location/scenery, synonym-tolerant — load-bearing for `SMOOTH_DIALOGUE` / `CLARIFY_TURN` / `RECOMPOSE`, which emit fresh prose).
- **`RECOMPOSE_BEAT_SAME_INTENT`:** emergency-only, schema-fenced (preserve beat/speaker/slot/scene/voice/visual_plan/cast/story-function; guards apply). Not in the normal path.
- **Forbidden actions** rejected at schema (rule 3 list).
**Gate:** an 800-word draft lands in band, every ledger key byte-identical except explicit Tier-2 reindex, turn/button beats intact, no new visual noun, `needs_render_realign` stamped where the segment count changed; micro-repair touches only flagged beats; one cycle.

### Sprint 3 — REJECT abort at the writer boundary (`_otr_story_spine.py` + `OTR_LedgerScriptWriter.py`)
**Spine:** on `REJECT`, unload the writer LLM, set `meta["story_verdict"]="REJECT"` + `meta["story_reject_reason"]`, **skip the scrub, return normally — never raise** (preserve PD1; a critic *crash* stays fail-soft too).
**Writer:** immediately after the spine call (~L3759, before the `meta["creative_model"]` stamping and the `return` at ~L3858):
```python
if meta.get("story_verdict") == "REJECT":
    raise RuntimeError(f"OTR reject gate: {meta.get('story_reject_reason') or 'story rejected'}")
```
Matches the writer's existing fail-loud pattern (~L2355/2366); the only new raise. Aborted run produces no node output -> graph stops before render.
**Gate:** REJECT aborts cleanly (VRAM unloaded, reason logged, no render); PASS proceeds; a QA crash does not break the run.

### Sprint 4 — Scrub verify + repair handoff (`_otr_ledger_scrub.py`)
**Goal:** confirm the scrub is mechanical-only, fail-closed, and runs LAST (after any micro-repair). Call it with `repair_available=False` (micro-repair already ran upstream; the scrub does not trigger repair). An un-normalizable mechanical defect aborts the run (fail-closed). It never rewrites the story.
**Gate:** forbidden-sweep clean; a forced leaked cast name is normalized or aborts; byte-identical audio on a clean run.

---

## 5. The loop (one sprint)

1. **REVIEW** — read the sprint section + target files; confirm the contract still matches the real tree; re-check rule list (§2) and standing constraints (§9). Contradiction -> report, do not code blind.
2. **CODE** — write to the contract; pure module, no GPU/network at import; add a `__main__` self-test (`SELF-TEST PASS: N/N`); land new logic dormant if not yet wired.
3. **WIRE** — into `_otr_story_spine.py` (and the writer for Sprint 3); workflow JSON unchanged (broadcast outputs / env bindings, not widgets); tag LLM calls creative/technical.
4. **REGRESS** — Bug Bible + core + audio byte-identical + the sprint A/B vs the 2026-05-31 baseline. No ship on a content opinion alone.
5. **COMMIT** — one atomic commit per sprint; git via Desktop Commander cmd + `.git\COMMIT_EDITMSG -F`; one push attempt; verify HEAD. Red regression -> do not commit.

---

## 6. Orchestration: parallel authoring, serial wiring

Subagents help only on **disjoint files**.

| Sprint | Owns (edit/create) | Reads/calls | Collision |
|--------|--------------------|-------------|-----------|
| 1 — QA router | `_otr_creative_qa.py` | `structured_call` | disjoint (new logic, own file) |
| 2 — editor | `_otr_radio_editor.py` | `compose_line`, `structured_call` | disjoint (own file) |
| 4 — scrub | `_otr_ledger_scrub.py` | leak/cast guards | disjoint (own file) |
| WIRE (1,2,4) + Sprint 3 | `_otr_story_spine.py`, `OTR_LedgerScriptWriter.py` | workflow JSON | **CONVERGENCE — serialize** |

Author the three modules in parallel; the spine + writer wiring is one thread, one sprint at a time. Parallel edits to the spine = conflicts + a byte-identity gate a subagent can't validate alone.

---

## 7. Wave plan

- **Sprint 0** first (verify A/B/E; add B if missing) -> regress -> commit.
- **Wave 1 — parallel authoring (3 subagents, dormant + self-test):** Sprint 1 (`_otr_creative_qa.py`), Sprint 2 (`_otr_radio_editor.py`), Sprint 4 (`_otr_ledger_scrub.py` verify/adjust). No spine wiring.
- **Wave 2 — serial wiring into the spine + writer (regress + commit each), in flow order:**
  1. Editor length pass (conditional) into the spine.
  2. QA router (always runs, cold-context + skeptical bar) into the spine.
  3. Micro-repair (on `MICRO_REPAIR_NEEDED`, flagged beats) into the spine.
  4. REJECT abort (spine signal + writer raise) — Sprint 3.
  5. Scrub call hardened to `repair_available=False`, confirmed last — Sprint 4 wiring.
- **Measure (from day one — QA runs on the single model too):** micro-repair rate, reject rate, dud rate. Use these to confirm QA earns its per-run call and to tune the reject bar. If you later bind a different technical model, expect higher recall on the writer's blind spots.

---

## 8. Acceptance

- Story protection retained: QA catches severe defects when a second model is bound; the editor fixes length/pacing; micro-repair is beat-local and one cycle.
- QA emits only `PASS / MICRO_REPAIR_NEEDED / REJECT`; no vanity scoring; **runs on any model configuration (model-agnostic), never auto-skips**. Same-model QA uses cold context + skeptical framing + a high reject bar.
- No whole-ledger / scene / object / speaker rewrites. Render contract intact: bindings/`scene_id`/`voice_preset` unchanged; Tier-2 count changes stamp `needs_render_realign`; no new visual noun; every line within word + char caps.
- REJECT aborts cleanly at the writer boundary (VRAM unloaded, reason logged, no render); a QA crash does not break the run.
- Scrub mechanical, fail-closed, last; never rewrites story.
- <= 1 micro-repair cycle; no loops; no model-specific branches.
- Bug Bible + core green; audio byte-identical on a clean (non-reject) run.

---

## 9. Standing constraints (every sprint, every subagent)

Audio byte-identical to the blessed baseline; 14.5 GB VRAM ceiling; wire changes into the workflow JSON as broadcast outputs (not new widgets); SFW + non-violent; never the word "dummy" (use "placeholder"/"test"); every LLM call tagged creative/technical and routed via the writer's two model widgets; all structured-JSON passes through `structured_call` (fail-closed, no new parser/retry logic); run the Bug Bible regression after every code change; git via Desktop Commander, one push attempt, verify HEAD.
