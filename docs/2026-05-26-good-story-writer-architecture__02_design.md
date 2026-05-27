# Good Story Writer Architecture — Design (revised, with wiring)

**Date:** 2026-05-26
**Pairs with:** `__00_question.md`, `__01_problem-statement.md`
**Replaces:** prior `__02_design.md`
**Status:** Design with real wiring + cheap pre-Wave-1 experiment (Wave 0) added 2026-05-26 evening. Sprint 10B implementation can dispatch to parallel subagents per the decomposition in section 4.

---

## 0. TL;DR

The writers' room (Director / Writer / Editor on one resident LLM) is the right pick. Before building all of it, run one cheap experiment to test the chat-style hypothesis on the surface where it costs the least to validate.

- **Sprint 10B Step 1 / Wave 0 -- wire what we already built.** `nodes/_otr_stage2_prompt.py` + `nodes/_otr_stage2_call.py` already implement the multi-turn roleplay dialogue composer with 30 green tests. The production writer at `OTR_LedgerScriptWriter.py:2872` and `:2938` does not call it; it calls the legacy one-shot `_OTRLC.compose_line`. One commit adds `use_multiturn_dialogue: bool = False` to the writer + dispatch at both sites. A/B-able from a workflow toggle. Detailed in Section 4, Wave 0.
- **Sprint 10B Wave 1 (5 parallel agents A-E)** -- news_interpreter semantic check, Stage 3 validators in legacy compose_line, critic-driven reroll escalation, Director agent, Editor agent. Detailed in Section 4.
- **Sprint 10B Wave 2 (sequential F + G)** -- Story Room loop, transcript-to-structured extraction. Detailed in Section 4.
- **Sprint 10B Wave 3 (operator)** -- full-budget listen-test gate. Detailed in Section 4.

**Worked example** of the writers' room conversation for "Spray of Hope" is in Section 2.5 -- the load-bearing concrete proof that the room catches the premise-clarity failure on draft 1 and fixes it on draft 2. **Acceptance criteria + A/B demo plan** (operator worksheet covering both the Wave 0 cheap experiment and the Wave 3 final gate) is in Section 6. **Explicit answers to the seven open design questions** from `__01_problem-statement.md` Section 6 are in Section 10.

Wave 0 is one focused commit. Wave 0 outcome reshapes Wave 2 scope but does not block Wave 1 -- Wave 1's five agents own orthogonal surfaces that improve the pipeline whatever Wave 0 finds.

---

## 1. Architecture (recap)

Writers' room. Three roles on the same small LLM, different system prompts per turn. Bounded free-form conversation. Schema only at extraction time.

- **Director** — reads news + cast + rubric; emits a director's brief (prose, not schema).
- **Writer** — drafts the episode against the brief; takes multiple turns; chooses its own working shape (plan-then-write OR draft-then-revise).
- **Editor** — scores against the rubric; passes or names what's broken per axis.

Loop: Director once → Writer 1-4 turns → Editor → if fail, Writer 1-3 more turns → Editor again. Hard cap: 3 Editor cycles, 18 total turns. Extraction (one constrained-decode pass against the final transcript) produces the structured artifacts the downstream pipeline already consumes.

**What this rejects:** per-turn JSON schemas (ChatGPT's 8-turn loop), open-ended chat, per-character actor agents (deferred), bigger model, human in the loop.

**What this preserves:** PD1 byte-identity in legacy mode, the entire downstream audio/video pipeline, Sprint 10A's diagnostic surface (Stage 1 plan + cast audit + Stage 3 validators + Stage 7 critic become components, not replacements).

---

## 2. Frozen contracts (modules no Sprint 10B agent may modify)

These ship today. New code consumes them; no agent touches them.

| Module | Public API surface | Used by Sprint 10B agent |
|---|---|---|
| `_otr_stage1_plan` | `Stage1Plan`, `Stage1Beat`, `Stage1CastMember`, `Stage1Arc` (pydantic) | D, E, F, G |
| `_otr_constrained_generate` | `make_constrained_generate_fn(cache_entry, schema_model) -> ConstrainedGenerateFn`. Closure signature: `(messages, *, temperature, max_new_tokens) -> str` | D, E, F, G |
| `_otr_model_loader` | `make_generate_fn(cache_entry)`, `make_polish_generate_fn(cache_entry)`, `ModelLoaderError` | F (Writer turns) |
| `_otr_critic_rubric` | `load_rubric() -> Rubric` with `axes: List[RubricAxis]`, `threshold: ShipThreshold` | D, E, C |
| `_otr_stage3_validators` | `validate_line(plan, beat, line_text, *, banned_phrases, min_on_beat_ratio) -> ValidationResult`. Codes: `length_drift`, `pronoun_mismatch`, `speaker_leak`, `banned_phrase`, `continuity_break`, `off_beat` | B |
| `_otr_whole_episode_critic` | `run_whole_episode_critic(*, plan, rendered_lines, generate_fn, rubric, max_attempts) -> CriticResult` with `verdict: 'ship'\|'discard'`, `failing_axes`, `regeneration_hint` | C |
| `_otr_name_gender` | `lookup_first_name_gender(name)`, `is_inversion(canonical, llm_gender)` | (no new agent needs it; Stage 1 cast audit already wired) |
| `_otr_legacy_to_stage1_adapter` | `legacy_ledger_to_stage1_plan(led_data) -> Optional[Stage1Plan]`, `extract_rendered_lines(led_data) -> List[dict]` | C |
| `_otr_stage2_call` | `compose_line(*, plan, beat, generate_fn, mode, n, ...) -> Stage2LineRecord`. Already dormant. Modes: `roleplay_multiturn`, `roleplay_single_turn` | F (optional, if Writer chooses line-by-line rendering) |
| `_otr_stage2_prompt` | `build_stage2_prompt_chain(...)`, `assemble_full_chat(...)`, `accept_turn1_ack(...)`, `accept_turn23_ack(...)` | (used internally by `_otr_stage2_call`; no agent imports directly) |

### Project conventions every agent must follow

- **LLM slot tagging.** Every `generate_fn(...)` call needs `# LLM slot: technical` or `# LLM slot: creative` within ±8 lines. The slot sweep test enforces this.
- **Structured / JSON passes** → `technical` slot. (Director brief generation, Editor verdict, extraction.)
- **Free-form dialogue / prose** → `creative` slot. (Writer turns.)
- **Writers'-room slot routing.** Writer turns route to `creative_writing_model`; Director and Editor turns route to `technical_model`. Both slots already exist as sockets on `OTR_LedgerScriptWriter` per the locked two-slot rule (PD6); Sprint 10B adds zero new model slots.
- **Constrained-decode calls** use the existing `make_constrained_generate_fn(cache_entry, schema_model)` pattern. Same closure signature.
- **No silent failures.** Retry ladders exhaust loudly with a stamped diagnostic on `meta.<agent_name>`.
- **PD3 (workflow JSON).** New nodes append widget values; existing widget order doesn't change. Width tests update with the new slot count (see Sprint 10A step 3-C precedent: 17 → 18 widgets).
- **Tests live in `tests/`** mirroring source structure. Each new module has its own test file. Full OTR suite + Bug Bible + LLM-slot sweep must pass before any commit.

---

## 2.5 Worked example -- "Spray of Hope" writers' room trace

The load-bearing concrete artifact: the minimum-viable writers'-room conversation for the exact news seed that produced the failure case in `__01_problem-statement.md` Section 1.1. The trace shows the room catching Spray of Hope's defining failure mode (the news premise never appearing in the dialogue) on the FIRST Editor cycle and producing a PASS draft on the second, inside 5 turns of the 18-turn budget.

**News seed (verbatim, 2026-05-26 ScienceDaily):** "Scientists say they've reversed brain aging with a simple nasal spray."

**Show.** SIGNAL LOST. Target ~500 words, two-hander plus announcer bookends.

**Cast (assumed locked upstream by Stage 1 cast pass).**
- REN BLACK -- late-shift signal operator, lost his sister to dementia. Skeptic.
- DR. MAEVE COLE -- the spray's lead researcher. Believer.

The pairing is the writers' room's first creative move: the news premise (a nasal spray that reverses brain aging) is converted into a two-character opposition (the researcher offering a free dose; the skeptic with a personal stake in dementia). This is what the one-shot composer cannot do -- it cannot decide that the second character should *be* the researcher, anchored to the news premise.

**Turn 1 -- Director (technical slot, `technical_model`, constrained-decode against `DirectorBriefSchema`).** Reads the news seed + cast names. Emits a structured prose brief: premise (nasal-spray trials reversing cognitive decline), dramatic question ("Would you trade your scars for a younger mind?"), opposed desires (Maeve wants Ren as the face of the program; Ren wants his sister's memory left alone), setting (comms shack at 3 AM, Maeve patched in from her lab), arc shape (temptation, confrontation, principled refusal with cost), audience feel (quiet weight, not action). Crucially, `forbidden_drift = ["code-red theatrics", "generic medical crisis", "override panic"]` -- the typed list of failure shapes the production pipeline currently produces unprompted. Buys us: a premise locked to the news, opposed desires named, AND an explicit typed list of failure modes the Writer must avoid.

**Turn 2 -- Writer (creative slot, `creative_writing_model`, free-form prose).** Draft 1. Writer leans on dramatic tension and underplays the premise. Opens with Ren noting an alert. Maeve patches in with "I have the results from the trial." Body is Maeve offering Ren a place in a follow-up study; Ren refusing because of the trial Sarah was in. Ending: Ren says "Maybe next time." 478 words. **The script never names the nasal spray, never names brain aging.** A listener who didn't read the news seed couldn't tell whether this is a cancer trial, a memory drug, a gene therapy, or a clinical study about anything else. This is the exact failure mode the production pipeline currently produces on Spray of Hope.

**Turn 3 -- Editor (technical slot, `technical_model`, constrained-decode against `EditorVerdictSchema`).** Reads brief + draft 1. Returns: `premise_clarity FAIL` ("nasal spray never named in dialogue; brain aging never named; listener cannot identify the news premise from the script alone -- episode could be about any medical trial"), `continuity PASS`, `pacing PASS`, `character_distinctiveness PASS`, `dialogue_naturalness PASS`, `emotional_arc FAIL` ("Ren's stakes told not shown; sister referenced as exposition, not as a present absence"), `resolution FAIL` ("'Maybe next time' is a tease, not a choice"). `pass_decision = False`. Three FAIL axes -- the load-bearing one is **premise_clarity**, exactly the failure mode Spray of Hope shipped. This is the signal the one-shot composer cannot produce because it has no second-pass surface.

**Turn 4 -- Writer (creative slot, `creative_writing_model`), draft 2.** Re-prompted with Editor's three FAIL reasons quoted in the user turn, with explicit emphasis on premise_clarity ("the nasal spray and brain-aging premise must be audible inside the first 30 seconds of dialogue"). Rewrites the announcer intro to name the vial. Rewrites Maeve's first speaking line to name the result (the spray worked, three trials, ninety-one patients, measurable reversal of cognitive aging). Rewrites Ren's interior so the sister isn't exposition -- she's the reason for the refusal. Rewrites the ending so Ren actually decides. 502 words.

**Turn 5 -- Editor (technical slot, `technical_model`), verdict 2.** All seven axes PASS. `pass_decision = True`. Cycle terminates clean at 5 turns of the 18-turn cap, 2 Editor cycles of the 3-cycle cap. The premise is audible in the announcer line ("a quiet miracle in a small glass vial") at second 0, and in Maeve's first speaking line ("The nasal spray worked. Three trials, ninety-one patients, measurable reversal of cognitive aging") at roughly second 18.

**Final draft -- "Spray of Hope," SIGNAL LOST**

ANNOUNCER: Good evening. This is SIGNAL LOST. Tonight, a quiet miracle in a small glass vial, and the operator who has to decide whether to use it. Stay with us.

REN BLACK: Station Seven, Black on the line. Doctor Cole, you are coming in clear. It is past three in the morning here. Tell me again why this could not wait.

DR. MAEVE COLE: Because by morning the embargo lifts and every newsroom in the country will be calling. I wanted you to hear it from me first, Ren. The nasal spray worked. Three trials, ninety-one patients, measurable reversal of cognitive aging. We did it.

REN BLACK: You did it.

DR. MAEVE COLE: We did it. I am holding a dose for you. One vial, your name on the label. For Sarah.

REN BLACK: My sister is gone, Maeve. A vial does not bring her back.

DR. MAEVE COLE: No. But you watched her forget your face. You told me once you could feel the same fog starting in yourself. The late shifts, the names you cannot place. Ren, this stops it. This rolls it back.

REN BLACK: And what does it cost.

DR. MAEVE COLE: Nothing. I am offering.

REN BLACK: Everything costs something. You know that. What do you need from me in return.

DR. MAEVE COLE: One interview. On the record. The man who lost his sister, saved by the spray. We need a face, Ren. Funding hearings start in June.

REN BLACK: There it is.

DR. MAEVE COLE: It is not a trick. It is the trade. Your story buys ten thousand more vials. Ten thousand more sisters.

REN BLACK: Maeve. I want to say yes. I want it so badly my hands are shaking on this console. But if I take that vial because I am afraid of forgetting, I will spend the rest of my clear-headed life knowing I sold the worst night of my life to a camera. Sarah deserved better than that. So do the next ten thousand.

DR. MAEVE COLE: So that is no.

REN BLACK: That is, give the vial to someone whose story you have not already written. I will come to your hearing. I will speak for Sarah. But I speak as her brother, not as your before-and-after.

DR. MAEVE COLE: Ren.

REN BLACK: Send me the hearing date. Station Seven, out.

ANNOUNCER: A signal broke through tonight, and a man chose which part of himself to keep. This has been SIGNAL LOST. Good night.

**Trace summary.** Script body: 502 words. Turns used: 5 of 18. Editor cycles: 2 of 3. **Premise visible at announcer second 0 (vial) and at Maeve's first speaking line (the nasal spray worked, three trials, ninety-one patients, measurable reversal of cognitive aging) at roughly second 18 -- well inside the "audible by minute 3" bar of Section 6's listen rubric.** Arc: temptation, confrontation, principled refusal with cost. No speaker-leak (every character line is in that character's voice; the announcer narrates only the bookends, not character action). Safe for work; no "damn." This is what the writers' room is supposed to produce, and it is what the one-shot composer cannot.

---

## 3. New contracts (frozen before any agent starts)

These dataclasses are the agent-to-agent interface. Must be agreed before dispatch. Frozen for the duration of Sprint 10B; any change requires both producer and consumer agents re-aligning.

### 3.1 `DirectorBrief` — owned by Agent D, consumed by Agents E, F

```python
# _otr_director_brief.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class DirectorBrief:
    """The Director's brief for one episode.

    Free-form prose at the field level: the LLM writes each field as
    one-to-three sentences. The structure exists so downstream agents
    (Writer, Editor) can pull specific fields without re-parsing.
    """
    news_premise: str           # what the episode IS about; 1-2 sentences
    dramatic_question: str      # what's at stake; 1 sentence
    opposed_desires: str        # whose desire opposes whose; 1-2 sentences
    arc_shape: str              # what changes between open and close; 1-2 sentences
    audience_feel: str          # what the audience should feel; 1 sentence
    forbidden_drift: List[str] = field(default_factory=list)
                                # things to NOT drift toward; 0-5 short phrases
    raw_brief: str = ""         # full prose brief; for context-hungry agents
```

### 3.2 `EditorVerdict` — owned by Agent E, consumed by Agent F

```python
# _otr_editor_pass.py
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class EditorVerdict:
    """The Editor's verdict on a Writer's draft."""
    pass_decision: bool                  # True = ship; False = revise
    failing_axes: List[str] = field(default_factory=list)
                                         # rubric axis json_keys
    per_axis_notes: Dict[str, str] = field(default_factory=dict)
                                         # axis_key -> 1-3 sentence note
    overall_note: str = ""               # free-form summary
    cycle: int = 0                       # 0-indexed Editor pass number
```

### 3.3 `StoryRoomTranscript` — owned by Agent F, consumed by Agent G

```python
# _otr_story_room.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class StoryRoomTurn:
    role: str                  # 'director' | 'writer' | 'editor'
    cycle: int                 # which Editor cycle (0 = pre-first-Editor)
    content: str               # free-form prose output

@dataclass
class StoryRoomTranscript:
    director_brief: 'DirectorBrief'
    turns: List[StoryRoomTurn] = field(default_factory=list)
    editor_verdicts: List['EditorVerdict'] = field(default_factory=list)
    final_draft: str = ""              # canonical Writer turn
    terminated_clean: bool = False     # True if Editor passed; False if cap hit
    total_turns: int = 0
    elapsed_seconds: float = 0.0
```

### 3.4 `StoryRoomExtraction` — owned by Agent G, consumed by downstream pipeline

```python
# _otr_story_room_extract.py
from dataclasses import dataclass, field
from typing import List, Dict
from ._otr_stage1_plan import Stage1Arc, Stage1Beat, Stage1CastMember

@dataclass
class StoryRoomExtraction:
    """Structured artifacts derived from a story room transcript.
    Field shapes match what the existing announcer pass / continuity
    ledger / music+SFX cue builder / bark generator already consume.
    """
    cast: List[Stage1CastMember]
    beats: List[Stage1Beat]
    dialogue: List[Dict]               # [{"beat_id", "speaker", "text"}, ...]
    audio_cues: List[Dict] = field(default_factory=list)
    running_facts: List[str] = field(default_factory=list)
    arc: Stage1Arc = None
    premise: str = ""
```

---

## 4. Parallel decomposition — four waves

### Wave 0 — One-commit cheap experiment (precedes Wave 1)

**Purpose.** Test the chat-style hypothesis on the smallest possible surface before committing to the full writers' room. Sprint 10A produced the Stage 2 multi-turn machinery and never wired it; this is the wire-up.

**Owns.** `OTR_LedgerScriptWriter.py` -- one new widget + dispatch at the two existing `_OTRLC.compose_line` call sites (lines 2872 and 2938 at HEAD `e8cf026`). One adapter helper. Workflow JSON appends the widget. No new files.

**Frozen against.** `_otr_stage2_call.compose_line(*, plan, beat, generate_fn, mode='roleplay_multiturn', n, ...)` signature (Section 2). 30 green tests in `tests/test_stage2_multiturn.py` already pin the multi-turn path.

**Code.**
1. Add `use_multiturn_dialogue` to `OTR_LedgerScriptWriter.INPUT_TYPES` as `BOOLEAN` defaulting to `False`.
2. At both `compose_line` call sites, branch:
   - `False` -- call legacy `_OTRLC.compose_line(req)` exactly as today.
   - `True` -- call `_otr_stage2_call.compose_line(plan=<adapted from req>, beat=<adapted from req>, generate_fn=<same>, mode='roleplay_multiturn', n=<bounded best-of-N, default 4>, ...)`.
3. Adapter `_line_request_to_stage2_inputs(req: LineRequest) -> tuple[Stage1Plan, Stage1Beat]`. Shared with Wave 1 Agent B (which needs the same helper as `_line_request_to_stage1_beat`); decide ownership in commit order (whichever lands first owns it, the other consumes).
4. Stamp `meta.dialogue_path = 'multiturn' | 'legacy'` per episode for audit.

**Wire.** Workflow JSON appends the new widget; `tests/test_workflow_audio_widget_vectors.py` length pin updates for slot N+1. PD3 satisfied.

**LLM slot.** Both branches are creative dialogue -- existing `compose_line` slot tags cover both. Slot sweep unchanged.

**Tests.**
- Unit: dispatch on `True` lands in `_otr_stage2_call.compose_line`; `False` lands in legacy.
- Unit: legacy path byte-identical to pre-change baseline (PD1).
- Integration: one episode runs end-to-end on both paths from the same Stage 1 plan.
- Regression: full OTR + Bug Bible + slot sweep + audio byte-identity (legacy path only).

**Acceptance.** With `use_multiturn_dialogue=True`, three full-budget episodes ship audio + video. Operator runs the Section 6.2 A/B against the same three Stage 1 plans on legacy.

**Outcomes.**
- **Lift (operator rubric mean delta ≥ +0.4):** Wave 1 + Wave 2 proceed as designed; Wave 0 ships as the dialogue default for episodes where the writers' room isn't yet active.
- **Tie or marginal (delta in [-0.2, +0.4]):** Wave 1 proceeds (orthogonal failure modes); Wave 2 scope reopened, possibly trimmed to "Director brief only" rather than full writers' room.
- **Regression (delta < -0.2):** Wave 0 ships as `=False` default permanently; Wave 1 still proceeds; writers' room redesign reopened with new evidence.

**Conflicts.** With Wave 1 Agent B (Stage 3 validators in `compose_line`): Wave 0 lands first; Agent B starts from the post-Wave-0 code. The `_line_request_to_stage2_inputs` adapter Wave 0 introduces is reused by Agent B.

**Why this is Step 1 and not a Wave 1 agent.** The machinery exists with 30 green tests; this is a wire-up commit, not a build. The A/B answer reshapes how much Wave 2 invests in the writers' room. Every other Sprint 10B step is more expensive than this one and should follow this signal.

---

### Wave 1 — 5 parallel agents, no inter-agent dependencies

Each agent is one commit following the Sprint 10A pattern: **Review → Code → Wire → Regress → Commit**. Each agent owns disjoint files; each must produce full OTR suite green + Bug Bible baseline held + LLM-slot sweep clean.

---

#### Agent A — News interpreter semantic check

**Owns.** `_otr_news_interpreter.py` (or wherever `build_news_briefs` lives). Hot zone: the post-validation that rejects key_terms not appearing verbatim in source body.

**Frozen against.** The `build_news_briefs` return schema. Downstream consumers must not need to change.

**Code.**
1. Replace exact-substring check (`key_term in article_body`) with semantic-presence check. Two options:
   - **(a) LLM-as-judge.** One yes/no call per term: "Does this article support the claim that X?" Returns bool; if yes, term is accepted. Cheap on small LLM, ~1s per term.
   - **(b) Embedding cosine similarity.** Use whatever embedding model is already cached in the project. Threshold tuned against the BUG-LOCAL-264 fixtures.
2. Agent picks (a) or (b) based on what's cleaner in the serving stack. Document the choice in the commit message.
3. Keep `build_news_briefs` return schema unchanged.

**Wire.** Pure module-internal logic swap. No node surface change. No widget. No workflow JSON change.

**LLM slot.** If (a): `technical` slot. Tag the call site (`# LLM slot: technical` within ±8 lines).

**Tests.**
- 3-5 fixtures of articles that historically tripped BUG-LOCAL-264 (brain-aging "Texas A&M", Artemis "SLS rocket", etc.). Each must now produce valid key_terms instead of exhausting the retry ladder.
- Regression: full OTR + Bug Bible + slot sweep.

**Acceptance.** A previously-failing fixture now produces a populated brief on first attempt.

**Conflicts.** None. Module is independent.

---

#### Agent B — Stage 3 validators wired into legacy line composer

**Owns.** `_otr_line_composer.py::compose_line` (the signature is in section 2). A small helper to construct a `Stage1Beat` from the legacy `LineRequest`.

**Frozen against.** The existing `compose_line` call signature. New parameters keyword-only with safe defaults so it's drop-in for existing callers.

**Code.**
1. Add to `compose_line`:
   ```python
   enable_stage3_validators: bool = False
   stage3_plan: Optional[Stage1Plan] = None
   stage3_beat: Optional[Stage1Beat] = None
   stage3_banned_phrases: Optional[List[str]] = None
   ```
2. After the existing draft + polish + strip pipeline runs (right before `LineResult` is built): if `enable_stage3_validators=True` AND `stage3_plan` AND `stage3_beat` are provided, run `_otr_stage3_validators.validate_line(stage3_plan, stage3_beat, cleaned, banned_phrases=stage3_banned_phrases)`.
3. If `result.errors` non-empty: regenerate the line once more with the failure reason injected as a `reroll_hint` (existing parameter). Accept whatever the regenerate produces; do not loop.
4. If `result.warns` non-empty: stamp them; don't regenerate.
5. Extend `LineResult` with a new field:
   ```python
   validation_findings: tuple[dict, ...] = ()
   ```
   Each dict is `ValidationFinding.to_dict()`-like (severity, code, message). Tuple so it's immutable.
6. Helper `_line_request_to_stage1_beat(req: LineRequest) -> Stage1Beat` — private to this module. Maps LineRequest fields (target_words, speaker, etc.) to a minimal Stage1Beat. Where fields don't exist, use placeholders.

**Wire.** One call site in `OTR_LedgerScriptWriter.py` updated to pass `enable_stage3_validators=True` + cast/plan context. Behind a new widget `enable_production_stage3_validators: bool = True` defaulting to True (live immediately, reversible). Workflow JSON gets the new widget appended.

**LLM slot.** The repair regenerate is creative — same slot as the rest of compose_line. Existing tags cover it; verify the slot sweep still passes.

**Tests.**
- Unit: lines that fail each validator code (`speaker_leak`, `pronoun_mismatch`, `length_drift`) trigger one repair attempt; findings stamped on `LineResult.validation_findings`.
- Unit: lines with only `warn` severity stamp findings without regenerating.
- Integration fixture: a known-bad line from past production (Spray of Hope beat 4: `Breathless, Ren Black mutters, "..."`) is now caught and either repaired or stamped.
- Regression: full OTR suite + Bug Bible + slot sweep.

**Acceptance.** Spray of Hope beat-4-style speaker-leak failures are caught in production output. PD1 holds (byte-identity for the legacy path on the canonical baseline).

**Conflicts.** With Agent C as noted below. Resolution: B operates INSIDE `compose_line()`; C operates AFTER `compose_line()` returns for every beat. No code overlap.

---

#### Agent C — Stage 7 critic verdict drives reroll escalation

**Owns.** New module `_otr_reroll_escalation.py` + wiring in `OTR_LedgerScriptWriter.py` (the reroll decision tree).

**Frozen against.** `_otr_whole_episode_critic.run_whole_episode_critic` signature and `CriticResult` shape.

**Code.**
1. New module:
   ```python
   # _otr_reroll_escalation.py
   from dataclasses import dataclass, field
   from typing import Literal, List, Optional

   EscalationScope = Literal["none", "line", "beat", "episode"]

   STRUCTURAL_AXES = {
       "premise_clarity",
       "continuity",
       "resolution",
       "emotional_arc",
   }
   LOCAL_AXES = {
       "character_distinctiveness",
       "dialogue_naturalness",
       "pacing",
   }

   @dataclass
   class EscalationDecision:
       scope: EscalationScope
       reason: str
       target_beat_ids: List[str] = field(default_factory=list)

   def decide_escalation_scope(
       critic_result,
       story_critic_targets: Optional[List[dict]] = None,
   ) -> EscalationDecision:
       """
       - critic.verdict == 'ship' -> 'none'
       - failing_axes ∩ STRUCTURAL_AXES non-empty -> 'episode'
         (use critic.regeneration_hint as context for next Stage 1 call)
       - failing_axes ⊆ LOCAL_AXES AND story_critic_targets names beats -> 'beat'
       - otherwise -> 'line' (legacy reroll path)
       """
   ```
2. Wire into `OTR_LedgerScriptWriter`: after all lines composed and the legacy `script_critic` runs, call `run_whole_episode_critic` (build inputs via `legacy_ledger_to_stage1_plan` + `extract_rendered_lines`). Pass `CriticResult` + the existing story-critic targets to `decide_escalation_scope`.
3. Dispatch:
   - `'none'` → proceed to render.
   - `'line'` → existing legacy line-reroll behavior.
   - `'beat'` → new path: recompose all lines in the named beats from scratch (loop calling `compose_line` for each target beat).
   - `'episode'` → stamp `meta.freeze_verdict='needs_full_rerun'` with `critic.regeneration_hint` stamped on `meta.regeneration_hint`. The cascade halt catches it.

**Wire.** Behind widget `enable_critic_escalation: bool = True` default. Workflow JSON gets the new widget. Tag the `run_whole_episode_critic` call site with `# LLM slot: technical`.

**Tests.**
- Unit: known-structural-failure-axes `CriticResult` → `EscalationDecision('episode', ...)`.
- Unit: known-local-failure-axes + story_critic_targets → `'beat'` or `'line'`.
- Integration: legacy ledger fixture with structural critic verdict triggers whole-episode regenerate path; line-only fixture triggers legacy line reroll.
- Regression: full suite.

**Acceptance.** A run that historically hit `needs_full_rerun` via 2-cycle legacy line reroll now routes to whole-episode regenerate after ONE critic pass with structural failing_axes.

**Conflicts.** With Agent B as noted: Agent B inside `compose_line()`, Agent C after `compose_line()`. With legacy reroll loop: don't delete it, branch around it via `EscalationDecision.scope`.

---

#### Agent D — Director agent

**Owns.** New files: `_otr_director_brief.py`, `nodes/OTR_DirectorBrief.py`, `tests/test_otr_director_brief.py`. Owns the `DirectorBrief` dataclass (section 3.1).

**Frozen against.** Nothing in the existing pipeline. New module, dormant until Agent F (Wave 2) wires it in.

**Code.**
1. Pydantic schema `DirectorBriefSchema` mirroring the `DirectorBrief` dataclass for constrained-decode output.
2. Prompt builder `build_director_prompt(news_seed, cast_names, rubric)` producing real prose direction. The system prompt reads like a director briefing a writers' room, not like a form. Include the rubric axis names as context so the Director writes briefs that set up episodes that can pass the critic.
3. Call site:
   ```python
   def run_director(
       news_seed: str,
       cast_names: List[str],
       *,
       generate_fn,        # constrained-decode fn bound to DirectorBriefSchema
       rubric: Optional[Rubric] = None,
       max_attempts: int = 2,
   ) -> DirectorBrief:
       """Returns a populated DirectorBrief or raises DirectorCallFailedError."""
   ```
4. Node `OTR_DirectorBrief`: takes `news_seed` (from upstream news_interpreter), `cast_names` (from cast lock), emits serialized `DirectorBrief` on its output socket.
5. Implementer's choice on whether to one-shot constrained-decode the full structured brief OR generate prose first then extract — both work, the latter is closer to "let the LLM dynamically create."

**Wire.** Register in `nodes/__init__.py` and `NODE_CLASS_MAPPINGS`. Workflow JSON gets new node entry positioned downstream of cast lock, output socket disconnected (Wave 2 connects it).

**LLM slot.** `technical` (constrained-decode pass).

**Tests.**
- Unit: 10 different news seeds; every output passes `DirectorBrief` Pydantic validation.
- Content audit: every brief contains identifiable references to news premise, human stakes, and "what changes." Audit is a deterministic keyword-overlap heuristic against the news seed and rubric axis names.
- Integration: node loads, runs end-to-end on a single news seed, emits the dataclass on its output socket.
- Regression: full suite + slot sweep.

**Acceptance.** 9/10 fresh news seeds produce usable briefs.

**Conflicts.** None. Fully isolated.

---

#### Agent E — Editor agent

**Owns.** New files: `_otr_editor_pass.py`, `nodes/OTR_EditorPass.py`, `tests/test_otr_editor_pass.py`. Owns the `EditorVerdict` dataclass (section 3.2).

**Frozen against.** Agent D's `DirectorBrief` shape (section 3.1).

**Code.**
1. Pydantic schema `EditorVerdictSchema` mirroring `EditorVerdict`.
2. Prompt builder `build_editor_prompt(director_brief, writer_draft, rubric, cycle)` producing real editor framing. Anchored to rubric axes by their `json_key`.
3. Call site:
   ```python
   def run_editor(
       director_brief: DirectorBrief,
       writer_draft: str,
       *,
       generate_fn,        # constrained-decode fn bound to EditorVerdictSchema
       rubric: Optional[Rubric] = None,
       cycle: int = 0,
       max_attempts: int = 2,
   ) -> EditorVerdict:
       """Returns an EditorVerdict or raises EditorCallFailedError."""
   ```
4. Node `OTR_EditorPass`: takes serialized `DirectorBrief` + draft string, emits `EditorVerdict`. Dormant.

**Wire.** Register node. Workflow JSON gets entry. No live consumer.

**LLM slot.** `technical`.

**Tests.**
- Unit: 10 known-bad drafts (including a hand-constructed Spray of Hope from `__01_problem-statement.md` section 1.1). Editor's `failing_axes` includes the axes the problem statement identifies.
- Unit: 3 known-good drafts (synthesized or from past successful episodes). Editor's `pass_decision=True`.
- Schema: every output passes Pydantic validation.
- Regression: full suite.

**Acceptance.** Editor correctly fails Spray of Hope on at least `premise_clarity`, `resolution`, `emotional_arc`.

**Conflicts.** Reads but does not modify the rubric. Frozen-contract clean.

---

### Wave 2 — Composition (sequential, blocks on Wave 1)

Wave 2 cannot start until Agents D and E have committed (the dataclasses must be importable). Wave 1 agents A, B, C don't block Wave 2 but should ideally land first so Story Room is measured against the improved baseline.

---

#### Agent F — Story Room loop

**Owns.** New files: `_otr_story_room.py`, `nodes/OTR_StoryRoom.py`, `tests/test_otr_story_room.py`. Owns `StoryRoomTurn` and `StoryRoomTranscript` (section 3.3).

**Depends on.** Agent D (`DirectorBrief`), Agent E (`EditorVerdict`).

**Code.**
1. Writer system prompt module: real prose framing for the LLM in Writer mode. Multiple turn templates: initial draft, revision based on Editor notes, optional expansion.
2. Loop function:
   ```python
   def run_story_room(
       *,
       news_seed: str,
       cast: List[Stage1CastMember],
       director_generate_fn,        # technical, constrained
       writer_generate_fn,          # creative, free-form
       editor_generate_fn,          # technical, constrained
       rubric: Optional[Rubric] = None,
       max_writer_turns: int = 4,
       max_editor_cycles: int = 3,
       max_total_turns: int = 18,
   ) -> StoryRoomTranscript:
       """
       1. Director turn -> DirectorBrief (one call).
       2. Loop:
          a. Writer takes up to max_writer_turns producing drafts.
             Each Writer turn uses writer_generate_fn (creative slot).
          b. Editor reads current draft + brief + writer turns.
          c. If pass_decision=True: terminate, terminated_clean=True.
          d. If pass_decision=False: feed EditorVerdict.notes to next Writer round.
          e. At max_editor_cycles: terminate with terminated_clean=False
             and the last Writer draft as final_draft.
          f. At max_total_turns (any role): hard terminate.
       3. Return transcript.
       """
   ```
3. Prefix caching: structure message lists so system prompt + DirectorBrief + cast cards are a stable prefix across Writer turns. Backend may or may not honor this; structure for it anyway.
4. Optional path: Writer can choose to call `_otr_stage2_call.compose_line` for per-line dialogue rendering on a beat-by-beat basis instead of producing whole drafts. This is opaque to the Editor — the Editor reads whatever draft the Writer produces. Implementer chooses whether to expose this as a Writer sub-mode or hide it.
5. Node `OTR_StoryRoom`: takes news_seed + cast + DirectorBrief node socket; emits `StoryRoomTranscript`. Feature flag widget `use_story_room: bool = False` default. When False, node passes through (no-op).

**Wire.** Register node. Workflow JSON connects Director → Story Room → (Agent G's Extract, dormant). Tag each generate call site by slot.

**Tests.**
- Unit: loop terminates clean on synthetic conversation where Editor passes cycle 1.
- Unit: loop terminates `terminated_clean=False` on synthetic conversation where Editor never passes.
- Unit: max_total_turns enforced (e.g., synthetic Writer that produces infinite drafts terminates at 18).
- Integration: run end-to-end on 3 fresh news seeds. Transcript completes within budget on at least 2/3.
- Regression: full suite + slot sweep.

**Acceptance.** Loop reliably terminates within turn budget across 20 trial runs; no observed infinite loops.

**Conflicts.** None.

---

#### Agent G — Transcript-to-structured extraction

**Owns.** New files: `_otr_story_room_extract.py`, `nodes/OTR_StoryRoomExtract.py`, `tests/test_otr_story_room_extract.py`. Owns `StoryRoomExtraction` (section 3.4).

**Depends on.** Agent F (`StoryRoomTranscript`).

**Code.**
1. Pydantic schema `StoryRoomExtractionSchema` reusing Stage1 types (Stage1CastMember, Stage1Beat, Stage1Arc) where possible.
2. Prompt builder: feeds full transcript + cast names + news_seed; asks for structured extraction. The LLM is transcribing, not creating.
3. Call site:
   ```python
   def extract_from_transcript(
       transcript: StoryRoomTranscript,
       *,
       generate_fn,        # constrained-decode fn bound to StoryRoomExtractionSchema
       max_attempts: int = 2,
   ) -> StoryRoomExtraction:
       """Returns StoryRoomExtraction or raises ExtractionCallFailedError."""
   ```
4. Implementer's choice: one big extraction call OR chain (cast → beats → dialogue). Pick based on constrained-decode output quality across 20 trial transcripts.
5. Node `OTR_StoryRoomExtract`: consumes `StoryRoomTranscript`; emits `StoryRoomExtraction` whose dict-shaped outputs (`dialogue`, `audio_cues`) match keys the downstream nodes already expect.

**Wire.** Register node. Workflow JSON connects Story Room → Extract → (existing announcer pass, continuity ledger, music+SFX cue builder, bark generator). When `use_story_room=True`, downstream nodes read from Extract; when False, they read from legacy outputs as today.

**Critical cross-check before commit.** Read the input shape of each downstream node (announcer, continuity, music/SFX, bark) and confirm Extract emits matching dict keys. Field names matter — `speaker` not `character`, `text` not `dialogue`, etc.

**LLM slot.** `technical`.

**Tests.**
- Unit: extraction on 20 synthetic transcripts produces valid Pydantic schema 20/20.
- Unit: cast names in extraction match cast in transcript.
- Unit: dialogue line count is plausible vs draft length.
- Integration: end-to-end Story Room → Extract → announcer pass; announcer produces a populated intro (proves downstream compatibility).
- Regression: full suite.

**Acceptance.** Extract produces downstream-compatible structured data from 19/20 transcripts. Announcer / music / bark all consume Extract outputs without changes.

**Conflicts.** Field-name compatibility with downstream nodes. Resolved by the pre-commit cross-check.

---

### Wave 3 — Operator gate (Step 8)

**Operator-owned.** Not an agent task.

Flip `use_story_room=True` in the workflow JSON. Queue a full-budget production episode (500-800 words, not a smoke run). Operator listens.

**Pass criteria.**
1. Audio + video ship end-to-end (PD1 invariant holds for the new path: bytes are reproducible against a stamped transcript).
2. News premise identifiable from audio alone (no metadata).
3. Episode has dramatic shape — something at stake, opposition, change.
4. Characters distinguishable.
5. Stage 7 critic `mean_score ≥ 4.0`.
6. Operator listen verdict: "yes, I'd ship this."

**If pass:** Sprint 10B done. Flip default to `use_story_room=True` in a follow-up commit. Sprint 10C scopes against whatever the next gap is.

**If fail:** Operator names the failing axis. Sprint 10B-revise tunes Director prompt, Editor prompt, or loop bounds based on which axis failed.

---

## 5. Dispatch order + parallelism map

```
                Wave 0 — one focused commit
                ┌────────────────────────────────────────────────┐
                │ wire _otr_stage2_call.compose_line behind      │
                │ use_multiturn_dialogue=False; one adapter,     │
                │ one dispatch, two call sites                   │
                └────────────────────────────────────────────────┘
                                      │
                                      ▼
                              operator A/B
                              (Section 6.2 demo plan)
                                      │
                                      ▼
                Wave 1 — 5 parallel agents
                ┌───────────────────────────────────────────────┐
                │                                               │
            Agent A         Agent B         Agent C
            news_interp    stage3 in       critic-driven
            semantic       legacy comp     escalation

            Agent D         Agent E
            Director        Editor
            agent           agent
                │              │              │              │              │
                └──────────────┴──────┬───────┴──────────────┴──────────────┘
                                      │
                                      ▼
                              all Wave 1 green
                              full OTR + Bug Bible
                              + slot sweep
                                      │
                                      ▼
                            Wave 2 — sequential
                                  ┌──────┐
                                  │ Agent F   Story Room loop  │
                                  └──────┘
                                      │
                                      ▼
                                  ┌──────┐
                                  │ Agent G   Extraction       │
                                  └──────┘
                                      │
                                      ▼
                              Wave 3 — operator
                                  ┌──────┐
                                  │ Step 8   E2E listen test  │
                                  └──────┘
```

**Wave 0 sequencing.** Wave 0 is a single commit and lands before any Wave 1 agent starts. The `_line_request_to_stage2_inputs` adapter it introduces is the same shape Wave 1 Agent B needs as `_line_request_to_stage1_beat`; Wave 0 owns it, Agent B imports it. Wave 1 Agent B's branch starts from the post-Wave-0 HEAD.

**Wave 1 parallel safety.** Each agent owns disjoint files. The only soft overlap is Agents B + C both interacting with `OTR_LedgerScriptWriter` — B inside `compose_line()`, C after `compose_line()` returns. Section 4 ordering convention keeps that conflict-free.

**Wave 2 sequencing.** Agent F imports `DirectorBrief` (D) and `EditorVerdict` (E). Agent G imports `StoryRoomTranscript` (F). Modules must commit cleanly before downstream agents pull them.

**Operator role between waves.** After Wave 1 completes, queue one smoke-budget run with all Wave 1 features enabled to confirm baseline ships. Then dispatch Wave 2.

---

## 6. Acceptance criteria -- A/B operator worksheet

Sprint 10B is done when all of the sprint-wide guards below hold AND the A/B in 6.2 comes back positive.

### 6.1 Sprint-wide guards

1. All Wave 0 + Wave 1 + Wave 2 commits land green: full OTR suite + Bug Bible baseline + LLM-slot sweep throughout.
2. Legacy pipeline (`use_multiturn_dialogue=False` AND `use_story_room=False`) produces byte-identical audio per PD1 against a pre-Sprint-10B golden.
3. Both feature flags reversible -- flipping to False restores legacy behavior in one widget change each.
4. Stage 7 critic `mean_score ≥ 4.0` on at least 2 of the 3 demo seeds (see 6.2) for the new writer cell.

### 6.2 A/B demo plan

Applies to BOTH the Wave 0 cheap experiment (legacy `compose_line` vs Wave-0-wired Stage 2 multiturn) AND the Wave 3 final gate (Wave 0 winner vs full Wave 1+2 writers' room). Same rubric, different writer cells.

**News seeds.** Freeze on the morning of run-day from a single ScienceDaily RSS pull; archive the three URLs in `docs/2026-05-26-10b-demo/seeds.txt` before any run. Working set for the first gate:
- **Space/tech:** "Astronomers report the first confirmed detection of water vapor in the atmosphere of a rocky exoplanet 40 light-years away."
- **Health/medicine:** "A new blood test detects pancreatic cancer up to three years before symptoms appear in a 4,000-patient trial."
- **Biology/wildlife:** "Researchers find that humpback whales off the Oregon coast are teaching each other a new bubble-net feeding technique never observed before."

Three different genres on purpose (wonder, dread, curiosity). If a writer only wins on one tone, the asymmetry shows.

**Stage 1 pin.** For each seed, run Stage 1 (planner + cast lock) ONCE on the legacy pipeline, save the resulting plan JSON to `docs/2026-05-26-10b-demo/plan_<seed>.json`, and feed it to every writer cell via `seed_override` on the writer node. Workflow JSON for the new writer paths points at the same plan file. No re-planning between cells. Isolates the writer-stage delta; removes Stage 1 variance as a confound.

**Run matrix per gate.** 3 seeds × 2 writer cells × **3 runs per cell** = **18 production-budget episodes per gate**. At ~10 minutes per episode on the RTX 5080, this is a ~3-hour overnight soak. Fewer than 3 per cell cannot distinguish "the architecture works" from "lucky roll." More than 3 per cell costs listen-test time geometrically (operator has to score every output) without adding signal at this sample size.

**Operator rubric (1-5 per axis, 5 = "I would ship this"):**
- News premise audibility: can a listener name the headline by minute 3?
- Dramatic shape: setup, escalation, turn, landing -- all four present?
- Character specificity: do the leads sound like distinct people, not interchangeable voices?
- Dialogue craft: subtext, interruption, rhythm -- or are they reading bullet points at each other?
- Resolution: earned ending, or does it just stop?
- Gut "good story?": one number, no hedging.

Mean the six axes per script; report per-cell mean and standard deviation across the 3 runs.

**Stage 7 critic threshold.** `mean_score ≥ 4.0` is the "good" bar. Reason: 4.0 is the lowest score where every sub-axis is at least "solid" rather than "acceptable" -- 3.x cells almost always have one broken axis hiding inside the average.

**Listen procedure.** Operator listens blind: filenames hashed, shuffled order, no metadata visible during scoring. Score on a single sitting per gate to control for mood drift.

**Decision rule.**
- **Ship as default:** new writer's operator-rubric mean ≥ legacy mean + 0.4 AND Stage 7 critic mean ≥ 4.0 on at least 2 of the 3 seeds.
- **Revise:** new writer ties or wins on 1 seed only, OR Stage 7 critic mean lands in 3.5-4.0.
- **Kill:** new writer loses on 2+ seeds, OR any single run scores below 2.5 on "news premise audibility" (lost the headline = lost the show).

---

## 7. What each agent decides for themselves

Design leaves these open on purpose. Each agent picks based on what's cleaner in their scope:

- **Agent A**: LLM-as-judge vs embedding cosine for semantic key_terms.
- **Agent B**: Where `_line_request_to_stage1_beat` helper lives (in `_otr_line_composer.py` or its own tiny module).
- **Agent C**: Whether 'beat' scope is its own new path or sugar for "rerun line composer on N beats."
- **Agent D**: Whether Director brief comes from one constrained-decode call OR free-form prose followed by a structured-extraction pass against that prose. The latter is closer to "let the LLM dynamically create."
- **Agent F**: Whether Writer uses `_otr_stage2_call.compose_line` for per-line rendering OR asks model for full draft prose. Both valid; Writer chooses its working shape.
- **Agent G**: One big extraction call vs chain (cast → beats → dialogue).

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Wave 1 agents B + C touch overlapping code in `OTR_LedgerScriptWriter` | Section 4 ordering: B inside `compose_line()`, C after. Code review confirms no overlapping line edits. |
| Pydantic schema drift between Agents D and E mid-sprint | Section 3 dataclasses frozen for sprint. Changes require both producer + consumer agents re-aligning. |
| Constrained-decode timeouts on long transcripts (Agent G) | 2-attempt retry budget at falling temperature. If both fail, loud diagnostic; operator can re-queue. |
| Story Room runs to `max_total_turns` without clean termination | Transcript still has `final_draft` from last Writer turn; Extract still runs; `terminated_clean=False` stamped on `meta.story_room`. Episode might be lower-quality but ships. |
| Downstream nodes reject Extract's outputs due to field-name mismatch | Agent G's pre-commit cross-check: read each downstream node's input shape, confirm Extract emits matching keys. |
| PD7 reproducibility breaks under multi-turn Story Room | Explicitly relaxed for `use_story_room=True`. Legacy mode preserves PD7. `meta.story_room.transcript` becomes the new reproducibility anchor — re-queueing with the same transcript produces the same audio. |
| One agent's commit fails Bug Bible | That agent blocks; other Wave 1 agents proceed. Sprint waits at Wave 1 boundary for the blocking agent to fix and re-commit. |
| Loop drift (model's "Ready" / "Confirmed" patterns don't terminate Editor cleanly) | Editor uses constrained-decode against `EditorVerdictSchema` — no ack-chain ambiguity. `pass_decision` is a typed bool. |

---

## 9. What this design explicitly rejects

- **Per-turn JSON schemas.** Schema lives at extraction (Agent G) and at structured-artifact agents (D, E). The Writer has no schema — the Writer does creative prose work.
- **Per-character actor agents** (Candidate B from `__00`). Deferred to 10C+ if character distinctiveness remains the gap after 10B.
- **Open-ended conversation.** Hard caps everywhere.
- **Bigger model.** Any chat-tuned small LLM under 16GB.
- **Human in the loop.** Director, Writer, Editor are all the same LLM.
- **Replacing legacy pipeline.** Feature-flagged; legacy preserved with byte-identity.

---

## 10. Answers to the seven open design questions

These are the seven from `__01_problem-statement.md` Section 6. Sections 2.5, 4, and 6 are the operational answers; this section is the short, explicit version that maps each problem-statement question to a load-bearing decision in this design.

1. **Scope of the chat-style change.** Staged. **Wave 0** wires multi-turn dialogue only -- the smallest surface that tests the chat-style hypothesis with one commit. **Wave 1-2** (Section 4) is the full writers' room: Director sets premise / arc / audience feel; Writer drafts beats and dialogue across multiple turns; Editor gates per-rubric. Both surfaces stay feature-flagged so the legacy one-shot composer is always reachable for PD1 byte-identity audio and as a fallback.

2. **Who is the "system" in the conversation?** One LLM, three roles, swapped by system prompt -- Director (technical slot on `technical_model`, constrained-decode brief), Writer (creative slot on `creative_writing_model`, free-form prose), Editor (technical slot on `technical_model`, constrained-decode verdict). VRAM keeps to a single resident model (~9-10 GB on Mistral-Nemo 12B), well under the 14.5 GB ceiling. PD6 satisfied: both slots route from `OTR_LedgerScriptWriter`'s existing `creative_writing_model` / `technical_model` sockets; no consumer node gets a `model_id` widget; zero new model slots introduced. Per-character actor agents (Candidate B from `__00_question.md`) are deferred to Sprint 10C if character distinctiveness remains the gap after 10B's listen test -- the 16 GB ceiling forecloses N-resident actor models, and serial persona-swap collapses into single-LLM-three-roles with extra plumbing.

3. **Where does the news-grounding contract live?** Wave 1 Agent A keeps `build_news_briefs` as the contract but replaces the brittle exact-substring `key_term` check with a semantic-presence check (LLM-as-judge OR embedding cosine; the agent picks based on what's cleaner in the serving stack). The contract's *shape* stays so downstream consumers do not change; only its *validator* loosens. Replacing the contract wholesale would force a re-derivation of every consumer's brief assumptions, and the contract itself is fine -- it is only its enforcement that brittles. The Director (Wave 1 Agent D) consumes the brief and converts it into prose direction the Writer can act on.

4. **How does Stage 7 critic enter the conversation?** Two layers, not one. **Layer 1: the Editor (Wave 1 Agent E) is a turn in the writer loop** -- scores each draft against the rubric and feeds typed `EditorVerdict.failing_axes` + per-axis notes back to the Writer for revision; up to 3 Editor cycles, hard-capped. This is the layer that catches Spray of Hope's `premise_clarity FAIL` on draft 1 (see Section 2.5 worked example) and lets the Writer fix it on draft 2 rather than shipping the failure. **Layer 2: Stage 7 critic stays as a post-pipeline gate** driven by Wave 1 Agent C's escalation logic -- structural failing_axes route to whole-episode regenerate, local failing_axes route to beat or line reroll. Two independent measurements with the same rubric, in-loop and end-of-pipeline. This preserves Sprint 10A's auditability instrument while giving the writer a tight-loop signal it lacks today.

5. **What's the minimum-viable demo?** Section 6.2. Three fixed news seeds picked from one morning's RSS pull (frozen URLs), identical Stage 1 plan per seed, **3 runs per cell** per writer, blind operator listen on shuffled order. Rubric in Section 6.2. "Good" = operator-rubric mean ≥ 4.0 AND Stage 7 critic mean ≥ 4.0 on at least 2 of 3 seeds.

6. **Reproducibility contract.** PD7 (byte-identical audio on C7 seed) stays binding when both `use_multiturn_dialogue=False` AND `use_story_room=False` -- the legacy path is the byte-identity contract. Under either flag, PD7 relaxes to **transcript-anchored reproducibility**: re-queueing with the stamped transcript (`meta.dialogue_path='multiturn'` payload, or `meta.story_room.transcript` payload) produces the same audio. The transcript becomes the reproducibility anchor for chat-style runs. Widgets default to `False` so the out-of-the-box behavior is byte-identity.

7. **Backward compatibility.** Sprint 10A's machinery stays. Stage 1 grammar planner stays -- it produces the structured plan both Wave 0 dispatch and Wave 2 Story Room consume. Stage 3 validators get *promoted* from shadow to in-line via Wave 1 Agent B. Stage 7 shadow critic stays for measurement AND becomes the escalation driver via Wave 1 Agent C. The legacy one-shot composer stays callable for `use_multiturn_dialogue=False` AND as the fallback when Wave 0 / Wave 2 paths error. Nothing from 10A retires until Sprint 10C earliest -- 10B is additive across every surface.

---

**End of revised design.** Sprint 10B can dispatch immediately to Wave 0 as one focused commit; then to 5 parallel subagents in Wave 1 with Section 3 dataclasses as their joint reference; then Wave 2 sequentially after Wave 1 lands; then Wave 3 operator-owned (against the Section 6.2 worksheet). The Wave 0 A/B result feeds into Wave 2 scope but does not block Wave 1.
