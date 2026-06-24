# OTR Story Architecture -- Research Kickoff (REV 2, grounded 2026-06-23)

**Mission:** get OTR stories to A+ (or as close as the local-model ceiling allows). The
schema/plumbing is done and good. The lever now is **structural variety at the PREMISE level +
closing the critic->planner loop**, not more line-level gates.

Open a fresh window on this file and run it through `/roundtable` (grounded against the real
files in S1). Produce a hardened spec at `docs/2026-06-23-story-architecture/SPEC.md`, then a
coder kickoff. **Do not write production code in the research window** -- plan + roundtable only.

---

## REV 2 grounding correction (2026-06-23) -- read this first

REV 1 of this kickoff was written from module headers, not from the wiring. A read-only
ground-check (Desktop Commander over the real Windows repo) found that **two of REV 1's three
"genuinely-missing" moves already exist in code and are wired.** Sending the panel REV 1 would
have had it design subsystems we already ship -- the exact CLAUDE.md S0 anti-pattern. The
corrections, with evidence:

- **A whole-script story-quality critic EXISTS and runs every episode.** `nodes/_otr_story_critic.py`
  (`run_story_critic`, `StoryCriticReport`: arc_verdict / flat_lines / reroll_targets /
  continuity_issues / render_priority, 6-section rubric). It is called **unconditionally** on the
  non-terminal path of the live freeze cascade (`nodes/_otr_freeze_cascade.py`, "Sprint 5B"), so
  the claim "no pass judges story QUALITY" is false.
- **A targeted reroll + escalation loop EXISTS.** 5B is advisory; **5C acts**: `run_targeted_reroll`
  (`nodes/_otr_reroll.py`) re-composes every line the critic named, threading the critic hint in as
  a hard REVISE instruction, capped at MAX_REROLL_CYCLES, re-running the critic each cycle. An
  **escalation router** (`nodes/_otr_reroll_escalation.py` `decide_escalation_scope`) sends
  STRUCTURAL failures (premise_clarity / continuity / resolution / emotional_arc) straight to
  whole-episode regenerate and only routes LOCAL failures to line reroll. An unconditional
  **anti-loop mechanical floor** (`nodes/_otr_anti_loop.py`, "A3") unions deterministic
  near-duplicate / "What if...?" targets in even on a silent critic failure.
- **A scene-level (grouped-exchange) composer EXISTS and is wired -- but dark.**
  `nodes/_otr_compose_exchange.py` (`compose_exchange`) renders a 2-3 voiced-beat group as one
  exchange with prior committed lines in context, repair-by-group, legacy fallback. It is wired to
  the live writer behind the `use_exchange` BOOLEAN (`nodes/OTR_LedgerScriptWriter.py`, INPUT_TYPES
  ~L1838, **default False**) and is present as an input on writer node id 1 in the canonical
  `workflows/otr_scifi_16gb_full.json` (link null -> runs at the False default). Its docstring still
  says "DRAFT / not imported" -- that is STALE; it is imported and wired.

**Net:** the quality apparatus is already rich. What is genuinely absent is **premise-level
divergence + a taste selector**, and a **loop that lets the critic's structural verdict drive a
*different* plan** (today the structural path regenerates the same structurally-samey planner).
That, plus the local-model ceiling and a non-monotonic refine loop, is the real A+ gap.

---

## 0. The core reframe (corrected)

The old reframe ("every layer only subtracts badness; nothing judges quality") is wrong -- see the
REV 2 banner. The accurate reframe is sharper:

**The pipeline already detects bad stories. It cannot manufacture good ones, because the one thing
it cannot vary is the premise, and the critic's structural verdict has no good plan to escalate
to.** The 5B critic correctly flags an uneven arc; 5C reroll then burns its cycles trying to fix an
ARC problem by rewriting LINES (it can't); Wave-1C correctly routes the structural failure to
whole-episode regenerate -- but regenerate re-runs the **same beat-planner**, which produces the
**same shape**. The loop is closed on detection and open on cure.

Tonight's soak is consistent with this: passes grade **42-72, almost all below B (75)**, and
revision is **non-monotonic** (a live gemma episode went 42 -> 72 -> drifted back to 65; see
`docs/2026-06-23-refine-loop/` + `docs/2026-06-23-multipass-refine/`). Keep-best masks but does not
fix non-monotonicity.

Two independent external reviews AND Jeffrey's own multi-model roundtables (2026-06-22/23)
triangulate on the **same root cause: the beat planner / structural sameness** ("every premise
collapses into a console standoff; climax off-stage; announcer narrates the outcome"). That
triangulation is the strongest signal in this effort -- and it points upstream of everything the
existing critic/reroll apparatus can touch. Act on it there.

---

## 1. What OTR ALREADY has (do NOT rebuild -- build ON these)

Ground every recommendation against these real files (Windows repo
`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`):

**Generation spine**

- `nodes/news_interpreter.py` -- one control-plane LLM call -> 4 briefs (casting_brief,
  **script_brief = premise arc + central tension**, news_close_brief, key_terms). Premise-first
  already. (The "extrapolate to dramatic extremes" spine framing lives in
  `nodes/story_orchestrator.py` LLMScriptWriter.)
- `nodes/_otr_outline.py` -- grammar-validated `Beat[]` hierarchy, **no dialogue yet** (macro
  premise/setting -> phase/speaker -> beat intent/mood). This IS Dramatron's hierarchy.
- `nodes/_otr_line_composer.py` -- **legacy default path**: one plain-prose line per beat (last N
  ledger lines as context); Python attaches the `[VOICE:]` tag at stamp time (`compose_line`). This
  is the line-at-a-time surface that `use_exchange` replaces when ON.
- `nodes/_otr_compose_exchange.py` -- **scene-ish grouped-exchange composer** (2-3 voiced beats as
  one exchange, prior lines in context, repair-by-group). WIRED behind `use_exchange` (default OFF).
- `nodes/OTR_LedgerScriptWriter.py` -- the conductor. Holds the `use_exchange` flag (INPUT_TYPES
  ~L1838) and the `_refine_loop` (v1 2026-06-23, **keep-best** over up to 5 passes; ships the
  highest-grade pass; ~L2050).

**Selection + quality apparatus (the part REV 1 missed)**

- `nodes/_otr_story_select.py` -- `score_outline` (pure, deterministic **best-of-N OUTLINE**
  selector over FRESH-generated structures; metric-level divergence, not taste), `grade_story`
  (0-100 structural grade of composed text), `resolve_refine_passes`.
- `nodes/_otr_story_critic.py` -- **whole-script dramatic-quality critic** (`run_story_critic` ->
  `StoryCriticReport`: arc_verdict, flat_lines, reroll_targets, continuity_issues, render_priority;
  6-section rubric; never raises).
- `nodes/_otr_reroll.py` + `nodes/_otr_reroll_escalation.py` + `nodes/_otr_anti_loop.py` --
  **5C targeted reroll** (re-compose flagged lines with the critic hint), **escalation router**
  (structural -> whole-episode regen; local -> line reroll; ship -> skip), **A3 mechanical
  anti-loop floor** (unconditional deterministic targets).
- `nodes/_otr_freeze_cascade.py` -- the live orchestrator that runs 5B -> A3 -> 5C -> Wave-1C in
  order every episode.
- `nodes/_otr_story_quality_l12.py` -- conflict palette / beat_role dramatic-function sequence.

**Plus:** `_otr_craft_floor`, `_otr_creative_qa`, `_otr_line_hygiene` (cliche / on-the-nose /
thesis-close / stage-business flags), `_otr_dramatic_state` (+ `_otr_dramatic_state_llm`), series
bible in `project_state.py`, constrained generation (`_otr_constrained_generate` / `_otr_lmfe_compat`),
the freeze cascade, SDH captions.

**So:** the hierarchy, the reviewer, a real dramatic-quality critic, a targeted-reroll +
structural-escalation loop, best-of-N outline selection, keep-best refine, and the QA gates ALL
exist. The gaps in S2 are what is genuinely still missing.

## 2. The genuinely-missing lever (the grade move)

1. **Premise-level divergence + a TASTE selector (the primary new lever -- attacks the triangulated
   root cause).** Best-of-N (`score_outline`) already diverges, but over OUTLINE STRUCTURES selected
   by a deterministic METRIC -- it cannot escape structural sameness because all N candidates answer
   the same premise the same way. Add a **pitch room upstream of the outline**: one news/script_brief
   -> 3 radically different episode takes (different genre, protagonist, emotional core), then a
   **showrunner taste pass** kills all but one. Greatness lives in premise variance; a real room
   pitches ten and shoots one. Feed the winner into the existing outline + best-of-N machinery.
2. **Close the critic -> planner loop.** Today Wave-1C correctly sends a STRUCTURAL critic verdict
   to whole-episode regenerate, but regenerate re-runs the same planner -> same shape (S0). Re-point
   the structural-failure branch at the **pitch room / a divergent re-plan** (a different premise or
   beat shape), not a same-seed regenerate. This makes the critic the apparatus already has actually
   able to buy a better story instead of spinning cycles.
3. **Flip on the scene-level composer (cheap, already built).** `use_exchange` is wired and dark.
   Validate it on a live N=3 (VRAM <= 14.5 GB, zero slot drift) and flip it ON in the canonical
   workflow JSON. Line-at-a-time produces locally-coherent ping-pong with no scene shape; grouped
   exchange already fixes part of that. **Follow-on (higher risk, sequence last):** extend
   beat-group -> whole-scene free prose, then a mechanical transcribe-to-ledger pass (the prose ->
   beats speaker-attribution parser is the hard part).
4. **Upgrade the critic from defect-auditor toward listener-taste.** The 5B critic emits typed
   verdicts; add a listener-voice signal ("where did I lean in / check out, quote the best line,
   the 3 flattest"). Lower-leverage than 1-3 -- do it after.

## 2A. Two axes, and the full option table (nothing off the table)

Two independent axes make a story better. Most proposals move on only one; the best plan moves on
both. Naming both is how we keep nothing off the table.

**Axis A -- WHAT story (premise / divergence).** Fixes SAMENESS (the triangulated root cause).
Nothing in the repo does this yet.

- A1 Pitch room: 3 different episodes -> taste-select one.
- A2 Theme + ending first; write toward it.
- A3 Write-long-then-compress: over-generate, select down to the audio budget.

**Axis B -- HOW it is written (surface granularity).** Fixes FLAT SHAPE (no build / turn / subtext).
A ladder; OTR sits on rung 1:

- B1 line-by-line (`_otr_line_composer`, today's default).
- B2 grouped exchange, 2-3 beats (`_otr_compose_exchange` / `use_exchange`, BUILT, default OFF) --
  flip after a live N=3.
- B3 whole-SCENE free prose -> transcribe to ledger.
- B4 whole-EPISODE / long-arc free prose -> transcribe (the operator proposal: "write the whole
  story first, then have the LLM parse it down to the ledger bits, instead of line by line").

**On B4 (write-whole-then-transcribe) -- honest read:**

- Why it is likely right: LLMs hold arc / setup-payoff / escalation / subtext far better across one
  draft than when filling isolated slots; it is how Dramatron / Open-Theatre and real writers'
  rooms work; it directly fixes "climax off-stage, announcer narrates the outcome" because a model
  writing the whole thing stages its own climax. It may also let much of the line-level repair
  machinery retire.
- Why it is NOT a silver bullet: (1) it fixes EXECUTION, not PREMISE -- a whole-story draft of a
  console standoff is still a console standoff, so B4 PAIRS WITH Axis A, it does not replace it;
  (2) the **prose -> ledger extraction parser is the whole ballgame** (per-line speaker attribution,
  beat segmentation, cast match to the voice bank, `[VOICE:]` tags) and EVERY downstream stage
  (freeze cascade, per-line voices, video shot roles, captions) breaks on one misattributed line --
  design it first, fail loud; (3) the local-model ceiling still caps prose quality (a B draft
  transcribes to a B ledger) -> the frontier-lane decision.
- Net: B4 is probably the right long-term SPINE. Climb B2 -> B3 -> B4 behind the parser; keep B1
  intact until B4 is proven.

**Other better-story levers (to complete the table):**

- Close the critic -> planner loop (the existing critic's STRUCTURAL verdict drives a DIFFERENT
  plan, not a same-seed regen).
- Distinct character voices (character interviews / per-character voice-print) -- fixes "everyone
  sounds alike".
- Every beat must TURN (someone gains / loses power) -- fixes the flat middle.
- Stage the climax ON-MIC (a beat-role rule) -- fixes the named "climax off-stage" symptom.
- Frontier OpenRouter lane for the draft -- raises the prose ceiling (the accept-B-vs-frontier call).
- Fix non-monotonic refine (42 -> 72 -> 65) so revision stops drifting backward.

The roundtable's R1 job is to rank / sequence this table. The two decisions that shape everything
else: the **B-ladder rung** (stay B1 / flip B2 / commit to B4) and the **local-model ceiling**
(accept B vs wire the frontier lane).

## 3. Candidate clean room (flip = exists, just enable; new = build; rest exists) -- with seeds

1. Assignment desk (`news_interpreter`) -> **THREE candidate seeds, not one (new)** ("surface the 3
   headlines that could become a great half-hour drama; the real fact + the human question").
2. **Pitch room -- DIVERGE (new)** ("pitch 3 completely different episodes from this seed -- different
   genre, protagonist, emotional core; 4 sentences each; make them fight for the slot").
3. **Greenlight -- SELECT (new)** ("showrunner with taste and no budget: pick the ONE that makes a
   listener sit in their parked car; one line why; kill the rest"). Winner -> existing outline path.
4. **Theme & ending first (new)** ("what is this really about, one human sentence? now write the
   last 20 seconds") -- write toward it.
5. Character interviews (augments `_otr_casting`) ("you ARE this character -- 3 lines in your
   speaking voice: what you want tonight, what you fear, what you'd never admit").
6. Beat the story (`_otr_outline` + `score_outline` best-of-N, keep grammar validation) -- add:
   **require every beat to change the temperature** (someone gains/loses power); a flat beat is cut.
7. Compose (`use_exchange` **flip** -> already grouped-exchange; **new** follow-on = whole-scene
   free prose then transcribe).
8. Critique + act (`_otr_story_critic` 5B + `_otr_reroll` 5C **exist**) -- **new**: route the
   STRUCTURAL verdict to the pitch room (step 2), not a same-seed regen; **augment** the rubric with
   listener-taste.
9. Transcribe / stamp to ledger (existing freeze-cascade stamp) -- mechanical, constrained.

## 4. External repos to MINE (prompts/patterns, not whole clones)

- **`johnnie193/Open-Theatre`** (interactive drama) -- director/actor agent modes, character memory,
  plot-chain tracking, real prompt files (`prompt_drama_v1_eng.md`, `prompt_drama_v2_eng.md`,
  director reflection, player/character prompts). **Highest-value lift** for the beat-planner /
  pitch-room upgrade + "who acts next" scene-state. Adapt the prompts; don't import the framework.
- **`XucroYuri/how-to-make-script`** (MIT) -- rubric / route->generate->review->self-check + sub-skills.
  Runtime not implemented, so treat as a **rubric library** to upgrade the critic's structural
  verdict into listener-taste guidance (S2 move 4).
- **`google-deepmind/dramatron`** (Apache-2.0) -- canonical hierarchy (logline->characters->plot->
  locations->dialogue). OTR already has this; reference only for the character/location layer.
- `stefanfrench/radio-drama-generator` (Apache-2.0) -- closest name, **least useful** (POC behind
  OTR); steal only doc->transcript->TTS wiring ideas. License caveat: OuteTTS v1 CC-BY-4.0, newer
  CC-BY-NC-4.0 -- matters for commercial-clean.
- `mozilla-ai/document-to-podcast`, `lfnovo/podcast-creator`/`open-notebook` -- explainer-audio, not
  drama; only transcript/TTS plumbing is relevant (OTR already exceeds it).

## 5. Hard constraints + honest caveats (the new window must respect these)

- **Do NOT rebuild the quality apparatus.** `_otr_story_critic`, `_otr_reroll(_escalation)`,
  `_otr_anti_loop`, `_otr_compose_exchange` exist and are wired -- the work is to **close the loop**
  (critic structural verdict -> divergent re-plan) and **add the pitch room upstream**, building ON
  these. Any proposal to "add a story critic" or "build a scene composer" is REV 1 drift -- reject it.
- **Local-model ceiling is real.** gemma-12b / mistral-nemo top out ~B even unconstrained (tonight's
  soak). Diverge-select raises quality by SELECTION -- the pool must contain a good one -- so pair the
  pitch room with the **frontier OpenRouter lane** (already built, opt-in, cost-guarded) OR accept B
  as the local ceiling. **Operator decision required before betting the campaign on local.**
- **Non-monotonic refine is a known defect** (42 -> 72 -> 65). Keep-best ships the peak but the loop
  still wanders; the SPEC should decide whether to harden the refine loop or rely on best-of-N +
  pitch-room divergence instead.
- **`use_exchange` flip is gated on a LIVE metric** (VRAM <= 14.5 GB, zero slot drift, N=3) that
  cannot be measured headless -- it needs a GPU run. Sequence it as a quick win but treat the
  whole-scene-free-prose EXTENSION as the high-risk item (prose->beats speaker-attribution parser;
  the whole downstream -- freeze cascade, per-line voices, video shot roles -- needs clean per-beat
  speaker tags). Extension goes LAST, behind a flag, line-at-a-time path intact until proven.
- **No-fallbacks pipeline + the working A/V are sacred.** The render (ltx_av bookends + visualizer
  beats, HuMo-free as of `267a53e`) and the audio voice bank ship today -- story work is upstream of
  the ledger; keep it there.
- **Ledger schema** = `nodes/production_ledger.py` (`Ledger`, `new_ledger()`); version source in
  `_otr_ledger.py`; section schemas in `_otr_{image,video}_engines/schemas.py`. Transcription writes
  into this; don't fork it.
- UTF-8 no BOM, ASCII where practical, SFW, commercial-clean licensing, default-OFF / byte-identical
  until proven (every story change ships dark behind a flag, like the refine loop + use_exchange did).

## 6. Deliverable

Run `/roundtable` (GPT + Gemini + DeepSeek/Grok per CLAUDE.md S8; Claude grounded panelist + sole
judge) over this brief, **grounded against S1's real files so the panel cannot propose rebuilding
the existing critic / reroll / exchange apparatus.** Output:
`docs/2026-06-23-story-architecture/SPEC.md` -- the hardened clean room + the 2-3 highest-leverage
sprints, sequenced by risk:

1. **Pitch-room divergence + taste select, and route the critic's structural verdict to it** (the
   primary lever; cheap; attacks the triangulated root cause).
2. **Validate + flip `use_exchange`** (already built; quick win; needs one GPU N=3 run).
3. **Climb the B-ladder to whole-story prose** (B3 whole-scene -> B4 whole-EPISODE free prose, then
   the prose->ledger transcription parser). The **parser is the make-or-break** -- clean per-line
   speaker attribution; all downstream audio/video depends on it. R1 picks the rung; keep
   line-by-line intact until proven.

Then a coder kickoff. **Confirm the local-ceiling decision (frontier lane vs accept-B) with the
operator before committing the campaign** -- it determines whether the pitch-room pool can contain
an A.
