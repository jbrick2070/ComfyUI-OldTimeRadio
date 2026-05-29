# Round-Robin Question — How Should OTR's Writer Pipeline Be Architected So Episodes Tell a Good Story?

**Date:** 2026-05-26
**Component:** the entire writer pipeline (`nodes/_otr_outline.py`, `nodes/_otr_line_composer.py`, `nodes/_otr_casting.py`, `nodes/_otr_story_brief.py`, `nodes/script_critic.py`, `nodes/_otr_reroll.py`, `nodes/_otr_announcer_pass.py`, `nodes/_otr_continuity.py`, `nodes/_otr_news_interpreter.py`)
**Round-robin gate:** yes — synthesis required before any architectural sprint scopes
**Status:** Claude-drafted seed question. ChatGPT and Gemini passes to follow.

---

## 1. The triggering observation

Two consecutive end-to-end runs on 2026-05-25 produced episodes the writer system itself self-diagnosed as "uneven" or "needs_full_rerun." Specific failure modes from the second run (`pending_20260525_223109`, episode "Solar Standby"):

- **`build_news_briefs` exhausted 3-attempt retry ladder** — Mistral-Nemo 12B could not produce schema-compliant JSON with valid `key_term` matches against source text at temperatures 0.7 → 0.35 → 0.1.
- **`run_story_brief_reflection` exhausted 3-attempt retry ladder** — model emitted `no decodable top-level JSON object found: line 1 column 1 (char 0)` at every temperature including the typed-repair pass.
- **Cast naming → gender inversion (recurrence).** COLE BOUVIER got `gender=female + voice=v2/en_speaker_4`; MIRA BOUVIER got `gender=male + voice=v2/en_speaker_3`. Cole reads strongly male in English; Mira reads strongly female. The Sprint 9 CHARACTER VISUAL CONTRACT (HEAD `4efd87e`) either did not fire in this Python process OR did not pin name→pronoun consistency. Previous run had the same class of bug (Reginald = female).
- **`LineComposer` length drift on every other call** — repeated `length drift: 11 words outside band [13..44] for target=26` then `shipping drifty line on final attempt 2` — the model is consistently undershooting target word counts.
- **Story critic verdict: `uneven`, 3 reroll targets, 3 flat lines, 1 continuity issue, 1 voice-drift note.**
- **Reroll loop got WORSE on cycle 2** — cycle 1 reduced flat lines from 3 → then critic re-ran and found **6 flat lines** plus 5 new reroll targets. Cycle 2 ran again, still uneven. System gave up: `restoring pre-reroll lines and stamping needs_full_rerun`.
- **Announcer intro AND outro both fell back to deterministic template** — empty `script_brief` (because brief reflection failed) starved the announcer-pass of context.
- **MusicGen confirmed Sprint 8.1 fallback path** — `story_brief_status=failed mood_terms=[] mood_source=v1_atmosphere_vocab`. The brief-consumer wiring (Sprints 8.1-8.7, HEAD `4efd87e`) IS firing structurally, but the brief itself never populated, so every consumer is on the v1 fallback path.

Compare against the previous run (`bioluminescent_trench_descent_20260525_182002`) which had:
- Geographic continuity break (announcer said "Indian Ocean", b004 said "Pacific")
- Premise/scene mismatch (cast = fisheries diplomats; scene = submarine emergency)
- Physics error ("ballast compensators reading zero Kelvin")
- Same gender inversion class (Reginald = female)
- Purple/abstract dialogue ("forfeit to the void", "expected toll the Indian Ocean will claim")

Two runs, two stories the operator describes as "not a good story." This is no longer a one-off — it's the writer pipeline's steady state.

---

## 2. What's already locked

Operator constraints — these are not on the table:

- **Audio is king (Prime Directive 1).** Any architectural change preserves byte-identical audio against the current baseline OR re-blesses the baseline through round-robin.
- **100% local, open-source, offline-first.** Default writer model must run on the operator's 16GB RTX 5080 Laptop without external API calls. **Exception currently up for debate:** see Candidate D below — a hybrid that routes high-leverage JSON-emitting passes to an API LLM while keeping bulk dialogue local. The operator named this as the unspoken option after the second failed run.
- **Two-model selector pattern (locked Sprint B).** `OTR_LedgerScriptWriter` exposes `creative_writing_model` and `technical_model` slots; every other node receives the model ID via socket. Any new node respects this.
- **PD4 — SFW, good narrative arc (beginning, middle, end).** "Good narrative arc" is the bar that's currently failing.
- **L3 ledger contract.** Cast / lines / meta / continuity_ledger / episode_canon fields stay; any new fields are additive.
- **Existing critic + reroll loop** is part of the architecture. The question isn't whether to have a critic — it's whether the dialogue generation surface upstream of the critic is correctly shaped.
- **Writer-phase wall-clock budget = unbounded (operator directive, 2026-05-26).** *"I do not care if it takes an extra hour for a good story, we need a good story."* Story quality is the ONLY optimization for the writer phase. Solutions that add LLM-call overhead (e.g., A2's 4× multi-turn priming, B's actor-agent state-update calls, D's API round-trip latency) are not penalized on the cost axis. Render-phase wall-clock (HuMo, LTX, FFmpeg mux, upscale) is unchanged — this directive applies to the writer pipeline only.

---

## 3. What the literature says (May 2026 state)

Quick survey of current academic + community work on character-driven LLM dialogue:

- **Rule-Based Role Prompting (RRP)** — arxiv 2509.00482 (Sep 2025). Character-card / scene-contract design with strict enforcement of function calling outperforms pure prompt-engineering approaches for tool-augmented dialogue agents.
- **RPGAgent** — CHI 2026 (Proceedings of the 2026 CHI Conference). Multi-agent LLM system specifically for "story-to-play generation." Encodes temporal-spatial context per scene: primary objective, scene type, key objects, locations, main characters, associated dialogues. Closest published architecture to what OTR is trying to do.
- **SNAP** — Plan-Driven Framework for Controllable Interactive Narrative Generation. Structures narratives into "Cells" with explicit "Plans" to prevent narrative drift. Per-cell context confinement.
- **HAMLET / LLMR** — Multi-agent orchestration with specialized agents (planner, scene analyzer, inspector, director, actors). Each actor agent maintains individualized state: persona, emotion, goals, episodic memory.
- **Persona-Infused Conditioning** — Big-Five personality vectors injected explicitly into prompts. Steers LLM persona expression through trait-level conditioning rather than free-text descriptions.
- **Dramaturge** — arxiv 2510.05188v3. Divide-and-conquer iterative narrative script refinement via collaborative LLM agents. Specialized critic agents (continuity / character / pacing) instead of one monolithic critic.

**Convergent pattern across all current work:** per-character ACTOR AGENT with persistent state (persona + emotional arc + episodic memory), driven by a DIRECTOR AGENT that decides scene-level intent, with specialized CRITIC AGENTS for refinement. OTR's current "one structured call per beat to generate one line in isolation" pattern is NOT what the field considers SOTA in May 2026.

---

## 4. Candidate architectures

Five candidates, narrowest-change to broadest-change. Not mutually exclusive — D explicitly combines B and one of A1/A2 and C.

### Candidate A1 — Status quo + single-prompt role-play (smallest change)

Keep current architecture, rewrite `OTR_LineComposer` prompts to be conversational / role-play framed instead of structured-emission framed. One LLM call per dialogue line, all context delivered in a single system+user prompt pair.

**Concrete shape:**
```
SYSTEM: You are COLE BOUVIER. You're a warm, level-headed person trying
to figure out what's happening during a power outage in your small town.

USER: The grid just went down. Previous exchanges:
  COLE: "Did you hear that pop from the substation?"
  MIRA: "Probably the transformer. Third one this month."

The next beat should reveal that something is wrong beyond the usual.
Speak Cole's next line, in character. Just the line.
```

Output is free-form text; post-processing extracts the spoken line. Length is approximate, not enforced.

**Pros:** smallest change, matches Mistral-Nemo's training (heavy on conversational data), persona-name pulls on pretrained associations (may fix Cole-as-female bug naturally), length cadence emerges from character rather than being fought, 1× LLM call per line (no VRAM/wall-clock overhead vs current).

**Cons:** harder to validate programmatically (no JSON schema), small risk of meta-commentary leaks (`"As Cole, I would say..."`), no persistent character state across calls, single-prompt may not give the model enough "warm-up" to fully inhabit the character, doesn't fix brief reflection / news interpreter failures (those still need JSON).

### Candidate A2 — Multi-turn conversational priming with acknowledgment chain (operator's refined instinct)

Same role-play paradigm as A1, but the model is walked through context in MULTIPLE turns with explicit acknowledgment between each, then asked for the dialogue line. The model's "yes" or equivalent confirmations are validated; only the FINAL turn's output (the actual dialogue line) is committed to the ledger.

**Concrete shape (per dialogue line, 4 turns):**

```
Turn 1:
  SYSTEM: You're an actor in a radio drama. You're going to play COLE BOUVIER.
          First I'll give you the story premise. Confirm you understand by replying "Yes."
  USER:   The premise: a power outage hits a small town, and as the night unfolds,
          siblings Cole and Mira discover the outage is hiding something worse than
          a transformer fault. Did you get that?
  ASSISTANT (expected): Yes.

Turn 2:
  USER:   Here's the overall arc. Setup: the lights go out, Cole and Mira start
          comparing notes. Complication: they realize neighbors are gone, the
          radio is broadcasting old emergency tapes on loop. Resolution: they
          decide whether to leave or wait it out. Did you get that?
  ASSISTANT (expected): Yes.

Turn 3:
  USER:   You are COLE: warm, level-headed, the older sibling. You don't panic
          easily but you notice things. You're outside the house with Mira (younger,
          sharper, more anxious). It's been 20 minutes since the grid went down.
          Did you get that?
  ASSISTANT (expected): Yes.

Turn 4:
  USER:   MIRA just said: "Probably the transformer. Third one this month."
          Now it's your turn. As Cole, what do you say next? Just the line, in
          character. This is the line we'll use.
  ASSISTANT: [DIALOGUE LINE — this is what lands in the ledger]
```

The first three turns "coax" the model into the role; the fourth elicits the actual ledger-bound output. Acknowledgments can be validated — if the model says something other than "yes" / "got it" / "understood," the chain catches the drift before the dialogue line is even attempted.

**Pros:** explicit context-loading through dialogue (the model's native conversation pattern), each acknowledgment FORCES the model to actively process that context (vs passive reading in a long single prompt), final dialogue prompt is short and focused because all context lives in conversation history, acknowledgments are validatable (catch drift early), chatbot-native = best match for Mistral-Nemo's training, character coherence likely strongest of A1 / A2 (model has "warmed up" before speaking).

**Cons:** ~~4× LLM call overhead~~ — **OPERATOR-CONFIRMED ACCEPTABLE per §2's writer-phase wall-clock directive ("I do not care if it takes an extra hour for a good story"). Cost axis is removed from the A1-vs-A2 evaluation; A2 is now ranked on quality alone.** Remaining cons: more VRAM pressure per call (longer conversation context). Risk that model breaks character earlier in the chain (Turn 2 or 3 instead of just at Turn 4). Validation of "yes" responses needs robust fuzzy-matching (model might say "Sure" / "Understood" / "Got it"). The acknowledgment turns produce no creative output by design — they only context-load.

**Operator framing (verbatim, 2026-05-26):** *"sort of like a multi-prompt conversation to coax the right dialogue."*

### Candidate B — Per-character actor agents (RRP / RPGAgent-style)

Replace per-beat `OTR_LineComposer` with per-character `OTR_ActorAgent` instances. Each actor maintains state across the episode: persona card, emotional arc, episodic memory of every line said + key facts learned. New `OTR_DirectorAgent` decides whose turn + scene-level dramatic intent. Critic remains a structured-output pass (judging IS structural).

**Pros:** matches 2026 SOTA pattern (RPGAgent, HAMLET), per-character memory eliminates voice-drift, persona cards explicitly carry name→pronoun consistency, scene contracts (à la SNAP) prevent geographic/premise drift, dialogue becomes character-driven rather than beat-driven.

**Cons:** substantial architectural rewrite (multiple new nodes, new ledger fields for actor state), more LLM calls per episode (state-update overhead), still bottlenecked by Mistral-Nemo's intrinsic quality on each call.

### Candidate C — API LLM swap for high-leverage passes

Keep current architecture; route the JSON-emitting structured passes that Mistral-Nemo is failing on to an API LLM (Claude / GPT-4 / Gemini Pro). Specifically: `build_news_briefs`, `run_story_brief_reflection`, `run_story_critic`. Keep dialogue generation (`OTR_LineComposer`) local.

**Pros:** API LLMs produce parseable JSON reliably at temp=0.3, catch gender/name consistency naturally, cost is ~$0.01/episode for these passes, smallest behavioral surface change (same node graph, different model behind specific calls), unblocks the Sprint 8.1-8.7 brief-consumer wiring which currently has no v2 brief to consume.

**Cons:** breaks the 100%-local rule (the operator named this as a real cost), creates a network dependency, adds API key management, doesn't address dialogue quality directly.

### Candidate D — Hybrid: actor agents on local, API LLM on JSON gates

Candidate B's per-character actor agents (role-play prompting, local Mistral-Nemo) PLUS Candidate C's API-LLM routing for the high-leverage JSON passes that are currently failing. Each piece does what it's best at.

**Pros:** every pass uses the model that fits its task — local for creative dialogue (which Mistral-Nemo can do well in role-play mode), API for structured judgment (which API LLMs do reliably). Matches 2026 SOTA on the dialogue side AND fixes tonight's JSON-validation failures. Most expensive but most likely to produce "a good story."

**Cons:** biggest scope. Likely 2-3 sprints. Requires the operator to formally accept the API-LLM exception to the 100%-local rule.

---

## 5. Specific questions for outside consultants

For ChatGPT and Gemini, please address each numbered question explicitly with a rank-ordered recommendation:

1. **Architecture ranking.** Given the constraints in §2 and evidence in §1, rank A1 / A2 / B / C / D in order of expected wow-people output quality. State the deciding factor for your ranking.

2. **A1 vs A2 specifically.** Per §2's writer-phase wall-clock directive, the 4× LLM-call overhead of A2 is operator-accepted (not a cost). Ranked on QUALITY ALONE, does A2 produce measurably better dialogue than A1, or is the acknowledgment chain mostly theatre? Cite published evidence if any. If you have a strong A1-or-A2 prior independent of OTR's specifics, name it.

3. **Local-LLM ceiling assessment.** Is Mistral-Nemo 12B at its intrinsic ceiling for OTR's writing task, or is the failure mode primarily a prompt-architecture problem (A1 or A2 would fix it)? Cite specific evidence from §1 that informs your answer.

4. **API LLM exception — should it be made?** The operator's "100% local" rule is a stated value, not a hard constraint. Given that tonight's `run_story_brief_reflection` failed to produce ANY valid JSON across 3 attempts at 3 temperatures, is the rule still load-bearing or has it become a self-imposed quality ceiling?

5. **Per-character actor agents — necessary or premium polish?** Candidate B is the largest architectural change. Is it required for "a good story" or is it an overengineering risk for a single-operator pipeline? Specifically — could (A1 or A2) + Candidate C (smaller cost, no actor agents) achieve the same wow factor?

6. **Gender-inversion bug — fix path?** Cole-as-female and Mira-as-male have now recurred across two consecutive runs. Is this best fixed by (a) explicit name→pronoun validation in the casting prompt, (b) persona cards in actor agents, (c) post-hoc pronoun audit pass, (d) acknowledgment chain in A2 catches it via Turn 3 ("You are COLE: warm..." — model self-corrects), or (e) something else?

7. **Critic + reroll loop — is it doing more harm than good?** Tonight's reroll cycle 1 made the script WORSE (3 flat lines → 6 flat lines). Should the reroll loop be removed, bounded harder, or replaced with a per-character regeneration pass (B/D)?

8. **Smallest unit of work that produces a measurable quality jump.** If the operator can only ship ONE change before the next round-robin, which single change moves "good story" the most?

---

## 6. Claude's seed recommendation (best guess pending consultation)

**Recommend: Candidate D (Hybrid) with A2 as the role-play variant, sequenced rollout.**

Reasoning:

- **Single-prompt role-play (A1) is necessary but probably not sufficient.** Role-play prompting will measurably improve dialogue line quality on Mistral-Nemo — that's well-supported by both the literature (RRP) and Mistral-Nemo's known strengths. But a single-prompt approach asks the model to absorb premise + arc + character + scene + previous-line ALL AT ONCE before producing a line. On a 12B model, that's a lot of context-pressure.

- **Multi-turn conversational priming (A2, operator's instinct) is the sharper version.** Each acknowledgment turn forces ACTIVE processing of one chunk of context. The model "warms up" into character across turns 1-3 before being asked to speak in turn 4. This matches Mistral-Nemo's chat-tuning even better than A1, AND the acknowledgment chain itself is validatable — if the model says something other than "yes" at turn 2, you've caught drift before any dialogue is generated. **The 4× call overhead is operator-accepted per the writer-phase wall-clock directive (§2): "I do not care if it takes an extra hour for a good story."** A2 is now ranked on quality alone, and on quality alone A2 has structural advantages over A1 that map cleanly to known weaknesses in Mistral-Nemo's behavior on tonight's runs.

- **Candidate C alone leaves dialogue quality on the table.** API LLMs would unblock the JSON gates, but if `OTR_LineComposer` keeps fighting Mistral-Nemo with structured prompts, the dialogue will still feel flat. The brief-consumer pipeline could be perfectly wired and the rendered episode would still be "purple monologues" because the line generation is the wrong shape.

- **Candidate B is the right shape but the wrong scope for v2.** Per-character actor agents with persistent state across calls is where the field is in 2026 — but it's a 2-3 sprint architectural rewrite. Notably, A2 already provides "per-character state" *within a single line composition* via the conversation history; B extends this to *across the whole episode* via persistent actor objects. A2 may be a good 80% solution that defers B until measured.

- **Candidate D phased = realistic.** Phase 1 (small): A2 multi-turn role-play in `OTR_LineComposer` + Candidate C API LLM on the three failing JSON passes. Ships v2. Phase 2 (post-v2.1): Candidate B per-character actor agents if Phase 1 still leaves quality on the table. This sequencing means the operator never ships a worse pipeline mid-flight, and each phase is independently validatable.

**Sequencing proposal:**

1. **Sprint 10A (Phase 1, ships v2):**
   - Rewrite `OTR_LineComposer` to use A2 multi-turn role-play prompts on local LLM (premise → arc → character → previous-line → dialogue, with acknowledgment validation between each turn).
   - Route `build_news_briefs`, `run_story_brief_reflection`, `run_story_critic` to API LLM via a new `OTR_APILLMClient` node (Candidate C). Operator approves API exception explicitly for these three passes.
   - Add explicit name→pronoun validator in casting prompt (addresses gender-inversion bug at lowest cost — though A2's Turn 3 character-card may catch it intrinsically).
   - Bound critic+reroll to 1 cycle max (current 2-cycle loop made things worse in tonight's run).
   - Acceptance: end-to-end run produces a populated v2 story brief, dialogue passes critic on first attempt without reroll, gender matches name connotation, A2 acknowledgment turns succeed >95% on first attempt, output is byte-identical against post-fix baseline.

2. **Sprint 10A-LAB (parallel A1 vs A2 measurement):**
   - Before committing A2 to production, lab-isolate both in `otr-tts-lab`: render 3 episodes via A1, 3 via A2 on identical outlines, operator listens blind, picks the winner. If A1 is "good enough," ship A1 — fewer LLM calls, simpler validation. If A2 is meaningfully better, ship A2 and accept the overhead.

3. **Sprint 10B (Phase 2, v2.1 candidate):**
   - Per-character `OTR_ActorAgent` with EPISODE-PERSISTENT state (Candidate B). Big sprint, lab-isolate first per existing `otr-tts-lab` rule. Only opens if Phase 1 still leaves quality short of ship bar.
   - Director + critic agent split (per HAMLET pattern).

---

## 7. What this does NOT decide

- **Which API LLM** (Claude / GPT-4 / Gemini 2.5 Pro) — separate evaluation if Candidate C or D is adopted.
- **Cost ceiling** — separate operator decision; tonight's estimate is ~$0.01/episode for the three high-leverage passes only.
- **Whether to add Big-Five personality vectors to cast schema** (literature-supported but premature without first measuring Phase 1 outcomes).
- **Whether to switch to a different local LLM** (Llama-3.3-70B / Mistral-Large / Qwen-2.5-72B) instead of going API. The local-only optimization track is its own conversation, parallel to but not blocking Candidate D.

---

## 8. Round-robin protocol

Per CLAUDE.md round-robin section:

1. ChatGPT (gpt-4.1 via `scripts/_consult_openai.py`) — first opinion + critique
2. Gemini (gemini-2.5-pro via `scripts/_consult_round_robin.py`) — agreement / corrections / additions
3. Claude — synthesize, flag disagreements, decide grounded answer
4. Loop step 2 if externals disagree materially — re-prompt with disagreement spelled out

Save all transcripts under `docs/2026-05-26-good-story-writer-architecture/` per the round-robin save discipline.

---

## 9. Sources surfaced during research

Literature backing this doc:

- Talk Less, Call Right: Enhancing Role-Play LLM Agents (arxiv 2509.00482) — RRP technique
- RPGAgent: Story-to-Play Generation with LLM-Based Multi-Agent System (CHI 2026)
- SNAP: A Plan-Driven Framework for Controllable Interactive Narrative Generation (arxiv 2601.11529)
- Plug-and-Play Dramaturge: Divide-and-Conquer Iterative Narrative Script Refinement (arxiv 2510.05188v3)
- A Persona-Aware LLM-Enhanced Framework for Multi-Session (ACL 2025 Findings)
- Multi-Agent Based Character Simulation for Story Writing (In2Writing 2025)
- Better Zero-Shot Reasoning with Role-Play Prompting (arxiv 2308.07702v2)
- An Updated Guide to AI Roleplaying In 2026 — community-side state survey

---

**Operator instruction:** review this draft, edit anything mischaracterized, then run the round-robin per §8. Two passes minimum (ChatGPT + Gemini); a third (NVIDIA / DeepSeek) optional if disagreement is material. Synthesis lands here as `__04_synthesis.md`.
