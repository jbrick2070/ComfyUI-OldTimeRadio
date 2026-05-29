# Problem Statement — Breaking the 3.7 Story-Quality Plateau (OldTimeRadio)

**Date:** 2026-05-28
**Purpose:** Round-robin seed (ChatGPT → Gemini → Claude). Self-contained; assume the reader has zero prior context on this project.
**Question in one line:** Our local, fully-automated radio-drama generator reliably produces *structurally complete but emotionally flat* episodes that plateau at a critic mean of ~3.7/5. How do we get to a genuinely engaging story without breaking the local-only constraints?

---

## 1. What the system is

OldTimeRadio (OTR) is a 100% local, offline ComfyUI pipeline that turns a news headline into a short old-time-radio drama (script → voices → audio). One model does all the language work: **Mistral-Nemo-Instruct-2407, 4-bit NF4 quantized**, on a single **RTX 5080 Laptop, 16 GB VRAM (hard 14.5 GB peak ceiling)**. No cloud, no API, no paid services. Output must be **safe-for-work, non-violent**, and have a real beginning/middle/end.

The same model is addressed through two logical "slots":
- **creative** slot — narrative passes (outline, cast, dialogue, polish).
- **technical** slot — structured/JSON passes (validators, critic, editor verdicts).

## 2. How an episode is built (current pipeline)

1. **News fetch + rank** → pick one headline (here: a NASA Curiosity rover drill-sample story).
2. **Style pick** (LLM invents + chooses a style descriptor).
3. **Cast lock** (2 characters + an announcer; LLM writes character descriptions, assigns TTS voices).
4. **Outline** — macro pass → per-phase pass → per-beat pass. Produces ~9 beats across 2 acts (setup, resolution) with a music interlude.
5. **Dramatic-state object** is stamped onto the ledger (a "dramatic question," opposed desires, a designated `costly_choice_beat`, an "ending changes" claim).
6. **Continuity ledger** (facts, who-knows-what, props).
7. **Dialogue compose** — per beat, a multi-turn roleplay composer generates **N=4 candidate lines and picks a best-of-N winner** by score.
8. **Announcer intro/outro**, **title**, **story brief**.
9. **Director brief** → **Story Room**: a writer/editor agent loop (budget: 1 writer turn, 2 editor cycles, 14 total turns). The editor scores the draft against the rubric and returns failing axes + per-axis notes; the writer revises.
10. **Commit** the room's draft to the dialogue slots, then **Stage-7 critic** scores the whole episode, then a **freeze** step applies "doctor edits."

## 3. The concrete failure (real data from episode "Mars Heartbeat Wait")

- Final length: **9 lines / 137 words** (target was 220).
- Critic verdict: **ship**, **mean 3.7 / 5**. Per-axis (1–5):

  | Axis | Score |
  |---|---|
  | premise_clarity | 4 |
  | character_distinctiveness | 4 |
  | continuity | 3 |
  | **naturalness** | **3** |
  | pacing | 4 |
  | **emotional_arc** | **3** |
  | resolution | 4 |
  | **specificity** | **3** |
  | sfw_adherence | 5 |
  | audio_readiness | 4 |

- The in-loop critic independently flagged the two dead lines:
  - **b002** — "exposition delivered instead of tension played; 'intriguing' and 'game-changing' could be shown, not told."
  - **b004** — "dramatically inert; the clock ticking could build tension but the line is static."
- Arc verdict: **"uneven."** Reviewer said the editor pass "improved" the draft, and the room mechanics were clean (8/8 rows committed, no fallback). **The machinery works; the writing is flat.**

### The actual script (so reviewers can judge directly)
> ANNOUNCER: Tune in as we peer into the red heart of Mars, where the Curiosity rover has just pierced the veil of 'Campo Marte'.
> ANTON CRANSTON: "We've encountered an anomaly unlike any we've seen before. It's... intriguing, and potentially game-changing. The world watches, holding its breath."
> TARIQ HALPERT: "Steady, everyone. Seven minutes is just a heartbeat in geologic time. Let's not get our hopes up, but... let's not rule anything out, either."
> ANTON CRANSTON: Checking the clock. Sixty seconds left.
> [music interlude]
> ANTON CRANSTON: "Thank goodness," Cranston whispered, shoulders slumping. "Let's see what we've got."
> TARIQ HALPERT: "Unbelievable. It's not just unlike anything we've seen... it's unlike anything I've seen."
> ANTON CRANSTON: "Composition analysis indicates an unknown, crystalline structure. We're looking at something potentially transformative, people. Let's bring it into focus."
> ANNOUNCER: And there it is, the first-ever color snapshot from within 'Campo Marte'.

## 4. Diagnosis (working hypothesis — challenge it)

The premise was the **seven-minute wait** for the first color image, but the script *never dramatizes the wait*: it jumps "sixty seconds left" → music → "thank goodness." The persistent low axes (**naturalness, emotional_arc, specificity** — all 3) point to one root cause: **the system labels drama in metadata (dramatic question, costly_choice_beat, "ending changes") but the dialogue states emotions instead of dramatizing them.** No character makes a costly choice; the two characters sound nearly identical; the reveal ("crystalline structure, transformative") is generic. It is competent and watchable but inert — a textbook 3.7.

## 5. Hard constraints any idea must respect

- **Local only.** Mistral-Nemo NF4 on one 16 GB GPU, 14.5 GB peak. No cloud, no API, no bigger model assumed available.
- **Audio is king.** Anything that risks the audio output is reverted. Text changes are fine; the deliverable is a voiced drama.
- **SFW, non-violent, full arc.**
- **Cost is latency.** Each LLM pass is a blocking decode at NF4 speed (the editor pass alone is ~100–200s at 4096 tokens). More passes = minutes. Ideas should be mindful of turn budget.
- Episodes are currently **short (~150–220 words, ~8 voiced lines)**.

## 6. What is already in place (do NOT re-propose these)

- A dramatic-state object, a designated costly-choice beat, and an "ending changes" check (they're *stamped* but not *realized* in the lines).
- A writer/editor Story Room loop with a rubric-bound editor that returns per-axis failing notes.
- Best-of-N=4 candidate selection per dialogue beat.
- A 10-axis whole-episode critic (the scorer above).
- Continuity ledger, cast voices, style picker.

## 7. Questions to brainstorm

1. **Where is the real bottleneck** — the outline (beats too generic), the per-beat dialogue composer (writes single lines in isolation, so subtext can't accrue), the editor (notes don't force concrete rewrites), or the dramatic-state object (labels vs realizes)?
2. **Should the unit of generation change** — e.g. compose *scenes with conflict and subtext* rather than one line per beat scored in isolation? Does line-by-line best-of-N actively prevent build?
3. **Is ~8 lines / 150 words too thin** to dramatize a wait? Would a longer target (e.g. 350–500 words, a genuine middle with a setback/reversal) help more than any prompt change?
4. **How do you make a small quantized model show-not-tell** within tight latency — few-shot exemplars of subtext, banned-word lists ("intriguing," "game-changing," "unbelievable"), a dedicated "punch-up / de-exposition" pass, or a craft-specific editor rubric (subtext, one concrete sensory detail per line, a turn/reversal per scene)?
5. **Should the critic rubric itself change** to reward craft (specificity, reversal, a costly choice that's actually paid) so the editor optimizes toward engagement rather than competence?
6. **Cheapest highest-leverage single change** — if you could make exactly one modification to move the mean from 3.7 toward 4.2+, what would it be and why?

## 8. How success is measured

The 10-axis critic above (mean of ten 1–5 scores). Ship gate: no axis < 3, mean ≥ 3.5, SFW ≥ 4. **Baseline to beat: 3.70.** Note the critic is itself an LLM and jitters ±0.2–0.4 per call, so any claimed improvement must hold across **N=3 episodes**, compared on means — a single 3.9 is noise.

---

### Instructions for the round-robin responders
Give the **single highest-leverage change** first, with a one-paragraph rationale tied to the evidence in §3–4. Then rank 2–3 alternatives. Respect every constraint in §5. Do not propose anything in §6. Prefer ideas testable in one or two episode runs over large re-architectures, but flag if you believe only a re-architecture will break the plateau.
