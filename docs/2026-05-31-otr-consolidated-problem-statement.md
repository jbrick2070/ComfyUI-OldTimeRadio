# OTR Consolidated Problem Statement — Story Quality + Bug Backlog (Sprint Feed)

**Date:** 2026-05-31 | **Branch:** `v2.0-alpha` | **HEAD:** `670636a`
**Purpose:** one cohesive problem statement blending *story-quality improvements* and the
*bug backlog* so the next sprint can round-robin independent chunks across sessions/models.
**Nothing is deferred** — the bugs are integrated as first-class work streams, not parked.
**This is a planning doc. No code is implemented here.**

---

## 1. Current state

**Quality baseline (7-axis / 35-pt rubric, organic news, no baked premise):**

| Config | Best score | Note |
|--------|-----------|------|
| Opus 4.8 remote (creative) + local Mistral (technical) | **31/35 (89%)** | ~$0.47/ep; overshot 350→829 words |
| local mistral-nemo (both slots), 350w | 27/35 (77%) / 25.5 (73%) | the local sweet spot; reliable |
| remote mistral-nemo (both slots), 350w | 23.5/35 (67%) | F3 name-leak |
| local 60w / 500w | 71% / 69% | 60w too cramped; 500w no better than 350w |

**Proven working (do not regress):** OpenRouter routing is clean — 30/30 creative calls
hit Opus (route=throughput), **0 fallbacks**; 5 technical calls stayed local, **0 leaked**;
C2 no-evict held. BUG-296 (per-run budget reset) fixed + live-confirmed. Fail-closed JSON
(C4), cost guard (C6, ~$0.47/Opus-ep under the 300k ceiling), and the 429 retry ladder all
hold under real runs.

**Reference commits:** `c2c1955` baseline scorecard · `d982821` BUG-296 fix · `7526b2d`
length sweep · `3696627` Opus comparison + improvement plan · `670636a` three-pass section.

---

## 2. The integrated goal

**A "good OTR episode" is:** *(story)* clearly grounded in the selected news, with a real
beginning/middle/end and a turn, distinct character voices, and a payoff that lands;
*(radio)* every line speakable in roughly one breath and legible by ear, not just on the
page; *(safe)* safe-for-work and non-violent; *(clean)* correct speaker attribution, no
cast-name leakage into the body, no crashes, audio byte-identical to the blessed baseline;
*(bounded)* cost and VRAM under a known per-episode ceiling.

**Acceptance criteria a reviewer can grade:**
1. Rubric **≥ 28/35 (80%)** on the 7 axes (Opus already clears this; lift local toward it).
2. Actual words within **±20%** of the radio target (~350) — no 800-word overshoot, no
   sub-floor undershoot.
3. **Zero** SFW slips (no "damn"/profanity), **zero** speaker-attribution errors, **zero**
   cast-name-in-body leaks.
4. **Zero** fail-closed aborts on a *valid* news story (news-gate + inventor robust).
5. **Zero** Bark/render crashes; audio byte-identical gate green; SFW + non-violent pass.
6. Cost ≤ an agreed per-episode cap (see §6).

---

## 3. The full problem space — blended (story + bugs)

Each item: **what's wrong · why it matters · evidence · candidate fix.** IDs match
`docs/openrouter-llm-call-improvement-plan.md` (C1–C6), `docs/2026-05-31-otr-story-quality-baseline.md`
(F1–F5), and `BUG_LOG.md`.

### P0 — Three-pass story architecture (the spine)
- **What:** generate → tactical edit → creative-QA. Pass 1 Opus writes; Pass 2 trims to
  radio length/cadence; Pass 3 judges "is it actually interesting" + can send work back.
- **Why:** single-pass Opus is the best output (89%) but overshoots length and over-writes
  for the ear; there's no taste gate, only a structural critic.
- **Evidence:** Opus "Green Book of Nights" — 89% but 829 words, dense comma-splice prose.
- **Fix:** the architecture in `openrouter-llm-call-improvement-plan.md` (model table,
  control flow, PD6 `editor_model`/`critic_model` broadcast outputs). **Supersedes C1; may
  obsolete C5** (a creative-QA pass should catch name-leaks too).

### P1 — C1 length / radio-legibility (collapses into Pass 2)
- **What:** lines are too long/dense for radio. **Why:** "audio is king"; a listener can't
  track 829 words of indirection. **Evidence:** Opus 829 vs 350 target. **Fix:** Pass 2
  editor — per-line spoken ceiling (~35 words/one breath) + whole-episode budget. **This is
  not a separate work item from P0; it is Pass 2.**

### P2 — C2 model-specific creative prompts
- **What:** one prompt feeds both a 12B local model and Opus; they need opposite nudges.
- **Why:** Opus needs restraint; mistral needs "stay on the news." **Evidence:** Opus
  sprawl; mistral's "potato that spits fire" drift off a climate-news seed. **Fix:** a thin
  per-model-class rider on the shared creative prompt (select off the resolved slug).

### P3 — C3 per-slot temperatures
- **What:** global per-call-type temps. **Why:** Opus likely wants a lower creative temp
  (control), small local wants its higher temp (color). **Evidence:** Opus over-invention.
- **Fix:** derive creative temp from model-class (no new widget). Bundle with P2 in one A/B.

### P4 — C4 news key-term gate fail-closed (BUG-264 family)
- **What:** `news_interpreter` requires key_terms to appear verbatim in source; good
  paraphrases get rejected and the whole run aborts. **Why:** kills valid episodes before a
  word is written. **Evidence:** T7 abort ('human-like errors'/'UCLA researchers'); the
  BUG-296 verify run aborted the same way. `news_interpreter.py:803`. **Fix:** accept a
  term the LLM-judge says is *semantically supported* by the source; hard-reject only true
  hallucinations; don't burn the 3-attempt budget on one term.

### P5 — F4 writer "inventor" pass fail-closed
- **What:** the inventor pass parse-fails on near-duplicate keys and aborts the writer.
- **Why:** another fail-closed that loses a valid episode. **Evidence:** T7 (local 150w):
  `inventor failed after 3 attempts; parse failed: 'city_lab_expansion_notice' /
  'initiative_expansion_notice'`. **Fix:** dedup/normalize the inventor keys before validate,
  or a more forgiving inventor grammar + one repair attempt. (Same family as P4 — validators
  too brittle.)

### P6 — F3 / BUG-295 cast-name leak into the body (remote)
- **What:** the ALL-CAPS cast name lands inside a stage direction / as a dangling noun.
- **Why:** nonsense lines that degrade TTS + reading. **Evidence:** T4 remote mistral —
  "* ERIN SPENDER the monkeys' enclosure*", "safe in the ERIN SPENDER". Local was clean.
- **Fix (C5):** extend the BUG-279 leak filter (`_otr_line_composer.py` ~L1663-1692) to
  retry when a **multi-word ALL-CAPS roster name** appears mid-phrase / inside `*...*`. **May
  be obsoleted by P0 Pass 3** (a creative-QA pass should reject such lines).

### P7 — C6 / F1 low-word-count attribution collapse
- **What:** at ≤60 words a multi-party scene collapses onto one speaker id. **Why:** breaks
  voice distinctness + coherence. **Evidence:** T1 remote 60w; clean at 350w. **Fix:** prefer
  ≥150w for multi-character scenes (350 is the sweet spot anyway), or a stronger per-line
  speaker binding. **Low priority** — mostly avoided by the 350w target.

### P8 — F5 "damn" slips the SFW guard
- **What:** "damn"/"Damn it" passed the SFW validator. **Why:** CLAUDE.md is explicit —
  safe-for-work, no profanity. **Evidence:** T6 + T8. **Fix:** add "damn(ed)" (+ review the
  wordlist) to the SFW validator, or make it an operator-tunable allowlist; trivial.

### P9 — BUG-276 / 271 cast-routing crash (recurrence)
- **What:** a character line resolves to the announcer with no Bark `v2/*` preset →
  `BatchBarkGenerator` Gate 3 crash. **Why:** hard crash, no episode. **Evidence:** T5
  (remote 350w) crashed at `batch_bark_generator.py:520` (char_id='announcer', line b018);
  `bypass_freeze_halt` let it reach Bark. Local didn't hit it. **Fix:** the parked deeper
  fix — mirror the announcer-exclusion in the reroll/Stage-3 cast pool + a Gate-1/2 pre-Bark
  reject of any character line whose char_id resolves to the announcer. See the extensive
  BUG-276/271 investigation in `BUG_LOG.md`.

### P10 — Dead-code / legacy cleanup
- **What:** accumulated legacy/scratch code + an unfinished cleanup audit. **Why:** clean
  names/logs matter (CLAUDE.md); drift hides bugs. **Evidence/pointers:** `ROADMAP.md`
  "ROADMAP staleness audit — 8 stale-actionable items, edits not yet applied" + the
  cleanbreak/legacy-debris Bible candidates in `BUG_LOG.md` (BUG-207/208/224/226…) + the
  forbidden-sweep tool `docs/_s28_forbidden_sweep.py` (hardened in BUG-LOCAL-293); **plus a
  concrete current item: this session left scratch helpers in the repo root**
  (`_otr_soak2.py`, `_otr_dump_scripts.py`, `_otr_show_episode.py`, `_otr_routing_audit.py`,
  `_otr_or_*.py`, `_otr_interrupt.py`, `_otr_soak_ids.json`, `_otr_last_prompt.json`,
  `_otr_headless_launch.bat.premopus`) that should be moved to a `tools/` dir or removed.
  **Fix:** apply the 8 ROADMAP staleness edits; sweep + delete dead paths gated by the
  forbidden-sweep; relocate/retire the scratch helpers. **NOTE — confirm the pointer:** I did
  not find a single doc named "dead-code synthesis"; if you meant a specific committed critique,
  point me at it and I'll fold it in.

### P11 — Caption text size reduction (−65%)
- **What:** the burned-in SDH open-caption letters render too large on screen; cut the
  caption font size by ~65%.
- **Why:** render/output hygiene — the oversized lower-third dominates the frame, fights the
  1940s-broadcast look, and crowds the HuMo/composite visual. A smaller caption sits cleaner.
- **Evidence:** `nodes/_otr_captions.py:66` — the default `sdh_standard` ASS style is **Arial
  52 px** (the A/B-only `otr_crt` style is **50 px**, `:80`); the `{size}` field is written
  into the ASS `Style:` line at `:192-195` and burned in at the video composite/blend stage
  (the SDH caption burn, ~Node 58). 52 px renders oversized at the output resolution.
- **Fix:** reduce the `sdh_standard` `"size"` **52 → ~18** (52 × 0.35 = 18.2; the 65% cut) at
  `_otr_captions.py:66`, and `otr_crt` `"size"` 50 → ~18 to match. One style-dict value each;
  no node-surface / widget / workflow-JSON change; caption-only, so audio stays byte-identical.
  **Side effects to validate:** (a) *legibility* — 18 px is small; confirm it still reads at the
  procgen_blended output resolution, and (SDH captions are an accessibility feature) that the
  ~65%-opaque box + outline keep contrast; (b) *line-wrapping* — smaller glyphs fit more
  characters per line, so existing wraps/margins may change (re-check the ASS wrap + `margin_v`);
  (c) *vertical layout* — a smaller caption block + box shifts the lower-third; confirm it stays
  inside the title-safe area. **Stream D** (render/output hygiene). **Gate:** A/B one caption at
  52 px vs 18 px on a real output frame — legible + correctly positioned; audio byte-identical.

### P12 — TTS model evaluation: Bark alternatives (research / eval)
- **What:** evaluate a different local TTS model that may beat Bark for OTR voices. Jeffrey
  recalled it only as *"something like Express to…"* by *"some company like express to"* — best
  decode is **XTTS-v2 (Coqui)** ("X-TTS" sounds like "express" dictated). **This is research /
  eval, not implementation** — Jeffrey picks the winner before any swap-out work begins.
- **Why:** Bark is today's TTS; a higher-quality / more controllable / faster local model could
  lift voice quality, character distinctness, and the period-authentic 1940s feel. Any candidate
  must be **100% local + open-source, no paid services** (CLAUDE.md).
- **Evidence:** Jeffrey's recollection (above) + Bark's known limits (instability/artifacts,
  speed). Open-source, consumer-GPU candidates:
  - **XTTS-v2 (Coqui)** — *leading match for "Express"*; strong Bark alternative, multi-speaker
    voice cloning, runs comfortably on 16 GB; mature + widely used.
  - **StyleTTS2** — very high quality + fast; expressive style control.
  - **F5-TTS** — recent, strong zero-shot quality.
  - **VoiceCraft** — zero-shot + speech editing.
  - **ChatTTS** — conversational TTS (less likely the "express" match).
- **Fix (eval plan, no code):** score each candidate on voice quality, 3-voice character
  distinctness, period-authentic 1940s broadcast feel, VRAM on RTX 5080 16 GB (under the 14.5 GB
  ceiling, co-resident with the writer/video budget), **license = open-source (required)**, and
  ComfyUI node availability vs implementation effort. **Acceptance gate:** render the SAME 1-line
  script across 3 character voices with current Bark and each candidate; side-by-side listen test
  vs the existing Bark baseline; Jeffrey selects the winner before any swap-out. **Stream D**
  (output hygiene / stability — eval only). **Confirm:** was **XTTS-v2** the model you meant? If
  yes, this tightens to XTTS-v2-vs-Bark and the rest drop to "also-considered."

---

## 4. Dependencies + ordering

| Item | Depends on / relationship | Blocks |
|------|---------------------------|--------|
| **P0 3-pass** | needs editor_model+critic_model slots (PD6) | absorbs **P1**; likely obsoletes **P6/C5**; houses **P2/P3** prompts |
| P1 C1 length | = P0 Pass 2 | — |
| P2 C2 prompts | rides P0 (lives in the Pass-1 prompt + Pass-2 editor) | — |
| P3 C3 temps | bundle with P2 | — |
| **P4 news gate** | independent | unblocks valid-news runs (highest "runs that should have shipped" hit-rate) |
| **P5 inventor** | independent (same brittleness family as P4) | unblocks valid-news runs |
| P6 F3/C5 leak | independent; **re-check after P0** (Pass 3 may catch it) | — |
| P7 C6/F1 attribution | independent; mostly moot at ≥150w | — |
| P8 F5 SFW | independent; trivial | — |
| **P9 BUG-276 crash** | independent; deep cast-routing fix | unblocks remote reliability at length |
| P10 cleanup | independent | reduces drift; do continuously |
| **P11 caption size** | independent; one caption-style value (`_otr_captions.py:66`) | render/output hygiene; no blocker |
| **P12 TTS eval** | independent; research/eval only (no code until Jeffrey picks) | gates any future Bark swap-out |

**Key arrows for round-robin:** P0 is the big rock and **supersedes P1, houses P2/P3, may
obsolete P6**. P4+P5 are a cheap robustness pair that should land *first* (they decide whether
a run produces a story at all). P9 is an isolated crash fix. P8/P10/P11 are independent hygiene;
P12 is a research/eval bake-off (no code until Jeffrey picks the TTS).

---

## 5. Suggested parallel work streams (round-robin)

Each stream is self-contained: a fresh session with **this doc + its stream brief** can
execute without coordinating with the others. Every stream's exit gate = Bug Bible + core +
audio byte-identical green **and** a scored A/B vs the 2026-05-31 baseline (no ship on a
content opinion alone).

- **Stream A — Creative quality core (P0 + P1 + P2 + P3).** Build the three-pass rig: add
  `editor_model`/`critic_model` writer broadcast outputs (PD6), author the Pass-2 editor
  prompt (length/cadence), re-aim the Pass-3 critic to creative-interest, add per-model
  creative prompt riders + temps. *Biggest effort; the headline win.* Owner needs the writer
  + slot-scheduler + critic code. Gate: 3-pass output beats single-pass Opus (89%) AND reads
  cleanly aloud.
- **Stream B — Robustness / fail-closed (P4 + P5).** Loosen the news key-term gate (semantic
  support, not verbatim) and harden the inventor pass (dedup/normalize + one repair). *Small,
  high-value, independent.* Owner needs `news_interpreter.py` + the inventor pass. Gate: the
  T7-style stories now produce episodes; regression proves hallucinations still rejected.
- **Stream C — Output hygiene + safety (P6/C5 + P8/F5 + P7/C6).** Extend the name-leak retry
  (multi-word ALL-CAPS in stage directions), add "damn(ed)" to the SFW guard, set the
  multi-char word-count floor. *Bounded, low-risk; re-check P6 need after Stream A lands.*
  Owner needs `_otr_line_composer.py` + the SFW validator. Gate: the T4 "ERIN SPENDER" case
  retries clean; SFW pass; byte-identical audio.
- **Stream D — Stability + cleanup + render/output hygiene (P9 + P10 + P11 + P12).** Land the
  BUG-276/271 cast-routing fix (announcer-exclusion in reroll + pre-Bark gate) and the
  dead-code/scratch cleanup; **reduce the SDH caption size ~65% (P11, `_otr_captions.py:66`,
  52 → ~18 px)**; and run the **Bark-alternative TTS evaluation (P12, XTTS-v2 lead) — research
  only, Jeffrey picks the winner before any swap-out.** *Isolated crash fix + continuous hygiene
  + output polish + a TTS bake-off.* Owner needs the reviewer/reroll/cast-contract code + the
  forbidden-sweep (P9/P10), `nodes/_otr_captions.py` (P11), and a TTS eval harness (P12). Gate:
  force a `needs_full_rerun` and assert no character line reaches Bark on an announcer voice;
  forbidden-sweep clean; caption A/B legible at the reduced size; TTS listen-test comparison
  delivered for Jeffrey's pick.

Streams A and B/C/D are fully parallel. Within A, do P0 scaffolding before P2/P3 tuning.

---

## 6. Open design questions (need Jeffrey's call)

1. **Pass 2 (editor) model:** local mistral-nemo (free, +VRAM) · Haiku 4.5 (~$0.008,
   Anthropic-consistent) · GPT-4o-mini (~$0.002)?
2. **Pass 3 (critic) model:** Sonnet 4.6 (~$0.02, recommended) · Haiku 4.5 (~$0.008) · local?
   (Opus judging Opus is discouraged.)
3. **Production default flip:** still local Mistral. Authorize OpenRouter+Opus as the default
   creative slot? (Quality win for <$1/ep, but a paid default + the radio-density caveat.)
4. **Per-episode budget cap:** single-pass Opus ~$0.47; 3-pass ~$0.50, ~$1 worst case (one
   Opus re-write). Set a hard `OPENROUTER_MAX_TOKENS_PER_RUN`/dollar ceiling for the default?
5. **Round-robin coordinator:** want a **kickoff-prompt template** for fresh sessions (points
   at this doc + a stream brief + the gates), so each parallel session starts identically?
6. **Slot count:** keep 2 model-slots (editor/critic reuse local) or go to 4 roles
   (writer/editor/critic/technical-JSON)? (More control vs more wiring + cost.)
7. **Is single-pass Opus already enough?** 89% is strong; the 3-pass rig buys radio
   legibility + a taste gate for +~$0.03 and real wiring. Ship single-pass Opus + a light
   C1/C2 prompt nudge first and measure, or build the full rig now?

---

## 7. References + appendix

- `docs/2026-05-31-otr-story-quality-baseline.md` — 6 scored runs, F1–F5 findings, length curve.
- `docs/2026-05-31-otr-story-quality-comparison.md` — all-runs scorecard + Opus 89% + full
  "Green Book of Nights" script + routing + cost.
- `docs/openrouter-llm-call-improvement-plan.md` — C1–C6 + the three-pass architecture
  (model table, control flow, effort, open questions).
- `docs/openrouter-remote-llm-go-forward-plan.md` + `docs/openrouter-setup.md` — the shipped
  OpenRouter feature + setup/routing.
- `BUG_LOG.md` — BUG-296 (fixed), BUG-295 (F3, proposed), BUG-276/271 (P9, investigation +
  parked fix), BUG-264 (P4 family), cleanbreak/legacy Bible candidates (P10).
- `ROADMAP.md` — canonical going-forward plan; "ROADMAP staleness audit — 8 items" (P10);
  AUDIO QUALITY + SFX/CLEAN-LEDGER tracks (adjacent, not in this statement's scope).
- Tooling: `docs/_s28_forbidden_sweep.py` (dead-code sweep), `_otr_soak2.py` /
  `_otr_dump_scripts.py` / `_otr_show_episode.py` / `_otr_routing_audit.py` (this session's
  soak + scoring + routing-audit harness — scratch, to be relocated under P10).

**Standing constraints (apply to every stream):** Prime Directives (audio byte-identical;
14.5 GB VRAM ceiling; wire every change into the workflow JSON; SFW non-violent; never the
word "dummy"; every LLM call tagged creative/technical and routed via the writer's two model
widgets — adding editor/critic models means new *broadcast outputs*, not new per-node
widgets). Run the Bug Bible regression after every code change. Git via Desktop Commander cmd
+ `.git\COMMIT_EDITMSG -F`, one push attempt, verify HEAD.
