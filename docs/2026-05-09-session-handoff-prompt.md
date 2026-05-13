# Session Handoff Prompt — paste into a new Claude conversation

> Copy everything between the `─────` rules below into the first message of a new Claude conversation. Augment with the round-robin responses you've collected from ChatGPT and Gemini once you have them.

─────────────────────────────────────────────────────

# Picking up an OTR v2.0-alpha session — need synthesis on a script-writer architecture round-robin

## Who I am

Jeffrey Brick. Solo dev on **OldTimeRadio (SIGNAL LOST)**, a ComfyUI custom-node pipeline that produces 1940s-style sci-fi radio drama videos end-to-end.

- Hardware: RTX 5080 Laptop, **16 GB VRAM**, Windows, torch 2.10.0+cu130, CUDA 13
- Constraints (`CLAUDE.md`): 100% local for the audio/video render path, no cloud services, MIT-licensed, C7 byte-identity audio guarantee
- Repo: `https://github.com/jbrick2070/ComfyUI-OldTimeRadio` branch `v2.0-alpha`
- Current HEAD: `485874b`
- Rollback tag: `v2.0-alpha-pre-humo-fix-2026-05-08` (commit `eb5bcfb`)
- Sister repo (Bug Bible): `https://github.com/jbrick2070/comfyui-custom-node-survival-guide`

## What just happened (today, 2026-05-09)

Two architectural problems caught and partially fixed:

### Problem 1 — HuMo Phase C 88× slowdown on large episodes (mostly closed)
- Symptom: HuMo 14B running at 5,284 s/it instead of 60 s/it on alien_whispers-class scripts (60+ audio chunks)
- Root cause: section-5 HuMo pre-pin (`force_full_load=True` before Phase A) fragmenting the cudaMallocAsync pool + `chunk["audio_emb"]` cleanup walk never firing (BUG-086 schema split that the cleanup wasn't updated for) → HuMo Phase C entered with a shredded allocator pool
- Fixes shipped:
  - **Fix 1** (commit `601ae35`): removed the section-5 pre-pin
  - **Fix 1.5** (commit `8247716`): cleanup walk now visits `chunk["audio_emb"]` per-chunk + Phase C count log repaired (was always reporting "0 lines" due to schema-stale predicate)
- Validation: 4-HuMo smoke (synthetic 4 char-clones of l002 with 4 audio chunks) ran clean at 62-66 s/it across all 4 clips, last clip *faster* than middle clips → no fragmentation creep at smoke scale
- Pending: real alien_whispers test still hasn't fired because of Problem 2

### Problem 2 — ScriptWriter produces prose, not formatted screenplay (open)

- Symptom: Mistral-Nemo 12B (4-bit NF4 local) reliably writes flowing narrator prose when prompted for `[VOICE: NAME, traits] dialogue` screenplay format. Parser falls through 5+ fallback passes accumulated over months.
- Today's fresh evidence:
  - **Failure 1** (FIXED): cyberpunk neon-noir 100w produced inline `[VOICE:]` format → critique counted 10 dialogue lines → revision kept it → FormatNorm reformatted to bare `CHARACTER:\ndialogue` → parser crashed PARSE_FATAL. Patched via Pass 3 in commit `485874b`.
  - **Failure 2**: psychological slow-burn 110w → produced ZERO character dialogue. Cast=3 declared (ANNOUNCER, JEHOSHAPHAT, LEV), only ANNOUNCER spoke (2 bookend lines). Final 35.9 MB mp4 with no story.
  - **Failure 3**: same style, fresh run → 835 tokens emitted, streaming counter showed 0 scenes / 0 dialogue lines / 0 character chars across the entire generation. Mistral wrote pure prose.
- Root: every parser patch since BUG-014 is whack-a-mole on Mistral's format drift. The architecture is wrong, not the regex.
- Solution: pivot to per-line composition with the ledger as source of truth (sketched in problem statement)

## What I need from you in this conversation

I'm going to feed you the round-robin responses from ChatGPT and Gemini on the script-writer architecture question. **I want you to synthesize their responses, identify disagreements, and recommend a single path forward for me to ship.**

The setup:
- Read `docs/2026-05-09-script-writer-architecture-problem-statement.md` if you have file access (Cowork mode). If not, ask me to paste it.
- Read `docs/2026-05-08-humo-phase-c-slowdown-v2-decision-point.md` for HuMo context.
- Six candidate architectures (A-F). My pre-synthesis vote was Architecture F (hybrid: outline + per-line fill + ledger).

When I paste round-robin responses, do this:
1. List which architecture each model picked
2. Flag any technical claims that disagree (especially around grammar-constrained sampling on NF4 models, context window strategy, API use upstream of audio)
3. Recommend ONE architecture to ship, with implementation order (what gets coded first)
4. Estimate ETA at ~6 hr/day pace for solo dev (me)

## Concrete next-action menu (so you know what's on the table)

- **Architecture D (prompt-only rewrite, ~30 min)** — bridge fix tonight if HuMo 88× test must complete first
- **Architecture F (hybrid: outline + per-line + ledger, ~2-3 days)** — my pre-synthesis vote
- **Architecture B (single JSON-constrained call, ~1-2 days)** — if grammar-constrained sampling works on NF4 Mistral
- **Architecture C (multi-model with Claude/Gemini API for outline)** — strongest reliability but might violate `CLAUDE.md` "no cloud services" rule (judgment call: ScriptWriter is upstream of audio, doesn't break C7)

## Files I'll likely paste in

- `docs/2026-05-09-script-writer-architecture-problem-statement.md` — the round-robin question
- `nodes/_otr_period_prompts.py` — the 1940s system prompt module (current attempt to steer Mistral)
- `FAILED_SCRIPT_DUMP.txt` — what Mistral wrote that crashed the parser today
- The round-robin responses from ChatGPT and Gemini

## What's NOT on the table tonight

- Reverting any of today's commits (Fix 1, Fix 1.5, parser Pass 3 — all proven non-regressive)
- Switching off Mistral-Nemo as the local LLM (CosyVoice 2/3 etc are TTS, not story; story-LLM upgrade is a v2.1 sprint per memory)
- Bumping HuMo soak cap above 0 (already verified clean at 4 consecutive renders)

## Style preference

Honest, factual, structured. No fluff. Minimal emoji. Same as the rest of my CLAUDE.md says.

─────────────────────────────────────────────────────
