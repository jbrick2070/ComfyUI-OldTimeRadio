# Overnight-soak roundtable -- pass01 judgment (Claude is the judge)

Panel: 3/3 (gpt-5.5, gemini-3.1-pro, deepseek-v4-pro). Spend ~$0.11.

## Strong consensus (accepted -> folded into SOAK_PLAN.md)
1. **CUT the scheduled/cron triage agent. Use ONE synchronous supervisor.** All 3:
   a separate timer racing the soak -> VRAM OOM collisions, editing `.py` while
   ComfyUI holds old code, restarting mid-render. The supervisor runs the loop and,
   on a failure, synchronously pauses -> fixes -> resumes. No concurrency.
2. **Append-only JSONL soak log** (the matrix driver overwrites one file/tag);
   atomic writes (tmp + os.replace); the triage reader tolerates a partial last line.
3. **The unit suite does NOT enforce audio byte-identity** (that gate is
   OTR_REGRESSION_RUNTIME/GPU, off by default). So unattended auto-fix MUST be
   PATH-restricted to writer/schema-validation files proven not to touch the audio
   render path. No edits to audio/loader/workflow-JSON/VRAM/BUG-276 code.
4. **Mechanical regex whitelist, NO LLM codegen.** Match specific exception strings
   (e.g. pydantic "longer than max length") -> pre-approved coerce template
   (truncate/clamp). Only schema-boundary coercions where the cap is already declared
   (the BUG-264/307 class). Everything else: FLAG, never fix.
5. **Night 1 = OBSERVE-ONLY** (soak + log + classify, NO writes). Enable auto-fix
   writes only after the classifier is proven to separate known-276 / infra / coerce
   / real-crash.
6. **BUG-276 = known-failure**: dedupe, count, report -- never propose a fix, never
   re-queue as "new".
7. git keep/revert = **trial-commit + `git reset --hard HEAD~1`** (NOT stash); kept
   fixes on `soak/auto-YYYYMMDD`, never main, one push attempt max, no force-push.
8. **Bounded log reads**: record the comfyui.log byte-offset at episode start; only
   scan the slice after (the driver's last_audio_done/err_tail scan global logs ->
   stale attribution).
9. Hard caps (max 2 kept / 3 attempted fixes/night; 1 red revert -> disable
   auto-fix), a re-entry lock, a hung-render watchdog (cancel_queue may not kill an
   executing node -> restart ComfyUI on no-progress), and a preflight/baseline gate
   before starting.

## Judge's grounding corrections (where the panel was slightly off)
- **The regression GATE does not need a ComfyUI restart.** Full `tests/` + Bug Bible
  are CPU-only (`CUDA_VISIBLE_DEVICES=''`, OTR_TEST_MODE=1) and run in a FRESH pytest
  process -> they import the edited `.py` and SEE the new code with no restart (I ran
  them headless ~10x this session). The restart is only needed to re-verify the fix
  in the LIVE soak. So: edit -> run the CPU gate (sees new code) -> if green, restart
  ComfyUI -> re-verify the failing combo -> resume.
- **bypass_freeze_halt is already OFF.** The driver does NOT patch it, so it stays at
  the canonical workflow default (node 11 = false, restored in BUG-293). So BUG-276
  halts gracefully by default; no patch needed -- just never set it True.
- **QUALITY status**: keep only machine-checkable detectors (audio_done length_sec <
  a floor; segments < 3) -- cut subjective quality judgement.

## Convergence
One pass, tight consensus, grounded. The hardened design is in SOAK_PLAN.md. No second
paid pass warranted.
