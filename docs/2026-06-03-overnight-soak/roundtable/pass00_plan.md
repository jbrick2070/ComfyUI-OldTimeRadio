# Design: autonomous OVERNIGHT OTR soak + Cowork auto-fix (harden this)

## Goal
Use the night to run MANY OTR episodes unattended, surface intermittent bugs, and
have the Cowork agent AUTO-FIX the easy ones behind hard guardrails -- leaving a
morning report. Must never make the codebase worse while unattended.

## Context (real)
- OTR = a local ComfyUI writer->audio->video radio-drama generator on one RTX
  5080 (16 GB), Windows. A 340-word episode is ~4 h WITH HuMo video; **pruned to
  the audio sub-DAG (closure of node 7 EpisodeAssembler) it is ~5-10 min** and
  costs no video. The intermittent bugs all live in the writer/audio path, so the
  overnight soak runs AUDIO-ONLY and loops many episodes.
- Driver exists: `scripts/_otr_soak_matrix.py <words> <combos.json> <tag>` -- per
  combo it patches model/engines, prunes to the 9 audio nodes, submits to ComfyUI
  (localhost:8000), polls with a per-run timeout, records PASS/FAIL + the
  `audio_done` marker + VRAM peak, writes `_otr_matrix_<tag>.json`. A failing
  combo is cancelled + logged, never fatal.
- Each episode draws FRESH OS-entropy RNG (cast/news/style) -> different every
  run -> that's how intermittent bugs (BUG-276 announcer-routing, BUG-295 name
  leak, BUG-264 news overrun) surface.
- Regression gates (CPU, ~35 s + ~2 s): full `tests/` (3675 passed / 13 skip / 0)
  + Bug Bible (`bug_bible_regression.py --pack-dir <repo>`, 23/1/2/0). A
  KNOWN-FAIL-GUARD in conftest flags any NEW test failure.
- Recently fixed this session (coerce-class, low-risk): BUG-264 (news_briefs
  count/length coerce), BUG-295 (name-leak scrub/retry), BUG-307 (key_term
  length). OPEN: BUG-276 (announcer line -> Bark crash; audio-path, deferred --
  do NOT auto-touch). gemma-4-12b runs via an Ollama sidecar (separate).
- Discipline (CLAUDE.md): audio byte-identical at every gate; run Bug Bible +
  full tests after EVERY code change; one git push attempt then hand a block;
  NEVER force-push; never the word "dummy"; UTF-8 no BOM; `.py` edits need a
  ComfyUI restart to load. Branch work on `v2.0-alpha`; only Jeffrey merges main.
- Tooling available to an overnight Cowork agent: Desktop Commander (Windows
  shell), scheduled-tasks (cron-like), the file tools, git via DC cmd shell.

## Proposed design (critique + harden this)
1. **Soak loop** (continuous, unattended): a driver loops 340w audio-only
   episodes back-to-back. Combos rotate {mistral-nemo (proven), gemma-2-2b (weak,
   stresses the news/cast gates = BUG-264 surface)} x {bark/kokoro + SA3 music}.
   `bypass_freeze_halt=OFF` so BUG-276 HALTS gracefully (no crash). Log each
   episode: PASS / HALT(freeze) / CRASH(exc) / QUALITY(flagged) + marker + VRAM +
   the failing phase + a short error tail.
2. **Triage + auto-fix agent** (a scheduled Cowork task waking every ~30-60 min):
   reads the soak log for NEW failures; classifies each; for a WHITELISTED EASY
   category only, applies the smallest fix, then runs the FULL regression (tests +
   Bible) and KEEPS the fix only if green, else REVERTS. Commits each kept fix to
   a dedicated `soak/auto-YYYYMMDD` branch (never main, never force-push), logs a
   BUG_LOG entry, and re-queues the soak. Caps total auto-fixes per night;
   one-fix-at-a-time.
3. **Auto-fix WHITELIST (safe categories only):** schema coerce-class (a new
   pydantic over-cap field -> truncate, like BUG-264/307); a clearly-bounded
   guard/known-pattern with an obvious fix + a regression test. **Always-FLAG (no
   auto-fix):** anything touching the audio render path / PD1 byte-identity
   (incl. BUG-276), VRAM/loader, workflow JSON wiring, anything needing a design
   decision, anything not on the whitelist.
4. **Guardrails (hard):** never break audio byte-identity; whitelist-only;
   regression-gated keep/revert; revert-on-red; cap N auto-fixes; one-at-a-time;
   no force-push; ComfyUI restart only if the agent can do it safely (else flag
   "needs restart"); everything logged for morning review.
5. **Morning report:** episodes run, PASS/HALT/CRASH/QUALITY counts, distinct
   bugs found (with repro seed/episode), auto-fixes KEPT (green, on the branch),
   auto-fixes REVERTED (red), items FLAGGED for human.

## Questions for the panel
1. Is this loop + scheduled-triage split the right architecture, or is a single
   long-running supervised driver safer? How to make the agent's auto-fix step
   genuinely safe unattended?
2. What EXACTLY is safe to auto-fix vs always-flag? Tighten the whitelist.
3. The keep/revert mechanic: best way to make every auto-fix atomically
   revertible + regression-gated without a human? (git branch + stash? a "trial
   commit" the agent reverts on red?)
4. How to avoid the classic failure: the agent "fixes" something that passes
   tests but degrades output (esp. audio)? What extra gate beyond the unit suite?
5. Scheduling/lifecycle on Windows: continuous soak script + periodic Cowork
   triage task -- pitfalls (VRAM contention, ComfyUI restart, a hung episode, the
   agent stepping on a mid-render)? 
6. What are we missing for a SAFE unattended overnight run?
