# Pass 1 judgment (Claude judge + panelist; grounded vs the real code)

Panel: gpt-5.5, gemini-3.1-pro, claude-opus-4.8 (deepseek-v4-pro starved on hidden
reasoning at both 2200 and 12000 out tokens -> dropped). Reviews + manifests under
pass01/ (gpt+gemini fragments @2200), pass01b/ (opus @12k), pass01c/ (gpt+gemini @12k).

## ACCEPTED (CONFIRMED against code -> folded into pass01_plan.md)
- License is the real decision: indextts2 `commercial_clean=False` -> not a safe
  commercial default. Commercial-clean cloner = chatterbox (MIT). (gpt, opus)
- No operator widget selects the engine; `_resolve_char_engine` returns the first
  `legacy_first_engines` entry -> indextts2 always wins for `voice_bank=default`.
  A selector (a `default_clean` bank routed to chatterbox) is the needed build. (opus, gemini)
- kokoro-as-char_voice CUT: (1) `prepare_text` identity -> reads stage directions
  aloud (gemini); (2) missing-`.pt` swap collapses chars to one announcer voice
  (opus, gpt, gemini); no char pool. Both confirmed in `eng_kokoro.py`.
- bark stays wired as `missing_ref_fallback` (not demoted). (opus)
- Plan inaccuracies: voice_ref_kind vs voice_ref_field; kokoro device="cuda" (not
  CPU-capable); bark DOES have per-char presets. (opus, gpt)
- Verify-at-build: chatterbox/dia consume delivery_vector; chain resamples
  24000/22050; kokoro base-model offline cache. (opus, gpt, gemini)

## REJECTED (panel was confidently wrong -- grounded out)
- "Promoting chatterbox is a no-op; vz_* refs are indextts2-namespaced; no
  chatterbox bank entries/profile" (opus MUST-1, gpt assumption). FALSE: the bank
  has 37 `engine="chatterbox"` refs + a `char_chatterbox_v1` profile
  (commercial_clean=true, allowed_voice_banks:[default]). Only a selector is missing.
- "kokoro begin_episode preflight fires for chars" (gemini): moot -- kokoro is
  announcer-only; only relevant under the (cut) char promotion.
- "delivery-vector regex drops punctuation cues" (gemini): partial misread -- it
  scores !/?/... separately and matches space-containing cues on the raw string.

## STILL OPEN (verify-at-build, carried)
chatterbox/dia delivery_vector + fallback re-smoke; timeline resample; kokoro
base-model cache. None block the decision; all are build-time checks.

## Convergence
All 3 panelists + the grounding converge: **chatterbox + indextts2** as the 1-2,
kokoro announcer-only, Qwen3 deferred, bark fallback retained. The only panel
"must-fix" that would have changed the build (chatterbox no-op) was grounded-FALSE.
No surviving new material item remains for the build decision -> converged on the
WHAT; the one open build task (the selector) is specified.
