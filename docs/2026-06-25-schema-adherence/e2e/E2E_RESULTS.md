# SCHEMA-ADHERENCE E2E VALIDATION -- BOTH WRITER LANES PASS (2026-06-25)

Operator-requested GPU validation: a 320-word, ALL-VISUALIZER episode on the
canonical `workflows/otr_scifi_16gb_full.json`, run on BOTH writer lanes (local +
frontier), confirming each renders end-to-end to an OBS final. This exercises the
`structured_call` core that Lever-1 touched -- and the frontier lane is the exact
path (`openrouter:slot-a`) the original Opus bug exhausted.

## VERDICT: BOTH PASS end-to-end -> OBS final + `audio_byte_identical OK`, zero tracebacks.

| lane | writer | OBS final | size | wall | audio_byte_identical |
|---|---|---|---|---|---|
| LOCAL | mistralai/Mistral-Nemo (in-process) | `signal_lost_frozen_awakening_20260625_154315_silent_procgen_blended_final.mp4` | 77.8 MB | 21:03 | OK (4065bb3c3006) |
| FRONTIER | `~openai/gpt-latest` via `openrouter:slot-a` (technical stays local mistral) | `signal_lost_thumb_on_the_relay_20260625_160635_silent_procgen_blended_final.mp4` | 80.7 MB | 17:42 | OK (53dacf07330b) |

Both in `C:\Users\jeffr\Documents\ComfyUI\output\otr\obs\`. Config both lanes:
canonical JSON, all visual roles = `visualizer` (via capability-profile role_overrides),
320 target words / 3 acts / 2 characters, captions burned. Local voice = bark;
frontier voice = kokoro (faster -- operator note). Box reset selectively before each
boot; `:8000` free + VRAM baseline confirmed; box reset clean after.

## Lever-1 fixes observed FIRING LIVE (the point of the test)

- **C3 (structural-rung skip) -- LOCAL.** `normalize_length[mistralai/Mistral-Nemo]`
  attempt 1 failed a Guard1 CONTENT check (`SHORTEN_LINE ... requires non-empty
  new_line` -- a `PostValidationError`, not JSON syntax) -> the ladder went STRAIGHT
  to `attempt 2/3: typed repair`, SKIPPING the structural retry. Exactly the
  token-burn fix.
- **C2 (tolerant clamp) -- FRONTIER.** GPT emitted `intent` fields longer than the
  schema `max_length`; `validate_tolerant_data` logged `coerced ... over-long
  field(s) ... to avoid an abort: intent` repeatedly instead of failing. Model-
  agnostic robustness on a verbose frontier model.
- **The proven bug path, now clean -- FRONTIER.** `normalize_length[openrouter:slot-a]`
  (the EXACT pass + slot the original Opus failure exhausted) ran: GPT's nested
  `BeatEdit` JSON **parsed** (no "Field required" exhaustion -> the alias/tolerant
  validation handled the frontier shape), then a Guard1 content failure routed
  straight to typed repair (C3). **No `StructuredCallFailedError`, no ladder
  exhaustion, no ~90k-token burn** -- the original failure mode is gone.

## Cost / notes

- Frontier OpenRouter spend: ~83,268 tokens total on `~openai/gpt-latest` (~under $1).
  Technical passes stayed on local mistral (free).
- The first local attempt aborted in 47s on a config error of mine (`act_count=2`
  below the 3-act floor for 320 words); re-run at `act_count=3` passed. Not a
  pipeline bug.
- Benign, expected log noise (not failures): `LTX-OPEN HEALTH` warnings (the radio
  open rendered on `visualizer` by design, soft open); `_slice_master_audio ...
  WITHOUT the master content hash` cache caveat; the ComfyUI-Manager sqlite DB init
  error at boot.
- Driver: `scripts/_otr_combo_soak.py` (local, gitignored) loads the canonical
  `workflows/otr_scifi_16gb_full.json` (its own HARD-RULE header) -- no stale copy.
  Two local env knobs added for the frontier lane (`OTR_COMBO_CREATIVE_MODEL`,
  `OTR_COMBO_OR_SLOT_A`).
