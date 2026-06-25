# R3 JUDGMENT -- dynamic coda segue, wiring (Claude, grounded). CONVERGED.

Panel: GPT-5.5, Gemini-3.1-pro, DeepSeek-v4-pro. Spend ~$0.1110. Convergence: HIGH --
ALL items were build-precision (no architecture change) -> CONVERGED at R3; no R4
(stop-at-convergence; an R4 would be a near-empty "looks good" pass).

## ACCEPTED (folded into the final pass03_plan)
- **Non-deterministic builtin `hash()`** (all 3): salted per-process -> breaks the
  deterministic fallback. Use `hashlib.sha256` mirroring `select_style`. The single
  most important catch (violates a hard invariant).
- **Frozen `LineResult`** (all 3): flags via `dataclasses.replace`, not in-place.
- **`fact` defined first** (GPT#4): clean the brief at the top, before
  generate/validate/fallback.
- **Early-branch ordering** (GPT#1): build `_outro_ending_change`/`_outro_final_char_line`
  INSIDE the else (only the fictional path needs them); the coda branch doesn't.
- **Coda-specific validator** (GPT#6): allow a trailing colon, reject only a LEADING
  speaker label -> do NOT reuse `validate_announcer_line`.
- **Punctuation normalization** (Gemini#5/DeepSeek#3): rstrip trailing punctuation
  before appending the colon (no "truth.:").
- **Reroll = fresh 2-message array** (Gemini#4): no consecutive user messages
  (role-stutter would 400 on strict APIs).
- **Enforce the total length cap** (DeepSeek#1) in the final clean_one_line.

## JUDGE CALLS
- **`premise` source** (GPT#2 bleed risk): use `outline.premise` (setup-framed
  dramatic premise), NOT `script_brief` (can hint the resolution). `intro_text` is
  the SAFE no-spoiler open -> keep it for tone. This closes the only residual bleed.
- **CUT the n-gram copy guard** (GPT CUT#1): the bridge never sees the brief, so the
  guard is low-value AND created a validator-signature mismatch all 3 flagged.
  Cutting it removes the mismatch entirely.
- **CUT the outcome-verb blocklist** (GPT CUT#2/DeepSeek): false-positive risk, low
  value; the structural split is the real safety.

## SIMPLIFICATION CONFIRMED (folds back into the main campaign)
The main-campaign STEP F **climax-line decoupling is now unnecessary**: the ON-flag
coda is a premise->news pivot that never touches the fictional climax, so "protect
the character climax" holds by construction; the OFF path is unchanged. DROP
`climax_character_line` from the build (one fewer edit). Keep `_climax_beat_id` for
KILL 3.

## CONVERGENCE
Architecture (R1) + mechanics (R2) + wiring (R3) settled; only build-precision
remained and is folded. The final spec is `pass03_plan.md`. NEXT: fold into
pass04_plan.md STEP F + CODE_MAP.md C3 + GO_FORWARD_PLAN + the tracker.

## TOTAL coda-segue SPEND (3 LIVE passes)
R1 $0.0898 + R2 $0.0961 + R3 $0.1110 = ~$0.2969. (Announcer campaign $0.51 +
coda-segue $0.30 = ~$0.81 across both.)
