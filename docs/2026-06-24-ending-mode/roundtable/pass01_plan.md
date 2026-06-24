# Ending-mode design — R1-hardened plan

Goal: stop the weak local writer collapsing every premise into the
console/kill-switch climax, by giving the FINAL beat a concrete, style-driven
ENDING that reframes the (still on-stage) climax away from the doomsday button.
Ships DARK / default-OFF / byte-identical. 100% local. Deterministic.

## 1. Ending taxonomy — a CLOSED enum (not prose)

The catalog's `ending_mode` strings ("a revelation, a reversal, or an unresolved
final sound") are flavor, NOT the machine contract. Add a closed enum; map every
catalog style to exactly ONE primary tag + keep the prose as `ending_flavor`.

`ENDING_TAGS` (8, mutually exclusive):
`revelation`, `reversal`, `unresolved_final_sound`, `reconciliation`,
`bittersweet_parting`, `ironic_twist`, `quiet_acceptance`, `confession`.

Each tag carries a CONCRETE final-beat template (what literally happens / the
last sound), e.g.:
- `confession` — "the character finally admits the thing they've hidden; the
  scene lands on the admission, not an action."
- `unresolved_final_sound` — "no resolution; end on one telling sound and a line
  that leaves the question open."
- `quiet_acceptance` — "the character chooses to stop fighting; a small, human
  decision, no machinery."
Add a field `ending_tag` to every `_otr_style_catalog.py` entry.

## 2. Where to enforce it — KEEP the role, REFRAME the choice

Do NOT replace `beat_role`. `_otr_story_quality_l12.validate_beat_roles` requires
exactly ONE `irreversible_choice` on the last voiced CHARACTER beat — that
invariant is GOOD (it keeps the climax on-stage; replacing it touches BEAT_ROLES
+ assign + validate + all consumers + tests). Instead:

- Thread the selected style's `ending_tag` + template into the FINAL character
  beat's prompt so the "irreversible choice" is rendered AS that ending (a
  confession / a refusal / a quiet decision), not as a self-destruct. The choice
  stays irreversible and on-stage; only its CONTENT changes.
- Enforcement point: the beat/line prompt for the last voiced character beat
  (`_otr_outline._build_beat_user_prompt` and/or the line composer's final-beat
  request). Gate strictly behind the flag.
- ANNOUNCER-OUTRO FIX (GPT R1): the announcer close is appended AFTER the climax
  character beat and can narrate the outcome, diluting the ending. The ending_tag
  governs the final CHARACTER beat; the announcer outro must stay generic
  (no outcome narration). This pairs with the existing T4 staging penalty (climax
  = final voiced beat).

## 3. Style SELECT — DETERMINISTIC, replaces the LLM inventor

Replace the Pass-1/Pass-2 LLM style picker with a deterministic, cheap selector
over the 100-catalog (no paid call, byte-identity-safe):

- Mirror `select_domain`: keyword-classify the premise + meta (title / logline /
  brief / theme) to a domain, then pick the best-fit style from the catalog.
- Default pool = `non_emergency_slugs()`; an emergency-tagged style is eligible
  ONLY when the article genuinely calls for one (explicit disaster/rescue
  keywords). This keeps the center of gravity off the console.
- Deterministic tie-break keyed off the existing cast/style seed so the C7
  byte-identity gate holds. Runs ONLY when the flag is on; off => the current
  picker path is untouched (byte-identical).

## 4. Anti-trope negative constraint — CUT

Drop the standalone "no countdown / self-destruct / kill-switch" ban. Panel +
Gemini converge: it's redundant and weak models obey negatives poorly. The lever
is the POSITIVE concrete ending tag (§2) plus L1 crisis-noun grounding (§6),
which deterministically substitutes console/lever/gauge for premise-specific
objects.

## 5. The flag + byte-identity

One flag (proposed `OTR_ENABLE_STYLE_GRAMMAR`, env + optional widget). OFF =>
no selector, no ending_tag injection, the current style string + prompts + beat
roles are byte-identical; the C7 audio path is unchanged (assert in a test). ON
=> deterministic selector + ending_tag injection at the final beat.

## 6. Interaction / ordering with existing levers

- Turn L1/L2 crisis-noun grounding (`OTR_STORY_QUALITY_L12`) ON together with the
  style grammar (they are complementary: grammar fixes the climax SHAPE, L1 fixes
  the trope VOCABULARY). Document them as one "story-grammar" bundle.
- T4 staging penalty (on-mic climax) complements §2's announcer-outro fix.
- T2 critic adapter is orthogonal (measurement); leave as-is.

## 7. Validation — baseline first, then A/B

Record baseline (current code) numbers FIRST, then measure the lever:
- crisis-noun density at the FINAL/climax beat (target 0; reuse
  `count_ungrounded_crisis` over `GENERIC_CRISIS_NOUNS`).
- distribution of distinct `ending_tag` across a soak (target: not all one tag;
  >= 80% non-doomsday).
- critic `arc_verdict` mix vs baseline.
- a small graded A/B (lever off vs on) over ~6 episodes on a couple of seeds.

## Hard constraints (carried)

100% local default; ships DARK / default-OFF / byte-identical; deterministic +
C7 seed path holds; edit canonical `otr_scifi_16gb_full.json` in the same change
as any node/widget change; full suite + Bug Bible green; UTF-8 no BOM; SFW; no
new heavy model, no extra paid call.

## Open for R2/R3 (build + wiring)

- Exact data shape for `ending_tag` + template (in `_otr_style_catalog.py`).
- The deterministic selector's home (new helper vs extend `select_domain`) and
  the precise prompt injection points + flag plumbing through `OutlineRequest`.
- The final-beat detection (last voiced CHARACTER beat) reuse from l12.
