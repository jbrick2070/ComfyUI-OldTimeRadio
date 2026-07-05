# Anchor review (Claude, code-grounded) -- Stage 3 visual_style sub-plan v1 -- r1

VERDICT: SOUND ARC, 2 self-flagged risks.

Grounding: the draft was written FROM a fresh grounded site map (Explore agent
sweep, spot-verified: finish_visual_prompt :524/:552, compose_still_prompt
:456/:504/:512/:517, build_radio_host_prompt :297/:363, STYLE_TAIL_DEFAULT :232
matches the lab sci_fi_radio.json positive_tail byte-for-byte).

SELF-FLAGGED RISKS (want panel eyes):
- R1: meta-as-threading-channel is the big architectural call (vs 2C's explicit
  param threading). Pro: every composer already takes `meta`; zero signature
  churn; the widget stamp is one line. Con: implicit -- a composer that builds
  prompts BEFORE the meta stamp lands (or from a meta copy) silently styles
  as sci_fi_radio. Need a grounded check of WHEN meta["visual_style"] is
  stamped vs when the first visual prompt is composed (portraits mint early
  in the writer? or all downstream of the ledger?).
- R2: byte-identity across "every composer" is a strong pin -- get_era_tail has
  3 profiles and compose_still_prompt has role-conditional broadcast-tail
  logic; the 3A tests must cover the matrix, not one happy path.
- R3: scattered direct IMAGE_GRADE_TAIL appends (derive_image_prompts,
  _compose_char_scene_prompt, background_plate, radio-host) -- the draft names
  2; the site map found ~4+. 3A must sweep ALL direct constant reads or the
  suite's byte-identity pins won't catch a missed one (they'd still pass at
  default). An AST/grep guard test (no direct STYLE_TAIL_DEFAULT/
  IMAGE_GRADE_TAIL/RADIO_BROADCAST_TAIL reads outside the pack module +
  extraction test) would pin it structurally.
- Q1-Q4 in the draft are genuine forks for the panel.
