# R2 judgment

Panel: GPT-5.5, Gemini-3.1-pro, DeepSeek-v4-pro. Spend $0.152. Strong convergence
again (both full reviews "not build-ready"; same coding specifics). Gemini save
truncated.

ACCEPTED (grounded CONFIRMED):
- OutlineRequest gains enable_style_grammar/ending_tag/ending_template defaulted
  fields (GPT1/DS3) — mirrors script_brief precedent.
- Catalog data contract: ENDING_TAGS/ENDING_TEMPLATES + per-entry ending_tag +
  domain, rename prose -> ending_flavor, external ENDING_TAG_BY_SLUG, self-check,
  default fallback tag (GPT4/GPT5/DS1/DS7).
- Final-beat detection is a COORDINATE computed in generate_outline, NOT
  validate_beat_roles (GPT2/DS-assumption). Pass is_final_character_beat into
  _build_beat_user_prompt; append-only (GPT1/DS4).
- Announcer-outro: gate the INTENT under the flag (off=exact, on=non-outcome);
  do NOT remove the close (budget validator #7). "last voiced CHARACTER beat"
  (GPT3/DS5).
- Selector needs a domain field per entry + cast_seed tie-break; pipeline order
  roles->ground->inject->stamp (DS2/DS8/DS9).
- Telemetry of selected ending_tag + crisis-noun count (DS6).

REJECTED: none material.

VERIFY-AT-BUILD / carry to R3:
- SEQUENCING: select_style needs the macro premise, which only exists INSIDE
  generate_outline — so the selector must run there (post-macro, pre-beat),
  not in the writer pre-outline. cast_seed must reach generate_outline. This is
  the main wiring item for R3.

Convergence: coding plan settled; R3 = wiring/sequencing.
