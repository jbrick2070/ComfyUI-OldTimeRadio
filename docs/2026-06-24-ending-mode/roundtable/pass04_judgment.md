# R4 judgment — CONVERGED

Panel: GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro (all returned at --max-tokens
9000). Spend $0.243. Gemini flipped to "yes-with-fixes" — design sound, final
concrete fixes only.

ACCEPTED (grounded CONFIRMED):
- Do NOT rename `ending_mode` -> `ending_flavor`: `render_style_grammar()` reads
  `s['ending_mode']` (CONFIRMED in the catalog). Keep `ending_mode`, ADD
  `ending_tag` + `domain` (Gemini1 / DeepSeek3).
- CUT `ENDING_TAG_BY_SLUG` (redundant; read via `get_style(slug)["ending_tag"]`)
  (DeepSeek-CUT / Gemini-verify).
- Gate the announcer intent via a DIRECT `os.environ` read inside
  `_assemble_outline` (honors the frozen OutlineRequest) (Gemini2).
- CUT the `meta.style` override (risks visualizer/LTX desync) (Gemini-CUT).
- `final_char_beat_id` = key where role == `BEAT_ROLE_IRREVERSIBLE_CHOICE`
  (imported) (Gemini-opt). Remove the stray "last voiced CHARACTER beat" line
  from the announcer section.
- Build chunk 1 authors the 8 ENDING_TEMPLATES + the 100 ending_tag/domain
  assignments (DeepSeek1/2 — build content, not a design hole).

REJECTED: none.

CONVERGENCE: R1 design -> R2 coding contract -> R3 sequencing -> R4 final fixes;
no new structural must-fix. STOP. The lever is DARK / default-OFF / deterministic;
the only risk is the empirical bet, settled by the §J A/B.

## Total roundtable spend (4 rounds): ~$0.66
R1 $0.184 + R2 $0.152 + R3 $0.084 (2 models reasoning-exhausted) + R4 $0.243.
