# 2026-09-19 -- the scanned Shakespeare lane, one day's briefs

Every file here is a claim about the tree AS IT STOOD when the file was
written, and the tree moved seventeen commits in the day. Read in this order
and trust the newest.

**PARKED (operator, evening): the scan lane is closed for the ship.** Final
state: 43 scenes vendored (pt 5 -- Tempestade 1.2 and 3.1 joined Macbeth
1.3, Lear 1.1 and Hamlet 1.1); `--fold` shipped (`fbfbe0d6`); the post-push
review batch `18ffc540` (SOENA head, Ariel's song, three fold gaps, the
window vote and edge) and its round two `23497b6e` (suffix rule, fold
refusal by any pass, `d'Ariel`, the loose head's space). Reviewer verdicts
are files: `RESULT_codex_fold_fbfbe0d6.md`, `RESULT_codex_batch_18ffc540.md`
(21,380 edge rows measured, five identity changes, all furniture). The
NotebookLM thesis material is `PROMPT9_six_lanes_notebooklm_segments.md`
and `notebooklm/SCHEMA_speaker_cue_oddities.md`; lane outputs land in
`notebooklm/`. Everything below is the day as it happened.

**State at the end of the day (`git log cc62b2c1..HEAD` is the receipt):**
* BOUNDARY HALF DONE. `--pages START-END` on main; all eight Spanish windows
  return their scene with the complete English roster (`76f88d8c`).
* SPEAKER HALF OPEN. A perfect span yields zero speeches because Spanish
  cues abbreviate and today's extractor DROPS what it cannot resolve. The
  standing ruling (`d9a5d82a`) says drop nothing: emit an unbound speaker.
  Not yet built.
* Reading order was wrong underneath everything; an opt-in reader fixes it
  (`ae792152`). Hyphen weld and split-heading rejoin fixed (`2bb45e9e`,
  `efee731a`). Both vendored Portuguese scenes regenerated (`2bb45e9e`,
  `2e9ff62e`); corpus glue census zero.

**Live briefs -- these are current:**
`PROMPT7_codex_*` `PROMPT5_grok_*` `PROMPT7_agy_*` `PROMPT_terra_*`
`PROMPT_flash_*` `PROMPT_composer_*` -- six lanes, allocated by what each
proved today (`9b2b55c9`).

**Superseded, kept for the record:**
* `PANEL_PROMPTS_spanish_abbreviated_labels.md` -- its section 1 is refuted
  by its own addendum; HEAD cited is stale; carries a banner saying so.
* `PROMPT2_*`, `PROMPT3_*`, `PROMPT4_*`, `PROMPT6_*` -- answered; their
  results are folded into the commits above and into GO_FORWARD_PLAN §2.

**Diagnoses and audits (still valid as measurements):**
`scan_sources_text_layer_audit.md`, `chinese_edition_diagnosis.md`,
`italian_rusconi_diagnosis.md`, `NEXT_SESSION_PROMPT.md`.
