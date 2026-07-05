# GO FORWARD PLAN - OTR Upstream Transplant

Updated: 2026-07-02 (overnight session, Fable). Single source of truth for
resuming this build. Repo: ComfyUI-OTR-UpstreamStoryLab (transplant
workspace), remote github.com/jbrick2070/ComfyUI-OTR-UpstreamStoryLab.

## CURRENT STEP

Execute the production transplant (chunk T1 below). Everything upstream of
production is DONE and green; production repo is UNTOUCHED.

## What is done (verified tonight)

- Workspace rebuilt: v1 lab archived in git (41c6512), production mirrored
  @ ComfyUI-OldTimeRadio d48a9d76 (SFX-free) with hash manifest
  (PRODUCTION_MIRROR_MANIFEST.md).
- Kibitz arc r1-r4 run over the architecture + coding plan (Codex, Claude
  Code, Antigravity r1; agy quota-died mid-r2 and was dropped per skill
  fallback). Finals: kibitz-runs/2026-07-02-upstream-transplant-v2/r*/final.md.
- v2 lab BUILT and green on the real ComfyUI venv. Gate commands (run these,
  do not trust historical counts - kibitz r4):
  `.venv\Scripts\python.exe -m pytest tests -q`
  `.venv\Scripts\python.exe scripts\validate_lab.py`
  `.venv\Scripts\python.exe scripts\smoke_nodes.py`
  `python scripts\verify_tree.py`
  - JSON owns ALL content: fixtures/banks.json (4 banks, defaults,
    interpreter bindings), fixtures/pipelines.json (legacy descriptive +
    simple_4 executable), 12 seam-complete story packs, 5 visual styles
    (sci_fi_radio byte-pinned to production tails), 3 source packets with
    fixture briefs, 3 PD source folders.
  - Python owns behavior only: contracts.py (strict models, per-seam
    template variable validation), registry.py (fail-loud loader/router,
    auditable Resolution), profiles.py (pack -> profile, no prose),
    interpreters.py (allowlisted bindings), bridge.py (spec + dual meta
    mirrors + adapter article + provenance hashes + round-trip emit),
    preview.py (pack-driven leakage scans), runner.py (simple_4 FakeLLM,
    loud per-pass failure), compat.py (pinned production shapes + AST
    drift extractors).
  - ComfyUI nodes v2: Validator / StoryPackPreview / BridgeArtifactEmit,
    all choices registry-discovered.
- Staged production modules (transplant_work/production_new_modules/, pure
  dict-in/dict-out, tested): _otr_ledger_input_adapter.py,
  _otr_story_prompt_profile.py, _otr_visual_style_policy.py,
  _otr_source_interpreter.py.
- transplant_work/PATCH_PLAN.md: module-level production edit map,
  FALLBACK_INVENTORY table (6 sites, per-lane decisions), widget default +
  append-only + explicit-socket rules. NO line hunks (generated at
  transplant against live HEAD).

## Chunk T1 - production transplant (tomorrow, in ComfyUI-OldTimeRadio)

0. Re-mirror check: compare production HEAD vs d48a9d76; if moved, refresh
   production_mirror + rerun lab drift tests (they fail loudly if shapes
   moved). ALSO (kibitz r4): verify the live RSS import in the production
   venv - `from nodes import story_orchestrator;
   story_orchestrator._fetch_science_news` - and exercise its failure path
   once; story_orchestrator is deliberately not mirrored.
1. Drop the four staged modules into nodes/ verbatim; add production-side
   tests (adapter validated against the real NewsBriefs class).
2. Writer edits per PATCH_PLAN item 1 (resolution before fetch, RSS gated
   science-only, article-dict entry for packet lanes, meta stamps, title
   label, coda facade call, style/style_custom precedence test).
3. line_composer coda facade + grounding (item 2); style_picker kwargs
   (item 3, NEW kwargs); outline/pitch/select/dramatic profile threading
   (items 5-6); science baseline byte-pin tests BEFORE and AFTER.
4. brief_helpers policy seam (item 7) - unstamped meta = byte-identical.
5. Whitelists (item 9), then workflow JSON LAST (item 10) behind the full
   gate list (docs/FABLE_FINAL_REVIEW_2026-07-02.md TEST/VALIDATION GATES).
Visual deep stage (meta_brief/shot_lock/render_driver) stays a separate
later chunk (item 8).

## Hard rules (unchanged)

Only edit production in the explicit transplant chunk. No fallbacks between
lanes ever; grandfathered science quality-floors per FALLBACK_INVENTORY.
No hidden models: slot plan declared, engine ids runtime-stamped. Workflow
JSON: append-only widgets, forceInput sockets, validators green first.

## Kickoff prompt for the next window

"Resume the OTR build. Read docs/GO_FORWARD_PLAN.md in
ComfyUI-OTR-UpstreamStoryLab; current step is chunk T1 (production
transplant). Start with the re-mirror check, then step 1."

## DEFERRED STORY-LLM FIXES (park here; operator 2026-07-02: no story-LLM changes in production until this refactor lands)

- **Director-note leak into spoken dialogue** (proof7 `Lab Race Against Time` b003,
  operator eyeball catch): the composer emitted `Can't hold back the surge much
  longer! Oya's voice should be more tense and urgent.` -- an UN-parenthesized
  writing note inside `line.text`, so TTS read it aloud and the captions showed it.
  The existing production scrubs only catch (...) / [...] spans (stage-direction
  scrub, OTR_LedgerScriptWriter I.6a) and leading self-vocatives (I.6b).
  Proposed fix (drafted + reverted out of production same day per operator hold):
  a pre-freeze scrub (c) that drops whole SENTENCES matching a high-precision
  director-note class -- `<name|his|her|their|the>('s)? voice (should|must|needs to)`,
  `should sound more`, `deliver this line` -- keeping the original text (LOUD warn)
  if the scrub would leave <2 words. Fold this into the v2 writer's line-hygiene
  stage instead (root-cause: the composer prompt should ban delivery notes in
  dialogue; the scrub is defense-in-depth).
