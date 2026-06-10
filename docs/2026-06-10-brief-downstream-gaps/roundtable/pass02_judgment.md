# Pass 02 judgment (Claude, grounded) -- CONVERGED

Panel reviewed pass01_plan against the IMPLEMENTED code (commit c51526b).
Spend: ~$0.02 metered. All three models converged on the same two items;
no new gap families. The loop stops here.

## Dispositions
- **Lipsync-base plan/code divergence** (GPT#1, DS#1 -- the only "must-fix"
  both agreed on): RESOLVED BY CORRECTING THE PLAN, per DeepSeek's framing.
  The face-forward default is FUNCTIONAL (the overlay's landmarker needs a
  mouth; a scene prompt re-breaks the combo lane's face-detect lottery).
  The implemented behavior stands: env override verbatim, else the
  face-forward default; documented in code and now in the plan.
- **get_era_tail "precedence" wording** (GPT#2): the LEGACY composer
  CONCATENATED all non-empty signals in the order atmosphere_line ->
  palette -> v1 lighting (legacy lines 222-229 append all); my port matches
  the legacy. The plan's word "precedence" was wrong, not the code. Plan
  wording fixed to "concatenation in legacy order".
- **max_chars hard guarantee** (GPT#3, DS-SF#1): FIXED -- a final hard cap
  after the clause re-append; pathological small caps can no longer exceed
  max_chars.
- **Trailing-only NO_TEXT preservation** (GPT-SF#1): FIXED -- endswith()
  check instead of substring relocation.
- **Docstring rot in helpers** (GPT-SF#2): FIXED -- module docstring +
  log_story_brief_disposition consumer-ID table refreshed to the live
  lanes (ltx_scene_open / shotlock_m4 / flux_portrait).
- **Acceptance must prove the scene branch ran** (GPT-SF#3): ACCEPTED --
  acceptance adds a grep for "LTX SCENE .* composed from the episode brief".
- **Disposition len() on non-list terms** (Gemini, leaked-reasoning item):
  FIXED -- isinstance guard.
- **_core_tokens empty-token edge in the consistency gate** (Gemini):
  PRE-EXISTING gate detail unrelated to the brief gaps; out of scope, noted
  for the backlog.
- **Writer-comment F4 verify** (GPT-SF#4, DS-opt): already done in c51526b
  (OTR_LedgerScriptWriter ~3988 names finish_visual_prompt); the file was
  not in the panel's grounding set.

## Convergence
Pass 2 produced zero unaddressed must-fix items after judging. CONVERGED.
