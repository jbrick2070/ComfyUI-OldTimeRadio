# R4 JUDGMENT -- convergence (Claude, grounded)

Panel: GPT-5.5, Gemini-3.1-pro, DeepSeek-v4-pro. Spend ~$0.1344. All three
verdicts: yes-with-fixes -- and EVERY fix was spec-PRECISION, not redesign. No new
architecture, no new interface, no reopened assumption. => CONVERGED.

## ACCEPTED R4 FIXES (folded into the final pass04_plan)
- STEP E/F literal `script_brief=script_brief` contradiction (Gemini#1, GPT#2):
  pass `"" if _style_grammar_on else script_brief` to BOTH composers; the coda
  also drops the fiction brief from user_parts/fallback (GPT#2 -- no fiction bleed).
- STEP D snippet bugs: `<meta/period or "">` SyntaxError -> `meta.get("period","")`
  (Gemini#3); capture must follow `outline = generate_outline(...)` (DeepSeek#1);
  guard under `if _style_grammar_on` (GPT SHOULD#2); cast from the LOCKED
  `character_cast`, not `led.data["cast"]` (GPT SHOULD#3).
- Self-containment (GPT#4/#6, Gemini#2, DeepSeek#3): inline the full StoryContract
  dataclass, the truncation formula, the role map, and define `NEWS_CODA_LEAD_IN`
  as ONE constant (GPT#3, DeepSeek#2) -- no cross-reference to pass02.
- `validate_news_coda_line`: drop the `key_terms` param, derive internally
  (DeepSeek#2); pin the ending_change overlap threshold (>=3 content tokens len>=4,
  GPT SHOULD#1). 
- Climax-line invariant documented: one climax-class beat today; take the LAST
  climax beat if KILL 3 lifts it (DeepSeek#3).
- Telemetry `sq = meta.setdefault("story_quality", {})`, primitives only (GPT SHOULD#4).
- KILL-4 consequence cleanly OMITTED from the map (not a stub) (DeepSeek CUT#1).

## VERIFIED (panel "verify" -> I checked or pinned)
- `resolved.get("news_seed")` is the established run() local (KILL-1 uses it at
  :3261) -> Gemini SHOULD#1 is a non-issue.
- `render_style_grammar` is now CONSUMED (macro prompt) -> the literal KILL-2
  "zero callers" defect is closed by the build itself (GPT SHOULD#5/CUT#3).
- sound_world appears ONLY in the macro grammar block; phase/beat get story_engine
  only -> audio vocab stays out of the dialogue path (DeepSeek CUT#2 documented).

## REGRESSION SWEEP (do the R3 cuts undercut the goals?) -- NO
- Per-line register cut: the first build proves STRUCTURAL style (outline
  injection) + grounding, not in-dialogue tonal style. Matches the operator thesis
  (structure now; line-craft is the deferred ceiling). Acceptance reads
  "structurally different", honestly.
- Spoiler belt deferred: input starvation is the deterministic guarantee against
  script_brief/news contamination; the only residual (setup-beat self-spoil) is
  low-risk and belt-covered later.

## RESIDUAL = build-time verification only (10-item checklist in the plan)
era source; locked-cast at STEP D; ledger beat_id at climax row; news_close_brief
distinct/non-empty; macro parse unaffected; phase/beat param threading; flag-gated
prompt text; OutlineRequest snapshot fixtures; sole reroll-rebuild site; OFF-flag
golden + audio-byte-identical. None reopen the design.

## CONVERGENCE CALL: CONVERGED at R4.
The design is stable across all four rounds; R4 produced only precision edits, now
folded. Hand the FINAL pass04_plan.md to a CODER window. One operator creative
decision remains (the `NEWS_CODA_LEAD_IN` wording) -- not a build blocker.

## TOTAL SPEND (all 4 LIVE passes)
R1 $0.0927 + R2 $0.0953 + R3 $0.1828 + R4 $0.1344 = ~$0.5052.
