# r2 kibitz — judgment log (no-fallbacks rip)

Panel this round: **Codex** (read-only sandbox) + **Claude anchor/judge**.
Antigravity = operator manual run (AGY_MANUAL_PROMPT.md, queued). Fable = deferred
to the R1-execution final gate (§9), not the doc.

Hardened deltas folded into `docs/2026-07-03-no-fallbacks-rip/PLAN.md` §E (E1-E9 +
rejected/scoped-out). Grounding results:

ACCEPTED (CONFIRMED against real files):
- E1 no `MISSING_REF` enum → use MISSING_MODEL + detail. (registry.py enum read.)
- E2 `_resolve_character_voices_fail_soft` is 3 behaviors; KEEP the announcer
  reroute (routing, not fallback), rip only preset-synth + orphan-reassign.
- E3 story_orchestrator.py = 2711 lines; plan's :4100-5393 LLM citations are stale
  (subagent hallucinated). RE-GROUND all R3 targets before coding.
- E4 body_score raise inert unless caller `except` narrowed. (same for grammar/
  contract soft-fails.)
- E5 no project `ValidationError` → ValueError / named WriterValidationFailure.
- E6 collapse R1a+R1b into one commit (cast_lock feeds the voice nodes).
- E7 add a writer-side pre-freeze assert for empty voiced lines (TTS raise = defense).
- E8 define "explicit-but-unresolved" precisely for the image slot raise.
- E9 verify `_bark_health_check_for_cast` has a live caller before ripping.

REJECTED / SCOPED:
- Workflow-JSON audit CUT from R1-R3 (runtime-only); applies to R5 only.
- C2 live-catalog dropdown not implementable live → operator "park a latest"
  (cached alias); C2 stays OUT of R1-R4.

Codex usage: 1 call, ~3.0M input / 12.6k output tokens, model auto-selected.
No cloud spend (local sandbox). MISREAD/hallucinated: none material — Codex's
citations were accurate; it CORRECTED the plan's own stale numbers.
