# R2 judgment (Claude, judge)

Panel: GPT-5.5, Gemini-3.1-pro, DeepSeek-v4-pro (truncated at token cap), Grok-4.3 (~$0.095). Now grounded
against `GROUNDING_EXCERPTS.md`. Convergence: STRONG and concrete -- the arc became a build spec.

ACCEPTED (folded into pass02_plan.md):
- New fields don't exist in the APIs -> specified exact homes: optional dataclass fields on `Beat`/`LineRequest`
  (default empty -> byte-identical), ledger-visible values ride `meta`/`compose_flags`.
- Do NOT overload `EpisodeBudget.arc_phases`; `beat_role` = separate field + a NEW validator preserving the
  first-failure contract (GPT/Gemini/Grok).
- L1b: deterministic `select_domain` keyword map + curated table + "general" fallback palette; no inline LLM
  (unanimous).
- L1 crisis-repair must not corrupt legit terms ("switch" is in a real title) -> allowed palette includes
  title/premise nouns; whole-token; generated-intent-only; deterministic substitution, no retry (GPT).
- beat_role enum vs "climax" inconsistency resolved (irreversible_choice = climax, last voiced beat).
- L2 content needs a real source: verify a structured character cost field, else a deterministic fallback table;
  field-presence markers not prose regex; full-field fallback beat factory (GPT/Gemini/DeepSeek).
- L3 use an explicit `ACTION:` marker (not bare brackets) + conservative regex; don't persist internal_action
  (GPT/Gemini).
- L4 final-text-only, anchored; mojibake verify-only.
- Flags named now + default-off; audio-affecting => golden re-baseline (GPT).
- Build split: scaffolding-first (flag off, byte-identical) then render-on (GPT).
- Acceptance metric made concrete (densities + cross-episode distinct conflict types) with L5a as prerequisite.

JUDGE CORRECTION (panel misread): Gemini's "ZERO workflow-JSON change is contradicted by adding fields" conflates
the internal Pydantic/dataclass schema with the FROZEN ledger schema (l3-2026-05-14) + the workflow JSON. Adding
optional `LineRequest`/`Beat` dataclass fields with empty defaults is neither -- R3's spine already added
`story_quality_v2_enabled` to `LineRequest` exactly this way. Confirmed allowed; the frozen things stay frozen.

VERIFY (now, for R3 wiring): exact dataclass names/shapes (Beat/OutlineBeat, LineRequest, EpisodeBudget); the
writer call site populating allowed_people/things; the critic too_many_edits/arc="?"/story_quality aggregation;
whether a structured character cost/fear field exists.

Convergence call: NOT yet (expected at R2). R3 must pin the exact signatures + the validator/critic insertion
points; R4 confirms no new must-fix.
