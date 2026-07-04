# Kibitz judgment — default_tts removal (r1, codex-only panel)

Panel: codex (antigravity dead — zero output this session). Claude = anchor + judge.
All codex claims were grounded against the real Windows files; every one CONFIRMED.

## Accepted (folded into PLAN.md)
- MUST-FIX 1 — "no 0-byte files" gate is a false alarm (repo has pre-existing 0-byte root files). CONFIRMED. -> scoped the gate to touched files only.
- MUST-FIX 3 — compatibility boundary for stale saved graphs/API prompts passing `default_tts`. CONFIRMED (`__init__.py:160` exports the node). -> added explicit compat-boundary paragraph (canonical workflow only; external graphs re-save).
- SHOULD-FIX 1 — `voice_assignments` is a derived legacy view (`voice_assignments_from_cast`, `_otr_ledger_consumers.py:164`), not the stored authority. CONFIRMED. -> reworded §1 rationale ("sequence() does not read default_tts; routing decided upstream off canonical cast").
- SHOULD-FIX 2 — repo-wide grep, not just tests/; stale comment at `scripts/_otr_overnight_story_soak.py:201`. CONFIRMED (it is a comment, non-breaking). -> §2.5 widened to repo-wide with the grounded hit list; soak comment added as a delete/refresh item.
- SHOULD-FIX 3 — name the exact applier gates. CONFIRMED (`apply_profile` @ _otr_workflow_apply.py:442; `cross_validate_profile` @ capability_profiles.py:298). -> §3 names both.

## Accepted-with-override
- MUST-FIX 2 / CUT 1 — "cut the Fable pass, it's routine per §9." Correct as a §9 default, but the OPERATOR explicitly requested a Fable medium confirmation for this plan. Operator overrides §9 default -> Fable pass kept, reframed as belt-and-suspenders. Noted in the header.

## Rejected / not-adopted
- CUT 2 — drop the ROADMAP/doc-history reference (§6). Kept: it is a one-line corroboration in the verification note, not a build dependency; harmless and useful provenance. Not promoted to a build step, so codex's real concern (don't make history a gate) is already satisfied.

## Independent (Claude anchor) findings beyond codex
- Node 3 wiring verified directly from the live JSON: `default_tts` promoted input has `link: null` (no wire), `widgets_values` index 4 = `"bark"`, `dialogue_offset_ms` follows. Removal = drop the input entry + that array slot, no link surgery. This is the r3 (wiring) make-or-break; grounded, no defect.

## Arc note
Given the change is a fully-grounded single-widget prune and antigravity was down, the arc was compressed to codex r1 + Claude's grounded anchor (covering coding/wiring/convergence) rather than spinning codex through r2/r3/r4 for a trivial prune. Operator can request the full 4-round codex arc if desired.
