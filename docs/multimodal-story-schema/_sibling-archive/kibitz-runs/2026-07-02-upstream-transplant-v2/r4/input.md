# r3 Final - Wiring (synthesized; fixes applied and verified)

Panel: Codex + Claude Code + driver anchor (agy still quota-dead). Round ran
over the LANDED code (fe0a33f). Every accepted claim was applied to the tree
during synthesis and verified: 46 pytest green + smoke + verify_tree on the
ComfyUI venv; validate_lab now covers the full 55-spec matrix.

## Applied (all grounded)

1. One handoff contract (Codex M1): adapter gains `load_bridge_artifact(path)`
   (explicit path -> UTF-8/BOM/JSON checks -> validate); e2e test proves
   emit -> file -> adapter for all three banks (tests/test_bridge_e2e.py).
2. No silent artifact overwrite (Codex M2): emit filename now carries all
   four resolved axes + canonical-hash8; filename-scheme test added.
3. Atomic callee-first transplant order + writer self-test widget-count
   update in the same chunk (Codex M3/M5): PATCH_PLAN ATOMICITY section;
   n_optional==16 assertion called out explicitly.
4. Honest caching (Codex M4 + driver anchor M1): node digest now includes
   production_mirror/ AND PRODUCTION_MIRROR_MANIFEST.md; registry instance
   cached behind the digest (Claude Code S5).
5. provenance.production_baseline is LIVE (Claude Code C1 resolved by
   wiring, not cutting): read from PRODUCTION_MIRROR_MANIFEST.md
   (d48a9d76...), tested. Provenance scope documented: fixtures + baseline
   hash; mirror file contents are validation-time concerns (drift tests +
   node digest) - Claude Code M3 resolved as documented-intentional.
6. Tail name-mapping documented at the seam (Claude Code M1): era/style/
   image_grade/radio_broadcast override keys mapped to policy fields and
   production constants; JSON field names style_tail/radio_broadcast_tail
   banned.
7. close_brief round-trip test across both rename sites (Claude Code M2).
8. validate_lab full matrix = pytest matrix (Codex S2). Whitelist keys named
   exactly + parity test required (Codex S1). Path-threading convention:
   the workflow carries an explicit path STRING to the writer's forceInput
   socket; adapter never scans folders (Claude Code S3; PATCH_PLAN).
9. Science coda double-injection guard noted (Claude Code S4): science lane
   ignores profile.coda_system_prompt (compose_news_coda owns it); the
   transplant adds a test pinning science coda output pre/post.
10. story_orchestrator._fetch_science_news not mirrored -> verify-at-
    transplant note (Codex S3). FetchRequest marked forward-declared
    (Claude Code C2) - kept: it anchors the declared fetcher protocol that
    banks.json names.

## Rejected

- Renaming the override-dict keys to match JSON field names (Claude Code M1
  alternative): the override vocabulary matches the PRODUCTION constant
  roles, which is what the seam consumer thinks in; documentation beats a
  contract change the day before transplant.
- Hiding custom_source_bank from lab dropdowns (Codex OPT): its visible
  fail-loud behavior is the designed UX for the guided lane.

## Carry to r4 (convergence)

- Any residual contradiction between PATCH_PLAN, GO_FORWARD_PLAN, and the
  landed code.
- Confirm the r4 gate list is complete for tomorrow's chunk T1.
