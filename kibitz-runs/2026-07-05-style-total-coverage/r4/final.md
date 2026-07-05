# r4 JUDGMENT (Cowork Claude, anchor + judge) -- style total-coverage. ARC COMPLETE.

## Accepted (folded into v5 FINAL)
- CODEX M1 + AG M1: ONE reconciled schema inventory -- 11 str + 4 dict (v5 header block
  is authoritative). AG M1's sharper catch: `_build_char_scene_request` has NO existing
  look text, so `scene_instruction_look` is EXEMPT from the non-empty rule -- sci_fi
  ships "" and the composer appends it ONLY when non-empty (byte-identity by
  construction).
- AG M2: `get_open_subject` maps role "announcer_visual" -> dict key "announcer" in
  Python (the pack key stays "announcer").
- CODEX M3: image prompt OBJECTS (:1731/:1771/:1787/:1808) also gain the additive
  `visual_style` + `prompt_field_source` keys (values per the r3 arm map), alongside the
  render-request observability keys.
- CODEX M2: exact insertion points + sci-fi default texts + `style=None` signatures for
  BOTH LLM request builders specified at build against :1061-1078 / :1094-1115 with
  seam-level byte-identity tests.
- CODEX S1/S2: INPUT_TYPES startup-load and trace-allowlist propagation added to the
  verify checklist. CODEX S3: "B7" defined = the forbidden-import sweep
  (tests/test_b7_forbidden_sweep.py), runs in-suite. CODEX CUT: full-episode identity
  stays operator-acceptance only.
- AG S1: static key set for the env-membership check (already r3; wording tightened).
- AG S2: still_word dicts validate EXACT keys {"noir","sci-fi","western","pulp",
  "default"} case-sensitive. AG OPT: v1-pack load error names the path + "upgrade to v2".
- Both agents' verify-at-build checklists MERGED into the plan's section 5.
- ANCHOR: B-chunk delta tests must assert fields CHANGED from the A1 sci-fi defaults
  (no lazy authoring); 3B 45-test suite passes UNCHANGED through A1.

## Convergence statement
r4 surfaced inventory/wiring reconciliation only -- no new architecture. No conflicts
across r1-r4. **v5 = FINAL, BUILD-READY** (chunks A1 -> A2 -> B -> C, commit+push per
green chunk). Arc: 8 agent calls (codex+antigravity x 4) + 4 anchors; $0 cloud spend.
