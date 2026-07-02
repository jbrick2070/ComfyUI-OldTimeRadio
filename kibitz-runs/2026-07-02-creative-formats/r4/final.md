# r4 synthesis -- CONVERGED

Panel across the arc: Claude Code + Codex (grounded, line-cited every
round); Antigravity benched after a 22-min r1 hang (skill fallback).
12 planned agent calls -> 9 delivered (3 codex + 3 claude + r1 codex +
r1 claude + r1 agy-failed). Zero cloud spend.

r4 verdicts: codex yes-with-fixes (4 must-fix, all contract-class),
claude yes-with-fixes (1 must-fix), anchor yes-with-fixes (3 small).
No cuts anywhere. All folded into CREATIVE_FORMATS_PLAN.md sec 1c
"r4 FINAL CONTRACTS" (a)-(k) + the corrected visual_format semantics
(explicitness vs EFFECTIVE default source -- codex's profile-patching
catch, _otr_workflow_apply.py:433-479, the last real design defect).

ACCEPTED r4: effective-default explicitness + 16gb_full test;
required_inputs=() + format_ctx-not-a-token; lines[] char_id+line_id;
sepia = local PIL only (wording contradiction removed); pin
cloud_kling_lipsync as THE row (avatar variant = V1-failure probe
alt); format_ctx_version absent-vs-stale; goldens home + gate; exact
smoke entrypoint; board-manifest sha to ledger; additive-safety
statement; F1 gate harmonized (S1 + lipsync adapter); status header
fixed.

VERIFY-AT-BUILD (consolidated from both agents + anchor):
1. format_composite added to FAMILIES + FAMILY_REQUIRED_INPUTS; import
   guard passes; schema round-trip.
2. FormatContext _Forbid submodel; build_request_from_shot copies it.
3. Precedence tests incl. 16gb_full profile role overrides.
4. kling payload PATH->VIDEO/AUDIO mapping vs pinned fields.
5. V1: still-frame silent clip accepted; tin-face texture; readability
   (fixture GLB must be CREATED -- none exists in-repo today).
6. V2: chosen multiview row (NEW pin) exports Blender-importable GLB.
7. V3: 4K board paste/scale/rounding sanity.
8. V4: crop/paste +/-2px + portrait face-similarity (verify
   portrait_ledger exposes a COMPARISON fn, not just the hash; add a
   small compare helper if not).
9. Workflow JSON: validator + round-trip + link integrity +
   widget-count + append-only audit on node 87; VERIFY whether
   forceInput gate_in occupies a widgets_values slot (determines the
   exact append position).
10. _SCENE_INIT_FAMILIES must NOT include format_composite.
11. Headless env preflight (flag/credentials/budget/concurrency).
12. Blender gate: follow eng_mesh_stage's existing selftest pattern;
    version >= 4.5.
13. Double-apply: OTR_VISUAL_FORMAT env + widget resolve at ONE point
    in direct(); profile applier does not re-apply.

STATUS: plan converged and BUILD-READY pending prerequisites (cloud S0
remainder -> S1 -> kling adapter; then F1, then F2). Queued in
GO_FORWARD_PLAN behind the cloud sprints; LTX-fixes window holds the
code baton.
