# Pass03 judgment log (judge: claude-fable-5) -- CONVERGENCE PASS
Panel: gpt-5.5, gemini-3.1-pro (verdict: yes-with-fixes), grok-4.3, deepseek-v4-pro on pass02_plan.md. Spend $0.3075.
Three passes running, ZERO challenges to the architecture (master + generated tiers + profile layer). Pass03 items are
implementation-precision refinements of pass02 mechanics. All folded into the FINAL plan; convergence declared below.

## ACCEPTED (folded into the final plan)
1. **Method name**: validator FUNCTION is `validate`, not `execute` (GPT#1). Folded.
2. **Env persistence across prompts**: long-running server keeps os.environ; stamped runs must SET the ceiling EVERY
   execution; conflict with an externally-set value = LOUD warn + SMALLER value wins (safety); wrapper_bridge reads env
   at dispatch time, not import time (GPT#2). Folded.
3. **Gate wiring corrected** (GPT#3/#4 + Gemini#2 merged): use the EXISTING `gate_in` (`forceInput=True`) on
   `OTR_VideoDirector` (node 87) -- zero schema change, no widget drift; S0 verifies whether the image lane
   (`OTR_ImageDirector` 88 -> `OTR_ImageGenDispatcher` 91) is downstream of 87's gate; if not, ONE forceInput-only
   gate input is added to the image dispatcher type. Never widget-backed gates.
4. **Node-63 widget padding**: master `widgets_values` padded to the new length in the SAME commit as the INPUT_TYPES
   change; widget-vector tests updated in the same S2 acceptance (Gemini#3, Grok#1). Folded.
5. **validate_anyway may NEVER skip the profile assertion**; CI check: no shipped snapshot has validate_anyway=false
   (GPT#5 + OPT). Folded.
6. **Precedence formulas made explicit**: normal availability = reality INTERSECT env-gates INTERSECT enabled(P);
   FORCED availability = reality INTERSECT env-gates (explicitly NOT enabled(P)), LOUD; `OTR_FORCE_ENGINE_MAP` stays
   VIDEO-scoped in v1 (it exists only in render_driver) (GPT#6/#7). Folded.
7. **Coverage test algorithm corrected**: enumerate from the PROFILE MAPPING in BOTH directions (every profile field
   has a graph target; every managed widget is applier-only), detect engine widgets by COMBO CHOICES intersecting
   registry ids (not saved values), include feature BOOLEANs + seed-policy widgets, type-keyed only -- my "nodes 19/88"
   id slip is exactly why raw ids are banned (GPT#8/#9/#10, DeepSeek#1/#7). Folded.
8. **Determinism contract narrowed and grounded** (GPT#11, DeepSeek#4): gate = given identical script/ledger fixture +
   pinned seeds (incl. OTR_CAST_SEED/OTR_STYLE_SEED set BY THE GATE HARNESS only), the render pipeline produces
   normalized-identical outputs. Writer/LLM stage excluded (OS-entropy + remote sampling are out of contract).
   Production default keeps OS entropy (invariant intact). Folded.
9. **Profile propagation to runtime**: validator (when stamped) exports `OTR_ACTIVE_PROFILE` + `OTR_VRAM_CEILING_MB` +
   `OTR_SNAPSHOT_HASH` (sha256 of the validated file); render_driver's ledger restamp + registries read env; headless
   sets the same vars directly (Gemini#4 + GPT#12 merged). `master_hash` = sha256 of master content at emit time
   (DeepSeek OPT). Folded.
10. **role_overrides = single engine id per role widget**; fallback chains stay registry-owned (GPT#13). Folded.
11. **Adapter parity**: INPUT_TYPES->object_info adapter must replicate `_serialized_slot_names` semantics (forceInput
    no-slot; seed companions) and is TESTED against that function; the OTR package root must be imported (registry
    self-registration) BEFORE schema read; CI identity gate runs on the offline adapter, live /object_info cross-check
    in soak lane (GPT S1/S4, Gemini#1). Folded.
12. **`_load_workflow` AND `IS_CHANGED` both get repo-root resolution**, merged before any .gen.json is emitted
    (GPT S2, Grok#4). Folded with explicit ordering note.
13. **Parity matcher concrete**: `del prompt["<node63 id>"]["inputs"][field]` on BOTH sides by node-type lookup;
    identity gate needs no ignore (both sides have empty stamps) (Gemini S1, GPT S5, DeepSeek#5). Folded.
14. **queue_smoke**: remote-slot pickers go through a whitelisted helper; prints resolved profile LOUD (GPT S3,
    Grok S1, Gemini OPT). Folded.
15. **Cold-import strategy made concrete** (DeepSeek#2 + ASSUMPTION): lazy imports inside functions; dep-free registry
    tables; FALLBACK defined -- an adapter that cannot be import-safe registers behind try/except and is simply absent
    from that env's roster (enable-set already tolerates absence; generator only saves enabled(P) ids), so the
    approach cannot be invalidated by one stubborn adapter. Folded into S1.
16. **Cold-load test reuses the admissibility paths** (`_is_openrouter_admissible`/`_is_comfy_admissible`) and runs for
    ALL tiers, not just cpu_floor (DeepSeek#3, Grok S3). Folded.
17. **cpu_floor profile committed in S0 but non-shipping** until the S1 cold-import + S3 cold-load gates pass (GPT S7). Folded.
18. **Manifest default = selected + fallback-chain engines** with optional "all compatible" mode; v1 registry metadata
    minimal (enable-set needs only class/toolchain/sidecar); downloader detail stays S5 (GPT S8 + CUT3). Folded.
19. **Downshift reason->suggestion table** (cuda missing -> cpu_floor; VRAM<10GB -> 8gb_lite; mac -> cpu_floor) (GPT S9). Folded.
20. **seed_policy covers every `request_seed` widget by NAME across node types** (ImageDirector has one too) (DeepSeek#8). Folded.
21. **Two-heavy-roles regression test** guarding the no-static-co-residency decision (Grok#3). Folded into S1 tests.
22. **Mapping doc = checked-in JSON consumed by applier + tests** (Grok#2). Folded into S0 deliverable.
23. Niceties folded: profile application report artifact (S3, optional); `.gen.json` banner in `extra.info`; audio-may-
    run-before-assertion note in validator docstring + LOUD queue_smoke log (Grok S1).

## REJECTED (with grounds)
- **DeepSeek CUT: drop the ordering wire because UI `order` fields (63=0) already guarantee execution order.** MISREAD:
  the UI JSON `order` field is a layout artifact; the SERVER executes by topology over the API prompt (which does not
  carry `order`). Output-node iteration order is version-dependent, not contractual. The explicit gate wire stays.
- **DeepSeek CUT: defer `master_hash` stamp field.** Kept: one generator line; it is the only way to answer "which
  master produced this snapshot" during support; cutting it saves nothing material.
- **Grok CUT: parity gate on 16gb only.** Rejected: tier files are exactly where drift bites; the check is a cheap dict
  compare per tier.
- **Grok CUT: defer the wizard.** Already S5/after core; no further deferral needed (GPT's matching cut said "after
  S4", which S5 already is).

## Convergence call: STOP (converged at planning altitude)
Pass01 challenged architecture seams; pass02 challenged design mechanics; pass03 only refined implementation details of
pass02's fixes -- a textbook diminishing-returns curve with the direction unchallenged throughout. Every pass03 item is
folded; what remains unverifiable is exactly the set of S0/S1 build-time gates (real widget names, cold-import reality,
image-lane gate topology), which only code execution can answer. A pass04 would re-review prose about facts the panel
cannot see. Final plan: `2026-06-10-switchable-workflow-architecture__decision-and-plan.md` (repo docs/).
Total campaign spend: $0.7798 (pass01 $0.2237 + pass02 $0.2486 + pass03 $0.3075).
