# A2 native capacity implementation receipt

Base: af9ccb09bd6380abcdbfe5f7bf16459b3fd02749. The pushed commit containing
this receipt is the A2 handoff checkpoint; its exact SHA is supplied with the
operator handoff. Final 21-file Python candidate hashes are in a2_snapshot.json.

Known native capacity now comes from the exact snapshot, AutoConfig and loaded
decoder configuration, nested text_config first. Missing later metadata cannot
downgrade known earlier capacity. Explicit positive context settings constrain
capacity without modifying max_position_embeddings or inventing a memory
allocation. The canonical HF root precedes discovery, and the captured setting
participates in both cache lookup and publication. Epoch/orphan and admission
ownership remain. Direct callers retain the old argument shape; the two new
keyword-only parameters are optional. Allocated GGUF context/reuse is unchanged.

One CPU prompt preparation owner supplies the writer, base, polish and standalone
constrained native paths. Fit occurs before device transfer. Structured inspection
uses the actual schema-enriched messages and explicit planned output budget;
it returns primitive measurements without counting a generation. Inspection may
acquire the configured native model; it does not generate. Remote/GGUF inspection
returns unsupported without acquiring a model.

The default remaining output minimum is one token; explicit atomic/full-budget
contracts remain. No truncation or minimum-model-window refusal. Google inherits
the shared one-token default too; OpenRouter/Comfy retain their explicit floors.
EOS scalar/list/tuple/set completion at exact capacity is accepted. Unmarked None
is a clear programmer TypeError, not permission to infer a larger output budget.
Real OOM/provider/cancellation failures are not replaced with story-size gates.

Validation:

- Final focused: 95 passed after all production and fixture changes.
- Full: 14,197 passed / 51 existing failures / 183 skipped / 1 xfailed.
  Baseline M1 full receipt: 14,154 passed / 53 failures / 183 skipped / 1 xfailed.
  No new failing IDs or changed assertion evidence. Two obsolete capacity-policy
  failing tests were replaced by the actual-capacity contract tests. Four shared
  tracebacks differ only in source excerpts/locations, listed explicitly in
  a2_full_comparison.json; their assertions and exception evidence are unchanged.
  One existing AMD dictionary failure changes display order only.
  The full run includes final EOS/None fixes and the corrected optional-signature
  and true-zero-room fixtures. No quarantine or conftest guard changes.
- Controlled candidate Bible: 30 passed / 10 unchanged failures / 11 skipped /
  3 xfailed, identical to clean af9ccb09 baseline under Bible a7a10b0e.
  No new failing IDs, assertion or payload changes. Candidate resynchronized
  after the final production/test changes; hashes are recorded.
- Actual canonical validator, JSON round-trip and link/widget audit pass:
  23 nodes / 63 links / 37 writer widgets. No node/socket/widget/wiring change.
  Unchanged workflow SHA256:
  d586a286aaee4c039e410ae9a10014c5c7f4ab82d00eac0e9e1cc0564415057c.
- One finished-code CLI review: Gemini 3.8 Flash (High) through Antigravity;
  root grounded every claim. Independent internal follow-up verified the fixes.
  Full dispositions: kibitz-runs/2026-09-10-my-story-a2-qa/r4/judgment.md.

Reproduce with the real Windows ComfyUI venv, PYTHONUTF8=1 and pytest -q
-p no:cacheprovider. Full JUnit: tmp/my_story_a2_final_full.xml. Focused files:
test_generation_budget, test_constrained_generate, test_context_window_precondition,
test_a4_capacity_phase_advances_the_ladder, test_loader_backend_protocol,
test_system_role_fold and test_vram_envelope_c4. Run the Bible's relative
tests/bug_bible_regression.py from its separate root with --pack-dir naming
the recorded controlled candidate. Comparison JSONs preserve the durable result.

Scope limits: no model/GPU/media run in this chunk. Native live-capacity proof,
cleanup conservation, adaptive source reading/review, visuals and Mac evidence
remain in GO_FORWARD. Unknown capacity and remote estimates still need their
separate transport/accounting work. Sci-Fi's legacy repair-fit heuristic remains.
No new PBUG/Bible entry is created from these static findings. This checkpoint
does not claim complete My Story source qualification or a Mac OS-kill cure.

Sonnet CLI completed the requested final synthesis; grounded dispositions are
under kibitz-runs/2026-09-10-my-story-final-sonnet/r4/. Main-tree and controlled
candidate hashes were rechecked before commit and match all 21 tested Python
files; see a2_precommit_verification.json. The current canonical audit passes.

Final operator amendment: finish ALL current campaign coding before prompt
preparation, coordination or live machine tests. The coordinator heartbeat is
paused and uncommitted handoff/hardware prompt drafts were removed. Root remains
sole production coder, beginning C1 next. Mac/4060 do nothing until Jeffrey's
explicit instruction. README matrix freshness passes for68engines; no new cell
was promoted. This code checkpoint is not a test-wave release.
