# Claude anchor -- r4 (convergence / residual defects), before the panel

VERDICT: yes-with-minor-fixes. After r1->r3 the plan is build-ready and code-grounded; no
new ARCHITECTURE is needed. Remaining items are spec-precision + two compatibility checks
that belong in the feasibility smokes, not new design.

## Residual items (converging -- none reopen the arc)
1. **GGUF + lightx2v LoRA compatibility (fold into the min-frame smoke).** Step B verifies
   audio cross-attn on a GGUF UNET, but the lightx2v distill LoRA is 14B-shaped and applied
   via `LoraLoaderModelOnly` to the loader's MODEL output. The 33f smoke must ALSO confirm
   the LoRA actually merges onto the GGUF-loaded model (same arch -> should, but verify) --
   else run the GGUF leg LoRA-free at higher steps. Not a blocker; a smoke assertion.
2. **177f worst-case may hard-OOM (expected; handle as data).** The matrix's 177f 14B cell
   can exceed physical VRAM and crash the server mid-render; the runner already records a
   leg error -- treat an OOM at 177f as a RESULT ("does not fit at max beat"), not a harness
   bug. Confirm the abort/teardown path leaves :8000 clean for the next cell.
3. **Probe node must actually be registered + spliced.** The honest-meter node
   (OTR_BakeoffVramProbe) lives in the sibling pack and must be wired into the diagnostic
   graph (post-VAEDecode) for the measured legs + dry-validate must assert it is registered,
   exactly like OTR_BakeoffReclaim. Otherwise the true-demand number silently never logs.
4. **Control caveat -> make it a one-line decision, not an open question.** Production pins
   `humo_1.7B` (portrait); the bakeoff control was `humo_1.7B_169` (wide). CONVERGE: keep
   the wide 14B-vs-1.7B_169 A/B as the aspect-matched quality comparison (what the operator
   already eyeballed), and DROP adding a portrait control -- it would be apples-to-oranges
   against the wide 14B and the operator's goal is the 14B look, not validating 1.7B.
5. **cfg override + no-LoRA env-set must not collide.** The literal cfg rewrite and the
   env-driven lora/steps are independent (different mechanisms, both reflected in the
   manifest cross-check) -- confirm the manifest asserts the FINAL effective values so a
   leg that sets both is still verified.

## Convergence call
No new must-fix expected from r4; if the panel raises only spec-precision (yes-with-fixes),
declare CONVERGED and emit the build-ready final.md (sequence A->B/C->D, kill-gates, the r3
wiring contracts). Stop at r4 -- do not add passes.

## Invariants still guarded
Single-resident <=14.5 (target <=13.5 true-allocated); in-process always-silent; cold-import
clean (probe imports torch lazily); harness diagnostic-only; promotion via
role_overrides.other_beats_visual + node 87, operator-gated; UTF-8 no BOM; SFW; v2.0-alpha.
