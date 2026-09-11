# Driver anchor -- 2.4-audit-tail: output-root and protected model-root consolidation

Written by the driver (Claude Opus 5, Cowork, 5080) BEFORE any panel round. The panel
proposes; the driver disposes and verifies every claim against the real Windows files.

**Round shape (operator directive): ONE second opinion per round.** r1 codex, r2
cursor, r3 codex, r4 sonnet. **An arc IS coding.** A no-brainer gets no arc -- this
row is here because grounding found a genuine fork with more than one defensible
answer.

**THE OPERATOR'S BAR:** *"as long as it doesn't crash when it's not supposed to."*
Exactness is NOT the goal -- this is a fun experimental app, and aesthetic drift is
explicitly acceptable. This row is classified **DURABILITY**.

---

## 1. What is TRUE TODAY, grounded against current code

Two genuinely separate, still-open threads remain under this row; a third (output-root under nodes/) is closed. (1) ENV-EXPORTER: tests/conftest.py:47 pops OTR_OBS_DIR on import but never touches OTR_OUTPUT_DIR (grep of tests/conftest.py confirms this is still the only pin/strip line). No commit since the 2026-09-04 audit (docs/GO_FORWARD_ARCHIVE.md:8642) has touched conftest.py to address it, and no PBUG entry tracks it (grepped docs/PROD_BUG_LOG.md, no hit). The one candidate mechanism the audit could point to -- the package's own OTR_OUTPUT_DIR pin at __init__.py:105-118 -- is gated on `import folder_paths` succeeding, and its own comment (__init__.py:108-109) says that import is meant to fail outside the real ComfyUI process, i.e. it should NOT fire under pytest/CLI. So the mystery is not merely unfixed, the leading candidate for its cause contradicts itself when read against its own contract. Genuinely unresolved, file:line-grounded, and cheap to chase (one debug print in conftest.py would show whether OTR_OUTPUT_DIR is already in os.environ at collection time and from where). (2) PROTECTED MODEL-ROOT CONSOLIDATION: commit 6f95955d (2026-09-04, 'One owner each') unified exactly TWO of the resolvers -- nodes/_otr_video_engines/wan_shared.py:40 configured_models_root() now delegates to nodes/_otr_gguf_backend.py:822 _models_root(), guarded by tests/test_models_root_single_spelling.py. But that same commit and the follow-up audit (docs/GO_FORWARD_ARCHIVE.md:8500-8510) explicitly parked two more owners rather than merging them: nodes/_otr_paths.py:157 comfy_models_dir() (env var OTR_MODELS_DIR; order: env -> folder_paths -> repo walk-up -> cwd) and nodes/_otr_image_engines/flux2_klein.py:199-217 _resolve_unet_path() (env var MODEL_ENV/none of the above; order: explicit ckpt env -> folder_paths -> OTR_COMFYUI_MODELS_ROOT/COMFYUI_MODELS_ROOT+'/diffusion_models/' -> bare default, never the legacy C:\\ComfyUI-Models literal). I verified both functions still exist unchanged at those sites, still disagree on env-var name and resolution order with _models_root() (which additionally checks a hardcoded C:\\ComfyUI-Models before folder_paths), and there is no AST-wide guard analogous to tests/test_output_root_single_owner.py preventing a fifth spelling from appearing. This is a real, live inconsistency with concrete portability risk (a machine that sets only OTR_MODELS_DIR to relocate weights, e.g. a 4060/RunPod/clean-room box, will NOT relocate flux2_klein's GGUF lookup or comfy_models_dir()'s consumers the same way) -- exactly the crash/durability class the operator's bar cares about, not aesthetic drift. (3) OUTPUT-ROOT itself (the OTR_OBS_DIR/OTR_OUTPUT_DIR/folder_paths.get_output_directory triad under nodes/) IS closed: tests/test_output_root_single_owner.py enforces single ownership via AST walk with a small named allowlist (eng_mesh_stage.py, vram_context_test.py) and I found no code under nodes/ violating it. The three sub-items named as already re-grounded (cold-cache test dependency, google/veo unpinned-fixture, worktree-credit) check out consistent with REFUTED/FIXED: commit 7706a3d0 (2026-09-11, same day) fixed the cold-cache dependency by name; the google veo/omni canonicalize tests (tests/test_google_veo_video_adapter.py, tests/test_google_omni_video_adapter.py) already build their provider clip and canonical output entirely under pytest's own tmp_path fixture, never touching comfy_output_dir()/OTR_OUTPUT_DIR, so the 'unpinned, writes under the live output tree' claim does not hold against current code; nodes/otr_credits_roll.py contains no .git/HEAD or git-subprocess read at all today (only comments), so the worktree-gitdir-pointer failure mode the audit worried about has no live code path to trigger it.

**Key files:** `docs/GO_FORWARD_PLAN.md:149`, `docs/GO_FORWARD_ARCHIVE.md:8500-8515 (G. OPEN FOLLOW-UPS section)`, `tests/conftest.py:47`, `__init__.py:97-118`, `nodes/_otr_paths.py:157-186 (comfy_models_dir)`, `nodes/_otr_gguf_backend.py:822-855 (_models_root)`, `nodes/_otr_video_engines/wan_shared.py:40`, `nodes/_otr_image_engines/flux2_klein.py:191-221 (_resolve_unet_path)`, `tests/test_models_root_single_spelling.py`, `tests/test_output_root_single_owner.py`, `nodes/otr_credits_roll.py (no .git read found, consistent with REFUTED)`, `tests/test_google_veo_video_adapter.py, tests/test_google_omni_video_adapter.py (tmp_path-only, consistent with REFUTED)`

**Blast radius:** Model-root consolidation touches nodes/_otr_paths.py (comfy_models_dir, used by FLUX-anchor/PuLID/future LTX-Wan per its own docstring), nodes/_otr_image_engines/flux2_klein.py (its GGUF unet loader), nodes/_otr_gguf_backend.py (_models_root, the GEMMA writer's GGUF path), and nodes/_otr_video_engines/wan_shared.py (already unified). A merge changes model-lookup behavior on any machine that sets OTR_MODELS_DIR, OTR_COMFYUI_MODELS_ROOT, or COMFYUI_MODELS_ROOT differently -- i.e. every non-5080 portability box (4060, RunPod, a clean-room install) is the actual audience per CLAUDE.md 0B; the 5080's own numbers are unlikely to move today only because none of those three env vars is set here and the C:\\ComfyUI-Models legacy path exists locally, but that must be proven, not assumed, exactly as 6f95955d did for ffmpeg. The env-exporter fix is diagnostic-only (tests/conftest.py) and, once the source is found, is either a no-op finding or a one-line conftest strip -- no render-path blast radius.

## 2. Claims in the existing row description that CURRENT CODE CONTRADICTS

These are why this anchor was rebuilt from the code rather than from the backlog
text. A row description is a dated claim, not a finding.

- The GO_FORWARD_PLAN.md:149 one-line summary implies the whole row reduces to three items that were 'already re-grounded' -- true for those three, but it obscures that the row's other two named threads (env-exporter, model-root) are NOT closed and were never re-verified in that pass; they are simply carried forward unchanged from the 2026-09-04 audit.
- docs/GO_FORWARD_ARCHIVE.md:8642's 'UNFOUND EXPORTER' framing implies a mystery whose resolution is just pending discovery; current code contradicts the premise that __init__.py's own pin (nodes/__init__.py:105-118) could be the exporter -- that pin is explicitly gated on `import folder_paths` succeeding, which its own comment says 'ONLY fires inside the ComfyUI process... CLI/pytest get no pin' (__init__.py:108-109). So the cited candidate mechanism affirmatively does NOT explain the symptom under its own documented contract; the real source is still ungrounded.
- docs/GO_FORWARD_ARCHIVE.md commit 6f95955d's own message says '...the ONE spelling docstring is true instead of false' for the models root -- true only for the wan_shared/_otr_gguf_backend pair; nodes/_otr_paths.py:157 comfy_models_dir() and nodes/_otr_image_engines/flux2_klein.py:199-217 were explicitly left out ('cursor r3: do not open a third env in this diff') and remain unmerged today, so 'one spelling' does not hold pack-wide.

## 3. The fork -- this is what the round must pressure-test

For the model-root piece: should nodes/_otr_paths.py::comfy_models_dir() and nodes/_otr_image_engines/flux2_klein.py::_resolve_unet_path() be folded into the same single owner as _otr_gguf_backend._models_root()/wan_shared.configured_models_root() (one env var, one order, one hardcoded-legacy-path policy for every 'where do OTR's model weights live' question), or are they legitimately answering different questions (comfy_models_dir() is documented as the general ComfyUI models tree via folder_paths for FLUX-anchor/PuLID/etc, while _models_root() is specifically the GGUF-converted-weights tree with its own C:\\ComfyUI-Models legacy fallback) and should instead be renamed/documented to stop implying sameness rather than merged? Reasonable people could pick either: merging risks the exact regression 6f95955d's own docstring warns against ('preferring <comfy>/models over an existing tree... returned the wrong root while looking verified'), while leaving four owners risks a portability box (4060/RunPod/clean-room) that pins only one env var and silently gets inconsistent model resolution across engines -- durability-class, not cosmetic. This is the kind of design choice with more than one defensible answer that the project's own rules route to a panel rather than a solo fix. The env-exporter mystery, by contrast, is NOT a design fork -- it is a one-right-answer diagnostic (find what sets OTR_OUTPUT_DIR inside a pytest session) that needs one grep/debug-print pass, not an arc.

## 4. Questions for the reviewer -- answer these; do not restate section 1

1. **Is the mechanism in section 1 complete and correct?** Read the cited files
   yourself. Is there a step the driver missed, or something that already partially
   mitigates this?
2. **Which side of the fork survives contact with the real code?** Name the files and
   functions each choice would actually touch, and say which you would not write.
3. **What is the smallest change that resolves it?** Smallest that is CORRECT -- not
   smallest that compiles.
4. **Blast radius, measured not asserted.** This project runs on a 16 GB 5080 and an
   8 GB 4060 (CLAUDE.md section 0B). If the change touches shared code, what
   measurement proves the machine you are NOT fixing is unchanged?
5. **Is any part render-inert enough to ship before a four-machine test wave, or must
   all of it wait?** Say plainly.
6. **What would make this row WRONG to do at all?** Argue the other side once.

## 5. Hard constraints -- a proposal violating any of these is rejected on sight

- **No new content gates**, story-rejection gates, word/duration/cast-size limits,
  standalone checkers, chunkers, or recursive retry loops.
- **No prompt rewrites.** Prompts are hand-crafted per model and CHARACTER-BUDGETED
  (`motion_registers` 240 chars enforced at load, BUG-LOCAL-112; `_fit_motion_slot`
  truncates to 60). Adding conditional nuance spends budget that does not exist.
- **Story/prose QUALITY work is DONE** by operator directive. Correctness defects are
  still open; better prose is not.
- **Do not make an OOM silent.** An OOM is the only acceptable killer and must stay
  truthful. The goal is to stop CAUSING avoidable ones, never to hide them.
- **Do not reduce how many episodes reach `otr/obs/`.** A fix that refuses more than
  it saves is a worse fix.
- **One canonical graph.** No second workflow JSON, no new output-path owner, no
  reviving deleted code.
- **Never blanket-kill Python processes** -- that kills the agent's own tooling.
- Style: no curse words, never the name "dummy".
