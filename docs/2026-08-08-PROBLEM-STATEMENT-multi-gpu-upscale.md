# PROBLEM STATEMENT -- system-agnostic multi-GPU upscale stage (queue item 8)

**Author:** Claude (Opus 4.7), coder window 2026-08-08.
**Status:** DRAFT for r1 fan-out (Fable cold r1 + Codex + Antigravity panel).
**Not a plan.** This is the framing document. r1 is the arc; r2 is the coding
plan; r3 is the wiring; r4 is convergence.

## 1. Motivation and origin

* `docs/GO_FORWARD_PLAN.md` queue item 8 (operator-ordered 2026-08-07):
  "system-agnostic multi-GPU upscale stage -- built against the profile and
  registry contracts, NEVER a resurrection of the retired NVIDIA-only node".
* `ROADMAP.md:193-196` (the promotion): "a new system-agnostic multi-GPU
  upscale stage -- PROMOTED 2026-08-07 to `docs/GO_FORWARD_PLAN.md` queue
  item 8, after the 23-episode disposition. The constraint travels with
  it: built against the profile and registry contracts, NEVER a
  resurrection of the retired NVIDIA-only node".
* `docs/2026-07-10-lean-mean-rip-final.md` Decision D-2 (operator ratified
  2026-07-10) codicil -- the authoritative constraints for the campaign:

  > rip NOW; a FUTURE system-agnostic multi-GPU upscale campaign is
  > planned for after this rip lands. Design constraints recorded for
  > that campaign:
  > (1) it is a REBUILD against the portability stack -- registry rows
  >     with device_backends per engine (nvidia/amd/mps/cpu), per-tier
  >     profile values, fail-loud on unsupported hardware -- NOT a
  >     resurrection of the vendor-locked RTX-VSR node (git keeps the
  >     reference);
  > (2) it plugs into the EXISTING name-only `upscale_stage` profile
  >     reservation the portability plan deliberately kept (panel tried
  >     to cut it 3x);
  > (3) HONEST-SWITCH LAW: its widgets/profile fields land in the SAME
  >     COMMIT as a working engine, `off` is a real enum value,
  >     selecting an unsupported engine raises -- no cockpit switch
  >     ever again precedes or outlives its machine (the spacesaver
  >     lesson);
  > (4) new widgets APPEND at the end per positional law, variants
  >     regenerate via build_variants.py.

* `docs/2026-07-09-platform-portability-brief.md` section 2.3 (the
  reservation the codicil is talking about):

  > 1080p is the output ceiling for now. Upscalers are acknowledged
  > future work: the switch spec may RESERVE one `upscale_stage`
  > switch (default `off`), but no upscaler design/eval in this
  > campaign. Note `nodes\rtx_upscale.py` exists and is RTX-coupled
  > -- audit it as a portability liability, do not extend it.

* `docs/2026-07-09-platform-portability-final.md` section 7 (the ship
  state of the reservation): "**upscale_stage**: NAME-ONLY reservation,
  default off, nothing built".

## 2. What is genuinely built today (grounded 2026-08-08 against HEAD ebe24bd4)

The reservation is name-only; nothing else in-tree matches the codicil's
contract shape. In particular:

* **`nodes/rtx_upscale.py` (923 LOC, `OTR_RTXUpscale`) is still in the tree
  and still registered.** `__init__.py:254` maps
  `OTR_RTXUpscale -> .nodes.rtx_upscale:RTXUpscale`. It is the "retired
  NVIDIA-only node" the operator warns against reviving; the D-2 rip was
  operator-ratified 2026-07-10 but has NOT shipped. Its dependency
  contract is a hard vendor lock: `import nvvfx` (the NVIDIA Video Effects
  SDK Python binding) at runtime, per-frame `sr.run()` calls through
  `nvvfx.VideoSuperRes`. It is Windows + NVIDIA only. On Mac / AMD / Linux
  it is not merely slow -- it does not import.
* **The canonical workflow does NOT USE it.** `workflows/otr_canonical.json`
  contains no `OTR_RTXUpscale` node (verified by python-parsed node list:
  the 23 nodes are `OTR_SignalLostVideo`, `OTR_CaptionBurn`,
  `OTR_VideoDirector`, `OTR_VideoRenderBatch`, `OTR_PostUpscaleProcgenBlend`
  and 18 non-video nodes). So the live pipeline is already free of the
  RTX-coupled upscale stage; the code file survives as unused registered
  dead weight.
* **No `upscale_stage` field exists in any profile file.** The name-only
  reservation lives in the two portability docs; the profile schema
  (`nodes/_otr_shared/capability_profiles.py:_TOP_LEVEL_KEYS`, and any of
  the 7 shipped profile files including `otr_g4_wan_ti2v.json`) does not
  yet know the word `upscale_stage`. The reservation must therefore be
  MATERIALIZED as a new profile section in the same commit as the working
  engine (HONEST-SWITCH LAW).
* **No `upscale_engines` registry exists.** The three shipped engine
  registries are audio, video, image (each a subclass of the dep-free
  `EngineRegistry` in `nodes/_otr_shared/engine_registry_base.py`). A
  new namespace can be added by instantiating one more registry against
  the shared base; the pattern is the shipped one, not a fresh
  divergence.
* **No `upscale` role or capability declaration in the video registry.**
  `nodes/_otr_video_engines/registry.py:CAPABILITIES` covers video-render
  engines (humo, ltx_video, wan_ti2v, ...) with cloud partners; nothing
  serves an upscale role. Retrofitting the video registry with an
  `upscale` role would blur "engine that renders a clip" with "engine
  that scales a finished clip", which is the misalignment the codicil is
  guarding against.

## 3. What "system-agnostic multi-GPU" actually means, concretely

Two readings live in the operator's brief and both must be closed before
r2 begins. The recommendation below is what this problem statement
argues for; r1 is where the panel disposes.

### 3A. System-agnostic == multi-VENDOR + multi-BACKEND (RECOMMENDED)

The dep-free portability contract already speaks this vocabulary
(`nodes/_otr_shared/capability_profiles.py`):

* `_DEVICE_BACKENDS = ("cuda", "cpu", "mps")` -- ROCm presents as `cuda`,
  so AMD is covered by the CUDA backend token.
* `_GPU_VENDORS = ("nvidia", "amd", "apple", "none")`.
* `_PLATFORMS = ("any", "win", "mac", "linux")`.

A "system-agnostic" upscale stage therefore means every registered engine
declares which backends it can run on (in the same shape the video
registry already uses: `"device_backends": ["cuda", "cpu", "mps"]`) and
the profile's `device_backend` value gates which engines are ELIGIBLE at
selection time. Fail-loud when the operator picks an engine whose
backend list does not include the profile's backend (`INCOMPATIBLE_PROFILE`
via the shipped `EngineUnusable` taxonomy).

### 3B. Multi-GPU == per-device selection (cuda:0 / cuda:1 / mps / cpu) (RECOMMENDED)

On a dual-GPU host (or any host with a torch device other than the render
device), the upscale stage should be able to run OFF the render device so
the render device's VRAM is not disturbed. The concrete surface is one
new profile field, e.g. `upscale_stage.device = "auto" | "cuda:0" |
"cuda:1" | "mps" | "cpu"`, defaulting to `auto` (== render device with a
graceful CPU fallback disabled -- fail-loud if the pick is unavailable,
per operator rule 6 of the portability brief: "NO FALLBACKS").

### 3C. Multi-GPU == distributed compute across HOSTS (REJECTED)

The 4060 laptop reaches `otr/obs/` over a direct Ethernet cable as a
READ-only viewer (mapped drive Z:); it is not a render target. Nothing
in-tree ships a work-queue or a network protocol for offloading a stage
to another machine. Interpreting "multi-GPU" as distributed compute
would create a genuinely new subsystem (transport, health-checks,
retries, artifact fetch) far beyond the codicil's scope. This
interpretation is REJECTED in this problem statement; if the operator
disagrees, r1 escalates.

## 4. Hard constraints (verbatim from the codicil, restated)

1. **No resurrection of `rtx_upscale.py`.** New adapters must not import
   `nvvfx`, must not call `nvvfx.VideoSuperRes`, must not `import` the
   `RTXUpscale` class. Whether the RTX file itself is deleted in this
   commit or left as dead code is an r1 question; the code file is a
   portability liability either way (its registration in `__init__.py`
   surfaces the id `OTR_RTXUpscale` in ComfyUI's node menu even though
   no shipped workflow uses it).
2. **Registry rows with `device_backends` per engine (nvidia/amd/mps/cpu).**
   Every registered upscale engine declares its backend list in the same
   shape the video registry uses.
3. **Per-tier profile values.** The `upscale_stage` reservation becomes
   a real profile section; every shipping profile must set it explicitly
   (defaulting to `{"engine": "off"}` for the tiers that do not use it,
   so the default matches today's behavior byte-for-byte).
4. **Fail-loud on unsupported hardware.** No silent CPU fallback. If the
   selection is unsupported, raise `EngineUnusable` (or the upscale
   namespace's parallel type) with a classified reason and a clear
   message.
5. **HONEST-SWITCH LAW.** Widget + profile field + at least one working
   engine ship in the SAME COMMIT. `"off"` is a real registered engine
   value (a pass-through), not an implicit absence. Selecting an
   unregistered / unsupported engine raises before any bytes are read.
6. **Positional widget law.** Any new writer widget APPENDS at the END
   of `widgets_values`; variants regenerate via `build_variants.py`.
   This is the BUG-LOCAL-097 anti-drift rule the codicil restates.
7. **Ledger discipline.** Every ledger field the upscale stage writes
   has ONE owner. Fields the RTXUpscale node used to write (episode
   dir, upscaled filename, mux report) must have a new owner or be
   explicitly retired -- no silent orphans.
8. **Canonical workflow discipline.** If a new node must be wired into
   `workflows/otr_canonical.json`, the JSON edit ships in the SAME
   change as the code (CLAUDE.md section 0). If the design chooses
   NOT to add a node (e.g. the stage is a helper called from an
   existing composite node), no JSON edit is required and `git diff --
   workflows/` MUST stay empty.

## 5. Success criteria (what "done" looks like)

A leg is the proof.

1. **Suite green.** The pre-existing 9222 / 111 / 1 baseline holds, plus
   the added unit + wiring tests for the new registry namespace, the
   new profile section validator, and the honest-switch guard.
2. **Bug Bible 17 green** at the current survival-guide commit
   (`3759ae5` at the time of writing; re-verify before push).
3. **`git diff -- workflows/` behavior is deliberate.** Either empty (the
   stage is invoked from a helper in an existing node, no wiring
   change) OR exactly the widget/link additions the design demands,
   with `OTR_WorkflowValidator` + widget-count vs `INPUT_TYPES` audit
   both green in the same change.
4. **Non-goal ledger hole test passes.** A test enumerates every ledger
   field the retired RTXUpscale wrote and asserts each has a NEW owner
   (deterministic Python, another pass, or an explicit default) OR is
   explicitly recorded as retired.
5. **Non-CUDA smoke test.** A CPU-only unit test invokes the working
   `off` engine and the working CPU engine on a tiny fixture (e.g. an
   8-frame 64x64 mp4) and asserts (a) bytes-out matches a golden hash
   OR (b) the file exists with correct dimensions and framecount, and
   (c) `obs_publish OK` semantics hold.
6. **NVIDIA leg (deferred to a live-render session, NOT gating the
   commit).** A single 120-word episode leg renders on the 5080 with
   the new stage set to a real upscale engine (recommended candidate:
   Real-ESRGAN via spandrel, per section 6), and shipping receipts
   land: `Prompt executed in ...`, `RESULT SUCCESS`, `obs_publish OK`,
   asset present at `otr/obs/<ep>.mp4`. This is a POST-commit proof;
   commit itself is unit-suite-gated.
7. **Non-NVIDIA leg (deferred, out of scope for this box).** A Mac /
   AMD proof leg cannot run on this box. The design's honest-switch
   guarantees that a Mac profile with `device_backend=mps` selects the
   MPS backend of an upscale engine that declared MPS in its
   `device_backends`; the CI unit for that path is the honest guard.

## 6. Candidate engine backends (menu, r1 fans out)

The problem statement does not pick. r1 chooses which of these ship in
the first commit (the operator wants at least ONE working engine per
the HONEST-SWITCH LAW).

* **`off`** -- pass-through. No processing. Must ship. Runs on every
  backend.
* **`ffmpeg_lanczos`** -- pure CPU ffmpeg `-vf scale=W:H:flags=lanczos`.
  Zero model dependency, runs anywhere ffmpeg does, deterministic,
  slow but reliable. Not a model upscaler -- an honest interpolator.
  Candidate for the "no GPU" and "no torch" tiers.
* **`spandrel_esrgan`** -- Real-ESRGAN (or any spandrel-supported
  architecture) loaded via ComfyUI's built-in
  `UpscaleModelLoader` / `ImageUpscaleWithModel` pattern. Runs on
  cuda, mps, cpu without changing weights. Weights are Apache-2.0
  (Real-ESRGAN) or per-model. Preflight declares the required
  model file id under `preflight.required_models`. RECOMMENDED as
  the first non-trivial engine.
* **`cloud_replicate_esrgan`** or similar -- an opt-in cloud partner
  slot for the cloud profiles (`otr_cloud_low` / `otr_cloud_hq`).
  Deferred as a separate row that lands with its own auth wiring
  once the local engines are proven. r1 says whether to draft the
  cloud row shape in this campaign or park it.

## 7. Non-goals (explicit)

* **Not going above 1080p in this campaign.** The portability brief's
  ceiling ("1080p is the output ceiling for now") still holds. Target
  resolution comes from the profile's `render.composite_w` /
  `render.composite_h`; the upscale stage may take a smaller source
  and reach that composite target, but does not exceed it.
* **Not distributed compute across hosts.** See 3C.
* **Not touching the audio path.** Audio remains `-c:a copy` from the
  source of the stage's input; C7 byte-identity holds. Whatever the
  new stage does, it does to VIDEO ONLY and re-muxes audio unchanged.
* **Not story-quality work.** The 2026-08-04 "story quality is done"
  directive holds. This campaign changes bytes at the visual level
  only; script / gender / voice text are not touched.
* **Not a resurrection of the SFX-bed pattern.** The 2026-08-06 rip
  banned five video engines via `RETIRED_ENGINE_IDS`; the upscale
  namespace mirrors the retired-ids policy from day one (a
  `RETIRED_UPSCALE_ENGINE_IDS` set, even if empty at launch, so the
  policy is in place for the first stale-id day).

## 8. Open questions r1 must close

1. **Rip the retired `rtx_upscale.py` in this same commit, or in a
   separate small commit before this one?** Argument for same-commit:
   HONEST-SWITCH LAW says the cockpit switch (the new `upscale_stage`
   widget) and the working engine ship together; leaving the retired
   `OTR_RTXUpscale` id in the ComfyUI menu next to the new
   `OTR_UpscaleStage` id (or whatever it lands as) invites confusion.
   Argument against: separating cleanly (rip commit -> rebuild
   commit) is easier to revert. Recommendation: SAME commit, and
   also add `rtx_upscale`'s id to a `RETIRED_NODE_CLASS_MAPPINGS`
   set consulted at boot so a stale saved workflow gets a NAMED
   refusal.
2. **New registry namespace or extend the video registry with an
   `upscale` role?** Recommendation: new namespace
   (`nodes/_otr_upscale_engines/`) via the shared `EngineRegistry`
   base, so responsibilities stay clean. r1 verifies.
3. **Stage form: dedicated ComfyUI node, or a helper called by
   `OTR_PostUpscaleProcgenBlend`?** Recommendation: dedicated node
   BEFORE `OTR_PostUpscaleProcgenBlend`, so the procgen blend keeps
   its role of overlaying the CRT flicker on a 1080p canvas. This
   requires the canonical workflow edit; r1 verifies whether the
   canvas math works out.
4. **Profile-section shape.** Recommendation: one new top-level key
   `upscale_stage` in the profile schema with `{engine: str,
   device: str, model: str}` fields; `off` engine ignores model.
   r1 refines.
5. **`IS_CHANGED` semantics.** The stage reads a source mp4 by path;
   a re-render replaces the same path with different bytes.
   Recommendation: `IS_CHANGED` returns the source mp4's sha256
   plus the engine/device/model tuple. r1 verifies against Bug
   Bible's `IS_CHANGED` guidance.
6. **Cloud engine row: draft in this campaign, or defer?**
   Recommendation: defer -- the cloud profiles do not yet run to
   green end-to-end (item 9 chunk 3 is the Macbeth probe blocker).
   Coming back once cloud is proven keeps this campaign focused.
7. **Ledger field ownership.** The RTXUpscale node stamps
   `perfect_run_spacesaver` cleanup, writes to
   `otr/episodes/<ep>/upscaled/<ep>.mp4`, and has never actually
   run in the canonical workflow. r1 enumerates fields to
   confirm no downstream reader silently depends on the retired
   path shape.
8. **Where does `off` mean "no file written" vs "copy the source
   through to the upscaled dir"?** Recommendation: `off` = source
   path passes through unchanged, no bytes copied; the downstream
   `OTR_PostUpscaleProcgenBlend` reads directly from the render's
   composite. Matches the current (RTXUpscale-absent) shape.

## 9. Deliverable shape for r1 (what fan-out receives)

* **Fable-cold input (per the 08-06 standing rule):** this document,
  verbatim, with no driver anchor attached.
* **Codex + Antigravity input (per the two-lane kibitz rule for
  Cowork-driven work):** this document + a `driver_anchor.md` written
  after Fable-cold, checked against the real Windows files via
  Desktop Commander, that grounds every claim above against
  file:line references. r1 folds the panel's judgment into
  `r1/final.md`; r2 opens against that.

## 10. Standing operator directives this campaign obeys

* Story quality is DONE (2026-08-04). No writer / prose work.
* No content guardrails on generated episodes (2026-08-03). No new
  filters.
* Never chase word count (2026-08-03). No length gates.
* Full kibitz arc, `kibitz-plugin:kibitz` by name (2026-08-04). Four
  rounds, Codex + Antigravity, driver-family excluded.
* Two-strikes-then-panel (2026-07-14). Third attempt on the same bug
  always gets a panel.
* Rip an LLM is allowed; a hole in the ledger is not (2026-07-14).
  Enumerate every field the retired stage wrote, give each an owner,
  prove on a live leg.
* Preserve operator-dirty paths (see next-window prompt): do not
  touch `config/profiles/otr_g4_wan_ti2v.json`,
  `config/profiles/otr_sbcov_*.json`, `tmp/*.ps1`, `kibitz/`,
  `config/source_banks/_corpus/`, `uv.lock`.
