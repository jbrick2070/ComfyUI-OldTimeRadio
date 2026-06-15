<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The pass01 summary contradicts both the current grounded `eng_ltx_video.py` and the included sprint plan in several build-affecting places; M0-before-code sequencing is also inconsistent with M1/CW-LTXAV-1.

MUST-FIX BEFORE BUILD:
1. [DECISION SUMMARY #4 / M0 / Architecture A] Wrong model/checkpoint target. The summary says “start Lane B on the distilled v1.1 / 8-step CFG=1 checkpoint” and “Gemma-3-12B encoder OFFLOADED to CPU,” but the sprint plan’s Hardware section says full fp8/distilled 22B-style artifacts are dead under the 14.5 GB ceiling and candidates are GGUF Q3_K_S/Q3_K_M or offloaded/block-swap rows. The grounded `eng_ltx_video.py` also shows the existing “distilled” 8-step path is a text-video 2B/v0.9/T5 or 22B-LoRA sampling mode, not an LTX-AV A2V checkpoint recipe. Concrete fix: rewrite M0 to probe the actual LTX-2.3 A2V graph/artifact set from official template, with candidate rows from the plan’s Hardware section, and do not hard-code “distilled v1.1 / 8-step CFG=1” as the Lane B starting implementation unless M0 graph spec proves that is the A2V node topology and weights.

2. [DECISION SUMMARY #4 vs TICKETS M0] Contradictory “no engine code until M0 passes” vs later tickets allowing CW-LTXAV-1 skeleton before/parallel to M0 in the sprint plan. This creates sequencing ambiguity for coders. Concrete fix: choose one rule. Smallest safe change: keep CPU-only skeleton/wiring allowed before M0, but gate all graph/render implementation behind M0 GO; change DECISION SUMMARY/TICKETS M0 text from “No engine code until M0 passes” to “No graph/heavy render code until M0 passes.”

3. [ARCHITECTURE A / WIRING B / INVARIANTS] Flag semantics conflict with “dropdown-visible” behavior. Architecture says one flag `OTR_ENABLE_LTX_AV`, `@register` unconditional, dropdown-visible, fails closed. But if current dropdown options are derived only from registered engines regardless of usability, that works; if options are filtered through `assert_usable`, flag-off would hide them. The document does not prove the dropdown path. Concrete fix: explicitly state and test that the static per-role dropdown options are populated from registry membership/roles, not usability, and that flag-off degradation happens only at render-time. If current code filters by usability, change the plan to edit that option-building path or abandon dropdown-visible-while-flag-off. [ASSUMPTION: option derivation is not shown in grounding.]

4. [WIRING B / TICKETS M3] Engine naming in force map is underspecified and likely wrong. M3 says `OTR_FORCE_ENGINE_MAP=*=ltx_av_*`, but the actual adapter names are `ltx_av_talk` and `ltx_av_music`, and role compatibility differs. A wildcard engine id is not defined anywhere in the grounding. Concrete fix: replace with explicit compatible mappings, e.g. `announcer_visual=ltx_av_talk,character_video=ltx_av_talk,music_visual=ltx_av_music`, matching CW-LTXAV-4.

5. [CLAUDE PANELIST CRITIQUE / ARCHITECTURE A / M2] Video-only decode path for LTX-2.3 A2V is asserted without grounded node names. The critique names `LTXVSeparateAVLatent` “or the video-only VAE decode,” but no grounded source shows these nodes/classes exist. Concrete fix: move the exact terminal/separation node names to M0 GRAPH SPEC output and make M2 consume those names; before M0, require a fail-closed node gate based on captured classes, not preselected `LTXVSeparateAVLatent`.

6. [WIRING B / prior plan Wiring (e)] New family `audio_conditioned_video` is required but pass01 only says “CAPABILITIES row” and “role_compat” at a high level. The sprint plan explicitly requires `schemas.py FAMILIES += "audio_conditioned_video"` and `FAMILY_REQUIRED_INPUTS` entry. Missing this will break schema/role compatibility tests if implemented only as registry metadata. Concrete fix: in pass01 WIRING/TICKETS M1, explicitly include edits to `schemas.py` family enum/list and required-inputs mapping, plus role_compat supply of `audio_ref` for `music_visual`.

7. [ARCHITECTURE A / TICKETS M1] “assert_usable (flag/Sage/NVML-required/node/weights/dims)” omits the exact ordered error classification from the sprint plan. The plan requires existing `EngineUsabilityReason` values only, no new reason, and dims violations re-raised as `EngineUnusable` rather than raw `ValueError`. Concrete fix: carry over the ordered gate list from “Adapters” in the sprint plan, including NVML fail-closed, node gate via lazy `NODE_CLASS_MAPPINGS`, realpath+size floors, template-None tolerance, and dims exception wrapping.

8. [WIRING B / Prod JSON] “dropdown OPTIONS only, NO new widgets” may be impossible if the workflow JSON stores dropdown options statically per node, but the grounding does not show the JSON/node format. The plan needs exact validation steps and node ids before editing. Concrete fix: add a pre-edit audit step: locate `OTR_VideoDirector` dropdown widgets/options in `otr_scifi_16gb_full.json`, update only option arrays if they exist, and re-run `OTR_WorkflowValidator` plus link/widget audit. If options are generated dynamically, no JSON edit should be made. [ASSUMPTION: JSON structure not shown.]

9. [M0 / Audio] M0 says “hash the output audio track (probe)” in the old sprint plan, while pass01 says LTX output audio is dropped and “never write LTX's audio to disk.” These conflict. Concrete fix: redefine the M0 audio probe to inspect the scratch A2V output only outside OTR for research, then require OTR engine M2 returns no audio path and encoded clip has zero audio streams. Do not make any OTR milestone depend on hashing an LTX-generated output audio track.

10. [DECISION SUMMARY #5 / INVARIANTS] Boomerang OFF for AV lane is correct, but the current `eng_ltx_video.py` has boomerang default ON for `ltx_video`. The pass01 invariant “Boomerang OFF for the AV lane” must not be implemented by changing shared env/defaults that affect `ltx_video`. Concrete fix: implement no boomerang code/path in `eng_ltx_av.py`; do not touch `OTR_LTX_LOOP_VIA_REVERSE`, `_LOOP_VIA_REVERSE_DEFAULT`, or `eng_ltx_video.py`.

SHOULD-FIX:
1. [DECISION SUMMARY #2 / ARCHITECTURE A] “two adapters in one file” is fine, but “one private core” risks sharing resident A2V weights between talk/music and violating the per-clip lifecycle/reclaim discipline from the sprint plan. Concrete fix: specify whether `_LtxAvCore` is per-render or shared singleton; if shared, it must still obey AS-3 lease, teardown reclaim, and post-render below-ceiling checks after each clip.

2. [CLAUDE PANELIST CRITIQUE / WIRING B] “fallback families must be re-verified” is left as a warning, not a build step. Concrete fix: add an M1/M3 checklist item to inspect current registry/fallback resolver and update tests for the exact `humo -> humo_1.7B -> latentsync -> still_kenburns` and `ltx_video -> still_kenburns` chains. Use “verify” until file contents are checked.

3. [TICKETS M2] “per-beat audio-slice conditioning input” omits the old plan’s cache-key bugfix for `_slice_master_audio` using master `mtime_ns + size`. Concrete fix: include that cache-key change and unit test in M2/M3, otherwise stale audio slices can condition the wrong render while byte-identical mux tests still pass.

4. [TICKETS M1] CPU unit tests “no GPU” need explicit heavy-import/cold-import enforcement. Grounded `eng_ltx_video.py` keeps heavy imports lazy; the same must be pinned for `eng_ltx_av.py`. Concrete fix: add cold-import test and AST/no-heavy-import test from the sprint plan.

5. [POLISH D / TICKETS M4] Optical-flow/framediff vs 5/30 gold is not sufficient for talk/lip-sync. The document says character lip-sync is the real win, but no measurable lip-sync acceptance is defined beyond “eyeball.” Concrete fix: require operator A/B labels against HuMo for identical audio/still/seed where possible, and park if not clearly better at acceptable wall/VRAM. [ASSUMPTION: no automated lip-sync metric exists in repo.]

6. [ARCHITECTURE A] Env names differ from existing `eng_ltx_video.py` conventions (`OTR_LTX_AV_TEXT_ENCODER` vs existing `OTR_LTX_T5_ENCODER` / checkpoint name handling). That is acceptable, but defaults/path resolution must be specified. Concrete fix: define exact default search roots and filenames only after M0 inventory; until then `assert_usable` should require explicit envs or documented shared-models defaults from M0.

7. [C BUGS/RISKS] “reset-before-run + watchdog” is not mapped to code. Concrete fix: identify whether this means existing lease/reclaim APIs, a launcher preflight, or a render timeout; otherwise remove it from implementation tickets or convert to a concrete preflight check.

OPTIONAL / NICE-TO-HAVE:
- Add a short README table only after M4, when Lane B has actual measured behavior; before then label it experimental and do not claim lip-sync quality.
- Keep GGUF/Q4 language out of user-facing docs until M0 proves the required ComfyUI-GGUF nodes are installed and stable.

CUT THESE (over-engineering):
1. [POLISH D] “audio-reactive ledger->prompt motion verbs” should be cut from this build. It is a separate Lane-A prompt feature, not required to prove or park LTX-AV, and it risks changing production look while this plan’s invariant says shipped engines/defaults stay untouched.

2. [M4] Optical-flow/framediff benchmarking for announcer/music can be cut from the hard gate. Lane B’s stated unique value is character lip-sync vs HuMo; motion magnitude is already addressed by current `ltx_video` ksampler+boomerang per pass01. Keep only a minimal motion sanity check for non-inert output.

3. [ARCHITECTURE A] “Expose two thin adapters over one shared model core” can be deferred. Build two adapters calling a common graph builder/config helper, but do not add cross-role resident sharing until M0/M4 proves reload cost is the bottleneck and memory residency is safe.

4. [WIRING B] Prod JSON option edits should be cut if the Director dropdowns are registry-driven. Do the audit first; avoid touching `otr_scifi_16gb_full.json` unless static option arrays actually require it.