
**Last updated:** 2026-06-09  
**Branch:** `v2.0-alpha`  
**HEAD:** `24e171b` (Phase 3: character_3d dark scaffold — NOT pushed)  
**Commits ahead of origin:** 17+

---

## ACTIVE MISSION

Build the OTR Video Engine per the execution plan:

- Repo: `docs/OTR_VIDEO_ENGINE__EXECUTION-PLAN_v1.4.md`
- Canonical: `C:\Users\jeffr\Documents\otr-video-roundtable\OTR_VIDEO_ENGINE__EXECUTION-PLAN_v1.4.md`

**Current step: Phase 3 character_3d dark scaffold CPU DONE. Next CPU work = operator decides (GPU smokes gating most lanes). Nearest unblocked CPU task = none defined; all remaining lanes are operator/GPU-gated or asset-gated.**

---

## HARD RULES

- Do NOT start / resume / "continue" any other sprint — NOT story-spine, NOT story-pipeline, NOT any audio sprint, NOT any other ROADMAP item. They are PARKED.
- Audio is SHIPPED; the audio script ledger is FROZEN (read-only). Never reopen or modify it.
- Ignore any stale `session_handoff.md` and any memory / ROADMAP entry implying other "active" work. The video engine is the ONLY active build until the operator says otherwise.
- Invariants in force at all times:
  - Byte-identical master audio + mux-LAST
  - Single resident heavy engine, VRAM peak ≤ 14.5 GB (3D engines: 14.0 GB)
  - Cloud / OpenRouter allowed (Jeffrey lifted 100%-local rule 2026-06-04; `feedback_cloud_lanes_ok.md`)
  - Determinism via seed-keyed cache
  - Every in-render fallback LOUD (log + ledger restamp; never silent)
  - V-6: all engines unconditionally imported in `__init__.py`; usability gated in `assert_usable` only
  - V-12: cold-import clean (no torch/diffusers/comfy at module scope)
  - UTF-8 no BOM; SFW; no "dummy" → use "placeholder"
- Commit per green chunk. Do NOT push unprompted.
- UPDATE otr-build-tracker artifact every session — preserve gauge + lanes styling.
- PRIME DIRECTIVE: never hand the operator a script/cmd/PowerShell block to run. Use Desktop Commander first, then Windows MCP. YOU run everything.

---

## WHERE WE ARE

### Phase 3 — character_3d dark scaffold — CPU DONE @ 24e171b

**10 files, 648 insertions. Suite: 3764 pass / 35 skip / 0 fail. Bug Bible: GREEN.**

| File | What |
|------|------|
| `nodes/_otr_video_engines/eng_character_3d.py` | NEW: `Hunyuan3DTalkEngine` + `TrellisTalkEngine`; both `default_roles=()`, `family="character_3d"`, `requires_flag`, `fallback_engine="humo"`; 4-stage `assert_usable` (flag → cu128 sidecar → mesh dir → ARKit template); `load()` → `RuntimeError("dark scaffold")`; `render_clip()` → `NotImplementedError` |
| `nodes/_otr_video_engines/schemas.py` | `FAMILIES` len now 8 (added `"character_3d"`); `FAMILY_REQUIRED_INPUTS["character_3d"] = ("audio_ref", "init_image")`; module-level sync assert |
| `nodes/_otr_video_engines/__init__.py` | V-6 unconditional import block for `eng_character_3d` |
| `tests/test_video_character_3d.py` | NEW: 18 tests (registration, family, role-fit, fail-closed x4, load/render_clip errors, fallback chain, cold-import V-12, V-6 with flags unset) |
| `tests/test_video_schemas_additive.py` | `len == 8`, `"character_3d" in sc.FAMILIES`, `test_character_3d_requires_audio_and_init_image` |
| `scripts/otr_video_dep_pilot.py` | `hunyuan3d_talk` + `trellis_talk` added to `OPT_IN_ENGINES` |
| `nodes/_otr_lmfe_compat.py` | Module-level `import transformers` wrapped in try/except (V-12 fix for base Python 3.11 test env) |
| `tests/test_b7_forbidden_sweep.py` | `assert sys.version_info >= (3, 12)` → `pytest.skip()` on 3.11 |
| `tests/test_voice_mixed_rate_resample.py` | `pytest.importorskip("scipy")` at module level |
| `tests/test_constrained_generate.py` | `pytest.importorskip("transformers"/"lmformatenforcer")` on 3 tests |

**Fallback chain B** for `character_3d` already in `SYNTH_FALLBACKS` (render_driver.py):
`hunyuan3d_talk → humo → latentsync → still_kenburns` — proven in A-S7.5 GPU soak.

### Prior session milestones (this branch)

| Commit | What |
|--------|------|
| `19afaea` | Phase 1: LTX GPU-VERIFIED (probe_f3; CLIPLoader+T5-XXL split; OTR_TEST_MODE VRAM guard at call-sites in eng_ltx_video + eng_humo + eng_wan_i2v; topology test) |
| `1c88c69` | Chunk E cleanbreak: EpisodeAssembler WAV save + link surgery (263/264) + tombstones |
| `8dfb6ca` | Per-beat audio slice |
| `9020ce3` | Image gate |
| `f2e603e` | M1 first watchable episode (tag: m1-first-episode) |
| `f003978` | B-SHIP (tag: B-ship, pushed; origin == local) |

---

## OPEN OPERATOR / GPU GATES

All remaining lanes require operator action before next CPU sprint can be defined:

| Gate | Status | Unblocks |
|------|--------|----------|
| GPU smoke on :8199 (`queue_smoke.py`) | READY — confirm `[EpisodeAssembler] master WAV saved` log + byte-identical mux + VRAM ≤ 14.5 GB | production wire GO-FORWARD #1 complete |
| Real full episode render (HuMo-2D HEAD) | READY — confirm 2D-refactor on real script/cast/audio | 2D-refactor confirm |
| Wan-i2v ckpt on disk (`OTR_WAN_I2V_CKPT`) | BLOCKED: no ckpt | Phase 2 Wan live-verify |
| cu128 toolkit + latentsync sidecar venv | BLOCKED: no cu128 toolkit | Phase 4 latentsync live-verify |
| ~25 real meshes (`OTR_B_MESH_DIR`) + ARKit-52 .npz (`OTR_B_ARKIT_TEMPLATE_NPZ`) + cu128 | BLOCKED: no assets | Phase 5 character_3d LIVE keystone |

---

## FIRST ACTIONS FOR NEXT SESSION

1. Read `docs/OTR_VIDEO_ENGINE__EXECUTION-PLAN_v1.4.md` in full.
2. Summarize the current step's build-vs-stub list + pass/fail assertions to prove comprehension.
3. Ask the operator which GPU gate was completed (if any) to determine whether new CPU work is unblocked.
4. Wait for operator go before writing any code.

---

## PARKED — NOT NOW

- Story-spine sprint
- Story-pipeline v4
- Any audio sprint (audio is SHIPPED)
- Any other ROADMAP item not in the GO-FORWARD video lanes above
