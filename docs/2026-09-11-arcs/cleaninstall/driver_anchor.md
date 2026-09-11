# Driver anchor -- 3.4-clean-install: auto-download coverage and where fetch code lives

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

Two entirely different things share the row's language, and only one of them is a real gap:

1. scripts/otr_canonical_api_run.py:261-347 (_assert_profile_models_present) is exactly what the row believed: a refuse-only gate for the AGENT/DEV headless API harness. It checks only `preflight.required_models` filenames declared in a profile JSON against `/object_info`'s server-visible names and raises SystemExit if a *filename* is missing (line ~336-343); non-filename (logical/repo-id) requirements are reported, not checked (line ~344-349). Zero download/fetch/urlretrieve/requests calls exist anywhere in the 514-line file (grep-confirmed). This script never ships to a clean-install user: .comfyignore's `scripts/*` (with only 3 narrow file negations, none of them this one) strips the whole scripts/ tree from every registry/Manager install. So this function's refuse-only nature is real but describes a dev tool, not the clean-install path.

2. The actual clean-install / queue-time-download path is nodes/_otr_visual_assets.py + nodes/_otr_workflow_validator.py + nodes/_otr_visual_asset_download.py -- all under nodes/, which DOES ship in the registry bundle. OTR_WorkflowValidator (wired as node 63, the first node, in workflows/otr_canonical.json) calls ensure_prompt_visual_assets(prompt, unique_id) before the writer executes (nodes/_otr_workflow_validator.py:572-573, 641-642). That function (nodes/_otr_visual_assets.py:560) resolves the LIVE dropdown selections for the three engines in its allowlist -- _COVERED = {\"z_image_turbo\", \"ltx_8gb\", \"stable_audio_3\"} (line 46) -- against a 7-file in-code MANIFEST (lines 19-40: 3 z_image_turbo files, 2 ltx_8gb files, 2 stable_audio_3 files, each pinned to a specific HF repo_id/filename), checks disk space, pins commit+sha256+size via HF metadata (huggingface_hub.get_hf_file_metadata), and downloads through nodes/_otr_visual_asset_download.py's fetch_verified() -- content-hashed, atomically no-clobber published via hardlink, destination resolved through folder_paths (ComfyUI's own model roots), never the hardcoded C:\\ComfyUI-Models. This is GPU-proven, not theoretical: docs/4060_DRILL_LOG.md Step 113 shows a real Comfy-Manager registry install (2.0.0-alpha.24 -> patched forward to the fixes) auto-fetching 36.8 GB and publishing to otr/obs/ with zero manual model placement.

Coverage is intentionally narrow, and this is the row's one genuinely surviving thread: _COVERED only lists z_image_turbo/ltx_8gb/stable_audio_3 -- the shipped canonical's own default-and-one-upgrade path. Every other engine (LTX 2.5, HuMo, Wan, AnimateDiff, flux_gen1, hidream_i1, lumina_image, ideogram4_local, the cloning TTS engines and their reference WAVs) still needs manual placement or the GitHub-only (not registry-shipped) scripts/ fetchers -- exactly as README.md's \"What to install\" table (line ~764) documents candidly, not silently.

ffprobe: not a code gap. nodes/_otr_shared/ffmpeg.py (resolve_ffmpeg, line 79) and nodes/_otr_shared/ffprobe.py (resolve_ffprobe, line 236) are a single, tested (tests/test_ffmpeg_single_resolution.py AST-walks nodes/ to enforce one owner) resolver honoring OTR_FFMPEG/OTR_FFPROBE env pins ahead of PATH, with Windows/macOS install-candidate fallbacks. README.md:97-105 and :764 document ffmpeg/ffprobe as a system prerequisite pip cannot install, with the one real (and openly documented, not hidden) friction point: \"a missing ffmpeg fails at render time with no earlier warning\" / \"fails at the mp4 encode, hours in.\" ensure_prompt_visual_assets never checks ffmpeg/ffprobe presence -- it is out of scope for that HF-weight-fetch mechanism entirely, and no other early (queue-time) ffmpeg-presence check exists anywhere in nodes/.

**Key files:** `scripts/otr_canonical_api_run.py:261-347 (_assert_profile_models_present -- confirmed refuse-only dev harness gate, zero fetch calls in the file, excluded from the registry bundle)`, `nodes/_otr_visual_assets.py:19-46,560 (MANIFEST, _COVERED, ensure_prompt_visual_assets -- the real queue-time auto-download orchestrator, ships in the registry bundle)`, `nodes/_otr_visual_asset_download.py (fetch_verified/download_verified -- hash+size verified, resumable, atomic-publish transfer library)`, `nodes/_otr_workflow_validator.py:572-573,641-642 (wires ensure_prompt_visual_assets into the canonical graph's first node)`, `workflows/otr_canonical.json (node 63 = OTR_WorkflowValidator, confirmed wired)`, `.comfyignore (scripts/* stripped from the registry bundle; nodes/ ships)`, `docs/4060_DRILL_LOG.md Step 112-113 (live proof: real Comfy-Manager registry install -> 36.8 GB auto-fetched -> obs_publish OK)`, `docs/4060_PORTABILITY_ANSWER.md (three-tier auto-download proof matrix, Floor/Stills/Motion)`, `docs/GO_FORWARD_PLAN.md:157 (current, already-self-correcting row text)`, `docs/GO_FORWARD_ARCHIVE.md:2510,8763 (original archived 3.4 spec and the now-shipped-and-archived Section 1.1 it was blocked on)`, `nodes/_otr_shared/ffmpeg.py:79 and nodes/_otr_shared/ffprobe.py:236 (single-owner ffmpeg/ffprobe resolver, tested, env-pinnable)`, `README.md:97-105,764 (documents ffmpeg/ffprobe as a system prerequisite and its late-failure known limitation)`

**Blast radius:** Any further work here touches nodes/_otr_visual_assets.py and/or nodes/_otr_workflow_validator.py, which are SHARED code called by OTR_WorkflowValidator -- node 63, the first node, in workflows/otr_canonical.json. Every machine that runs the canonical workflow (5080 shipping surface, 4060 portability surface, Mac, any fresh registry install) executes this same path, so CLAUDE.md section 0B applies to any change here: prove the untouched machine's behavior is measured-unchanged, not asserted. Extending _COVERED to larger/gated engines (LTX 2.5, HuMo, Wan) additionally risks multi-GB unplanned downloads and disk-margin failures on 8 GB-class machines if not scoped carefully.

## 2. Claims in the existing row description that CURRENT CODE CONTRADICTS

These are why this anchor was rebuilt from the code rather than from the backlog
text. A row description is a dated claim, not a finding.

- "Blocked on Section 1.1" (the row's own archived text, docs/GO_FORWARD_ARCHIVE.md:8763) IS stale, and the CURRENT GO_FORWARD_PLAN.md:157 already says so itself ("there is no 1.1 row") -- confirmed: docs/GO_FORWARD_ARCHIVE.md:2510 shows Section 1.1 ("THE SANCTIONED-GAP CONTROL PATH") shipped and was archived, and the earlier Section 1.1 ("KOKORO-ONNX BACKEND", archive:4118/4796) is also fully archived. No live Section 1.1 exists anywhere in docs/GO_FORWARD_PLAN.md today.
- The row's premise that clean-install has NO queue-time download (only scripts/otr_canonical_api_run.py's refuse-only preflight) is FALSE for the shipped-canonical-default engines. scripts/otr_canonical_api_run.py:261-347 (_assert_profile_models_present) IS confirmed refuse-only with zero fetch/download/urlretrieve/requests calls in the whole 514-line file -- but that script is a DEV-ONLY headless harness (its own docstring: "the small, boring headless entrypoint agents should use") and .comfyignore's `scripts/*` line strips the ENTIRE scripts/ tree from the registry install bundle (confirmed: only 3 narrow negations survive -- _otr_chatterbox_worker.py, _otr_dia_worker.py, otr_mesh_stage_blender.py). It was never the mechanism a clean-install user's preflight runs.
- The archived row's own problem statement ("only the writer LLMs, bark, musicgen and the kokoro voices fetch themselves... every image engine, every video engine, Stable Audio 3... are manual placement behind two fetchers under scripts/ that the registry bundle does not ship") is STALE for exactly the engines the shipped canonical graph uses/upgrades to. A real queue-time auto-download mechanism now lives under nodes/ (ships in the registry bundle, unlike scripts/): nodes/_otr_visual_assets.py's MANIFEST (lines 19-37) + ensure_prompt_visual_assets() (line 560), called from nodes/_otr_workflow_validator.py:572-573 and 641-642, which is OTR_WorkflowValidator -- confirmed wired as node 63, the FIRST node, in workflows/otr_canonical.json (`grep type/Node name for S&R -> OTR_WorkflowValidator`). Backed by nodes/_otr_visual_asset_download.py's hash/size-verified, resumable fetch_verified()/download_verified(). This was built in commits 2f67fc4e/2ca14536/222227b0/9a9a0535/a52e2aef/2aed1e10/e4b5dfec/f6fbb598 (2026-09-06 to 09-07) and published as registry version 2.0.0-alpha.28 (git commit 3117757a, 2026-09-07). It was then PROVEN end-to-end on a real Comfy-Manager registry install (not a git checkout) on physical 8 GB hardware: docs/4060_DRILL_LOG.md Step 112-113 -- Run 3, 2026-09-07 01:06-01:48, 36.8 GB of visual assets (z_image_turbo + ltx_8gb) auto-fetched with pinned revisions and sha256 into the correct model dirs, then `[OTR_MasterAudioMux] obs_publish OK -> otr\obs\`, captioned explicitly: "The FIRST episode ever produced on this card from a registry install rather than a hand-tended tree." docs/4060_PORTABILITY_ANSWER.md separately documents Tier 1 (procgen, 0 visual weights) and Tier 3 (ltx_8gb+z_image_turbo, ~46.5 GB) as PROVEN complete combinations, both with zero manual file placement, no HF token, no API key.

## 3. The fork -- this is what the round must pressure-test

Two narrower, genuine forks survive under the now-closed headline concern -- neither is crash-class, both are "more than one defensible answer":

1. SCOPE OF AUTO-DOWNLOAD COVERAGE. Status quo: _COVERED is deliberately 3 engines (the shipped canonical's own default+upgrade path) -- narrow, proven, low blast-radius. The archived row's original ask ("ONE MANIFEST" covering every image/video engine, every cloning TTS engine, via a single config/model_manifest.json feeding the preflight AND the pod provisioner AND the matrix generator, plus a download_policy widget: auto/ask/never) was NOT built that way and remains a real design choice if anyone wants it: keep the current narrow, provably-correct per-engine allowlist (nodes/_otr_visual_assets.py:_COVERED/MANIFEST), or generalize toward the original unified-manifest + policy-widget design. The former has a live proof (obs_publish OK on real 8GB hardware); the latter is unbuilt and would touch every machine that runs otr_canonical.json (CLAUDE.md 0B: shared-code change, must prove other machines' behavior unchanged) for engines nobody has proven auto-downloading (LTX 2.5 is gated and multi-GB; HuMo/Wan/AnimateDiff are large and OOM-risk per README's own compatibility matrix).

2. FFMPEG/FFPROBE EARLY-PRESENCE CHECK. Status quo: documented external prerequisite, discovered only at final mux/encode (potentially hours into a render). The codebase already has a working precedent for exactly this kind of cheap early gate -- scripts/otr_canonical_api_run.py:261 explicitly exists to "Refuse in SECONDS what would otherwise fail seven minutes into a render" for model weights. Whether an analogous fail-fast ffmpeg/ffprobe presence check belongs in OTR_WorkflowValidator (so a clean-install user without ffmpeg on PATH finds out in seconds, not hours) is a genuine, small, undecided design call -- not a crash bug, since the current failure is loud and documented, just late.

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
