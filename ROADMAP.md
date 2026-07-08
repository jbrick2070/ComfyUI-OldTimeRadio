# OTR Roadmap

**Branch:** `v2.0-alpha`
**Owner:** Jeffrey A. Brick
**Updated:** 2026-07-08
**Purpose:** release runway after the current and next sprint.

This file starts after the active handoff in `docs/GO_FORWARD_PLAN.md`. Keep
current sprint and next sprint there; keep later sprints here. Old sprint logs
and archaeology belong in `docs/ROADMAP_HISTORY.md` or dated docs.

## North Star

Ship OTR v2 as a user-friendly ComfyUI pack that can generate short old-time
radio episodes with story, voices, visuals, captions, credits, and final video.

The release should not only work on Jeffrey's machine. It should offer clear
paths for:

- no-GPU / procedural users
- all-cloud users
- RTX 8 GB, 12-16 GB, and 24 GB+ users
- Mac users
- AMD users where practical
- RunPod/cloud GPU users

Everything must fail loud. No silent local fallback, no surprise model loads, no
wrong-shaped Partner-node or API calls, and no hidden downgrade that changes the
user's chosen route.

## Before This Roadmap Starts

`docs/GO_FORWARD_PLAN.md` currently owns:

- current sprint: Google BYO API build, starting with `google_tts`
- next sprint: media archive drama seed deck

When those two are done, continue here.

## Release Arc After GO_FORWARD

### 1. Finish Remaining Source Packs

Build the remaining source-bank lanes before broad portability work.

Source packs:

- `public_domain_story`: public-domain text -> faithful condensed radio
  adaptation.
- `shakespeare`: Shakespeare scene/act -> faithful theatrical radio adaptation.
- `original_radio`: fully original LLM-generated radio drama with no external
  source.

#### Public Domain

Public domain is a condensation/adaptation problem, not a seed-invention lane.

Target flow:

```text
public-domain source
-> faithful digest around the configured word target
-> small radio cast
-> ledger
-> dynamic story-native visual style
```

Rules:

- Preserve the original plot path, central conflict, tone, and ending where
  practical.
- Prune cast to a small number of speaking roles.
- Compress exposition and side characters.
- Fail loud if no coherent plot/scene/ending can be identified.
- Default mode should be faithful, not loose "inspired by."

Visual style:

- If the user chooses a specific style, adhere to it.
- If the user chooses `llm_creative`, generate a unique episode style with a
  strict schema and safety cleanup.
- The style should fit the native story rather than impose sci-fi or archive
  grammar.

#### Shakespeare

Shakespeare should be its own parallel source pack, not only generic public
domain.

First scope:

- scene adaptation
- act adaptation

Later:

- whole-play condensation

Rules:

- Preserve iconic characters and relationships more aggressively than generic
  public domain.
- Compress acts/scenes rather than arbitrary chunks.
- Keep the central turn of the scene.
- Modernize only enough for radio clarity.
- Allow elevated diction without flattening the language.
- Prune cast hard, but do not casually merge iconic roles.

#### Original Radio

Fully original LLM-generated radio drama.

No RSS, no source text, no seed words by default. Let the LLM create title,
premise, cast, plot, ending, and optionally a unique visual style.

Target passes:

1. Generate candidate concepts.
2. Score for radio clarity, originality, small-cast feasibility, visual
   potential, and safety.
3. Pick one.
4. Build cast.
5. Build outline.
6. Fill ledger.
7. Repair continuity and voice separation.
8. Pick or generate visual style depending on user setting.
9. Final safety, leak, and cliche cleanup.

Visual style:

- If user selects a registered style, the story must honor it.
- If user selects `llm_creative`, generate a custom visual style for that
  episode with a strict schema.

## 2. Thirty-Word Smoke Sweep

After the source packs are coherent, run short 30-word/random smokes across
each lane.

Goal is not endless testing. Goal is to find and fix the obvious leaks:

- source-bank drift
- story leaks
- gender/cast mistakes
- weak cast separation
- stale sci-fi phrasing in non-sci-fi lanes
- bad coda/source note behavior
- style mismatch
- forbidden content
- overused smoke/fog language
- dull or incoherent episode structure

Stop when the lane is good enough to proceed to portability. Do not polish
forever.

## 3. Cloud And BYO API Paths

Keep "cloud" meanings clear:

- Comfy Cloud / Partner nodes: Comfy Credits path.
- Google BYO API: direct Google/Gemini API path.
- Future BYO providers: separate direct API paths.

Near-term Google ids:

- `google_llm`
- `google_music`
- `google_img`
- `google_vid`

`google_tts` is the current sprint in `docs/GO_FORWARD_PLAN.md`. Later Google
work should follow the same direct-BYO, fail-loud, no-cross-provider-fallback
contract.

## 4. Workflow Portability Profiles

Once source packs are usable, turn portability into explicit profiles/toggles.

Target profiles:

- `no_gpu_procgen`
- `all_cloud`
- `amd_8_16gb`
- `mac_8_16gb`
- `mac_32_64gb`
- `rtx_8gb`
- `rtx_12_16gb`
- `rtx_24gb_plus`

Each profile should document:

- expected hardware
- required models
- optional models
- local vs cloud dependencies
- approximate disk use
- expected render mode
- what fails loud if missing
- known unsupported paths

The canonical workflow remains:

- `workflows/otr_scifi_16gb_full.json`

Exported tier workflows should be generated from the canonical workflow by a
script, not hand-maintained:

- `workflows/exported/otr_no_gpu_procgen.json`
- `workflows/exported/otr_all_cloud.json`
- `workflows/exported/otr_rtx_8gb.json`
- `workflows/exported/otr_rtx_12_16gb.json`
- `workflows/exported/otr_rtx_24gb_plus.json`
- `workflows/exported/otr_mac_8_16gb.json`
- `workflows/exported/otr_mac_32_64gb.json`
- `workflows/exported/otr_amd_8_16gb.json`

Any real node/widget/wiring change must still update the canonical workflow in
the same change.

## 5. RunPod And Cloud GPU Harness

After local profiles exist, build a RunPod-friendly deployment harness.

Target scripts:

- `scripts/runpod/bootstrap.sh`
- `scripts/runpod/start_comfy.sh`
- `scripts/runpod/check_env.py`
- `scripts/runpod/smoke_profile.py`
- `scripts/runpod/collect_logs.py`

Target docs:

- `docs/RUNPOD.md`
- `docs/MACHINE_PROFILES.md`

RunPod matrix:

- cheap 8 GB NVIDIA
- 12-16 GB NVIDIA
- 24 GB+ NVIDIA
- all-cloud/no-local-model path on a small pod

Goal:

- install cleanly
- run at least one tiny profile-specific episode
- fail loudly with useful diagnostics
- collect logs and environment facts for optimization

## 6. Installation Path

Make the project approachable for people using ComfyUI and vibe-coding tools.

Target scripts:

- `scripts/install/windows.ps1`
- `scripts/install/linux_runpod.sh`
- `scripts/install/check_system.py`
- `scripts/install/download_models.py`
- `scripts/install/verify_install.py`

Target docs:

- `docs/INSTALL.md`
- `docs/FIRST_RENDER.md`
- `docs/TROUBLESHOOTING.md`
- `docs/MACHINE_PROFILES.md`

The install docs should answer:

- Which workflow do I load?
- Which models do I need for my machine?
- What API keys or credits are optional?
- What stays local?
- What costs money?
- What should I do when a loud error appears?

## 7. README Polish

The README is the final product pass, after profiles and install scripts are
real.

It should be accurate, friendly, and user-first:

- what OTR is
- screenshots/video examples
- pick-your-machine path
- first render in practical steps
- story source choices
- model/API cost clarity
- troubleshooting
- developer details at the bottom

The README should feel like a usable product guide, not an internal engineering
notebook.

## 8. Ship v2

Ship when:

- source packs are usable
- short smokes catch no obvious leaks
- cloud/direct API paths fail loud and route correctly
- canonical workflow validates
- exported workflows are generated and tested
- install scripts work on at least the main paths
- RunPod harness works on representative pods
- README and first-render docs match reality
- full repo suite and Bug Bible are green

Tagging/promotions remain operator-gated. Pushes to `v2.0-alpha` remain normal
green-chunk workflow.

## Active Principles

- Fix root causes, not shims.
- Fail loud; no silent fallback.
- Keep workflow JSON wired when node/widget surfaces change.
- JSON owns content/config. Python owns validation/routing/execution.
- Prefer small, testable chunks.
- Keep user-facing docs accurate and kind.

## References

- `AGENTS.md`
- `CLAUDE.md`
- `docs/BUG_LOG.md`
- `docs/ROADMAP_HISTORY.md`
- `docs/multimodal-story-schema/MEDIA_ARCHIVE_QA_HANDOFF.md`
- `docs/google_tts_ideas.md`
- `workflows/otr_scifi_16gb_full.json`
