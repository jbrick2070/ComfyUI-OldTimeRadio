# Story-Architecture Increment-1 -- LIVE SMOKE RESULTS

**Date:** 2026-06-23 overnight. **Box:** RTX 5080, headless ComfyUI :8000, LTX
lane (HuMo-free). **Build HEAD:** `bbd0943f` (v2.0-alpha, pushed). **Verdict:**
PASS -- full episode composed -> critic-graded -> rendered -> published an OBS
final with the new levers ON. **No crash. Zero bugs surfaced at runtime.**

## What ran

One episode on the REAL canonical `workflows/otr_scifi_16gb_full.json`, writer
mistral-nemo, 280 words, 2 characters, server booted with the new levers ON:

```
OTR_ENABLE_PITCH_ROOM=1   OTR_ENABLE_CRITIC_ESCALATION=1   OTR_BYPASS_FREEZE_HALT=1
```

(The bypass was set so a deliberate critic EPISODE-escalation could not block the
render -- it lets the critic compute + stamp its telemetry, then ship anyway, so
the smoke proves BOTH the lever firing AND an OBS final. `use_exchange` left at its
workflow default -- see T3 below.)

Driver: `scripts/_otr_smoke_storyarch.py` (throwaway). Full telemetry:
`docs/2026-06-23-story-architecture/smoke_result.json`. Episode: **"Akira's
Resolution"** (`signal_lost_akiras_resolution_20260623_225458`). Wall time ~22.5
min.

## T1 pitch room -- PROVEN LIVE

`[pitch_room] greenlit pitch 1/3 (domain=space, source=local:llm)`. The local
writer generated **3 genuinely divergent premises** and the local greenlight
taste-selected one with a real rationale:

| # | logline | conflict_type | archetype/genre | standoff risk |
|---|---|---|---|---|
| **1 (greenlit)** | geologist rallies crew vs. mission control's timeline to explore a Martian formation | Person vs. Society | naive idealist / sci-fi | 2 |
| 2 | reluctant engineer confronts the team over when to abort on a mystery signal | Person vs. Person | reluctant hero / thriller | 3 |
| 3 | morally-ambiguous geologist races to prove a risky theory as lives hang | Person vs. Nature | anti-hero / noir | 1 |

The winner's brief replaced `script_brief` via `dataclasses.replace`; the premise
carried through to the rendered episode (Mars crater exploration vs. mission
control). This is the primary lever working: it changed WHAT story got told, and
the three premises are NOT the same "console standoff." `meta.story_quality.pitch`
stamped with the full slate + ranking + rationale.

## T2 critic adapter + escalation -- PROVEN LIVE

The 5B story critic returned `arc_verdict=uneven`. The adapter mapped it:

```
[LFC] Wave 1 Agent C escalation: scope=episode reason=Stage 7 critic
verdict='discard' with structural failing_axes=['emotional_arc'] -- the problem
is the arc, not the lines. Whole-episode regenerate.
```

`meta.story_quality.critic_failing_axes = ["emotional_arc"]` and
`critic_regeneration_hint = "Show Akira's determination through action, not just
words.; Reveal new information about the crater...; Akira should face consequences
for defying mission control."` -- synthesized from the critic's reroll-target
hints. `meta.reroll_escalation.scope = "episode"`. With the bypass, the episode
still froze (`frozen_with_doctor_edits`) and shipped. The lever does exactly what
the SPEC intended: a non-strong arc routes to whole-episode escalation instead of
burning line-reroll cycles.

## Full pipeline -- PROVEN

compose -> stage-direction scrub (4 fixed) -> 5B critic (graded, not halted) ->
escalation (episode) -> freeze (`frozen_with_doctor_edits`) -> audio master
assembled -> LTX-AV bookends + visualizer beats (b000-b009, HuMo-free) ->
PostUpscaleProcgenBlend + burned captions -> **OBS final published**:
`otr/obs/signal_lost_akiras_resolution_20260623_225458_silent_procgen_blended_final.mp4`
(48.6 MB, 8 voiced lines). VRAM healthy throughout (pre-render free -> 14.3 GB
free; single resident heavy well under 14.5 GB).

## Bugs found + fixed

- **Build-phase (caught by tests, not the smoke):** a real infinite-loop bug in
  the pitch room's `_distinct_pick` (a modular step sharing a factor with the pool
  size only visited a subset and never reached `count` distinct items) -- found via
  pytest `faulthandler_timeout`, fixed at root with a seeded shuffle. 3 suite
  regressions from the new code (an LLM-slot-tag gap in the pitch room + two
  source-window stage7 wiring tests pushed out of window by the cascade insertion)
  -- all fixed at root (slot tags added; cascade logic extracted to a compact
  testable helper `build_escalation_signal`).
- **Runtime smoke:** none. The levers ran end-to-end on the first try.

## Pre-existing baseline (NOT introduced by this sprint)

The full suite carries **5 pre-existing failures** unrelated to this work, from the
operator's 2026-06-23 HuMo-free workflow UI-save (`267a53e`): pinned 16gb-profile /
workflow-structure / audio-wiring fixtures drifted from the hand-saved JSON
(`test_capability_profiles::test_16gb_profile_extracted_from_master_values`,
`test_workflow_apply` x2, `test_workflow_live_passes_validator::test_production_workflow_visual_structure_pinned`,
`test_full_workflow_v2_audio_wiring::test_force_input_sockets_have_no_widget_key`).
Verified pre-existing by stashing this sprint's edits and re-running. These need a
profile/fixture re-pin (operator's workflow domain), separate from Increment-1.

## T3 (use_exchange) -- DEFERRED with a note

`use_exchange` is a writer BOOLEAN widget (default False), NOT on the
`CREATIVE_WHITELIST`, so headless scripts cannot patch it via `patch_creative`
(it is a managed feature widget). T3's acceptance (a dedicated N=3 run asserting
effective `use_exchange=True` reaches the composer, VRAM <= 14.5 GB, zero slot
drift) needs a separate single-variable GPU run. Left OFF this session; the
operator can flip + validate it during the morning N=3 eyeball soak.

## Net

All three CPU tickets (T4 staging penalty, T1 pitch room, T2 critic adapter) are
BUILT, TESTED (+58 tests), default-OFF / byte-identical, committed + pushed
(`bbd0943f`), and the two primary levers are PROVEN LIVE to a published OBS final.
The build ships dark; the operator flips the flags + eyeballs an N=3 re-soak in
the morning. prod/main GATED.
