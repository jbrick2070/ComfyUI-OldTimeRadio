# Widget Ownership Legend

One page, four labels. Every widget on the canonical graph belongs to exactly
one primary audience below -- but ownership is FOUR AXES, not one column: a
widget has a UI audience, may have a profile baseline, may accept a sanctioned
direct headless override (`-Set` via `patch_creative`), and may be further
constrained by a runtime resolver. Tooltips state the axes that matter for
that widget; this page defines the labels the tooltips use.

Documentation only (2026-08-28 census first build): nothing is hidden,
removed, renamed, or reordered. All 132 serialized widget slots are unchanged.

## Episode

Choices that change what you hear or see in ONE episode, meant for anyone.
Writer: `episode_title`, `num_characters`, `custom_premise`,
`include_act_breaks`, `act_count`, `creativity`, `lemmy_cameo`, `source_bank`,
`visual_style`, `source_ref`. Audio/pacing: `tape_emulation`, Episode
Assembler `crossfade_ms`. Presentation: Caption Burn `burn_captions` /
`caption_style`, Scene-Aware Scopes `landscape_bars`. Reproducibility: Video
and Image Director `seed_mode` / `request_seed`; Image Director granularity.
A stable corpus value does not demote these -- captions, seeds, pacing and
bars remain understandable choices even when production rarely varies them.

## Profile

Machine and execution policy: your hardware profile (variant) pins these, and
they move by profile edit, not per-episode. Writer creative/technical model
and the LLM device/attention/quant/VRAM/context/GGUF policy; voice banks,
cast policy, voice engines/devices, theme engine; video/image role models,
canvas/FPS, device/dtype, max render frames, upscale and final-resolution
policy; validator generation stamps. Note the axes: a profile-owned field may
still have a sanctioned direct override (the writer models do), and a visible
episode choice may still carry a profile baseline.

## Template

Valid advanced or deployment configuration -- pinned in the shipped template,
edited only when you know why. Validator strictness, Writer sampling details,
exchange/validator/news switches, provider slot bindings, `story_scaffold`,
custom model JSON, linked JSON/path fallback slots, sample rate, spatial
width, Haas delay, warmth, LPF, opening/closing timing, freeze readiness,
voice reuse, FFmpeg paths, output paths, blend policy, suffix, shadow/green
policy, manifest/path fallbacks. Pinned means configuration, not deletion.

## Diagnostic

Harness and debugging surfaces that can desynchronize or truncate an episode
if treated as episode knobs. Scene Sequencer `start_line` / `end_line` /
`dialogue_offset_ms`; N7 `episode_title` (an Assembler info/log label, not
the published title) and N12 `episode_title` (last-resort fallback after the
ledger's title fields); N12 `draw_scopes` (legacy, canonical value false);
N92 `mode` / `beats` / `oom_index` / `frame_count` / `engine` /
`portrait_path` / `audio_path` (per-mode consumption is stated in each
tooltip); N93 `bypass` (A/B diagnostic).

`N12 draw_scopes`, `N93 audio_bars`, and `N94 landscape_bars` are three
distinct controls and never substitutes for one another.

## FFmpeg resolution, stated once

Caption burn, master mux, and silent composite resolve: the node's `ffmpeg`
widget value, then the `OTR_FFMPEG` env var, then PATH. Credits has no widget
and resolves env then PATH. `OTR_CAPTION_STYLE` changes caption STYLE only --
no environment variable can enable caption burning.
