# Question -- 2026-05-10

# Round-robin question — 2026-05-10 — OTR v2.0-alpha downstream visual chain L3 audit

## Context

OTR (Old Time Radio) is a ComfyUI custom-node project that generates 1940s-style sci-fi radio dramas with both an audio path (Bark/Kokoro/AudioGen TTS + procedural SFX + MusicGen) and a visual path (FLUX stills + HuMo lip-sync + LTX motion + composite + RTX upscale).

We just shipped a 7-of-7 "Ledger Consumer Rewrite" sprint (commit eec4718 on `v2.0-alpha`) that ported every audio + critic + video-engine node from the legacy parser-list `script_json` shape to the new L3 production-ledger contract:

```
ledger = {
  "cast": [{"char_id": "c01", "name": "MANFRED", "voice_preset": "v2/en_speaker_6"}, ...],
  "lines": [{"line_id": "l001", "char_id": "c01", "speaker_role": "character", "text": "...", "start_s": 0.0, "dur_s": 4.2, "shot_id": "...", "traits": "..."}, ...],
  "meta": {"gen_params_initial": {"style": "noir mystery", ...}, "schema_version": "l3-2026-05-08", ...},
  "episode_id": "...",
  ...
}
```

The seven shipped consumers (script_critic, batch_bark_generator, kokoro_announcer, scene_sequencer, batch_audiogen_generator, batch_procedural_sfx, video_engine) all use the new helper module `nodes/_otr_ledger_consumers.py`:

```python
from . import _otr_ledger_consumers as _OTRLC
led = _OTRLC.load_ledger(script_json)         # raises ValueError on legacy list shape
plan = _OTRLC.production_plan_or_empty(production_plan_json)
for line in _OTRLC.iter_lines(led, roles={"character"}):
    text     = (line.get("text") or "").strip()
    line_id  = line.get("line_id")
    name     = _OTRLC.speaker_name(led, line)
    preset   = _OTRLC.voice_preset(led, line)
```

And write back via `nodes/_otr_ledger.py` (`patch_line_fields(led_disk, line_id, {...})`, `save_ledger_safe(path, led_disk)` atomic via tempfile + os.replace).

## The current sprint scope

The remaining downstream pieces are the **visual chain** + **utility supporting nodes** + **post-process tail** + **helper API tests** + **B4 LLM prompt audit** + **fresh workflow JSON** + **dry-run gates**.

The visual chain in the production workflow `workflows/otr_scifi_16gb_full.json`:

```
OTR_LLMScriptWriter → OTR_LLMScriptCritic → audio fan-out → ...
                                          → OTR_VideoPlan → OTR_ShotDurationCalculator → OTR_BatchFluxRender (radio bookend + dead env stills)
                                          → OTR_BatchFluxPortraitRender (per-cast portraits)
                                          → OTR_SignalLostVideo → OTR_BatchHumoRender (character lip-sync) → OTR_BatchLTXRender (non-character motion) → OTR_VideoComposite (1080p mux) → OTR_RTXUpscale → OTR_PostUpscaleProcgenBlend
```

The previous session's ROADMAP entry (line 67) predicted:

> "After all 7 audio/critic consumers ship, recon the 4 video files (batch_flux_render.py, batch_humo_render.py, batch_ltx_render.py, video_composite.py). All read ledger from disk (not wire script_json), so they should 'just work' with the L3 format. Confirm. If recon surfaces text-matching or list-index access on ledger.lines[], write a per-file mini-spec; otherwise mark 'AUDITED CLEAN, no rewrite needed'."

## My recon findings

I just completed STEP 0 recon on every active downstream visual node by grepping for the danger patterns (`payload.get("tokens")`, `for x in payload`, `[VOICE: NAME]` regex, `item.get("type") == "dialogue"`, list-index access on `ledger.lines[]`, legacy field names like `character_name` / `voice_traits`).

Findings file by file:

### visual/batch_flux_render.py
- DEAD path `_parse_env_prompts(script_json, ...)` looks for legacy `[{"type": "environment", "description": "..."}]` shape. On L3 ledger dict input it returns `[fallback]` (no crash, just degrades). Default `skip_env_stills=True` (widget) bypasses this entirely.
- LIVE path `_render_and_save_radio_bookend()` reads ledger from disk via `production_ledger.get_ledger()` singleton + `_OTRL.load_ledger_safe()`. Uses `led["meta"]["gen_params_initial"]["style"]` (L3-correct) with `gen_params.style` back-compat. `led.get("scenes")` lookup is L3-orphaned (no `scenes` array in L3) but degrades safely (returns []). Stamps top-level `radio_bookend_path` + `meta.radio_bookend_path` (no per-line writes).
- Verdict: AUDITED CLEAN.

### visual/batch_flux_portrait_render.py
- Reads ledger via `_OTRL.in_flight_ledger_path()` + `_OTRL.load_ledger_safe()`.
- Walks `cast[]` for char_id, name, voice_preset, portrait_path. Uses `iter_lines` semantically by char_id grouping + `resolve_speaker_role(ln)` for the BUG-094 "skip announcer" filter. All L3-native fields.
- Verdict: AUDITED CLEAN.

### nodes/batch_humo_render.py
- Reads ledger via `_load_ledger_with_path()`. Walks `lines[]` using `line_id`, `char_id`, `speaker_role`, `start_s`, `dur_s`, `text`, `word_count`, `shot_id`. All L3-native.
- Has an orphan-rescue helper `_rescue_orphan_line_char_ids` that falls through `ln.get("speaker") or ln.get("name") or ln.get("character_name")` ONLY when char_id misses cast[]. On clean L3 data, char_id always resolves and the fallback chain never fires. Even when it does fire, it's a fuzzy match into the cast[] table (no parser-list assumptions).
- Verdict: AUDITED CLEAN.

### nodes/batch_ltx_render.py
- Reads ledger via `_load_ledger`/`_OTRL.load_ledger_safe`. Walks `lines[]` using `line_id`, `speaker_role`, `dur_s`. All L3-native.
- `_build_ltx_role_prompt(role, line, ledger)` returns a static prompt by `speaker_role` (no field interpolation).
- Verdict: AUDITED CLEAN.

### nodes/video_composite.py
- Reads ledger via `_load_ledger_with_path`. Walks `lines[]` using `line_id`, `speaker_role` (default "character" on missing), `start_s`, `dur_s`. All L3-native.
- BUG-LOCAL-129a static-radio fill triggered when LTX clip missing or `dur_s == 0`. BUG-135 motion-loop fill uses an existing LTX clip for music/sfx/gap segments.
- Verdict: AUDITED CLEAN.

### nodes/rtx_upscale.py
- Path-in/path-out wrapper around RTXVideoSuperResolution. The only ledger read is for the spacesaver cleanup feature: reads `meta.perfect_run_spacesaver` flag + `episode_id`. Both top-level/meta — fully L3 compatible.
- Verdict: AUDITED CLEAN.

### nodes/otr_post_upscale_procgen_blend.py
- Path-in/path-out wrapper. Uses `_OTRL.in_flight_ledger_path()` + `load_ledger_safe()` for episode_id discovery. No `lines[]` reads.
- Verdict: AUDITED CLEAN.

### nodes/otr_save_to_episode_workspace.py
- IMAGE save sink. Uses `_OTRL.in_flight_ledger_path()` + episode_id only. No `lines[]` reads.
- Verdict: AUDITED CLEAN.

### nodes/otr_video_plan.py, nodes/otr_shot_duration_calculator.py
- Pre-FLUX adapters. Take `production_plan_json` (Director output) and emit shot/compose plans. Don't touch script_json/ledger directly.
- Verdict: AUDITED CLEAN.

### nodes/post_audio_video_pipeline.py
- RETIRED per __init__.py comment ("RETIRED in favour of in-graph batch nodes"). Kept registered for backward-compat workflow loading only. Not in the active production workflow.
- Verdict: RETIRED, audit moot.

## My consolidated read

The entire active downstream visual chain is already L3-aware. ROADMAP's prediction holds: every video node reads ledger from disk via `in_flight_ledger_path()` / `load_ledger_safe()` (not from `script_json` wire), uses L3-native field names exclusively (`line_id`, `char_id`, `speaker_role`, `start_s`, `dur_s`, `text`, `word_count`, `shot_id`, `cast[].char_id`, `cast[].name`, `meta.gen_params_initial.style`), and degrades gracefully on missing fields with `.get(...)` defaults.

This means STEPS 1-7 of my sprint plan (rewrite tasks) collapse into a single recon-verdict deliverable: AUDITED CLEAN, no rewrites needed.

## Question for the round-robin

Given the recon findings above, is the **AUDITED CLEAN, no rewrites needed** verdict for the entire downstream visual chain correct, or are there hidden bugs / drift / edge cases I'm missing?

Specifically:

1. **Is the dead-code `_parse_env_prompts` in batch_flux_render.py worth a defensive tightening** (e.g., delete the function + remove the `script_json` widget input) even though the default workflow has `skip_env_stills=True`? Or leave it as inert legacy?

2. **Is the orphan-rescue fallback chain `ln.get("speaker") or ln.get("name") or ln.get("character_name")` in batch_humo_render.py a future trip wire** if a downstream LLM upgrade changes how the ledger writer stamps lines? Or is it correctly defensive?

3. **Are there any L3 field names I should be writing tests for** that visual nodes are reading but aren't covered by `tests/test_otr_ledger_consumers.py` (which I'll write next)? Specifically, any visual-only fields like `radio_bookend_path`, `meta.gen_params_initial.style`, `cast[].portrait_path`?

4. **Workflow JSON migration:** the production `workflows/otr_scifi_16gb_full.json` still uses `OTR_LLMScriptWriter` (legacy writer that emits parser-list `script_json`). The L3 consumers (Bark, Kokoro, Sequencer, AudioGen, ProcSFX, Video) raise `ValueError` on legacy list input. So today the workflow is broken end-to-end for the L3 consumers — they'd all crash on Round 1 `load_ledger`. The fix is to swap node #1 from `OTR_LLMScriptWriter` to `OTR_LedgerScriptWriter` (the v2 ledger writer registered in `__init__.py`). **Is the `OTR_LedgerScriptWriter` socket count + types compatible with the existing critic socket wiring (`script` + `script_json` outputs)?**

5. **Any other downstream-of-writer concerns** I'm not seeing — ProductionLedger singleton lifecycle issues, multi-episode state bleed-through, save_ledger_safe atomicity edge cases?

Be concrete. Cite file paths + line numbers when prescribing changes. If a piece is genuinely OK as-is, say so explicitly so I don't go fix something that doesn't need fixing. The goal is to not break a working pipeline by editing inert code.
