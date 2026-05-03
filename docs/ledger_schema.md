# OTR Production Ledger schema

**Owner:** Jeffrey Brick
**Current version:** `l3-2026-05-02`
**Source of truth:** `nodes/_otr_ledger.py::CURRENT_SCHEMA_VERSION`

## What the ledger is

A single JSON file written for every episode the OTR pipeline produces. Tracks the cast, scene structure, dialogue lines, audio cues, music cues, video clips, and final deliverable paths. Lives at `<ep_dir>/audio/<episode_id>_ledger.json` (per-episode workspace layout, post 2026-05-02 EVENING reorg) or `output/audio/<episode_id>_ledger.json` (legacy flat layout).

Two write paths:

1. **`Ledger` class** in `nodes/production_ledger.py` — the in-memory authoritative writer. Owns the canonical filename, calls `_merge_with_disk` before each save to preserve fields written by audio nodes between its saves.
2. **`save_ledger_safe(path, ledger)`** in `nodes/_otr_ledger.py` — the schema-l3 helpers used by audio nodes (Bark, MusicGen, AudioGen, AudioEnhance, EpisodeAssembler) to round-trip the JSON.

Both stamp `schema_version` (top-level + nested under `meta`) and `meta.paths` (Phase E, BUG-LOCAL-018) on every write.

## Top-level fields

| Field | Type | Set by | Required |
|---|---|---|---|
| `schema_version` | string | both writers | yes |
| `episode_id` | string | LLMScriptWriter (initial) → SignalLostVideo (finalized via `Ledger.rename_episode`) | yes |
| `commit` | string | Ledger init | yes |
| `total_episode_dur_s` | float | SignalLostVideo / SceneSequencer | no until finalized |
| `total_char_count` | int | Ledger._recompute_totals | yes |
| `total_word_count` | int | same | yes |
| `total_dialogue_lines` | int | same | yes |
| `total_beats` | int | Ledger.set_beats | yes |
| `cast` | list[dict] | LLMScriptWriter / LLMDirector | yes |
| `scenes` | list[dict] | LLMDirector | yes |
| `shots` | list[dict] | LLMDirector | yes |
| `beats` | list[dict] | SceneSequencer / build_silent_test_episode | yes |
| `lines` | list[dict] | LLMScriptWriter / SceneSequencer | yes |
| `sfx` | list[dict] | LLMDirector / SceneSequencer | yes |
| `music` | list[dict] | LLMDirector / MusicGenTheme | yes |
| `clips` | list[dict] | BatchHumoRender / BatchLTXRender | yes |
| `final_audio_path` | string | SignalLostVideo | yes (post-finalization) |
| `final_video_path` | string | SignalLostVideo / RTXUpscale | yes (post-finalization) |
| `meta` | dict | both writers | yes |
| `audio_gates` | list[dict] | audio nodes (l3 expansion) | optional |
| `transitions` | list[dict] | VideoComposite | optional |
| `radio_bookend_path` | string | VideoComposite | optional |

## `meta` block

| Field | Type | Set by | Notes |
|---|---|---|---|
| `meta.schema_version` | string | both writers | mirrors top-level for redundancy |
| `meta.paths` | dict | both writers | NEW in `l3-2026-05-02` (Phase E, BUG-LOCAL-018) |
| `meta.perfect_run_spacesaver` | bool | LLMScriptWriter (widget stamp) | RTXUpscale reads this to decide cleanup |
| `meta.gen_params_initial` | dict | LLMScriptWriter | forensics |
| `meta.news_seed` | dict | LLMScriptWriter | forensics |
| `meta.log_paths` | dict | LLMScriptWriter | forensics |
| `meta.phase_ms` | dict | audio nodes (l3) | per-stage timing |
| `meta.vram_test_results` | list | vram_test_results node | test-only |

## `meta.paths` block (Phase E, l3-2026-05-02)

Resolved at every save from the actual on-disk ledger location. Self-correcting — if `Ledger.rename_episode` (Phase B) moves the per-episode dir between saves, the next save's `meta.paths` reflects the new location automatically.

Per-episode workspace layout (`meta.paths.layout == "per-episode-workspace"`):

| Field | Type | Example |
|---|---|---|
| `meta.paths.layout` | string | `"per-episode-workspace"` |
| `meta.paths.ledger_path` | string | `C:/.../output/otr/episodes/<ep>/audio/<ep>_ledger.json` |
| `meta.paths.episode_root` | string | `C:/.../output/otr/episodes/<ep>` |
| `meta.paths.audio_dir` | string | `C:/.../output/otr/episodes/<ep>/audio` |
| `meta.paths.stills_dir` | string | `C:/.../output/otr/episodes/<ep>/stills` |
| `meta.paths.portraits_dir` | string | `C:/.../output/otr/episodes/<ep>/portraits` |
| `meta.paths.videos_dir` | string | `C:/.../output/otr/episodes/<ep>/videos` |
| `meta.paths.composited_dir` | string | `C:/.../output/otr/episodes/<ep>/composited` |
| `meta.paths.obs_dir` | string | `C:/.../output/otr/obs` (only if `output/otr/obs/` exists) |
| `meta.paths.obs_final` | string | `C:/.../output/otr/obs/<ep>.mp4` (only if `output/otr/` is detected) |

Legacy flat layout (`meta.paths.layout == "legacy-flat"`):

| Field | Type | Example |
|---|---|---|
| `meta.paths.layout` | string | `"legacy-flat"` |
| `meta.paths.ledger_path` | string | `C:/.../output/audio/<ep>_ledger.json` |
| `meta.paths.episode_root` | string | `C:/.../output/audio` (degenerate; same as audio_dir) |
| `meta.paths.audio_dir` | string | `C:/.../output/audio` |

### Why `meta.paths`

Before Phase E, downstream nodes reconstructed paths from `episode_id` (e.g. `f"{ep_id}_treatment.txt"`). That works when the slug round-trips perfectly between writers and readers, but breaks when:

- `episode_id` slug differs from the actual on-disk filename (the slug-mismatch trap closed by Phase B)
- The per-episode dir was renamed by `Ledger.rename_episode` and the in-memory `episode_id` advanced but a downstream node still uses an old reference
- A test fixture or recovery scenario placed the episode dir somewhere non-standard

`meta.paths` resolves this once, at write time, from the actual on-disk truth. Readers look up by name (`led["meta"]["paths"]["audio_dir"]`) instead of reconstructing.

### Reader contract

All consumers of `meta.paths` MUST use `dict.get(...)` with a default, never direct subscript:

```python
# CORRECT
audio_dir = led.get("meta", {}).get("paths", {}).get("audio_dir")
if audio_dir is None:
    # Fall back to legacy reconstruction or skip
    ...

# WRONG (will KeyError on l3-2026-04-28 ledgers)
audio_dir = led["meta"]["paths"]["audio_dir"]
```

This guarantees back-compat with `l3-2026-04-28` ledgers (which have no `meta.paths` block) and any future bumps that move/rename fields under `meta.paths`.

## Lineage

| Version | Date | Adds | Notes |
|---|---|---|---|
| `l1-2026-04-24` | 2026-04-24 | baseline | `cast`, `scenes`, `shots`, `lines`, `sfx`, `music`, `clips` |
| `l2-2026-04-25` | 2026-04-25 | `beats[]` hierarchy | `Scene > Shot > Beat > Clip` per HuMo continuity brief |
| `l3-2026-04-28` | 2026-04-28 | diagnostic expansion | `meta.phase_ms`, `audio_gates[]`, `text_for_tts`, `bark_render_ms`, `warmup_pad_ms`, `transitions[]`, `radio_bookend_path` |
| `l3-2026-05-02` | 2026-05-02 | `meta.paths` block | Phase E, BUG-LOCAL-018. Additive only. Old readers ignore it. |

## Hard rules for downstream nodes

1. **Always use `dict.get(...)` for `meta.*`**, never `meta[key]`. This is the only thing that lets us bump the schema without breaking old ledgers.
2. **Treat the on-disk filename as canonical.** Discover via `audio_dir.glob(...)`, never reconstruct via `f"{episode_id}_..."`. Phase C codified this rule with a regression guard (`tests/test_filename_pattern_audit.py`).
3. **Never write to `meta.paths` from outside `_build_meta_paths`.** It's owned by the ledger save path. If you need a derived path, compute it locally.
4. **Don't bump the schema version without a BUG_LOG entry.** Date-suffix bumps within `l3-` are additive; `l4-` would mean a breaking change, which requires migration code.

## References

- `nodes/_otr_ledger.py` — `CURRENT_SCHEMA_VERSION`, `save_ledger_safe`, `_build_meta_paths`
- `nodes/production_ledger.py` — `Ledger` class, `Ledger.save`, `Ledger.rename_episode`
- `tests/test_ledger_rename.py` — Phase B regression
- `tests/test_filename_pattern_audit.py` — Phase C regression guard
- `docs/2026-05-02-rtx-upscale-qa-pass.md` — Phase E section (this block's design rationale)
- `docs/BUG_LOG.md` — BUG-LOCAL-014, 015, 016, 017, 018 entries
