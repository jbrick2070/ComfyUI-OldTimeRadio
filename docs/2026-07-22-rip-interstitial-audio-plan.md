# Rip Interstitial Audio Only (Quick-Win 7)

## Goal
Remove the interstitial audio insertion path from `OTR_SceneSequencer`, while retaining opening/closing synthesis in `EpisodeAssembler` and retaining the `music_inter` semantics in the story/visual layer.

## User Review Required
> [!IMPORTANT]
> The user requested to use `/kibitz` for second opinions. I will run this plan through the Kibitz panel (Codex 5.6-sol + Gemini Flash) before execution. Please approve this baseline plan so I can begin the Kibitz hardening pass and then implement it.

## Proposed Changes

### `workflows/otr_canonical.json`
- **[MODIFY]** Remove links that route node 83's (`OTR_StableAudioTheme`) outputs to node 3 (`OTR_SceneSequencer`).
  - From the JSON, `OTR_SceneSequencer` receives `music_cue_audio` on slot 7 and `music_cue_manifest_json` on slot 8.
  - Delete the corresponding links and update the `links` array and node references.
- **[MODIFY]** Retain links routing node 83 to node 7 (`OTR_EpisodeAssembler`).

### `nodes/OTR_SceneSequencer.py`
- **[MODIFY]** Remove `music_cue_audio` and `music_cue_manifest_json` from `INPUT_TYPES`.

### `nodes/scene_sequencer.py`
- **[MODIFY]** Remove `music_cue_audio` and `music_cue_manifest_json` from the `scene_sequencer` signature.
- **[MODIFY]** Remove the `_CM` import, parsing logic, and `_music_positions` accumulation (~lines 856-897).
- **[MODIFY]** Rip the insertion path logic in the rendering loop where interstitial cues are glued into the timeline.
- **[MODIFY]** Remove the ledger write-back for `music_positions` at the end of the file.

## Verification Plan
1. Run `pytest` to ensure unit tests pass (especially tests targeting the sequencer).
2. Run `scripts/otr_canonical_api_run.py --dry-run` to validate the patched workflow graph matches schemas.
3. Validate JSON integrity with `OTR_WorkflowValidator`.
