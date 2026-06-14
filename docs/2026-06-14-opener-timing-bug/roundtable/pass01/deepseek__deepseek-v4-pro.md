<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. The root cause is insufficiently diagnosed; the proposed fix direction lacks concrete integration steps and does not address why the existing music-mirror code (BUG-LOCAL-130) failed.

MUST-FIX BEFORE BUILD:
1. [Root cause / missing music cue] The EpisodeAssembler’s music-mirror code (BUG‑LOCAL‑130) depends on `ledger.music[]` being non‑empty; if the writer emits no opening‑music cue, the mirror never creates a `music_open` line. The ledger in the bug report has zero `music_open` lines, meaning the mirror skipped. Fix: ensure the writer always includes an `"opening"` music cue in `ledger.music[]`, or have EpisodeAssembler inject one when missing. Without this, no timed line will ever appear.
2. [Manifest clip gap] Adding a `music_open` line to `ledger.lines[]` fixes timing, but the composite’s clip manifest must also include a clip row for that line (e.g., the `b000_music_open` still). The plan does not describe how the manifest builder (OTR_VideoRenderBatch / BatchHumoRender) translates the new line into a clip entry. If missing, the head‑gap will still be filled from the floor, which (if the fallback title‑card bug persists) remains black. Specify that step.
3. [Title‑card fallback silent failure] The document states the volume‑envelope fallback in `video_engine._resolve_title_timing` should have drawn the card on the floor’s first 238 frames but did not. This is a separate defect—even if the primary path is restored, the fallback must work. Verify why it is not rendering (possible causes: `_CRTRenderer` card‑window conditions, dock position off‑screen, font loading failure, or the floor video itself missing the card). Fix or document the limitation.
4. [Wrong timing space] Option‑A (SceneSequencer) stamps timing in scene‑audio space. That would be incorrect because the opening theme is prepended later by EpisodeAssembler, shifting all scene‑audio times. The proper home is EpisodeAssembler’s write‑back, which already shifts `lines[].start_s` to master‑mix and has the true opening duration. Any fix must extend that write‑back to guarantee a `music_open` line with `start_s=0, dur_s=opening_theme_dur`.

SHOULD‑FIX:
- [Composite mode] Clarify that once a timed `music_open` line exists, the composite’s POSITIONED mode will place the clip and eliminate the black‑opener. The plan implies the composite falls to SEQUENTIAL mode because `start_s` is missing on a `music_open` row, but if the row is absent entirely, POSITIONED mode still holds—the gap fill is floor/black, not a broken mode. Correct that statement.
- [Deterministic duration] Derive `dur_s` from the actual length of `opening_theme_audio` (available in EpisodeAssembler), not a hard‑coded 9.5 s guess.

OPTIONAL / NICE‑TO‑HAVE:
- Audit the `_envelope_intro_end` heuristic; it may return 0 for faint openings, leaving the fallback unusable.

CUT THESE (over‑engineering):
- **Option C** (composite‑side untimed‑music handling) is a shim that works around the missing data model. It adds complexity without fixing the root ledger gap and should be rejected in favor of making the ledger correct.