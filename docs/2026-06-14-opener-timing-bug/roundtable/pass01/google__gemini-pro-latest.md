<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. EpisodeAssembler prepends the audio but fails to stamp the timing if the writer omitted the `ledger.music[]` row, leaving the beat untimed and triggering a catastrophic fallback in the composite.

MUST-FIX BEFORE BUILD:
1. [EpisodeAssembler] Missing authoritative timing stamp. `EpisodeAssembler` prepends the opening theme but only stamps timing if a `ledger.music[]` row already exists. If missing, the beat remains untimed.
   Fix: Synthesize the missing music row before the stamping loop so the existing mirror logic can propagate it to `lines[]`.
   ```python
                    _music_rows = _led.get("music") or []
                    if opening_theme_audio is not None and not any(m.get("cue_id") == "opening" for m in _music_rows):
                        _music_rows.insert(0, {"cue_id": "opening", "title": "Opening Theme", "shot_id": "b000"})
                        _led["music"] = _music_rows
   ```

2. [OTR_SilentComposite] Hidden A/V sync destruction. In `plan_timeline_segments`, an untimed `b000` shot forces the WHOLE composite into SEQUENTIAL mode (`all(r.get("start_s") is not None)`). This concatenates dialogue starting at 0s, skipping the head-gap entirely and overwriting the floor's title frames.
   Fix: Relax the condition to `any()` and filter untimed rows out of the positioned placement loop.
   ```python
    positioned = (target_total_frames is not None and rows
                  and any(r.get("start_s") is not None for r in rows))
    if positioned:
        valid_rows = [r for r in rows if r.get("start_s") is not None]
        for r in sorted(valid_rows, key=lambda x: float(x.get("start_s"))):
   ```

ANSWERS TO PANEL QUESTIONS:
1. **Which home?** `EpisodeAssembler`. It is the only node that authoritatively knows the exact duration of the prepended audio.
2. **Right data model?** Yes, stamping a `music_open` line is correct. `_resolve_title_timing` explicitly looks for this line, and the existing mirror logic in `EpisodeAssembler` is already designed to create it from `ledger.music[]`.
3. **Fixes both?** Yes. The timed line gives the composite a beat to place the `b000` still (fixing the black gap) and gives `_resolve_title_timing` the exact window to draw the card.
4. **Hidden bug?** Yes. ANY untimed clip forces the entire episode into SEQUENTIAL mode, destroying A/V sync.
5. **A/V Sync risk?** None. `EpisodeAssembler` already shifts all scene lines forward by `_shift_s` (the opening theme duration). The new `music_open` beat at `[0, 9.5)` perfectly fills the space before the shifted dialogue.
6. **Why did fallback fail?** Because the untimed `b000` forced SEQUENTIAL mode, which placed the first dialogue clip at `cursor=0`. The head-gap was skipped entirely, so the floor's title frames were overwritten by dialogue video.