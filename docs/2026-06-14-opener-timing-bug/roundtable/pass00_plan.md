# Root-cause + fix convergence: opening-music beat carries no timed line -> ~9.5s black opener + no title card (BUG-LOCAL-403 / 404)

Panel: critique the ROOT CAUSE and the proposed FIX. Claude grounds every claim against the real
code and synthesizes. The goal is the correct, PROPER (non-shim) fix the coder window should build.

## Symptom (operator look-QA, 2026-06-14)
- The published episode opens on **~9.5 s of BLACK** while the opening theme music plays: no opener
  still, no title card, no reactive CRT. Then the first character beat (a portrait clip) appears.
- The §4C big-bold decode->reveal->dock episode-title card NEVER draws.
- (Separate, already FIXED: BUG-402 -- the §4D blend emitted `format=gbrpformat` and fell back to
  source-copy, losing the scopes + burned captions. Not in scope here.)

## Confirmed evidence (grounded against the real files)
- Episode ledger `signal_lost_across_the_room_20260614_153812_ledger.json`: `lines[]` = **5 lines, ALL
  dialogue** -- `b001` announcer `start_s=9.5`/`dur_s=8.83`, `b002-b004` character, `b005` announcer.
  **ZERO `music_open`/`music_visual` lines.** First spoken line starts at `start_s=9.5` -> a 9.5 s
  un-anchored lead-in.
- A `b000_music_open` STILL exists on disk (`stills/still_b000_music_open_*.png` + `stills_manifest.json`)
  but has **no timed line/shot**.
- `nodes/scene_sequencer.py` (~L688 + L708-712): PASSTHROUGH on `music_open`/`music_close`/`music_inter`;
  explicitly does NOT stamp timing. Verbatim comment: *"Sequencer never handled music timing ... music
  lines retain whatever the writer initialized them with (start_s=None, dur_s=None). EpisodeAssembler /
  MusicGen handle music timing downstream."*
- The opening theme audio is prepended by `OTR_EpisodeAssembler` SEPARATELY (audio-only); there is no
  timed visual line for it.
- Composite `nodes/otr_silent_composite.py::plan_timeline_segments`: places beats by line `start_s`.
  POSITIONED mode requires ALL rows have `start_s` (`positioned = all(r.get("start_s") is not None ...)`);
  ANY `None` forces SEQUENTIAL mode. With no timed `music_open` beat at `[0, 9.5)`, the head-gap is
  floor/black gap-fill and the `b000` still is never placed -> black opener.
- Title card `nodes/video_engine.py::_resolve_title_timing`: PRIMARY path reads a `music_open` line's
  `start_s`/`dur_s` (absent here). FALLBACK = first-dialogue onset -> window `[0, round(9.5*25)=238)`
  @25fps, which SHOULD fire and draw the card on the floor's first ~238 frames -- but the card is not
  visible.

## Proposed fix direction (converge on the RIGHT one)
Give the opening music beat a TIMED representation: `speaker_role=music_open`, `start_s=0.0`,
`dur_s = opening-theme length (= first_dialogue.start_s ~= 9.5 s)`. Candidate homes:
- **A) SceneSequencer** music branch stamps `start_s=0`/`dur_s=opening_theme_dur` on the `music_open` line.
- **B) Upstream shot / `OTR_ImageDirector`** path (which already mints the `b000_music_open` still) emits a
  TIMED shot.
- **C) Composite** handles untimed leading music differently (place the `b000` still over the head-gap
  without requiring a timed line).

## Open questions for the panel
1. Which home (A/B/C, or other) is the PROPER non-shim fix? Where does the opening-theme DURATION
   authoritatively live -- EpisodeAssembler (it prepends the theme), the writer/outline, or the audio?
2. Is stamping a `music_open` LINE the right data model, or should music beats be SHOTS only (visual
   plan) decoupled from audio `lines[]`? (Note `b000` has a still but no line.)
3. Does fixing the timing alone fix BOTH the black opener AND the title card, or does the title-card
   fallback / composite head-gap need its own fix?
4. Does an untimed `b000` shot force the WHOLE composite into SEQUENTIAL mode (dropping positioned
   placement for every beat)? Is that a second, hidden bug?
5. A/V SYNC: the master audio already has ~9.5 s of music then dialogue at 9.5 s. If we add a timed
   `music_open` beat `[0, 9.5)`, does the composite's total length + beat placement stay aligned with the
   FROZEN master audio? Any double-count / off-by-one risk?
6. Why did the volume-envelope / first-dialogue FALLBACK in `_resolve_title_timing` not produce a visible
   card today -- is it not firing, or is the composite head-gap not showing the floor's title frames?

## Invariants to guard (reject any fix that breaks these)
- **Audio spine FROZEN:** byte-identical master, mux-LAST, no `-shortest`; `test_audio_byte_identical`
  stays GREEN. The fix is VISUAL / timing-metadata ONLY -- it must not touch the master mix bytes.
- Determinism (seed-keyed). The composite contract + clip-manifest schema stay stable.
- PROPER root fix, not a shim (operator rule). Do not hardcode 9.5 s -- derive from the real theme length.
- 100% local; UTF-8 no BOM; SFW; single resident heavy <= 14.5 GB.
