# ARCHITECTURE ROUND r1 -- MULTI-CLIP BEAT COVERAGE

**Repo:** ComfyUI-OldTimeRadio, `v2.0-alpha`, HEAD `a1d810f1`. Suite
`6454 passed / 27 skipped / 1 xfailed`; Bible `17 passed`; canonical
`5377914B`.

**This is an ARCHITECTURE round.** Produce the right SHAPE and the reasoning,
not a patch list. **Read the real files. Every claim cites `path:line`.**
A claim you cannot anchor is discarded, not weighed.

---

## 1. The requirement (operator, verbatim, 2026-07-25)

> "we need as much video to capture the beat. If that means using last frame to
> first frame of next clip (preferred for continuity) or jump-cut style new
> clip, either is fine. **We need enough clips per the beat for moving video.**"

Supporting rulings from the same session:

- "lip synch needs to have continuity, no render backwards, that doesn't work."
- "for video, if beat is multi-clip we can't reuse the first still, unless you
  conditioned the first still to be the last still of the last clip as well."
- "visual continuity -- either continuous movement or jump cut to new still,
  either is fine, but continuous using last-to-first frame is preferred."
- "for the 'still' paths -- easy, they can always use one still per beat, no
  continuity issues, no movement."
- "for video, if it needs a still, it's ALWAYS a still per beat."

**Ranked policy of record:** CHAIN (last frame -> next clip's init) preferred;
JUMP CUT (fresh still) acceptable; REUSE only if loop-closed; `still_*` lanes
are one still, always.

## 2. What actually exists today -- GROUNDED, do not re-derive, but DO correct me

A previous round established these against the code. Two panelists and I
verified each; they are the premises this round builds on.

1. **Nothing renders more than one clip per beat.** `render_driver.py:2627`
   stores `clips[out_shot["shot_id"]] = clip` -- one clip per shot id.
2. **Beats are filled by PING-PONG, not by more clips.**
   `eng_wan_ti2v.py:521-535` calls `_wb.extend_frames_to_target(...)`;
   `wrapper_bridge.py:435-462` builds the mirror cycle
   `[0,1,..,N-1,N-2,..,1]`, tiled and trimmed. The back half of every cycle is
   the render in REVERSE.
3. **`max_render_frames` is a render WORKLOAD ceiling, not a segmenter.**
   `eng_wan_ti2v._floor_length` (`:349-388`) caps the render, and the tail
   ping-pongs it back up to the beat's audio-derived target.
4. **The render budget reads LIVE FREE VRAM.** `_floor_length` calls
   `_MC.compute_real_frame_budget(_MC.free_vram_mb(), ...)` (`:378-388`). **So
   a clip count derived from it is unstable across the image/video phase
   boundary** -- stills would be minted against a VRAM reading that no longer
   holds at render time. Do NOT propose
   `ceil(target_frame_count / max_render_frames)`.
5. **Execution groups are per ROLE and the render ignores them.**
   `otr_shot_lock.py:1058-1089` -- *"CW-1 emits one consumer group per role
   that has beats (no base-clip providers yet -> no edges)"*; `run_episode`
   renders directly from shots. The provider half of the
   `depends_on` / `produces_base_for` DAG (`schemas.py:296-299`,
   `resolver.py:18-28`) is DESIGNED AND EMPTY.
6. **Veo's `last_frame` is NOT clip chaining.**
   `eng_google_veo_video.py:277-293` sends `instance["lastFrame"]` paired with
   `init_image` -- first/last-frame INTERPOLATION inside one clip.
7. **`ShotRow` is closed** (`schemas.py:302`, `class ShotRow(_Forbid)`), so any
   per-clip plan is an explicit schema addition.
8. **Audio lanes may no longer mirror** (landed `a1d810f1`):
   `fit_frames_to_target(..., allow_mirror=False)` raises
   `MirrorExtensionForbidden`; HuMo (`eng_humo.py:479-500`) uses it. Trimming
   is still legal.
9. **The route lock landed** (`57f4983a`): `resolve_final_shot_engines`
   resolves force map + radio-host redirect BEFORE
   `validate_and_repair_still_spine`. Effective engine per shot is settled
   before still validation.
10. **Audio-synced lanes:** `audio_driven_face` (HuMo family, cloud avatars)
    and `audio_conditioned_video` (`ltx_av`). `ltx_av` renders
    `next_8n1(target_frame_count)` natively (`eng_ltx_av.py:949`).

## 3. The questions

**Q1. What replaces ping-pong, and where does it NOT get replaced?** The
operator wants moving video. Is boomerang deleted outright, kept for lanes
where motion is decorative, or kept only as an explicit operator-selected
mode? Under the project's no-fallback law, what happens when an engine cannot
produce enough clips -- fail closed, or is a shorter real clip preferable to a
mirrored full-length one? Answer for scene lanes and audio lanes separately.

**Q2. Who computes the clip count, from what inputs, and WHEN?** Fact 4 says
it cannot come from live VRAM, and the count must be known before stills are
minted (the image phase runs first). Name the pure contract each adapter must
expose (legal frame lengths -- discrete? min/max? quantization?), the single
partitioner that consumes it, and the phase in which it runs. Deterministic
final-segment handling included.

**Q3. What is the ledger and manifest shape for N clips per beat?** Today one
clip per shot id and one manifest row per beat (`render_driver.py:2628-2662`,
`:3052-3129`). Does a beat become N shots, or one shot with an ordered
`clips[]`, or N execution groups? Weigh against fact 5 -- the provider/consumer
DAG exists, is validated, and is empty, but is per-role. Downstream (captions,
timeline, credits, `obs_publish`) must keep seeing one beat.

**Q4. How does CHAIN actually work, per engine?** The last frame of clip k must
become the init still of clip k+1. Name the extraction seam, where the frame is
persisted (it is an image asset the still spine will validate), and how it
interacts with `validate_and_repair_still_spine`. Which registered engines can
accept a chained init at all, and what does a lane that cannot do (jump cut)?

**Q5. Audio slicing.** For `audio_driven_face` / `audio_conditioned_video`,
each clip needs ITS OWN slice of the beat's audio, or lip sync drifts across
clip boundaries -- the defect class the operator just ruled on. Where does
per-clip audio slicing live, and how does it stay consistent with
`_cumulative_beat_start` (`render_driver.py:1489+`) and the frozen master mix?

**Q6. What is the smallest FIRST chunk that delivers moving video?** The
operator wants working output, not a cathedral. Name a slice that is shippable
alone, provable on a live leg, and does not strand the rest.

## 4. Invariants (violating one is an automatic fail)

- **THE LAW:** an audit may improve a story, never fail one for length,
  language, style, visual vocabulary or quality.
- **Fail closed. No shims, no fallbacks, no silent degradation.**
- **Never reverse an audio-synced render** (just landed; do not undo it).
- **Per-adapter ownership** -- a central function keyed on engine id is the
  shape this build exists to kill.
- Geometry (Python, engine safety) vs LOOK (pack-owned) stays split.
- Adapter imports are wrapped in `try/except: pass`
  (`_otr_video_engines/__init__.py`) -- anything raising at class-body or
  decorator time SILENTLY DELETES the engine from the menu. Validation stays a
  post-registration audit.
- Any node/widget/link/schema change edits `workflows/otr_canonical.json` in
  the SAME commit.
- Assets land in `otr\episodes\<ep>\`, final in `otr\obs\`; never tmp.
- UTF-8, no BOM, ASCII where practical, SFW.

## 5. Return format

VERDICT, then MUST-FIX, then SHOULD-FIX, then CUT. Each item: the claim, the
`path:line` anchor, the consequence if ignored. Say where you DISAGREE with
the premises in section 2 -- a grounded objection is worth more than agreement.
