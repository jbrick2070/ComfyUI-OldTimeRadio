# ROUND r3 -- WIRING: per-beat multi-clip coverage

**Repo:** ComfyUI-OldTimeRadio, `v2.0-alpha`. Code baseline `a1d810f1`; docs
`d3308e43`. Suite `6454 / 27 / 1`; Bible 17; canonical `5377914B`.

r1 and r2 are JUDGED (`docs/2026-07-25-multiclip-coverage-r1-judgment.md`,
`-r2-judgment.md`). **This round is WIRING ONLY**: exact seams, exact call
order, exact files, what breaks when each edit lands, and what the tests must
pin. Do not re-argue architecture. **Every claim cites `path:line`.**

---

## 1. SETTLED -- do not reopen

**From r1:** one `ShotRow` per beat; multi-clip CONTAINED inside the beat's
render (beat emits ONE clip, one start, one duration, so manifest / SFX bed /
captions / timeline / `obs_publish` are untouched); ExecutionGroup DAG
expansion CUT; planning is pure and static; forward-only (no mirror, no loop,
no hold as coverage); CHAIN > JUMP CUT > REUSE-only-if-loop-closed;
`still_*` lanes are one still; first slice is `ltx_8gb`.

**From r2:**
- **PER-ADAPTER:** its own video PROMPTS + its own frame-contract numbers.
  **SHARED:** ONE phrase-aware splitter and ONE assembler (operator's own
  division: *"each video model declare its own video prompts and reuse a
  phrase multi-clip beat splitter and putter-together"*).
- **TWO prompt hooks**, sharing one typed context:
  `build_jump_still_prompt` (frozen before minting) and `build_segment_prompt`
  (may depend on the prior clip's terminal artifact). They differ by PHASE.
  Do NOT repurpose `StillPlanRow.framing_geometry`
  (`still_plan_helpers.py:141`) -- it is stored authored text.
- **THE PAUSE MAP RANKS, IT NEVER CHOOSES.** The partitioner enumerates the
  LEGAL cut points from the adapter's frame contract; a pause map (derived
  from VOICE-ONLY per-line audio, before master mixing) only picks the nearest
  legal boundary. No map / no nearby pause / phrase longer than the cap ->
  plain quantum cut. **The pause map is a LATER chunk; slice 1 is quantum-only
  and still correct.**
- `FrameContract` = `min_frames`, `max_frames`, `quantum`,
  `discrete_durations`, `allow_tail_trim`, declared per adapter and added to
  the `VideoEngine` Protocol (`registry.py:51-98`).
- Roster audit must compare an independent expected roster
  (`registry.py:253`) against registered names -- a broken adapter's import is
  swallowed (`__init__.py:16-44`) so a post-registration audit alone is blind.
- Beat-session lifecycle: ONE model load, N segments, teardown in `finally`.
  Assert LOADER-CALL COUNT, not prepare-call count -- `ltx_8gb.prepare()` only
  resolves node classes; loaders are inside the per-clip graph
  (`eng_ltx_8gb.py:328`, `:370`, `:408`).
- Terminal-frame persistence is SYNCHRONOUS AND FATAL (today
  `persist_episode_clips` is `except: pass`, `render_driver.py:3024-3035`).
- CHAIN successors' `init_image` is a DEFERRED TOKEN, not an up-front path.
- Seam arithmetic: `sum(render - drop_head - trim_tail) == target_visible_frames`;
  trims applied BEFORE terminal-frame extraction.
- No new ComfyUI node for splitting or assembly.

## 2. Ordering of record (wire in this order)

1. Declaration surface + roster/declaration audit. No behaviour change; every
   adapter `single_only` until proven.
2. Partitioner + `CoveragePlan` (pure, quantum-only, exact-sum, seam trims).
3. Beat-session lifecycle (one load, N segments, `finally` teardown).
4. Transactional persistence + assembly (`.partial` -> validate -> hash ->
   atomic rename; ffprobe proves `frame_count == target_visible_frames`).
5. `ltx_8gb` live slice: one beat over 161 frames, >= 2 forward-only clips,
   one heavy load, no ping-pong, `RESULT SUCCESS` + `obs_publish OK`.
6. Pause map (ranking layer). 7. Further adapters; audio lanes last.

## 3. THE WIRING QUESTIONS

**W1. Where exactly does the partitioner run?** It needs audio timing and the
frozen route, and must run BEFORE stills are minted. Name the function and the
line. Candidates to evaluate against the real call order: `otr_shot_lock`
(`build_execution_plan`, `:1058`), the route lock that already landed
(`render_driver.resolve_final_shot_engines`, called from
`otr_video_render_batch.py` before `validate_and_repair_still_spine`), or the
image dispatcher. Trace the ACTUAL node order in
`workflows/otr_canonical.json` and say which node owns it.

**W2. Where does the `CoveragePlan` live on the ledger, and does the schema
enforce it?** r2 found `ShotRow` (`schemas.py:302`) is NOT enforced on real
rows -- production shot dicts carry `role` / `char_id`, which `ShotRow` does
not declare, and only execution groups are validated at lock
(`otr_shot_lock.py:1081`, `:1116`). So: is adding a typed nested plan to
`ShotRow` real enforcement or decoration? If the schema must become
authoritative first, say what that breaks today.

**W3. The frozen render-mode capture.** LTX-8GB reads an env ceiling
(`eng_ltx_8gb.py:69`), Wan reads env AND live free VRAM
(`eng_wan_ti2v.py:350`), LTX-AV captures an env ceiling at import
(`eng_ltx_av.py:58`), Veo's legal duration depends on resolution and reference
mode (`eng_google_veo_video.py:245`). Can these be captured at the EXISTING
route lock (`resolve_final_shot_engines`, commit `57f4983a`), or does capture
need its own point? Name it.

**W4. `profile_max_render_frames()` treats 0 as unlimited**
(`motion_common.py:442`). r2 wants an absent ceiling REJECTED. **Verify that
does not break the 8-GB WAN launch contract shipped at `f914f0a4`**, where
canonical ships `max_render_frames = 0` meaning "unpinned" and only
`otr_8gb_wan` declares the key. Reconcile, or say the rejection must be scoped.

**W5. The deferred CHAIN `init_image` token.** Give it an exact spelling and
enumerate EVERY validator, builder and consumer that must learn it --
`build_request_from_shot` (`render_driver.py:1495-1508`),
`_assert_family_inputs_satisfiable` (`:2439`),
`validate_and_repair_still_spine` (`:653`), the request schema
(`schemas.py:216`), role_compat's `init_image` token, and anything else you
find. A missed consumer is a crash before clip 0.

**W6. Assembly seam.** Where does concatenation live so that the beat still
returns ONE clip from `render_shot` (`render_driver.py:2424-2460` region)?
What guarantees CFR, dimension/fps/pixel-format/colour equality, and the
exact final frame count? Name the ffprobe assertion.

**W7. What must the canonical workflow do?** r2 says no new node and no new
widget. CONFIRM against `workflows/otr_canonical.json` that slice 1 needs zero
JSON change -- and if any input IS required, say exactly which node, that the
widget is APPENDED (never inserted -- positional `widgets_values`,
BUG-LOCAL-097), and that the JSON changes in the SAME commit.

**W8. Test wiring.** Name the test FILES (existing or new) for: contract
purity, roster audit, partition property sweeps, seam-frame ownership,
deferred-token handling, loader-call-count, transaction rollback, assembly
frame count. Say which existing tests will BREAK and must be updated in the
same chunk (e.g. anything asserting ping-pong fill on `ltx_8gb`).

## 4. Invariants (violating one is an automatic fail)

- THE LAW: an audit may improve a story, never fail one for length, language,
  style, visual vocabulary or quality.
- Fail closed. No shims, no fallbacks, no silent degradation.
- Never reverse or loop an audio-synced render (landed `a1d810f1`).
- Per-adapter prompts; shared splitter/assembler.
- Any node/widget/link/schema change edits `workflows/otr_canonical.json` in
  the SAME commit; widgets APPENDED only.
- Assets straight to `otr\episodes\<ep>\`, final to `otr\obs\`; never tmp.
- Full suite + Bug Bible after every chunk; UTF-8, no BOM, SFW.

## 5. Return format

VERDICT, then MUST-FIX, then SHOULD-FIX, then CUT. Each item: claim,
`path:line`, consequence if ignored. Wiring specifics beat opinions.
