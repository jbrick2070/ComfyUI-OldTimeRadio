# Look-QA Round 5 fix plan -- pass02 (FINAL build spec, panel-hardened x2)

Mission unchanged: make the SAVED-workflow production render pass the operator
eyeball (opening visual present and real; varied, brighter LTX scenes; every
talking beat shows its OWN cast member) WITHOUT touching the frozen audio spine.
Forensics: `docs/2026-06-10-look-qa-round5__problem-statement.md`. Panel record:
`roundtable/pass01..02/` + judgments. Workflow note (operator directive): all
fixes are node/writer CODE -- no new widgets, no graph changes; if any build step
turns out to need a graph-level value it is baked into
`workflows/otr_scifi_16gb_full.json` IN PLACE (never a runner patch, never a
second json).

Ground truth (probed, not assumed):
- `meta.visual_plan.scenes == []` (writer emits no scene layer post-CW-1);
  per-beat variety must come from line signals `beat_intent`/`arc_phase` + role.
- `otr_silent_composite` hold-fills short clips (`tpad=stop_mode=clone`).
- `plan_timeline_segments` requires ALL manifest rows to carry `start_s` for
  positioned mode (L222-223) -- one None row silently degrades the whole episode
  to sequential concat. Tonight's b000 row WAS None (lines-only lookup).
- Each accept30 run renders a FRESH episode; writer pre-freeze fixes bind it.
- run_episode `trace` rows carry only shot/attempts/engine today; node-92 report
  is the durable /history record.

## F1 -- LTX frame cap (the mud open)

`eng_ltx_video.py`: add module helper `_env_int(name, default, floor)` --
parse `os.environ.get(name)`, ValueError/absent -> default, clamp to
`max(floor, value)` with a LOUD warning on invalid or below-floor input.
In `render_clip`, between the existing `max(_LTX_MIN_FRAMES, ...)` and the 8n+1
snap:

```
cap = _env_int("OTR_LTX_MAX_FRAMES", 121, _LTX_MIN_FRAMES)
if length > cap:
    _LOG.warning("[eng_ltx_video] frame ask %d exceeds cap %d -- capping "
                 "(window fill = composite hold-last-frame)", length, cap)
    length = cap
```

The snap then yields <= cap and 8n+1 (121 -> 121; cap=120 -> 113). The composite
fills the rest of the window via its existing tpad clone. No Ken Burns addition.
Wan: NO cap this round (operator-gated lane); one-line code comment pointing here.

## F2 -- Per-beat LTX scene prompts + prompt observability

`render_driver.build_request_from_shot`, brief-composed branch:
1. Synthetic-open detection = empty `source_line_ids` OR shot_id ends with the
   OPENING_MUSIC_BEAT_ID -- never role alone.
2. core = `get_story_brief_ltx(meta)` (existing fallback chain);
3. role clause FIRST among clauses (survives the 240 trim):
   announcer_visual -> "a vintage radio set glowing warmly, lit dials and tubes";
   synthetic music open -> "opening establishing shot, the radio warming up,
   warm glowing dial light"; other text-engine roles keep the current set;
4. beat clause from the shot's line: `beat_intent` via a small fixed table with
   the loose fallback `"a beat of {intent}"` + one INFO line for unmapped
   intents (never silent); `arc_phase` tone clause; absent fields skip silently;
5. finish unchanged: `finish_visual_prompt(meta, p, max_chars=240,
   style_tail=False)`.

Observability (the "did it land" teeth):
- One INFO line per text-engine shot: source (m4|env|brief+beat), chars, sha8,
  `beat=` via `_beat_id_for_shot(shot)`.
- `run_episode` trace rows for ltx/wan shots gain `prompt_sha8`,
  `prompt_source`, `prompt_chars` (lands in the node-92 /history report).
- Diversity gate reads the trace: the episode's brief-composed LTX sha8s must
  not all be equal; prompts with `prompt_source=env` (OTR_LTX_RADIO_PROMPT)
  exempt the gate (warn-only) -- an operator override may legitimately repeat.

## F3 -- M4 person anchor on EVERY talking-head prompt path

`otr_shot_lock.py`, inside the char_beats loop (the only caller of the guard --
document that in `_prompt_is_consistent`'s docstring):
- `subject_anchor = f"{appearance}, face visible, speaking to camera"` --
  prepended to the final text on ALL paths (llm_text, LLM-directives compose,
  deterministic template) BEFORE the consistency guard and finishing.
- `_prompt_is_consistent` additionally requires, within the first 160 chars,
  at least one core appearance token AND one of face/portrait/speaking/camera.
  Object-only prompts fail -> anchored deterministic template (existing
  fallback). Prompt-text level only; no image detection this round.
- `_build_batch_prompt` instruction gains: "Describe the named character as the
  visible subject (face-forward, mid-shot or closer); never scenery or props
  without the character."
- Order preserved: guards -> finish_visual_prompt -> prompt_hash (c51526b).

## F4 -- Writer self-vocative re-attribution (pre-freeze, deterministic)

`OTR_LedgerScriptWriter.py`, at the existing pre-freeze scrub site, BEFORE
casting/voice mint (so the corrected speaker gets the minted voice):
- Detect: line text opens with the SPEAKER's OWN display name (first or full,
  case/punct-normalized) + comma. (Speaker-conditioned: "Gulliver, ..." spoken
  by GULLIVER fires; the same text spoken by Hayes never does.)
- Repair: when the cast has exactly TWO character rows, re-attribute to the
  other character -- update `char_id` AND every speaker-identity field on the
  line (`speaker_role` etc.; enumerate at build), LOUD log
  `[writer] self-vocative re-attribution bNNN cXX->cYY`. Three+ characters ->
  LOUD log, keep (ships; eyeball judges).
- ShotLock backstop: warn when a locked talking-head beat's text still opens
  with its own speaker's vocative.

## F5 -- Join hardening

- ShotLock `build_execution_plan`: stamp `"char_id": b["char_id"]` on every shot
  row; the announcer's char_id resolved from the CAST table by name match
  ("ANNOUNCER"; rows have no role field) -- normalized on the SHOT row only,
  frozen line rows untouched.
- `render_driver.build_request_from_shot`: resolve
  `char_id = str(shot.get("char_id") or line.get("char_id") or "")` (shot first:
  it carries the normalized id); LOUD warning when a shot whose engine family is
  `audio_driven_face` resolves a char_id with no portrait-index entry (gated to
  talking-head families -- synthetic/music beats stay quiet).
- `build_clip_manifest`: row `start_s` falls back to `shot.get("start_s")` when
  the line has none (kills the silent positioned->sequential degrade); rows gain
  `char_id` + `init_image` so the face-acceptance check is mechanical.

## F6 -- Tests (CPU, suite-resident)

1. eng_ltx cap: 238->121->121 snap; cap=120 -> 113 (<=cap and 8n+1); env
   override respected; invalid/below-floor env clamps LOUD; helper unit-tested.
2. Driver prompts: announcer/music/synthetic-open compositions differ; beat
   clause from beat_intent/arc_phase with both-absent fixture AND one-absent
   fixtures (silent degrade pinned); unmapped intent -> fallback clause + INFO;
   bright tokens survive the 240 finisher; trace rows carry sha8/source/chars;
   diversity check flags all-equal, exempts env source.
3. ShotLock: anchor leads on all three prompt paths; object-only prompt fails
   the guard -> anchored template; guard tokens within first 160 chars; shot
   rows carry char_id (incl. normalized announcer).
4. Writer: 2-character cast re-attribution (b004 shape) moves char_id +
   speaker fields pre-casting; 3-character ambiguity keeps + warns; detector is
   speaker-conditioned (no fire on other-speaker vocatives); frozen rows
   untouched post-freeze.
5. Manifest: synthetic-row start_s fallback restores positioned mode (fixture
   with a gap); rows carry char_id/init_image; the existing histogram and
   caption/credits behavior unchanged (pin tests stay green).

## Invariants (binding)

Frozen audio / mux-LAST / byte-identical green; fail-soft never fail-episode;
env overrides verbatim; guards -> finish -> hash; no new widgets; the SAVED
workflow `workflows/otr_scifi_16gb_full.json` is the only path (in-place edits
only, none expected this round); UTF-8 no BOM; SFW; suite + Bug Bible green at
every commit. Release procedure (operator rule, not a code gate): the gated
commits stay unpushed until the eyeball passes.

## Acceptance (the round-5 re-render gate)

ONE fresh 30w production render (words=30 the only patch):
- the cap log line fires for any >121f ask; the b000 window shows a real scene
  (YAVG-stddev as a DIAGNOSTIC, plus the eyeball);
- trace prompt sha8s for brief-composed LTX shots are not all equal; bright
  tokens present post-trim;
- every talking-head beat's manifest row has char_id == staged portrait ==
  cast table, and its mid-beat spot frame shows that character's face;
- no unresolved self-vocative ships from a 2-character cast (3+ ambiguity is
  LOUD-logged and acceptable);
- the nine existing log gates stay green; audio byte-identical; obs gains
  exactly ONE new AAC final; the operator eyeball gates the verdict.
