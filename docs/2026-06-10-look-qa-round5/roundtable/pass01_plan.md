# Look-QA Round 5 fix plan -- pass01 (panel-hardened, code-grounded)

Mission: make the SAVED-workflow production render pass the operator eyeball that
failed tonight (no opening visual / bad-dark-samey LTX scenes / beats without the
right people), WITHOUT touching the frozen audio spine. Evidence and forensics:
`docs/2026-06-10-look-qa-round5__problem-statement.md`. Panel record:
`roundtable/pass01/` + `pass01_judgment.md`.

Ground truth that shapes everything (probed live, not assumed):
- `meta.visual_plan.scenes == []` -- the legacy per-scene visual_prompt layer has
  NO data source post-CW-1. Per-beat variety must come from the frozen line's own
  signals: `beat_intent` + `arc_phase` + role (all confirmed present on lines).
- `otr_silent_composite.py` ALREADY hold-fills short clips into their window
  (`tpad=stop_mode=clone`, truncate-long/hold-short, ~L264-305). No composite work.
- Each accept30 run writes a FRESH episode: writer-side pre-freeze fixes bind the
  re-render. Frozen ledgers are never re-rendered; no frozen row is ever mutated.

## F1 -- LTX frame cap (D1: the mud open)

`nodes/_otr_video_engines/eng_ltx_video.py::render_clip`, between the existing
`max(_LTX_MIN_FRAMES, ...)` (L279-280) and the 8n+1 snap (L281):

```
cap = _env_int("OTR_LTX_MAX_FRAMES", 121)      # defensive parse, clamp >= _LTX_MIN_FRAMES
if length > cap:
    _LOG.warning("[eng_ltx_video] frame ask %d exceeds cap %d -- capping "
                 "(window fill = composite hold-last-frame)", length, cap)
    length = cap
# existing snap follows: length = ((length - 1) // 8) * 8 + 1
```

- Default 121 (proven-good range 49-121f; 121 = 8*15+1 survives the snap intact).
- Invalid/short env values clamp LOUDLY. Same cap wired into `eng_wan_i2v` if it
  shares the ask path (verify at build; wan beats are operator-gated today).
- The composite's existing tpad-clone hold fills the rest of the window. No zoom
  addition this round.

## F2 -- Per-beat LTX scene prompts (D2: one terse dark prompt x3)

`nodes/_otr_video_engines/render_driver.py::build_request_from_shot`, the
brief-composed branch (~L428-456) becomes a per-beat composition:

1. core = `get_story_brief_ltx(meta)` (unchanged fallback chain);
2. + role clause: announcer_visual -> "a vintage radio set glowing warmly, lit
   dials and tubes" (FIRST among clauses -- survives the 240 trim);
   music_visual (synthetic open) -> "opening establishing shot, the radio
   warming up, warm glowing dial light"; other text-engine roles keep today's
   clause set;
3. + beat clause derived from the shot's LINE (when one exists):
   `beat_intent` mapped through a small fixed table (e.g. revelation ->
   "a moment of revelation", dread -> "gathering tension") + `arc_phase` tone
   ("early scene-setting calm" / "rising stakes" / "aftermath hush") -- absent
   fields skip silently (fresh-episode safety);
4. finished as today: `finish_visual_prompt(meta, p, max_chars=240,
   style_tail=False)` -- era tail stays, "no on-screen text" stays trailing.

Acceptance teeth: one INFO manifest line per text-engine shot
(`[OTR.render_driver] prompt source=brief+beat sha8=xxxxxxxx chars=NNN beat=bNNN`)
and a DIVERSITY gate: the sha8s of the episode's LTX prompts must not all be
equal (b000/b001/b005 differ structurally by role+beat clause).

## F3 -- M4 person anchor for talking-head shots (D3: the no-person HuMo beat)

`nodes/otr_shot_lock.py`:
- `_deterministic_template` / the join at ~L486-494: PREPEND the cast anchor --
  `"<appearance>, face visible, speaking to camera"` -- before setting/beat text
  so the subject leads the prompt (HuMo follows leading tokens hardest).
- `_prompt_is_consistent` (~L339): for `CHARACTER_BEARING_ROLES` additionally
  require a person-anchor (any core appearance token AND one of
  face/portrait/man/woman/person/speaking); an object-only prompt fails ->
  existing fallback to the (now anchored) deterministic template. Fail-soft,
  CPU-only, no image-level detection this round.
- `_build_batch_prompt` instruction text gains: "The prompt MUST describe the
  named character as the visible subject (face-forward, mid-shot or closer);
  do not describe scenery or props without the character."
- Finishing order unchanged (guards -> finish -> hash; the c51526b contract).

## F4 -- Writer self-vocative re-attribution, pre-freeze (D3: two beats, one face)

`nodes/OTR_LedgerScriptWriter.py`, with the existing pre-freeze scrubs
(stage-direction / self-vocative family, a5f4763 site):
- Detection: line text begins with the SPEAKER's OWN display name (first or full,
  case/punct-normalized) followed by a comma -- the b004 pattern.
- Repair (deterministic, no LLM): if the scene/exchange has exactly ONE other
  character row, re-attribute the line's char_id to that interlocutor and LOUD-log
  `[writer] self-vocative re-attribution bNNN cXX->cYY ("Name, ...")`; otherwise
  LOUD-log and keep. Runs BEFORE casting/freeze so audio + video stay coherent
  (the voice is minted for the corrected speaker).
- ShotLock backstop (cheap): warn when a locked talking-head beat's text starts
  with its own speaker's name (catches anything that slips to render).

## F5 -- Join hardening (D3 latent)

- Announcer char_id: resolve from the cast table by NAME match ("ANNOUNCER") --
  never hardcode c01 (cast rows carry no role field; probe-verified). Normalize
  at the ShotLock JOIN (shot rows), never on frozen line rows.
- `render_driver.build_request_from_shot`: LOUD warning when a talking-head
  shot's char_id misses the portrait index (today it only fails closed at
  eng_humo with no upstream hint).

## F6 -- Tests (CPU, suite-resident)

1. eng_ltx cap: 238 -> capped 121 -> snap 121; env override respected; invalid env
   clamps LOUD; cap < _LTX_MIN_FRAMES clamps to MIN.
2. Driver prompts: role+beat composition differs across announcer/music/other;
   absent beat_intent/arc_phase degrade silently; bright tokens survive the 240
   finisher; manifest line + diversity sha8s present.
3. ShotLock: object-only M4 prompt rejected -> anchored template; anchored clause
   leads the deterministic prompt; consistency guard passes legit prompts.
4. Writer: b004-shaped exchange re-attributes to the interlocutor; 3-speaker
   ambiguity keeps + warns; frozen rows untouched post-freeze (guard test).
5. Announcer join: cast-table name resolution; missing-portrait warning fires.

## Invariants (binding)

Frozen audio / mux-LAST / byte-identical green; fail-soft never fail-episode;
env overrides verbatim; guards before finishing before hash; no new widgets; the
SAVED workflow is the only path; UTF-8 no BOM; SFW; suite + Bug Bible green at
every commit; 7 gated commits stay unpushed -- operator pushes after the eyeball.

## Acceptance (the round-5 re-render gate)

ONE fresh 30w production render (words=30 the only patch):
- log shows the cap line for any >121f ask AND stddev(per-second YAVG) > 2.0
  over the b000 window of the silent composite (tonight's mud: ~0.2);
- the episode's LTX prompt sha8s are not all equal (diversity gate) and the
  announcer/music prompts carry the bright-radio tokens post-trim;
- every humo beat's mid-point frame shows its OWN cast member's face (manifest
  char_id == staged portrait == cast table; spot frames extracted);
- no line ships whose text opens with its own speaker's vocative name;
- all nine existing log gates stay green; audio byte-identical; obs gains exactly
  ONE new AAC final; operator eyeball gates the verdict.
