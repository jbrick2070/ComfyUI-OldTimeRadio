# Story quality R2 -- unified BUILD PLAN DRAFT (coding pass input)

8 levers (3 structural + 5 creative), "best story" the north star. Ledger {cast,lines,meta} schema
l3-2026-05-14 FIXED (content-only); audio frozen; MODEL-AGNOSTIC (every gate is one opus already
passes -> lifts the weak end, never rewrites opus); craft-ONLY (no word/beat-count or budget/schema
change); cheap setup-time LLM calls + TARGETED rerolls only (NEVER a full per-line rewrite). Each
lever is its own green chunk: full suite + Bug Bible -> commit + push. Harden the SEAMS below.

## STRUCTURAL (do first -- kills the universal warts)

### S1. Music/non-dialogue beats must not render as spoken/caption text
- `_otr_outline._assemble_outline` mints `music_inter` (speaker=NARRATOR, role=music_inter) with a
  placeholder `intent` that leaks verbatim into every transcript/caption. ROLE-BASED suppression:
  the line/caption render path emits NO voiced text for `speaker_role == "music_inter"` (and confirm
  music_open/music_close/sfx stay non-voiced too) while KEEPING the beat + timing + the music row.
  Keep a valid internal intent (Beat.intent min_length=4); set it neutral ("Bridge with music only").
  Key on the ROLE, not `dialogue_slot_id is None` (music_open/close/sfx also have none).
- SEAM TO PIN: where the rendered line `text` is set from the beat for a music_inter beat
  (production_ledger.init_lines_from_outline? the composer? the caption burn reads ledger line.text).
- TEST: no rendered transcript/caption contains "Musical interlude bridging"; music_inter row count +
  voiced slot ids unchanged.

### S2. Announcer CLOSE dramatizes, never summarizes
- Change the close-beat intent in `_assemble_outline` ("Close the episode and tag the broadcast.")
  to a final-image contract: "Close on a concrete final image showing what changed; no moral, thesis,
  or news-summary tag." + a deterministic banned-thesis scan ("Tonight's revelation", "the lesson",
  "reminding us", "proving * right", "this shows", "* is now shared") -> reroll via the DEDICATED
  ANNOUNCER composer (critic excludes announcer lines). SEAM: the announcer close composer fn.
- TEST: the 3 grounded close failures reroll/reject.

### S3. Cliche + stage-business reject gate + opposed-wants in the line prompt
- SMALL deterministic gate in `_otr_line_hygiene` (grounded, high-signal, NOT a big ban-list): exact
  cliches ("you're playing with fire", "this changes everything", "we're not leaving anything to
  chance") + stage-business-without-pressure ("I'll go check...", "I'll double-check...", "I'll lock
  down...", "I've got this, no need..."). Flagged -> TARGETED reroll (beat intent + speaker + opposed
  want + prev/next + the reject REASON; NOT the ban-list). CAP ~3-5 char rerolls/episode.
- Inject the opposed wants into the line-composer prompt for voiced beats ONLY when DramaticState
  wants are SOURCE-DERIVED / NON-DEFAULT; require each beat intent -> an ACTION VERB UNDER PRESSURE
  (reveal/refuse/demand/bargain/accuse/conceal/choose).

## CREATIVE (the lift -- all selected)

### C1. Specificity anchors (highest leverage)
- One cheap SETUP-time call (or deterministic extraction from the news brief): derive 3-5 concrete
  anchors (proper place names, a physical object, a number, a named bystander). Stamp on meta. Inject
  into the line prompt ("use these concrete anchors") + a gate that flags an all-generic line
  (no proper noun / no anchor) on high-content beats for a targeted reroll. SEAM: where the news
  brief is built (news_interpreter / story_brief) + the line prompt.

### C2. Central story-object
- Derive one `central_object` (e.g. "the green ledger") onto the dramatic state at setup; act-1 beat
  objective introduces it, mid complicates, the close lands it as the final image (pairs with S2).
  SEAM: dramatic_state derivation + beat-objective build + the close prompt.

### C3. Voice distinctness (promote F5)
- At CastLock, derive CONTRASTING `speech_signature`s (clipped vs verbose, plain vs ornate) so two
  characters never share a register; promote the signature to a hard per-line constraint in the line
  prompt + a soft same-voice flag. SEAM: CastLock speech_signature + the line prompt.

### C4. Escalation contract
- Per-act/phase: the beat objective must RAISE the concrete stake vs the prior phase (a bigger
  number / closer threat / higher cost). Deterministic where possible (a phase-stake check), else a
  prompt constraint. SEAM: the phase/beat objective build in _otr_outline / the spine.

### C5. Subtext nudge (lightest, turn/climax beats only)
- Line-prompt instruction on the TURN/climax beats: "imply the pressure, don't name it"; a gate flags
  on-the-nose emotion ("I'm scared", "this is dangerous") for a targeted reroll on those beats only.

## WIRING (pass 03)
- Hypothesis: NO workflow-JSON / node / widget change -- all content inside OTR_LedgerScriptWriter +
  its modules (_otr_outline / _otr_line_hygiene / _otr_dramatic_state / CastLock) + the news brief.
  CONFIRM (grep the JSON; verify no new widget/INPUT_TYPES). The new setup-time LLM calls reuse the
  resident writer slot (no new model widget, V-11).

## FINAL QA (pass 04)
- Re-soak: 2-3 weak-local legs + 1 frontier leg, visualizer (cheap), read the scripts. Before/after
  metric scan: the 4 structural counts (music-placeholder / meta-close / cliche / stage-business) +
  craft signals (proper-noun density, central-object recurrence, per-act stake escalation, voice
  distinctness). GATE: weak-end metrics DROP and the opus/frontier leg does NOT regress (still passes
  every gate -> untouched). Add the metric scan to `scripts/story_quality_scan.py` (the F1-F8 harness).

## Open questions for the coding panel
- Q1 [S1]: the EXACT seam where a music_inter beat's text reaches the rendered transcript/caption --
  is it ledger line.text (set in init_lines_from_outline) read by the caption burn, or a composer?
  Where is the single safest suppression point that keeps timing + the music row?
- Q2 [S3/C1-C5]: the EXACT line-composer function + its prompt builder + the targeted-reroll seam
  (so wants/specificity/voice/action-verb injection + the gates all hook ONE place).
- Q3 [C1/C2/C3]: best setup seam for the 3 cheap derive calls (specificity anchors / central object /
  contrasting signatures) -- the news_interpreter, the story brief, CastLock, or the dramatic-state
  derivation? Which already runs once per episode on the resident writer slot?
- Q4 [anti-regress]: confirm every gate is one the opus sample passes (proper nouns, object,
  escalation, distinct voices) so opus is never rerolled.
- Q5: ordering/dependencies between chunks (e.g. C2 central_object must exist before S2's close
  final-image can reference it); the minimal safe build order.
