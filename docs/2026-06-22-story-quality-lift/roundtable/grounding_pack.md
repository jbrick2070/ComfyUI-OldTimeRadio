# GROUNDING PACK -- verified against the real Windows repo + the frozen ledger

All claims below were checked against `C:\...\ComfyUI-OldTimeRadio` source and the frozen ledger
`output/otr/episodes/signal_lost_chandras_echo_20260622_141546/audio/..._ledger.json` (18 lines).
Panel: verify your claims against THIS pack; do not invent file contents beyond it.

## A. REAL CORPUS -- the full 18-line arc (id | speaker_role | char_id | text)

- b001 announcer -- "Gather 'round, dear listeners... celestial echoes of a distant cosmic dance..."
- b002 character c03(Mali) -- "Cross-checking with Chandra's last known star charts... we're looking at..."
- b003 character (boss) -- "Keep it running, Mali. Whatever you need, get it done. We're suspending maintenance until we understand this."
- b004 character -- "Mali, secure the signal. We can't risk any eavesdroppers."
- b005 character c03 -- `"Not before I amplify it. The world deserves to hear this." adjusts dials on the console`
- b006 music_inter -- "" (empty -- S1 music suppression working)
- b007 character -- "I've isolated the frequency... but it's... it's pulsating, like it's alive."
- b008 character -- "I've seen pulsars dance like that, Mali. It's nothing extraordinary."
- b009 character -- "You've got a leak, Mali. Your numbers are all over the web."
- b010 character c03 -- `"I assure you, Manfred, my work is entirely my own. My husband's theories have no bearing here." clutches her wedding ring tightly`
- b011 announcer c02(Manfred) -- "I've just sent a copy of your research to 'The Chronicle'. They were very interested in your 'alive' frequency."
- b012 character c04 -- `"Gentlemen, let's not jump to conclusions. Mali's work, while fascinating, is purely theoretical." taps his cane impatiently`
- b013 music_inter -- "" (empty)
- b014 character -- "I've taken the liberty of sharing your 'frequency' with the press, Mali. It's a public service, really..."
- b015 character c03 -- `Well, Manfred, it seems I'll be presenting my findings to the UN sooner than expected." tightens her scarf, a nervous gesture "I do hope they find my 'theoretical' work as... persuasive as you seem to.`
- b016 character -- "I'm overriding the UN's block. Prepare to transmit."
- b017 character c02 -- `Sherlock, stop this at once! overrides systems, fingers dancing on the console I won't let you jeopardize Mali's life's work over a... difference in interpretation.`
- b018 announcer -- "As the observatory's gaze turns away, the Galactic Center's fiery tapestry now bears the mark of our newfound..."

Cast (inferred from text; exact names in ledger `cast`): c03=Mali (protagonist scientist),
c02=Manfred (the antagonist who flip-flops; b011/b014 leak-to-press, b017 defends her), c04=a
cane-tapping skeptic (b012). "Sherlock" = an AI that appears abruptly at b016/b017.

## B. STAGE-DIRECTION LEAK seam (verified file:line)

`nodes/_otr_line_hygiene.py`:
- `_leading_stage_strip(text, max_words)` (246-312) is **LEADING-ONLY**. It does `body=s.lstrip()`,
  strips one optional leading quote, then `if not body or not body[0].islower(): return s` (263-266)
  -- so any line NOT starting lowercase is returned UNCHANGED. It scans from the START for the
  dialogue boundary and strips the prefix. It never inspects text after a closing quote or mid-line.
- `scrub_leading_stage_direction` (315-318) wraps it with MAX_STAGE_PREFIX_WORDS=6.
- `detect_leading_stage_business` (321-327) returns (hit, hint) with _DETECT_MAX_PREFIX_WORDS=10;
  hint `_BARE_STAGE_HINT` = "write only the spoken words; do not prefix the line with an action
  description (no stage directions)".
- Conclusion: trailing (b005/b010/b012), embedded-between-quotes (b015), and embedded-no-quotes
  (b017) all PASS THROUGH unchanged. The corpus proves it.

`nodes/_otr_ledger_scrub.py`:
- `_strip_stage_directions(text)->Tuple[str,bool]` (381-412): applies `_STAGE_DIRECTION_RES`
  (151-157) = `\[[^\]]{1,80}\]` (brackets), `\*[^*]{1,80}\*` (asterisks), `\(([^)]{1,80})\)`
  (parens, cut only if inner matches `_CUE_VERB_RE`) -- all UNANCHORED so they cut delimited
  directions ANYWHERE. THEN applies the SEAM-A leading bare floor (408-411). No bare
  trailing/embedded handling. Stamps `CODE_STAGE_DIRECTION="stage_direction_stripped"`.
- `is_spoken_role` (135-144): `_SPOKEN_ROLES = {"character","announcer"}`. Only spoken rows are scrubbed.

`nodes/_otr_line_composer.py`:
- `compose_line` (1931-2261): stage-direction reroll block 2015-2060, guarded by
  `_stage_dir_repair_attempted` (ONE reroll level). Runs `detect_leading_stage_business` +
  flag_cliche/stage_business/on_the_nose; concatenates into `reroll_hint`; recursive compose;
  on failure keeps the draft ("freeze floor is the backstop"). `compose_line_draft` (1689-1928,
  max_attempts=2) does formatting/leak/oversize repair but NOT the stage-direction reroll.

## C. ANTAGONIST-ARC seam (verified)

`nodes/_otr_story_critic.py` `run_story_critic` (505-605): ONE LLM call, never raises. 5 craft
dimensions (253-335): knowledge / pressure / relationship / decision / obstacle, + TENSION FIT vs
`target_tension=N/5`. Arc verdict enum: `strong | uneven | flat | mid_collapse` (only mid_collapse
is an arc-shape detector). SECTION 1 CONTINUITY = factual contradiction, line-scoped. SECTION 2
VOICE DRIFT = per-character voice/register, NOT plot stance. **No per-character arc/stance-reversal
axis.** Reroll: `_otr_reroll.py` `run_targeted_reroll` (486-536), MAX_REROLL_CYCLES=2, scoped
re-score (`scope_line_ids`), `failed_dimension` folded into hint (STEP 5), repair-then-ship on halt.

## D. ROLE-STAMP seam (b011 mis-stamp; verified)

`nodes/production_ledger.py` `init_lines_from_outline` (684-805): `role = speaker_role or "character"`
(724); char_id by role (761-766): role=="character"->cast id; role=="announcer"->cid="announcer".
b011 has role=announcer but char_id=c02 (a cast id) -> INTERNALLY INCONSISTENT (the announcer branch
would have set cid="announcer", so the role was set to announcer on a character-charid row elsewhere).
`set_lines` (839-906) also defaults role to "character" (885). The only writer of announcer onto a
non-announcer row: `_otr_ledger_reviewer.py:1054-1070` role_mismatch repair honoring LLM
`expected="announcer"` (passes `_ALLOWED_SPEAKER_ROLES`={character,announcer,music_open,music_close,
music_inter,sfx}). No consistency assert ties char_id (cast id) to role=character.

## E. ESCALATION seam (UN jump; verified)

`nodes/_otr_outline.py` `_build_beat_user_prompt` (1166-1236): escalation is a SOFT PROMPT directive
only ("RAISE THE STAKE... escalate, never tread water", 1226-1234). `intent_is_action_under_pressure`
(1251-1257) is MEASUREMENT-ONLY, non-binding (comment 1239-1242). `compute_beat_tension_ramp`
(`nodes/_otr_slot_drama_contract.py:762-792`) = deterministic ordinal ramp 1->5 by beat position,
DECOUPLED from semantic scope. No proportion/setup gate; nothing prevents an abrupt scope jump
(observatory -> UN) between adjacent beats.

## F. Invariants (verified in repo)

Ledger schema id `l3-2026-05-14` confirmed in 5 files (production_ledger.py:316, _otr_ledger.py:49,
_otr_ledger_freeze.py:85, news_interpreter.py:84, _otr_ledger_scrub.py:903). Canonical workflow nodes:
writer=`OTR_LedgerScriptWriter` id 1; freeze cascade host `OTR_LedgerFreezeCascade` id 62;
CastLock id 80; character TTS `OTR_BatchCharacterVoices` 81; announcer TTS `OTR_AnnouncerVoice` 82.
