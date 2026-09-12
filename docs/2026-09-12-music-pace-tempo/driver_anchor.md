# Driver anchor -- the music pace/tempo follow-ups, finished diff

Driver: Claude Opus 5 (Cowork, 5080), sole judge. ONE reviewer (codex, a
file-grounded code claim), briefed to REFUTE. READ-ONLY: no git writes, no
edits, no GPU (a canonical render may be in flight on :8000), no full suite.

## What is being reviewed

The uncommitted diff in `nodes/_otr_music_palette.py`,
`nodes/_otr_music_prompt.py`, `nodes/_otr_audio_engines/eng_stable_audio_3.py`,
`scripts/otr_music_ab.py`, `tests/test_music_palette.py`,
`tests/test_music_prompts_are_musical.py` and the new
`tests/test_music_ab_binds_its_own_episode.py`.

## Context, measured, do not re-derive

The operator said the music sounded like "a loop-a-loop tape deck". Established
earlier tonight and already pushed: the episode has no repeated audio segment;
what repeats is a ~0.25 s pulse in the loudness envelope; nothing downstream
loops the music; Stable Audio Open is built for loops and one-shots; the cure
that measured best was an anti-loop negative prompt plus sustained-instrument
wording plus a conditioning window longer than the cue.

`loopiness` below is the strongest autocorrelation of the 10 ms loudness
envelope between 0.25 s and 6 s, measured by `scripts/otr_music_ab.py` on real
canonical episodes. Reference points: the episode that drew the complaint
measured 0.485 (opening) and 0.713 (closing).

## The five changes

1. **Majority pace.** `_mood_devices_with_pace` used to let the FIRST matching
   mood term decide the cue's pace. A brief lists moods in no order, so
   "pastoral, playful, reflective" gave a comedy a rubato lullaby. Now every
   matching TERM votes (including two terms that share one device -- counting
   devices read a two-thirds majority as a tie), the majority pace wins, and
   devices that contradict the winner are dropped.
2. **Tempo per device group, not per pace class.** `_MOOD_DEVICES` entries
   gained a fourth element. Grief keeps rubato; dread gets a taut sustain;
   both stay "slow" for the contradiction rule.
3. **No tempo phrase may name a pulse.** The first attempt at (2) said "a held
   and unwavering pulse" and the next ominous-brief leg measured **0.864**, the
   worst of the campaign. It now says "sustained and taut, no rubato", and a
   test bans pulse / beat / steady / metronome / driving / throb / ostinato /
   loop / rhythmic from every tempo phrase.
4. **`OTR_SA3_CONTEXT_RATIO`** makes the 3x conditioning-window floor tunable
   so the harness can A/B the change most likely to have over-corrected the
   music into formlessness.
5. **The harness only attributes an episode it can prove is its own**, because
   a render leaves `pending_*` directories that are newer than the finished
   episode and two arms in a row measured an empty one.

## Round 1 (codex): VERDICT "no". What was folded

codex returned three must-fixes. All three grounded; all three are in the diff
now. Its two should-fixes and one rename are in too; both cuts were taken.

1. **The tempo was still order-dependent.** `tempo_phrase` returned the first
   SURVIVING device's tempo in MOOD order, so `["heroic","playful"]` resolved
   to a FAST pace and then asked for heroic's neutral "unhurried tempo, broad
   phrasing", and `["grief","tense"]` and `["tense","grief"]` disagreed.
   Two arbitrations are now explicit and neither reads the brief's order:
   * `_winning_pace(votes)` -- ties fall to the earliest row of `_MOOD_DEVICES`
     (the only order in the module that is not the brief's), not to the
     earliest ballot. This CHANGED a shipped expectation:
     `mood_pace(["playful","melancholic"])` was `"fast"` and is now `"slow"`.
   * `_arbitrate_tempo(candidates, winner)` -- the tempo comes from every
     MATCHED device that agrees with the winning pace, ranked by a declared
     `_TEMPO_PRIORITY`. Every matched device, not just the ones that survived
     `limit`, so a truncated prompt cannot change the tempo.
   * `_TEMPO_PRIORITY` is stated as an ARGUMENT, not a measurement: when two
     devices disagree, the phrase that denies metre most explicitly wins.
2. **Ratio 1.0 contradicted the anti-loop invariant.** It stays reachable --
   a control arm has to reproduce what it controls for -- but it now warns on
   every call, the comment says it restores the DEFECT rather than "the old
   behaviour", the `total > dur` claim is scoped to the default, and the test
   that pins `(0.0, 12.0)` says so in its name and asserts the warning.
3. **The harness could attribute another render.** Three bindings, all failing
   closed: `episode_from_log(log, started)` now requires the named episode's
   music to have been written after the arm began; `newest_episode` REFUSES
   when more than one fresh episode appeared instead of warning and picking;
   and `receipt_matches(episode, env)` refuses an episode whose recorded
   sampler / scheduler / steps / cfg are not what this arm asked for -- which
   is also the check that would have caught the inert arms.
   `OTR_SA3_CONTEXT_RATIO` is bound only from below (the window may never be
   shorter than the ratio asks), and that limit is stated in the docstring.
4. **should-fix, folded:** `_MOOD_TAGS` in `nodes/_otr_music_prompt.py` was
   still pushing "tighter rhythm, percussive accents" and "rhythmic accents"
   into the same row text the negative prompt spends its length forbidding.
5. **should-fix, folded:** `limit=0` / `-1` coerced to one device contrary to
   the docstring. `_device_limit` is now the single decider, the docstring
   states the floor and why (a cue with no device has no prompt), and junk
   cannot raise inside a render.
6. **rename, taken:** `_HELD_PULSE` -> `_TAUT_SUSTAIN`.
7. **cuts, taken:** the speculative "a future per-pace cue length" API
   justification on `mood_pace`, and the provenance narrative inside the
   constants.

## Round 2 questions -- answer with file:line

1. `_winning_pace` and `_arbitrate_tempo`: is there ANY input for which the
   returned tempo still moves with the order of `mood_terms`? Walk
   `["heroic","playful"]`, `["grief","tense"]`, `["tense","warm","grand"]`,
   `["urgent","warm"]`, `["playful","melancholic"]`, a brief with four moods
   and `limit=2`, and repeated terms. Is the tempo ever a phrase whose
   device's pace contradicts the returned pace?
2. The tie-break now depends on `_MOOD_DEVICES` order. Is that order used for
   anything ELSE whose behaviour I have just coupled to the tie-break, so that
   reordering the table to fix one would break the other?
3. `_device_limit` / the device loop: for `limit` of 0, -5, None, "two", 1.9,
   `True`, and a `limit` larger than the table, what comes back? Can
   `mood_devices` ever return an empty list, and would anything downstream
   raise if it did?
4. `receipt_matches`: can it pass an episode that is NOT this arm's? Consider
   an arm that sets only `OTR_SA3_NEG_PROMPT`, a ledger with an empty
   `params`, a receipt whose `cfg` is absent, and two episodes rendered
   seconds apart with identical settings. Does its docstring overstate what it
   proves?
5. `cues_written_after` uses `after - 1.0`. Justify or refute that second.
   Does `all()` over an empty cue list behave as the caller expects?
6. Argue the other side: the smallest thing here that is still WRONG to ship,
   and anything in the comments that overstates what was measured.

## Hard constraints

Deterministic; no new dependency; a tempo phrase is words and never a BPM; the
operator's ear is the verdict on music and nothing here qualifies anything.

## Round 3 -- CURSOR IS THE TIEBREAKER (operator, 2026-09-12: "ask cursor for a tie")

codex has now returned **"no" twice**: r1 on the design of the fold, r2 on the
fold itself. Both were grounded and both were folded in full. You are not being
asked to repeat either review. You are being asked ONE question, and the answer
is a verdict:

> **Is this diff shippable as it stands, or is there a defect left that would
> lose an episode, report a false number, or make the music worse?**

Default to "not shippable" for anything you cannot ground in the files.

### What round 2 changed, so you review the CURRENT tree

codex r2's four must-fixes, all on `scripts/otr_music_ab.py` and all folded:

1. **An absent receipt field passed.** `if actual is None: continue` meant a
   ledger with `params={}` satisfied every check an arm made. It now fails
   closed per field, with a message naming the field.
2. **An arm the receipt cannot prove is refused outright.** `_RECEIPT_FIELDS`
   is the list of server-side settings the receipt records; a server-side
   variable outside it (say `OTR_MASTER_*`) is refused with a message saying to
   record it first, rather than measured and silently unattributable.
   **codex's own concrete fix for the negative prompt was wrong and was not
   taken as written:** it said to compare `receipt_of`'s `negative_prompt`,
   which is the COMPOSER'S, while `OTR_SA3_NEG_PROMPT` overrides it inside the
   engine -- comparing them would have refused every legitimate arm. The engine
   now records the negative text it actually used, and the denoise, in its own
   receipt (`params`), which is where the harness reads everything else.
3. **The check was per receipt row, the measurement per wav.** Every measured
   cue stem must now have a receipt row of its own.
4. **The one-second freshness grace is gone.** A proof with a tolerance is not
   a proof.

Also: the pace tie-break moved out of `_MOOD_DEVICES` order into its own
`_PACE_TIE_ORDER`, because that table already decides classification (a term
takes the FIRST pattern it matches) and one order should not silently do two
jobs.

### Where to look hardest

* `nodes/_otr_music_palette.py` -- `_mood_devices_with_pace`, `_winning_pace`,
  `_arbitrate_tempo`, `_device_limit`. Is the pace or the tempo still reachable
  from the brief's ORDER by any input? Can `mood_devices` return an empty list?
* `scripts/otr_music_ab.py` -- `receipt_matches`, `cues_written_after`,
  `episode_from_log`, `newest_episode`. Can any of them still accept an episode
  that is not this arm's, or refuse one that is?
* `nodes/_otr_audio_engines/eng_stable_audio_3.py` -- the receipt now carries
  two more fields. Does that reach any identity hash, any ledger seal, or any
  frozen fixture? (`cue_spec_sha256` is documented as excluding the receipt --
  check that, do not take it from me.)
* Every `log.` call touched: placeholder count against argument count. A
  mismatch of exactly this kind killed a render hours ago
  (`nodes/stable_audio_theme.py`, fixed in `988e6b7e`), and it raises ONLY when
  the level is enabled, so the suite cannot see it.

### Evidence already in hand, do not redo it

21 neuter mechanisms, 21 of 21 turn a test red, markers back to zero. The music
test group is green (197 + 14). The full suite identity diff runs against the
final tree before the push. Nothing here has been qualified by a render: the
operator's ear is the verdict on music, and this diff claims only that the
request is better formed and the harness cannot lie about which episode it
measured.
