# r1 judgment -- PBUG-20260817-04: the announcer invented a title it was handed

**Driver:** Claude (Cowork), panelist and sole judge. **Date:** 2026-08-17.
**Anchor written BEFORE the fan-out** (scratchpad; corrected copy summarized
here). Seat returns: `pass00/*.json`.

**PANEL PROVENANCE, exact.** Codex: quota-held, absent. Both agy lanes were
run **manually by the operator** (one review returned; pasted into the
window). Panel seats run by the driver: **Fable** (narrative, COLD -- defect
facts without the driver's hypothesis), **Sonnet** (mechanics), **Opus**
(validator/LAW). **This is r1 with four seats, one operator-driven. Not a
full arc.** Every claim below was grounded by the driver at the real files
or the shipped ledger before adoption.

**OPERATOR RULING (mid-window, binding):** the announcer names the REAL work
plus its own locator -- "The Tempest, Act One Scene Two" / "Moby-Dick, 'The
Quarter-Deck'" -- or, with no locator, work + EPISODE SUBTITLE ("The Tempest:
A Hot Take"). The work-title half is never the model's to compose.

## THE DRIVER'S OWN ERRORS -- three, all confirmed, one twice

1. **WRONG ACTIVE SEAM (found by agy, sharpened by Fable and Opus).** The
   anchor cited `_otr_line_composer._ANNOUNCER_INTRO_SYSTEM_SAFE` ("Do not
   invent facts"). That constant did not run: `compose_announcer_intro` sets
   phase `announcer_intro_safe_system`, `_resolved_closing_prompt` routes it
   through `_otr_creative_prompt_router.resolve_creative_system_prompt`, and
   `_PHASE_TO_PACK_SEAM` lands it in
   `nodes/story_packs/public_domain/faithful_radio_adaptation.json` via
   `get_pack_prompt`, which RAISES rather than falling back. **And the
   driver's first correction was itself wrong:** a truncated read said the
   pack seam had no anti-invention clause. The FULL seam ends *"Use ONLY the
   WORK title and the proper names in the cast list below; invent none."*
   The strongest wording was live and failed -- prompt-only fixes are dead
   from the correct exhibit now. Seam edits belong in PACK JSON.
2. **WRONG LEDGER SHAPE (Opus).** `meta.announcer_intro_rewrite` is a DICT
   -- `{'status': 'announcer_intro_rewritten', 'reason': None}` -- not a
   string. A validator written from the anchor's `==` comparison would never
   fire and report clean forever.
3. **"INVENTION, NOT TRANSCRIPTION" OVER-CLAIMED (Opus, confirmed on the
   ledger).** The announcer prompt itself carries no excerpt text -- that
   half stands. But `_otr_story_brief._build_produced_open_input` feeds the
   VERBATIM scene-1 dialogue into the derive that produces `setting` /
   `opening_status_quo`, only `cast` is roster-gated there, and **the scene-1
   dialogue of this episode already speaks "Watson" and "Mr. Holmes"** (see
   the second defect below). The invented title is downstream-consistent
   with dialogue that had already gone Holmes.

## A SECOND FIDELITY DEFECT IN THE SAME ARTIFACT (Opus; driver-verified)

Cast rows are `THE GREAT DETECTIVE` and `SECRETARY`, and
`meta._adaptation_character_names` carries Leacock's roster. Lines b002-b005
nonetheless address "Watson" and "Mr. Holmes" BY NAME, in shipped audio, on
the invent-nothing lane -- Doyle's characters spoken into a Leacock parody
whose whole joke is that it never names them. The markup gate cannot see it:
`UNKNOWN_SPEAKER` checks speaker KEYS, not names spoken inside a line.
**Logged as PBUG-20260817-06, OPEN, its own item.** It is upstream of the
announcer and likely feeds it; fixing PBUG-04 does not fix it.

## THE FIX SHAPE (r2 input) -- convergent across all four seats

**The composite-row precedent already ships.** `compose_news_coda` /
`_assemble_news_coda_surface` return a row whose Python-owned span carries
`PROTECTED_FACT_COMPONENT_FLAG`, which `_otr_ledger_clean` skips -- verified
live on this episode's b006. The operator's ruling is this pattern applied
to the intro. Constraints the seats proved:

* **Post-J.5 splice, not an in-compose change (Sonnet).** The subtitle
  branch needs `meta["episode_title"]`, which binds at J.5 -- AFTER both
  announcer compose sites (in-loop and rewrite). A fix inside
  `compose_announcer_intro` ships the subtitle branch broken; a fix at one
  call site misses the rewrite producer.
* **Protect the spliced span or the clean pass eats it (Sonnet + Opus).**
  The shipped intro row already carries `unclean_spoken_text` from
  `run_ledger_clean`; unprotected splices replay PBUG-20260815-01.
* **Consume the already-gated `_work_title`, never a fresh
  `identity_from_meta` read (all seats)** -- or the media_archive
  publication-as-work collision reopens (item F r3).
* **Do NOT render public_domain `unit_label` unqualified (Opus + Fable, two
  roads).** `identity_from_meta` refuses it BY DOCUMENTED DESIGN, and a
  share of labels are curator scene descriptors ("The night camp"), not
  piece titles -- speaking one makes the DETERMINISTIC half invent. Locator
  = shakespeare `act`/`scene` INTS rendered as ordinal words ("Act One,
  Scene Two" -- roman numerals are a TTS hazard, no int-to-roman helper
  exists); everything else falls to the subtitle form until the manifest
  gains an explicit piece-title flag (data change, operator-visible).
* **The subtitle must be marked as OURS (Fable).** "an episode we call 'The
  Blackwood Enigma'" -- otherwise an LLM title sits in exactly the slot
  where radio announced real chapters, and a listener hears a chapter that
  does not exist. Milder version of the same lie.
* **The subtitle branch gates on `title_source` (Opus).** `--title` still
  maps onto the widget in `otr_canonical_api_run.py`, and under this ruling
  a planted harness label would be SPOKEN ON AIR, not just printed on a
  card. PBUG-05's guard vocabulary is the gate.
* **Any validator: `run_gap_audit` WARNINGS, never errors; dict-shaped
  rewrite read; and the new SafeOpenBrief locator field inherits
  `work_title`'s never-required/never-starves contract (Opus).** THE LAW.
* **The join problem (Fable).** The model's ask becomes "continue a sentence
  you did not write": doubled "Tonight..." framings and blind pronouns are
  the new failure modes; the rendered sentence must be in-context as
  immutable material. Fable's example lines are in its seat file and are
  the r2 target register.

## DISPOSITION

1. r2 (coding plan) designs the post-J.5 composite splice + protection +
   gates above. **No code before r2.**
2. PBUG-20260817-06 (Doyle names in dialogue) is its own queue item --
   upstream, different producer, different fix surface.
3. The seam text in the PACK may gain the label-echo instruction as
   belt-and-braces, never as the fix.
4. Register-confabulation + container-to-piece survive as the explanatory
   mechanism, now with the upstream-dialogue route alongside; the a/b
   falsification (genre-plausible vs bland real titles) remains cheap and
   optional.
