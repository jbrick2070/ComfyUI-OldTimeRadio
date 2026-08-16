# Lemmy Chunk B -- the cameo ROLL on the content-owned lanes (r1 input)

Status: DRAFT for the kibitz panel. Chunk A (the cast contract) is pushed at
`da44f642` and live-proven on the 2026-08-16 sweep. This plan adds the ROLL.

## Ground truth the panel should not re-litigate (measured 2026-08-16)

* Both lanes derive their cast FROM the finished script and GATE on it:
  fable2's gate (b) demands speaker set == cast rows
  (`_otr_scifi_fable2.py:2858-2864`); codex's `cast_coverage` gate demands
  every cast id scheduled (`_otr_scifi_codex.py:1461-1493`). So the cameo
  decision MUST precede script authoring. Post-script injection fails
  structurally.
* PBUG-20260811-01 ("forcing the cameo kills the fable2 writer") is CLOSED AS
  MIS-ATTRIBUTED: the widget was provably inert on the lane at the repro
  commit, and both leg logs contain zero "lemmy". There is NO writer/cameo
  interaction to design around. The proof obligation is simply that a
  cameo-bearing cast passes both gates on a live leg.
* Codex is SCHEMA-LOCKED: `char_id`/`voice_slot` are
  `Literal["announcer","c01","c02","c03"]` and `CastPlanV4.cast` is
  `max_length=4` (`:277-286`); the id vocabulary repeats in `BeatPlanV4`
  (:488), `RadioScoreDraftBeatV4` (:551, grammar-decoded in P3),
  `ScriptLineV4` (:797) and `_DRAFT_SPOKEN_CHAR_IDS` (:1221, enforced :1352).
  The LMFE grammar on the local provider is rebuilt from the pydantic models
  automatically. No test pins the Literals; the lock lives in the module.
* Fable2 has headroom: `MAX_SPEAKING_CAST = 10`, no id vocabulary, no grammar
  binding. A cameo needs no schema edit, but consumes one of the 10 live
  voices and MUST speak (the casting validator demands one entry per script
  speaker).
* Lemmy's identity is FIXED DATA: `config/cast_pools.lemmy_row()` -- name
  LEMMY, male, `v2/en_speaker_8` (plus the qualified IndexTTS2 route),
  Cockney `speech_signature`, canonical description. The model must never
  invent his identity.
* Exclusion policy is inherited: `_source_bank_excludes_lemmy` covers
  `public_domain`/`shakespeare` families only; the scifi banks may cameo.
* The 11% roll is OS-entropy (`cast_pools.roll_lemmy()`), never the seeded
  rng, and `_LEMMY_CAMEO_FORCE` maps the `lemmy_cameo` widget to a
  None/True/False force knob -- today computed only on the legacy path,
  AFTER lane dispatch returns, so it is unreachable for these lanes.

## Proposed design

### Shared (both lanes)

1. **Roll placement:** at runner entry, before any authoring pass. Natural
   roll = `cast_pools.roll_lemmy()`; forced via a `lemmy_force` value plumbed
   through `resolved` (None = natural, True/False = force). The widget stops
   being inert on these lanes: `_resolve_inputs` carries `lemmy_cameo`
   through to `resolved` so dispatched runners see the same knob the legacy
   path sees.
2. **Exclusion first:** `_source_bank_excludes_lemmy(bank)` short-circuits
   the roll exactly as `assemble_pre_locked_rows` does, so a future
   content-owned adaptation lane inherits fidelity for free.
3. **Contract truthfulness (updates chunk A):** once a lane actually rolls,
   the stamped `lemmy_policy` becomes `"operator_cameo"` (the roll RAN) with
   `lemmy_hit` recording the outcome; `content_owned_cast_no_cameo_roll`
   remains only for the exclusion-free case where the roll could not run
   (and, historically, for pre-chunk-B ledgers). `casting_attempts` stays
   owned by the lane's own pass receipts.
4. **Identity pinning:** the cameo row is built from `lemmy_row()` -- never
   model-authored. Gender male, preset pinned, speech signature pinned,
   description pinned. Downstream voice allocation must respect the pin and
   never double-allocate his preset.
5. **Regression guards:** per lane, a unit test proving a forced cameo cast
   passes that lane's own gates; plus the live acceptance below.

### Fable2 (`scifi_news_pro`)

On a hit, LEMMY enters as a REQUIRED cast shape at the treatment boundary:
the treatment prompt receives him as a fixed, must-speak character (name,
role "gravelly engineer", speech signature) alongside the model's own
invented shapes, capped so shapes stay within `MAX_SPEAKING_CAST`. The script
pass then writes him from the start, so gate (b) holds by construction.
`_assign_voices` pins his row from `lemmy_row()` (preset override before the
menu deal; the menu allocates for everyone else). The casting pass must
return an entry for him like any speaker; his voice fields are overridden by
the pin regardless of what the menu proposed.

### Codex (`scifi_news`) -- THE FORK, panel input wanted

* **Option A -- DISPLACE (driver's lean):** schema untouched. On a hit, the
  P2 prompt instructs that one story row MUST be LEMMY with his pinned
  identity; he occupies c01-c03 like any story character, mirroring how
  `lock_cast` has him "consume a slot". `_validate_cast_plan` gains a
  conditional structural check (roster checks are sanctioned fail-closed):
  when the roll hit, exactly one row named LEMMY; on miss/exclusion, zero.
  `_assemble_ledger`'s voice allocation pins his row's preset instead of
  drawing from the gender pool.
* **Option B -- WIDEN:** add `c04` across the five schema sites +
  `_DRAFT_SPOKEN_CHAR_IDS`; the cameo rides the new id and never displaces a
  story character. More honest to "cameo = extra flavor", but touches the
  grammar-decoded vocabulary in three result types and every consumer that
  iterates the fixed set.

Driver's r1 lean: Option A. The cameo is small by design; displacement is
the established semantic in the legacy picker; Option B's schema surgery
lands in the P3 grammar path where a mistake truncates generation silently.

## Out of scope (do not expand)

Part SIZE/fidelity (later, quality-side, operator-deferred); the
`media_archive`/`original` cameo (already live via `lock_cast`); the 11%
rate; any change to voice qualification or the IndexTTS2 route.

## Acceptance

* Unit: forced-hit and forced-miss cast construction passes each lane's own
  gates; contract stamps truthful (`operator_cameo` + real `lemmy_hit`).
* Live, per lane: one forced-hit leg -- `RESULT SUCCESS`, `obs_publish OK`,
  a LEMMY cast row with pinned identity, at least one spoken line resolved
  via his `char_id` (never name-matching on lines), and the frozen contract
  showing `lemmy_hit: true`.
* The existing 11% statistical pin (`tests/lemmy_rng_check.py`) is untouched.

## Open questions for the panel

1. Displace vs widen on codex (the fork above).
2. Exact fable2 entry point: treatment `cast_shapes` (proposed) vs pitch, and
   how `n_max` interacts with a reserved cameo shape.
3. Should the `lemmy_cameo` widget plumbing land in this chunk or separately?
   It is the only `resolved` change, and it is what makes the force testable
   headlessly.
4. Downstream surfaces of a pre-locked row on these lanes: portrait/visual
   plan, captions, credits -- anything that assumes every cast row was
   model-authored?
5. The contract policy transition (chunk A's string becoming rare): is the
   proposed three-state truth (`operator_cameo` / `source_fidelity_exclusion`
   / `content_owned_cast_no_cameo_roll`) the right taxonomy, or should the
   third state be retired entirely once the roll exists?
