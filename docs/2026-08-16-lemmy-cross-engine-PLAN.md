# Lemmy on every TTS engine -- PLAN (item 1.1)

**Status:** hardened through r1 of a full `kibitz-plugin:kibitz` arc (Codex
gpt-5.6-sol high + Antigravity; driver Claude, sole judge). Every claim was
checked against the real Windows tree at HEAD `5662cd16`.

**Operator directive (2026-08-16):** *"we need to be sure we have lemmy working
on all tts engines"* -- NO-SKIP, and *"audition" means BUILD IT*. A later ruling
the same day defers his ears: *"we can get eyes after all sprints are coded
since I will be remote."* Those two together are the whole shape of this plan.

## 1. What is actually wrong, measured

Lemmy has a provable identity on **2 of 7** character engines.

* `indextts2` -- QUALIFIED. A real blinded audition (G1 Test A, 2026-08-10)
  with a complete receipt at `config/cast_pools.py:521-592`. Do not re-train,
  do not re-render, do not touch its harness.
* `bark` -- pinned by `lemmy_row()` (`config/cast_pools.py:733-734`) to
  `v2/en_speaker_8`. A ROUTING fact, not an audition: its receipt is explicitly
  `None` (`:484`). Stable, and unproven.

On the other five his voice is **redrawn per episode** --
`assign_voice_for_slot` folds `episode_seed` into `stable_cast_seed`
(`nodes/_otr_voice_bank.py:461-472`) and the pick is a weighted draw
(`:488-492`).

**Stated precisely, because the broad version is false.** The defect bites
FRESH, SEEDED, non-bark renders. The historic corpus was *accidentally pinned*:
33 of 35 reference-carrying LEMMY rows named the same reference because
`episode_seed` was `None` and the selector derived an identical seed
(`scripts/otr_g1_lemmy_audition.py:21-26`). So this is a defect that arrived
with correct seeding, not one that has always been visible.

## 2. The constraint that shapes the whole design

`QUALIFICATION_RECEIPT_REQUIRED_FIELDS` (`config/cast_pools.py:388-401`)
includes **`operator_verdict`**, commented *"a human said yes; nothing else can
supply this"*. `is_qualified_route` is fail-closed on it (`:404-421`); the real
validator in `nodes/_otr_voice_route.py` is stricter still.

**His ears are deferred, so no new route can become QUALIFIED this window, and
nothing here may pretend otherwise.** The comment at
`config/cast_pools.py:382-387` records the last violation: bark once claimed
`qualification_receipt: "canonical_bark_preset_v1"`, a bare string asserting an
audition that never happened (BUG-12.86). A driver-signed verdict would
recreate that bug with better prose. Forbidden.

## 3. The design -- a second tier that cannot impersonate the first

Add `LEMMY_VOICE_POLICY["provisional_native_routes"]`, a **separate key**.

* **Separate key, so no existing validator can be fooled.** The three readers of
  `approved_native_routes` -- the dormancy gate (`nodes/cast_lock.py:817`), the
  key-set discriminator (`:886-889`) and the only engine-indexing site
  `select_policy_route` (`nodes/_otr_voice_route.py:383-396`) -- keep reading
  exactly what they read today.
* **A second key inside the existing dict would break a shipped tripwire, which
  argues FOR separation.** `tests/test_otr_dialogue_policy.py:92` asserts
  `set(routes) == {"indextts2"}`. That test is correct: it fires when someone
  widens the QUALIFIED set. Provisional rows must not trip it.
* **The key-set discriminator stays unwidened.** `cast_lock.py:886-889` uses
  qualified engine names to decide whether to RAISE when no character engine
  resolves. Adding five provisional engines would widen a fail-closed raise on
  the strength of unauditioned rows.

### 3A. THE RULE THAT PREVENTS A RENDER-KILLER (r1, found independently by both reviewers)

**A provisional route is NEVER stamped into `cast_row["voice_route"]`.**

`resolve_and_verify_reference` treats ANY non-empty `voice_route` dict as a
route claim (`nodes/_otr_voice_route.py:595-597`) and RAISES `VoiceRouteError`
unless `status` is exactly the selectable/qualified status (`:626-629`). The
natural implementation -- reuse `_route_payload` for both tiers -- would
therefore have killed **every render** on a provisional engine.

So the provisional path stamps:

* the ordinary bank identity (`voice_ref_id`, `voice_engine`) exactly as a
  normal bank-drawn row does, and
* `lemmy_route_tier` (`qualified` / `provisional` / `unrouted`) and
  `lemmy_route_id` **on the cast row itself**, which is what persists to the
  ledger.

**No edit to `resolve_and_verify_reference` or to any validator.** The rejected
alternative was teaching the validator a `provisional` status -- that widens the
one deliberately brutal fail-closed check to admit unaudited rows and churns
`tests/test_voice_route_validation.py`, whose job is pinning that a
non-qualified status may not render.

**Telemetry, honestly:** a provisional row resolves to `LEGACY_REFERENCE`, so it
gets no route-level render receipt. That is not a regression -- every ordinary
bank-drawn character row today does exactly the same. The tier is legible in the
ledger through the cast row, which is where a listener would look anyway.

### 3B. The rest of the tier

* **A shape that cannot be mistaken for a receipt.** `provisional_receipt`
  carries only machine-suppliable facts: `engine`, `identity_kind`,
  `identity_id`, `artifact_path`, `artifact_sha256`, `rendered_utc`, `state`.
  **No `operator_verdict` field exists** -- not empty, absent -- so a
  provisional row cannot be promoted by filling in a blank. Promotion means a
  human listens and a qualified row is written.
  **No `decided_by` field either.** A driver-signed decision field is the
  surrogate verdict this design exists to refuse; a mechanical timestamp says
  everything true and nothing false.
* **Batch facts live once.** The frozen lines, seed and shared settings go in
  ONE audition manifest; each route references its own clip and hash. Copying
  them per row invites the copies to drift.
* **It DEGRADES; it never raises.** The qualified path is deliberately brutal --
  exactly one bank entry or `VoiceRouteError` with *"there is no fallback"*
  (`:476-483`). A provisional route must not inherit that: killing a render over
  an unauditioned convenience row inverts the risk the tier exists to reduce,
  and a render must not die (Law 7). Unresolvable -> today's draw, stamped
  `unrouted` with the reason.
* **The dormancy gate must test BOTH keys.** `cast_lock.py:817` returns `None`
  early when `approved_native_routes` is falsy. Today indextts2 keeps it
  non-empty, so the coupling is invisible -- but it silently makes the
  provisional tier's reachability depend on an unrelated dict. If the qualified
  route were ever demoted, every provisional route would go dormant with no
  error.
* **A local-clone allowlist replaces unenforced prose.** `rights.scope` is read
  only as a non-blank string (`:216-220`) -- a route whose scope read *"cloud
  engines only"* would validate identically. Provisional `local_wav` routes are
  restricted to an explicit allowlist of local clone engines; provider targets
  are refused. Not natural-language parsing -- a list.

**The strongest argument against, kept visible:** this makes an unauditioned
voice Lemmy's standing identity on up to five engines, and if one lands badly he
is now *reliably* wrong there. Held anyway: the alternative is not "no wrong
voice", it is a different stranger every episode. The recast happens at most
once per engine, and the tier stamp makes it a one-line change.

## 4. The per-engine matrix, corrected against the tree

Three states, never blurred: **`qualified`**, **`rendered_pending_listen`**,
**`configured_unrendered`**.

| engine | route | state after this sprint |
|---|---|---|
| `indextts2` | QUALIFIED -- untouched | `qualified` |
| `bark` | **no route -- already stable** | `rendered_pending_listen` |
| `kokoro` | `bm_george`, provisional | `rendered_pending_listen` |
| `chatterbox` | mirrored clone row, provisional | `rendered_pending_listen` |
| `dia` | mirrored clone row, provisional | `rendered_pending_listen` |
| `google_tts` | `gt_algenib`, provisional | `configured_unrendered` |
| `elevenlabs` | `el_daniel`, provisional | `configured_unrendered` |

**The sprint report says "configured" for the cloud pair -- never "rendered",
never "working".** `eng_google_tts.py:82-110` resolves an API key and fails
loudly without one; `eng_cloud_elevenlabs.py:167-180` goes through
`invoke_partner_node` (credits + auth). CLAUDE.md scope discipline is *100%
local, no cloud services, no API keys, no paid services*. They appear on the
listen page as pending, never as an audition arm.

**The validator already has their shape.** `reference.kind` is
`("local_wav", "provider_voice")` (`nodes/_otr_voice_route.py:54`), and the
`provider_voice` branch (`:295-302`) demands `provider` + `provider_voice_id`
and **no local file**. Both cloud rows carry a `provider_voice_id` in the bank
(`Algenib`, `onwK4e9ZLuTAKqWW03F9`).

**google_tts is better off than the table suggests, honestly so.** THREE real
Algenib clips from 2026-08-08 sit at
`output/otr/episodes/voice_audition_cockney/` -- `1_algenib_plain.wav`,
`2_algenib_cockney.wav`, `3_algenib_cockney_angry.wav` -- alongside
`4_charon_plain_control.wav`, which is a CONTROL VOICE and not Lemmy at all.
(An earlier draft called all four Algenib; the listen page must enumerate exact
filenames and hashes rather than globbing the directory.) The operator approved
that accent, including under emotion. They cannot complete a receipt because the
LINES were never recorded (`config/cast_pools.py:424-437` records exactly this
mistake). They go on the page labelled for what they are: right voice, different
words, prior approval.

**`el_daniel` is the operator's provisional pick and stays flagged.** The bank
tags `el_harry` `american`, contradicting the recollection that made him a
candidate; both go on the page so one session settles it.

**BARK GETS NO ROUTE, AND THAT IS CORRECT, NOT AN OMISSION.** Bark has **zero
bank rows** -- deliberately; it uses presets and its adapter reads
`voice_preset` (`eng_bark.py:32`). The route machinery cannot express it: a
route requires a non-blank `voice_ref_id` and exactly one bank entry for
`(voice_ref_id, engine)` (`:234-236`, `:476-483`). And bark **does not have the
defect** -- `lemmy_row()` pins him at writer time and `_stamp` never overwrites
`voice_preset` (`nodes/cast_lock.py:1016-1049`). Bark needs a clip on the listen
page and nothing else.

### 4A. THE MIRROR GENERATOR IS STALE AND DESTRUCTIVE -- DO NOT RE-RUN IT

The obvious move was to re-run `scripts/_otr_mirror_clone_refs.py`, which
re-tags every indextts2 `char_voice` row onto `cb_*` / `dia_*`. **It would
delete three rows that nine assertions pin.**

Measured: the bank holds FOUR clone-engine announcer rows --
`cb_announcer_male`, `cb_announcer_female`, `dia_announcer_male`,
`dia_announcer_female`. The generator drops every chatterbox/dia row and
recreates exactly ONE of them (`:62-74`), its docstring stating *"dia stays
char_voice-only this pass"*. Pinned by `tests/test_voice_bank.py:237,240,269,
270,311,339,421,453` and `tests/test_tts_engine_sidecars.py:247`.

The counts corroborate the history exactly: chatterbox 42 and dia 42 = 40
mirrored char rows + 2 announcers each, against **41** indextts2 rows today. The
generator last ran at 40, the announcer rows were added outside it, and
`idx_lemmy_algenib_cockney_v1` landed afterwards.

**So: append the two Lemmy clone rows surgically** (`cb_lemmy_algenib_cockney_v1`,
`dia_lemmy_algenib_cockney_v1`), copying the generator's own `_COMMON` field set
from the indextts2 Lemmy row -- same `ref_path`, same `ref_sha256`, one wav
serving all three engines. **And put a guard in the generator** so the next
person to trust its "idempotent" docstring does not silently lose three rows.

Adding one row to each pool still shifts the weighted draw for other characters
on those two engines -- a small, declared re-baseline on two **non-default**
engines. indextts2, the production default, is untouched.

**The approved wav is not where its path literally says.** The bank stores
`models/TTS/refs/indextts2/lemmy_algenib_cockney_v1.wav`; on this box that
resolves to `C:\ComfyUI-Models\TTS\refs\indextts2\`. A documented hazard
(`nodes/_otr_voice_route.py:144-155`) -- resolution goes through
`resolve_voice_ref_path` / `_resolve_ref_to_disk`, never a repo-root join. The
file is real: 298,170 bytes, PCM mono 24 kHz 16-bit, ~6.2 s, sha256
`47e733d5...a60db2`, matching the bank row and the qualification record.

## 5. Build order

1. **The tier.** `provisional_native_routes` + `PROVISIONAL_RECEIPT_FIELDS` +
   `is_provisional_route` in `config/cast_pools.py`, beside the qualified
   machinery and deliberately not sharing its field set. Unit-pin that
   `is_qualified_route` returns **False** for every provisional row -- that test
   is the safety property. Pin the mirror too: an engine in BOTH dicts resolves
   qualified.
2. **Resolution.** Consult provisional after qualified, before the draw. Stamp
   `lemmy_route_tier` + `lemmy_route_id` on the cast row; **never** write
   `voice_route`. Widen the dormancy gate to both keys. Qualified behaviour
   byte-unchanged.
3. **The bank rows -- DONE 2026-08-16, and the generator was the fix.** The
   root repair landed: the mirror generator now owns only the
   `(engine, voice_ref_id)` keys it produces, refreshes them in place and
   passes every other row through untouched, so it is idempotent by ownership
   and needs no `--force` bypass. `_new_id` also strips `idx_`, so the mirrors
   read `cb_lemmy_algenib_cockney_v1` rather than `cb_idx_lemmy_...`.
   Dry-run proof: `mirrored=83 added=2 preserved-unmanaged=3` -- it mints
   exactly the two Lemmy clone rows this sprint needs while preserving the
   three announcer rows the old version would have deleted. **Re-running the
   generator is now the correct way to create the rows; do NOT append them by
   hand.** The write itself lands with the tier, so the two-engine draw
   re-baseline arrives in one change.
4. **The routes.** **Five** provisional rows: kokoro (`bm_george`), chatterbox
   and dia (the appended clone ids, `local_wav`), google_tts (`gt_algenib`,
   `provider_voice`), elevenlabs (`el_daniel`, `provider_voice`). No bark row.
5. **The harness -- a NEW script; G1 stays frozen.**
   `scripts/otr_g1_lemmy_audition.py` is a durable blinded three-arm instrument
   whose output manifest is referenced BY SHA in the one qualified route
   (`config/cast_pools.py:559-563`). Generalizing it risks rewriting that
   manifest and invalidating the only real evidence we have. Write a separate
   cross-engine audition that speaks the frozen `LEMMY_AUDITION_LINES` on bark,
   kokoro, chatterbox and dia, switching on adapter metadata
   (`voice_ref_kind` / `voice_ref_field`) rather than a hand-written table, and
   **preflights each engine** (weights, sidecar venv, reference on disk) before
   the batch -- `assert_usable` proves registration only, never installation
   (`registry.py:141-174`). Hash every clip at write time.
   Verified feasible: all three sidecar venvs exist (`ComfyUI\chatterbox\.venv`,
   `ComfyUI\dia\.venv`, `ComfyUI\index-tts\.venv`), and chatterbox and dia take
   the identical `generate_voice(text, ref_clip_path, delivery_vector, seed)`
   signature as indextts2.
6. **The listen page.** One HTML: the four fresh engines, the existing qualified
   IndexTTS2 clips (referenced, not re-rendered), the historic google_tts clips,
   and the two cloud rows as `configured_unrendered`. Verify each clip's hash
   against the manifest at assembly. Beside it, a keep/demote checklist
   persisted **as data**, so one session produces decisions rather than
   impressions. Post-listening transition is defined: keep promotes only via a
   real human qualification receipt, demote deletes the provisional row, and no
   decision leaves a row in limbo.

## 6. What is NOT built (explicit)

Engine-conditional WRITING; suppression in any form; phonetic respelling;
re-training, re-rendering or re-harnessing indextts2; any cloud render or paid
call; loosening `is_qualified_route`; teaching any validator a `provisional`
status; a driver-signed `operator_verdict` or `decided_by` in any shape.

Also deferred: **migrating bark's `lemmy_row()` pin into the route map.** Still
true that the pin and a route would state one fact twice; still not this
sprint's job -- the pin is consumed far upstream at writer time
(`nodes/_otr_casting.py:1438`), every reader of `lemmy_row()["tts_model"]` needs
grepping first, and bark is the one engine with no defect to fix.

## 7. Gates

Full suite (baseline **10529/110/1**), Bug Bible (**20/26/3 at 284**),
`build_variants.py --check` (**50/0**), AST parse on touched `.py`, BOM check,
Sonnet QA on the finished diff BEFORE the push, HEAD == origin after it.

**A harness clip is not production proof.** The canonical graph pins one
character engine -- node 80 `OTR_CastLock` and node 81
`OTR_BatchCharacterVoices` both carry `indextts2` -- so a direct-harness render
proves the engine speaks but proves nothing about routing in production.
Therefore:

* **ONE canonical leg with both engine widgets switched to a PROVISIONAL
  engine**, proving tier resolution, the ledger stamp, and a non-fatal render
  through `workflows/otr_canonical.json`.
* **The qualified-path regression proof is folded into chunk B's forced-hit
  acceptance leg**, which already runs the canonical graph on indextts2.

Five legs, one per engine, were proposed and **rejected as disproportionate**:
the routing plumbing is engine-independent code, so that exercises one branch
five times at 15-40 GPU-minutes each while the soak is paused, and the
engine-specific risk is exactly what the harness covers.

**A leg landing `lemmy_route_tier == "unrouted"` FAILS the sprint** -- even
though production stays fail-soft. The seeded draw is the defect being fixed;
a green render on it is not feature success. Runtime behaviour and gate
strictness are different questions.

## 8. Questions carried into r2 (coding plan)

1. Where does the provisional consult belong -- inside `select_policy_route`
   (one resolution site, but it must return a differently-shaped claim that the
   caller must not hand to `_route_payload`), or a sibling resolver beside it
   (more duplication, no chance of the two payloads being confused)? r1 made
   this sharper, not settled.
2. `count_locked_characters` uses RAW equality on `row.get("name") ==
   "ANNOUNCER"` (`nodes/_otr_casting.py:1351-1352`) while the rest of the tree
   normalizes. Not this sprint's defect -- but does the provisional stamp touch
   any comparison with the same wart?
3. What exactly guards the mirror generator: a refusal when the on-disk bank
   contains rows it would not regenerate, or a `--force` flag? The first is
   safer and needs a diffing pass the script does not have today.
4. Does the cross-engine harness belong in `scripts/` as a sibling of G1, and
   should it be able to render a SUBSET (one engine) so a failed engine can be
   re-run without re-rendering the batch?
