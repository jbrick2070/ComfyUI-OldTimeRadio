# TTS Voice Preflight

Run this checklist whenever a character/announcer TTS engine is added, a voice
route is added or re-tiered, or the voice bank is regenerated. Format and
acceptance protocol follow `SOURCE_BANK_PREFLIGHT.md` (the house pattern) and
`VIDEO_LANE_PREFLIGHT.md` (the sibling that named this file): every hard item
receives `PASS`, `FAIL`, or an explicitly allowed `N/A`, plus evidence -- a file
and line, test name, validator output, or receipt path. Save an
`ID | status | evidence` matrix; the final receipt names that matrix and its
SHA-256. Any hard `FAIL` stops the work.

Machine enforcement lives in `tests/test_tts_voice_preflight_matrix.py`. This
document narrates those gates and is **never a substitute for running them** --
the `vram-recipe-lab/PREFLIGHT.md` rule. Written 2026-08-16, enforcement first,
per the family rule that a sibling is never created as an empty paper checklist.

**Every gate below exists because something real bit** during the 2026-08-16
cross-engine Lemmy work. Where a gate has no twin assertion, it says why.

---

## Gate 1 -- The dispatch surface

- **P1.1 The degraded fallback equals the profiles table.** A voice node's
  `LEGACY_FIRST_FALLBACK` must equal `legacy_first_engines(role)`, in order.
  *Why:* `build_engine_combo` prefers the profiles helper and drops to the
  node's hardcoded tuple only when the import RAISES -- exactly when nobody is
  watching. That tuple had drifted two engines short, so a degraded boot offered
  a character-voice dropdown with **no elevenlabs and no google_tts** and
  nothing logged it. A fallback that is not equal to the thing it stands in for
  is not a fallback; it is a second, worse answer.
  Order matters as much as membership: index 0 is the byte-identical default
  for the role. *Twin:*
  `test_p1_the_degraded_fallback_matches_the_profiles_table`, parametrized over
  both voice nodes.

  **The MUSIC node has the same drift and is NOT fixed here -- flagged for the
  operator.** `nodes/stable_audio_theme.py:36` declares
  `("musicgen", "stable_audio_music")` while the profiles table for `music` is
  `("stable_audio_3", "musicgen", "stable_audio_music", "sonilo",
  "google_lyria")`. It is short by three AND its index 0 disagrees with the
  shipped default, so a degraded boot would render music on `musicgen` instead
  of the promoted `stable_audio_3`. That is a silent DEFAULT change rather than
  a missing menu entry, it is outside the TTS surface this preflight governs,
  and it deserves its own decision rather than a drive-by edit inside a voice
  sprint. `MUSIC_AUDIO_PREFLIGHT.md` is the right home for the gate.
- **P1.2 Every name in the dropdown is a registered adapter that serves the
  role.** A menu entry no adapter answers to is a queue-time crash wearing a
  menu entry. *Twin:* `test_p1_2_every_char_voice_engine_is_registered_...`.
- **P1.3 Registration is NOT installation.** `assert_usable` checks
  registration and role compatibility and does **no IO**
  (`nodes/_otr_audio_engines/registry.py:141-174`). Before any render batch,
  preflight each engine's real prerequisites: sidecar venv python on disk,
  weights present, reference resolvable. *No twin:* the check needs disk and
  differs per box; run it in the harness, not the suite.

## Gate 2 -- Bank truth

- **P2.1 Every routed `voice_ref_id` resolves to EXACTLY ONE bank row for its
  engine.** The route resolver demands exactly one and has **no fallback**
  (`nodes/_otr_voice_route.py:476-483`); zero or two is a hard failure at cast
  time. *Twin:* `test_p2_1_every_routed_voice_ref_resolves_to_exactly_one...`.
- **P2.2 A preset engine may legitimately carry ZERO bank rows.** Do not read
  "no bank rows" as "engine is broken". Bark is preset-driven by design and its
  adapter reads `voice_preset`, so a route for it is *inexpressible* -- a route
  requires a bank-resident `voice_ref_id`. Check `voice_ref_kind` /
  `voice_ref_field` before concluding a row is missing.
  *Twin:* `test_p2_2_a_preset_engine_may_carry_zero_bank_rows`.
- **P2.3 The `ref_sha256` sentinels say which kind of row this is.** A real
  64-hex digest for clone rows; the literal `"cloud"` for cloud presets (which
  also carry `provider_voice_id`); `"pending"` for local presets. Feeding a
  sentinel to a byte check is how `gt_algenib` once looked usable as a local
  IndexTTS2 reference. *Twin:* `test_p2_3_the_sha_sentinels_say_which_kind...`.
- **P2.4 Adding a row to an engine's pool is a RE-BASELINE on that engine.**
  The pick is a weighted draw sorted by score then `voice_ref_id`
  (`nodes/_otr_voice_bank.py:474-492`), so one new row moves which voice every
  other character draws on that engine. Declare it; run the suite before and
  after the bank edit as separate steps so a moved assertion is attributable.
  *No twin:* this is a procedure, not a state.
- **P2.5 The bank's paths are relative to the MODELS root, not the repo.** On
  this box `models/TTS/refs/...` resolves under `C:\ComfyUI-Models\`, and a
  repo-root join produces a path that has never existed
  (`nodes/_otr_voice_route.py:144-155`). Always resolve through
  `resolve_voice_ref_path` / `_resolve_ref_to_disk`.
- **P2.6 There is no accent or nationality field.** The schema has none; accent
  rides informally in `timbre` (`el_daniel` -> `"british"`) or `style_tags`
  (`bm_george` -> `"british_leaning"`), and the qualified Lemmy clone row
  carries **no Cockney tag at all** -- its accent lives in the id string and the
  wav bytes. Never assert an accent from a bank row; it is not recorded there.

## Gate 3 -- Generated data

- **P3.1 If bank rows are GENERATED, prove a re-run reproduces the bank as it
  stands -- before trusting the word "idempotent".**
  *Why, and this one nearly cost three rows:* `scripts/_otr_mirror_clone_refs.py`
  drops every chatterbox/dia row and regenerates them from the indextts2 rows.
  The bank had since gained four clone-engine announcer rows; the generator
  recreates exactly one. A well-meaning re-run -- which its own docstring
  invited -- would have permanently deleted `cb_announcer_female`,
  `dia_announcer_male` and `dia_announcer_female`, which nine assertions across
  `tests/test_voice_bank.py` and `tests/test_tts_engine_sidecars.py` pin, and
  the resulting red would have looked unrelated to "I refreshed the mirror".
  **A generator may only be trusted to own rows it can actually recreate.** The
  script now refuses to write when it would destroy an unregenerated row;
  `--force` overrides deliberately.
  *Twin:* `test_p3_the_mirror_generator_refuses_to_delete_what_it_cannot_recreate`.

## Gate 4 -- The route contract

- **P4.1 Qualification requires a HUMAN.** `operator_verdict` is in
  `QUALIFICATION_RECEIPT_REQUIRED_FIELDS` and the comment says why: *"a human
  said yes; nothing else can supply this"*. No automated pass may fill it. A
  driver-signed verdict is the evidence-shaped-but-not-evidence pattern: bark
  once claimed `qualification_receipt: "canonical_bark_preset_v1"`, a bare
  string asserting an audition that never happened.
  *Citation caveat (2026-08-16):* the comment at `config/cast_pools.py:382-387`
  attributes that to `BUG-12.86`, but Bible `12.86` is
  *receipt-keyed-on-a-string-the-producer-never-emits*
  (`BUG_BIBLE.yaml:7070-7082`), which is a different shape. The lesson stands on
  its own evidence; the id is unverified and is deliberately not repeated here
  until someone traces the real record.
  *Twin:* `test_p4_1_qualification_still_requires_a_human` plus the fail-closed
  parametrize in `test_p4_2_an_unproven_route_is_never_qualified`.
- **P4.2 NEVER stamp a non-qualified route into `cast_row["voice_route"]`.**
  `resolve_and_verify_reference` treats ANY non-empty `voice_route` dict as a
  route claim (`nodes/_otr_voice_route.py:595-597`) and **raises**
  `VoiceRouteError` unless `status` is exactly the qualified status
  (`:626-629`). A second-tier route stamped there kills **every render** on that
  engine. Carry a lower tier as ordinary bank identity plus your own fields on
  the cast row instead.
- **P4.3 `rights.scope` is prose and is never parsed.** It is read only as a
  non-blank string (`:216-220`); a route whose scope read *"cloud engines only"*
  would validate identically. If an engine-class restriction matters, enforce it
  with an explicit allowlist, not a sentence.
- **P4.4 `audition_manifest.path` / `.sha256` are shape-checked only** and never
  opened or verified against disk (`:305-312`). Hash audition artifacts at write
  time if the receipt is meant to be trustworthy later.
- **P4.5 A new route TIER must be added to the CastLock dormancy gate.**
  `nodes/cast_lock.py:817` returns early when `approved_native_routes` is falsy.
  A second tier not named there goes silently dormant the moment the first dict
  empties -- no error, no log.
- **P4.6 A qualification manifest is EVIDENCE; do not point a general tool at
  its output directory.** The qualified IndexTTS2 route references
  `g1_lemmy_test_a/MANIFEST.json` by sha256, so a generalized harness that ever
  wrote to that hardcoded directory would silently invalidate the only real
  audition receipt in the tree. Give a new harness its own output root.

## Gate 5 -- Engine runtime declarations

- **P5.1 Every engine declares how its identity ARRIVES, and the two
  declarations are not interchangeable.** `voice_ref_kind` is how the reference
  is *passed* (the three clone engines say `wav_path` and take it as a call
  argument, carrying **no** row field); `voice_ref_field` is which *cast-row
  field* holds the identity (bark reads `voice_preset`, kokoro reads
  `voice_ref_id`). A generic harness must switch on this metadata, never a
  hand-written engine table. *Twin:* `test_p5_1_every_engine_declares_how_its_identity_arrives`.
- **P5.2 Sample rate is declared per adapter and must match its profile.**
  `pack_audio_batch` raises on any clip whose rate differs, i.e. long after the
  render is paid for. Today: indextts2 22050, chatterbox 24000, dia 44100,
  bark 24000, kokoro 24000, elevenlabs 44100, google_tts 24000.
  *Twin:* `test_p5_2_every_char_voice_engine_declares_a_positive_sample_rate`.
- **P5.3 Clone references are engine-agnostic.** All three clone engines share
  one pool of reference wavs under `refs/indextts2/`; the per-engine
  `refs/<engine>/` convention was retired deliberately. One approved wav clones
  on all three.

## Gate 6 -- Cloud engines

- **P6.1 A cloud engine cannot be rendered under the local-only scope rule.**
  `google_tts` resolves an API key from `OTR_GOOGLE_API_KEY` / `GEMINI_API_KEY` /
  `GOOGLE_API_KEY` and fails loudly without one; `elevenlabs` goes through
  `invoke_partner_node` (credits + auth). CLAUDE.md scope discipline is *100%
  local, no cloud services, no API keys, no paid services*.
- **P6.2 Say "configured", never "rendered" or "working".** A cloud row that was
  mapped but never heard is `configured_unrendered`. It may appear in a listen
  page as pending; it may never appear as an audition arm.

## Gate 7 -- Proof

- **P7.1 A harness clip is not production proof.** The canonical workflow pins
  ONE character engine -- `OTR_CastLock` (node 80) and
  `OTR_BatchCharacterVoices` (node 81) both carry it -- so a direct-harness
  render proves the engine speaks and proves nothing about routing in
  production. At least one acceptance leg must go through
  `workflows/otr_canonical.json`.
- **P7.2 Scale the legs to what actually differs.** Routing plumbing is
  engine-independent code; N legs across N engines exercise one branch N times
  at 15-40 GPU-minutes each. Prove the engine in the harness and the routing in
  one canonical leg.
- **P7.3 A degraded success is not success.** If the acceptance leg lands on the
  ordinary draw rather than the intended route, the gate FAILS even though the
  render is green -- production may stay fail-soft, but the gate may not.

## Receipt

Save `ID | status | evidence` for every gate above, name the matrix file and its
SHA-256, and record which engines were rendered versus configured. A gate marked
`N/A` states the reason in the evidence column.

## The family

- `SOURCE_BANK_PREFLIGHT.md` -- the format authority.
- `VIDEO_LANE_PREFLIGHT.md` -- the video sibling; enforced by the S8c suite.
- `TTS_VOICE_PREFLIGHT.md` -- this file; enforced by
  `tests/test_tts_voice_preflight_matrix.py`.
- Still future, each backed by its own enforcement code before the doc is
  written: `STILL_LANE_PREFLIGHT.md`, `MUSIC_AUDIO_PREFLIGHT.md`,
  `LLM_WRITER_PREFLIGHT.md`, `UPSCALER_PREFLIGHT.md`.
