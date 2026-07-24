# Extending OTR -- add your own source bank

**Status:** requirements contract of record (2026-07-24). Companion plan:
`docs/2026-07-24-independent-source-banks-v1-plan.md`. Deep reference:
`docs/SOURCE_BANK_GUIDE.md` (the shipped-bank playbook this doc builds on).
Every field name below is grounded against the live code; re-verify pins at
build time.

OTR ships six source banks. Every bank is INDEPENDENT and EQUAL -- its own
definition, its own fetch/interpret strategy, its own story pack. Adding your
own bank means adding a seventh peer, not plugging into a special "user" tier.
Your bank runs through the same trusted shared writer and the same production
tail as the shipped six.

This is not foolproof and is not meant to be. You own your bank. OTR gives you
one honest contract, loud failures that name the broken field, and a cleanup
pass that helps you meet the contract. If your bank ships a broken episode
anyway, that is a bug in your bank: fix it and re-run.

---

## 1. The one rule that matters: the ledger must be COMPLETE

Everything downstream of the writer -- TTS, per-beat audio slicing, shot
direction, captions, the credits roll, final assembly and `obs_publish` --
reads LEDGER FIELDS, not intentions. An episode with a hole in its ledger is a
broken render, no matter how good the story is.

Your bank's job is simple to state: after the writer finishes, the episode
ledger (`{"cast": [...], "lines": [...], "meta": {...}}`) must be complete.
The shared writer and its ledger-cleanup pass (section 4) do most of this FOR
you -- but they can only clean what your source material and story pack let
them build. Learn this contract before you author a pack.

### What each consumer reads (the contract, grounded)

**Voice / TTS** (all voice nodes share one loop): per line -- `line_id`,
`char_id`, `text` (non-empty unless the row is an explicit skip), tension.
Cast row -- `char_id`, `name`, `gender`, `voice_preset` / `voice_ref_id`,
timbre/age when present. Meta -- `episode_seed` (drives deterministic voice
casting), `cast_lock_revision`. Content-owned lanes must stamp verified
`text_for_tts`; a stale or missing stamp is a terminal error BEFORE audio
generation.

**Scene sequencing / per-beat slicing:** `lines[].speaker_role`, `line_id`
(this is the key that matches every audio clip back to its script row),
`text`; music rows -- `cue_id`, `placement`, `description`, `title`; the
stamped timeline (`start_s`, `dur_s`) and the clips manifest.

**Shot direction / video:** `speaker_role` maps to a video role with NO
FALLBACK -- an unknown role stops the render loudly. Also read: `line_id`,
`skip`, `boundary`, `beats[].beat_id`, cast appearance (`portrait_prompt` /
`appearance` / `character_description`, else `name`), `meta.episode_id`,
`meta.story_brief_terms.setting`, music `cue_id`, `visual_prompt`.

**Captions:** per line `start_s`, `dur_s`, `text`, `speaker_role`, `char_id`;
the speaker label comes from the cast row's `name`. A missing `start_s`/`dur_s`
means zero-width cues; a missing cast row means unlabeled dialogue.

**Credits roll (NO-FALLBACK -- raises on any missing receipt):**
`meta.episode_title`, `meta.style` (or the explicit no-scaffold receipts),
`meta.render_engines`, `meta.image_engines`, `meta.music_engine`,
`meta.cast_contract.cast_seed` OR `meta.episode_seed` (one of the two is
REQUIRED), `meta.gen_params_initial.seed_source`.

**Final assembly / publish:** the clips manifest (`clips`, per-row `shot_id`,
`beat_id`, `start_s`, `target_frame_count` > 0, manifest `fps` > 0); the mux
stamps `final_video_path` and publishes to `otr/obs/` (`obs_publish OK`).

**Allowed `speaker_role` values:** `character`, `announcer`, `music_open`,
`music_close`, `music_inter`. Nothing else. A skip row must carry empty text
plus a `tts_skip_reason`.

The authored-inputs table (exact minimum fields for cast / scenes / shots /
beats / lines / music) lives in `docs/SOURCE_BANK_GUIDE.md` section 7. That
table plus the consumer list above IS the complete-ledger contract.

---

## 2. What YOU provide (one folder, one bank)

```
user_packs/source_banks/<bank_id>/
  bank.json          # {"schema_version": "v2.0", "bank": { ...the row... }}
                     #   the row is EXACTLY a shipped banks.json row: id,
                     #   label, source_kind, fetcher/interpreter entry points,
                     #   default_story_pipeline, default_story_model,
                     #   defaults, required_seams, runnable, guide_ref.
                     #   Every key is required; unknown keys are rejected.
  <bank_id>.py       # single file: fetch_source + interpret_source +
                     #   check_compatibility (keyword-only, typed contracts)
  story_packs/
    <model_id>.json  # >=1 story pack, filename == its story_model_id
  fixtures/          # deterministic samples for activation preflight
  .otr_receipt.json  # WRITTEN BY --activate; never hand-edit
```

- `<bank_id>` is the folder name, the row's `source_bank_id`, and the dropdown
  value -- all three must match. Use lowercase letters, digits and underscores,
  starting with a letter. The six shipped ids (and `custom_source_bank`) are
  protected: a bundle that tries to shadow one is quarantined, and the shipped
  bank is untouched.
- Your bank row is parsed by the SAME parser that validates the shipped six and
  held to the same cross-reference contracts: the default pipeline must be
  registered, the default pack must exist under your own `story_packs/` and
  declare that pipeline, every `required_seams` entry must be present in it,
  and a `runnable` bank must have a real execution lane.

- **`fetch_source`** returns the exact seven-key payload envelope --
  `headline`, `summary`, `full_text`, `source`, `date`, `link`, `seed_text`
  (all strings, `seed_text` non-empty; unknown key = hard error) -- plus the
  `source_meta` / `source_rights` provenance sidecars. This is the same
  contract every shipped fetcher obeys (`SOURCE_BANK_GUIDE.md` section 5).
- **`interpret_source`** returns the interpreter surfaces the shared writer
  consumes (casting brief, script brief, close brief, key terms).
- **`check_compatibility`** accepts or refuses a request (word target, refine,
  source_ref, custom premise) with a structured reason.
- Python discipline: single file, stdlib + the OTR contracts leaf at import
  time, heavy imports lazy inside functions. If your code needs a third-party
  library that is not installed, the run HARD-FAILS with the ImportError --
  install the library and re-run. There is no dependency manifest.
- Your code runs in-process, user-trusted, like any custom node. `--activate`
  is the consent act.

## 3. What OTR provides (so you cannot break the pipeline)

- **The shared writer builds the ledger.** Your bank supplies source material
  and prompts; the trusted writer authors the script, owns every ledger write,
  and runs the shared tail. Your code never touches the canonical ledger, so a
  buggy bank cannot corrupt an episode's durable state.
- **Validation + quarantine.** `otr_check bank <path> --activate` validates
  your JSON and contracts, imports your Python in a bounded child process, and
  runs your fixtures. A broken bundle is QUARANTINED with a named, actionable
  issue -- it never appears in the dropdown and never breaks ComfyUI boot.
  Activation writes a content-addressed snapshot plus `.otr_receipt.json`; boot
  admits your bank only when the bundle's authoring bytes still hash to that
  receipt AND the snapshot is present. Edit anything and the bank goes STALE
  (quarantined, named) until you re-activate. The digest covers every authored
  file's path, size and content and ignores timestamps, so re-activating
  unchanged bytes is a no-op.
- **One quarantine never spreads.** Each bundle is judged alone: a broken bank
  costs its own dropdown row and nothing else. Every shipped bank and every
  healthy client bank still loads. The refusals print at load and are readable
  programmatically from the routing layer's validation-issue list.
- **Bounded fetch.** Network access goes through one bounded fetch seam
  (timeouts, redirect cap, size cap, https-only, loopback/private reject).
  A tripped bound fails loudly; there is no silent retry-forever.
- **Loud, named failures everywhere.** No silent fallback to another bank,
  model, feed, or asset -- ever.

## 4. The ledger-cleanup pass (your safety net, not your license)

After your fetch/interpret and the writer's passes, the shared tail runs a
ledger cleanup/completion pass: deterministic checks plus as many LLM passes
as needed to fill gaps, normalize rows, and sanitize content IN PLACE.

- **Content is repaired, never fatal.** Profanity or unsafe content in a line
  gets sanitized in place. Length, style, vocabulary, and quality NEVER fail a
  story (the standing law: an audit may improve a story, never fail one).
- **Structure is fatal.** A required field with no owner and no value after
  cleanup -- a line with no `line_id`, a speaker with no cast row, a missing
  `episode_seed` -- hard-fails the run with the field named. Fix the bank, not
  the symptom.

## 5. Author checks before you activate

1. Read `docs/SOURCE_BANK_GUIDE.md` sections 5 and 7 end to end.
2. Run your fetcher against your REAL source at both size extremes -- prompts
   silently tune themselves to the source shape they were tested on.
3. Extract cast FROM your source where the source has one; never let the
   model invent an adaptation's characters.
4. Confirm every generation-time seam in your pack says what YOUR source
   needs (adaptation packs: carry the source's own words; original packs:
   invent freely). The seam that misbehaves is the one in the prompt at that
   moment -- trace it, do not guess.
5. `RESULT SUCCESS` is not proof. Proof is `obs_publish OK` plus the episode
   file in `otr/obs/`.
6. Change one thing at a time and re-run; a bank is qualified by artifacts
   (ledger + episode asset + publish receipt), nothing softer.

*Sections 2-3 describe the v1 authoring surface being built under the plan of
record; the ledger contract in section 1 is live today and applies to every
bank, shipped or client-authored.*
