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
  <bank_id>.py       # single file: fetch_source + interpret_source
                     #   (keyword-only; check_compatibility is a RESERVED
                     #   name with no consumer yet -- see section 2)
  story_packs/
    <model_id>.json  # >=1 story pack, filename == its story_model_id
  fixtures/          # optional: recorded fetch payloads, one JSON each,
                     #   validated at --activate by normalize_fetch_result
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

- **Routing an entry point to your own code.** Set the row's `fetcher` and/or
  `interpreter` to the reserved value `"self"` -- "my bundle owns this lane".
  OTR then calls your module's `fetch_source` / `interpret_source`. Only a
  CLIENT bundle may say `"self"`; the shipped registry never learns that value,
  so you can neither shadow nor replace a shipped entry point. You may instead
  name a registered shipped id (`science_rss`, `media_archive_rss`,
  `public_domain_source`, `shakespeare_folger` / `news_interpreter`,
  `media_archive_interpreter`, `public_domain_interpreter`,
  `shakespeare_interpreter`) and reuse that lane wholesale, or mix -- a `"self"`
  fetcher with a shipped interpreter is fine. Any other value is a typo and
  quarantines the bundle. A bank only ever executes its OWN bundle.
- **`fetch_source(*, bank, technical_model, source_ref="", load_config=None,
  policy=None)`** returns the exact seven-key payload envelope --
  `headline`, `summary`, `full_text`, `source`, `date`, `link`, `seed_text`
  (all strings, `seed_text` non-empty; unknown key = hard error) -- either as a
  plain dict or wrapped in `SourceFetchResult` to carry the
  `source_meta` / `source_rights` provenance sidecars. This is the same
  contract every shipped fetcher obeys (`SOURCE_BANK_GUIDE.md` section 5), and
  your return value is validated by the same `normalize_fetch_result`. Your
  lane's ledger `seed_source` stamp is `user_bank:<bank_id>`, so client-sourced
  provenance is never mistaken for a shipped fetcher's.
- **`interpret_source(*, bank, payload, technical_fn, model_id)`** returns the
  interpreter surfaces the shared writer consumes: an object exposing
  `casting_brief`, `script_brief`, `key_terms`, `attempts` and `model_dump()`
  (with `news_close_brief` in the dump). The same
  `validate_interpreter_result` that judges the shipped interpreters judges
  yours, and the writer -- not you -- writes the result into the ledger.
  If your interpreter exhausts its own structured-output repair ladder and
  raises `SourceInterpretError`, the writer does NOT abort: it derives a
  deterministic same-source brief from your bank's label and the validated
  payload, exactly as it does for the shipped four, and stamps
  `meta.source_interpreter.status = "deterministic_same_source_fallback"`.
  That brief carries your source forward verbatim and invents no genre --
  it keeps the episode alive, it is not a substitute for a working
  interpreter. Any other failure (bad config, backend down, a contract
  violation in your return value) still propagates loudly.
- **`check_compatibility`** is a RESERVED NAME with no contract. Nothing calls
  it, and `otr_check bank --activate` deliberately does not inspect it -- not
  even for callability. There is no request type, no decision type, and no
  runtime consumer, so there is nothing about it that activation could check
  honestly; enforcing a shape now would freeze an interface with nobody to keep
  it true. Define it if you like and it will be ignored. If a wave ever gives
  it a real consumer, that wave defines its types and its checks together.
- **Activation binds your lanes' signatures.** For every entry point your row
  routes to `"self"`, `--activate` checks that the function can ACCEPT the
  keywords listed above -- by binding them, never by calling anything. A
  `**kwargs` catch-all satisfies this. A function that imports cleanly but
  takes the wrong keywords is refused at activation rather than minutes into
  your first render.
- **Going to the network: use the bounded seam, not your own client.** If your
  `fetch_source` reads a feed or scrapes a page, import
  `nodes._otr_feed_fetch` and call `fetch_feed(url)` or `fetch_article(url)`.
  It is stdlib-only, so it costs you no dependency, and it is the same seam the
  shipped banks use:

      from nodes._otr_feed_fetch import (
          FeedFetchUnavailable, fetch_article, fetch_feed,
      )

      document = fetch_feed(url)      # document.text is decoded and bounded

  Every fetch is https-only (no silent upgrade from `http://`), connect 5s /
  read 10s, at most 3 redirects, at most 2 MiB of DECODED body, 2 retries, one
  ~25s monotonic deadline for the whole call, and a media-type check (a feed
  must be RSS/Atom/XML, an article HTML/XHTML). Loopback, private, link-local,
  multicast and reserved addresses are refused -- on every redirect hop, not
  just the first -- and a name that resolves to a mix of public and private
  addresses is refused outright.

  **The two failure classes are different and you should treat them
  differently.** `FeedFetchRefused` means a bound of OURS tripped: the URL or
  the configuration is wrong. Let it propagate -- do not catch it, and never
  turn it into an empty result. `FeedFetchUnavailable` means the remote simply
  did not deliver (404, paywall, timeout, 503 after retries); if you have
  another candidate to try, catching this one and moving on is legitimate.
  `.reason` on either carries a stable machine-readable code.

  Nothing forces you through the seam -- your module is ordinary Python and
  could open its own socket. Do not. A bank that brings its own HTTP client
  inherits none of the above, which is exactly the trap wave 5 exists to close:
  network hardening is NOT inherited, it has to be wired on purpose.

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
  your JSON, your bank row, your story packs and every cross-reference boot
  enforces; imports your Python in a child process bounded by wall time and
  killed as a process tree; binds your lanes' signatures; and validates each
  `fixtures/*.json` as a recorded fetch payload through the very
  `normalize_fetch_result` your live `fetch_source` output will meet. (It does
  not CALL your functions -- fixtures are checked as data, not replayed as
  cases.) A broken bundle is QUARANTINED with a named, actionable issue -- it
  never appears in the dropdown and never breaks ComfyUI boot.
  Run `otr_check bank <path>` with no `--activate` for the same report without
  executing a single line of your code, and `otr_check bank --all` for every
  client bank at once.
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

After your fetch/interpret and the writer's passes, the shared tail runs
`nodes/_otr_ledger_cleanup.run_ledger_cleanup` -- deterministic completion,
then safety repair, then a bounded LLM fill -- and stamps its receipt at
`meta.ledger_cleanup`. It is the last thing that touches canonical line text.

- **It fills what it can derive.** A row with no `line_id` gets one minted; a
  duplicate id is renamed, not dropped; a blank `speaker_role` is resolved
  from a `char_id` that names a real cast row; a row with nothing sayable
  becomes an EXPLICIT skip carrying its reason instead of a silent hole; stale
  word/char counts are re-derived. Nothing here authors prose.
- **Content is repaired, never fatal.** Profanity and explicit weapon or
  sexual language in a delivered spoken row is rewritten in place by the
  shared same-story cleanup. If repair itself fails, the residual hits are
  reported and the freeze gate's G9 check remains the last-resort backstop --
  the cleanup pass never adds a second terminal content policy. Length, style,
  vocabulary and quality NEVER fail a story (the standing law: an audit may
  improve a story, never fail one).
- **One prose field is filled for you.** A blank `meta.episode_title` gets one
  bounded same-story LLM title, then a title derived from your source
  headline. `otr_credits_roll` raises on a missing title, so a hole here would
  otherwise surface as a crash minutes later in a node with no idea which bank
  caused it.
- **Structure is fatal.** A required field with NO owner and NO value after
  all of the above hard-fails with every offending field named at once: a
  blank `speaker_role` that no `char_id` could resolve, a role outside the
  five allowed values, a voiced row with no `char_id` at all, a skip row with
  no reason, a cast row with no `char_id` or no `name`, an unfillable
  `episode_title`. Fix the bank, not the symptom.
- **What it does NOT judge.** Fields another producer stamps later are not
  holes here: `meta.style`, `meta.render_engines` / `image_engines` /
  `music_engine`, the timeline (`start_s` / `dur_s`) and the clips manifest.
  Nor is the episode-seed receipt, which is owned by the cast picker or by the
  writer's content-owned branch. And a voiced `char_id` need not name a cast
  row -- the ANNOUNCER speaks on nearly every episode with no cast entry at
  all.

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

---

## 6. Selecting your bank in ComfyUI

There is nothing to wire. The workflow already carries the surface your bank
arrives on, and adding a bank changes no node, no widget and no link.

- **Your bank is a row in the `source_bank` dropdown** on node 1, the Story
  Writer (`OTR_LedgerScriptWriter`). That dropdown is not a stored list: its
  choices are read LIVE from the routing registry every time ComfyUI asks the
  node for its inputs, and activated client banks are folded into that registry
  beside the shipped six. Activate, restart ComfyUI, and your `<bank_id>` is
  simply there.
- **Restart is the refresh.** The registry is built once per process and cached,
  so a bank activated while ComfyUI is running does not appear until you restart
  it. The same is true in reverse: edit your bundle without re-activating and
  the bank goes STALE and drops out of the list at the next start.
- **Your story pack needs no widget.** The pack comes from your own row's
  `default_story_model`, resolved inside your own bundle's `story_packs/`
  folder. There is no pack selector to set and no shipped directory your pack
  has to be copied into -- ship more than one pack if you like, the row's
  default is the one that runs.
- **`+ Add Your Own` is a signpost, not your bank.** That row ships
  non-runnable on purpose (its id is `custom_source_bank`). Picking it fails
  loud before any story work, and the error repeats the path in this document.
  Your bank is its own row; you never select this one.
- **A quarantined bank is simply absent.** If your bundle fails validation it
  does not appear in the dropdown at all -- it does not appear greyed out, and
  it never blocks boot. Run `otr_check bank <path>` to see the named reason.

*The ledger contract in section 1 applies to every bank, shipped or
client-authored. Sections 2-6 describe the v1 authoring surface as it is
LANDED: bundle integrity and admission, the self-owned entry points, the
`otr_check bank --activate` CLI, the bounded fetch seam, the ledger-cleanup
pass, and the dropdown surface above.*
