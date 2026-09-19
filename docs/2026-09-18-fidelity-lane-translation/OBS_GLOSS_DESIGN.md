# The obs filename carries a two-word English gloss — design, not yet built

**Operator, 2026-09-18:** *"they could say, this is the episode title. I need
you to summarize it in two words for a file name, index"*, and *"I need to
summarize it in English. Two words."*

> The obs filename is an index, not the title: two English words that tell me
> which episode it is, then the language and the clock. The real title lives in
> the ledger, on the card, and in the announcer's mouth — never in a filename I
> cannot read.

Designed by the Fable pass, 2026-09-18. **Nothing here is built yet.** Every
file:line below is a claim to be checked before it is coded.

## Why a gloss and not the alternatives

| candidate | readable | bounded | anchored to the episode |
|---|---|---|---|
| **two-word English gloss** | yes | yes, structurally | yes |
| translated title | yes | **no** — `the_christening_gown_beneath_the_door` is English and 37 chars | yes |
| invented name | yes | yes | **no** — drifts off the episode |
| truncated native title | **no** — `嘘の夜` is still Japanese | yes | yes |
| hash | **no** | yes | yes |
| nothing | **no** | yes | — |

The argument that decides it: **wrongness is cheap.** A gloss is openly a
paraphrase, so a slightly-off gloss (`stage_fright` for *The Understudy*) is not
a defect of the same kind as `XunoYeMingke` — a romanisation that is confidently
wrong and announces nothing. Romanisation was measured and rejected: `anyascii`
gives the CHINESE reading of Japanese kanji and drops Devanagari vowels;
`misaki.cutlet` returns IPA, not romaji.

Applied to **every** language including English. The length problem is real on
English titles too, and one shape in the folder is the "more logical" that was
asked for.

## Where the gloss is minted

**Correction to the original plan:** the title is NOT emitted in one place.
`nodes/_otr_writer_tail.py:842-950` resolves it from THREE branches — a typed
widget, a custom-lane `final_title_override`, and an LLM regen via
`_generate_title_from_script` — and `nodes/_otr_ledger_cleanup.py:334` can fill
it again later. So this cannot be one more field on an existing call; that call
fires on only one branch.

- **Site:** `_otr_writer_tail._run_writer_tail`, after `canon.title = final_title`
  (~line 956) and before the stamp at ~line 1030, where all three branches have
  merged and `final_title` is non-empty by construction.
- **Slot:** technical (`ctx.technical_fn`) — the same choice
  `_otr_ledger_cleanup._llm_episode_title` makes for naming an artifact that
  already exists. Wrap in `slot_scheduler.helper_context("gloss_title")`.
- **Call:** `structured_call` with a one-field schema, `post_validator`,
  `max_attempts=2`, `max_new_tokens=32`. Prompt carries the title and the
  language label and nothing else — the operator said *summarize the title*, not
  *summarize the episode*. Deliberately NOT the premise: it is the one lever
  that lets the gloss drift off the title.

**Ledger fields**, stamped beside `episode_title`:
- `meta.obs_title_gloss` — the accepted string, or `""` when the fallback took over
- `meta.obs_title_gloss_source` — `llm_two_word_gloss` or `native_title`, plus
  the last rejection reason when it fell back. That is the receipt.

Stamped ONCE and never rewritten. A later tool changing it would drop the
pointer to the planned path on the next save, which is the PBUG-06 failure.

## Validation, and what each rejection costs

Normalise: strip, NFKD, drop combining marks, lowercase, runs of `[^a-z0-9]` to
one space.

| check | reject when | catches |
|---|---|---|
| alphabet | a letter is still non-ASCII after NFKD | native script returned anyway; the title verbatim on ja/zh/hi. `café` folds to `cafe` and passes |
| word count | 0 tokens, or 4+ | a sentence. Ask for two, TOLERATE one or three — rejecting a good three-word answer costs a retry and then an unreadable fallback |
| length | a token over 16 chars, or joined over 24 | run-on / concatenated output |
| frame echo | tokens ⊆ {title, episode, summary, filename, label, words, untitled} | a small model repeating the instruction |
| profanity | **no check** | operator directive 2026-08-03 — no content filter on the generation path |

**Accepted leak, on purpose:** a Latin-script non-English gloss (`reloj
inquieto`) passes. It is readable, and this repo already refused a language
detector for exactly this case (`_otr_verbatim_translation.py:14-15`).

**Fallback = the native title exactly as `_obs_title` renders it today.**
Deterministic, never empty (`_obs_title` returns `"episode"`), already covered by
64 tests, and **visibly** a fallback — a native-script name says at a glance that
the gloss did not happen. Rejected alternatives: a bank name looks intentional
while distinguishing nothing the `sspr` code does not already carry; the
timestamp alone is the PBUG-20260918-09 bug shape; and `work_title` is NOT safe,
because on `media_archive` it holds the PUBLICATION, not the play
(`_otr_writer_tail.py:880-885`).

## The binding — it cannot be left untouched

`_otr_ledger._published_obs_path:209-213` is `stem == form or
stem.startswith(form + "_")` against the native id. Any name not leading with the
native title fails it, and not leading with the native title is the entire point.

**Identity tail**, derivable from the episode id by one anchored regex:
`_(?:[a-z]{2}_)?\d{8}_\d{6}(?:_replay_\d{8}_\d{6}_\d{6})?$`. The iso is absent on
English; the replay suffix comes from `production_ledger.replay_episode_id`, and
`replay_from` deep-copies `meta` so the gloss rides along.
`_safe_episode_title_slug` output cannot contain that pattern.

**Option A — tail only.** Accept any stem containing the identity tail. No
signature change, but the validator can no longer tell two episodes rendered in
the same second apart, and `tests/test_meta_paths.py:336` flips from refused to
accepted. **Rejected.**

**Option B — gloss + tail. RECOMMENDED.** Pass the ledger's gloss into
`_published_obs_path(..., obs_title_gloss="")`; both callers already hold `meta`
(`_otr_ledger.py:474-478`, `production_ledger.py:1770`). When non-empty, accept a
third form: `sanitise(gloss) + identity_tail(id)`. **Exactly as strong as today**
— another title at the same second is still refused, another timestamp refused, a
prefix of the id refused, and the three existing parametrised refusals stay green.

## One shared module, because this has now drifted twice

`_obs_title`, `_trim_title`, `_FILENAME_FORBIDDEN`, `_WINDOWS_RESERVED` live in
`otr_master_audio_mux.py:1207-1286`. The validator that must agree with them
lives in `_otr_ledger.py`, **which cannot import the mux.**

PBUG-20260904-06 and PBUG-20260918-09 were both *two modules spelling one rule
differently*. Move the rule to `nodes/_otr_shared/obs_name.py` and add
`validate_gloss`, `identity_tail(episode_id)`, `obs_stem(gloss, episode_id)`.

## Wired in the same change — all four consumers

1. `_otr_writer_tail.py` ~956 — mints the gloss.
2. `otr_master_audio_mux.py::_obs_basename` ~1376 — reads `meta.obs_title_gloss`
   from the ledger it already loads at 1338-1340. Non-empty → `<gloss><identity
   tail>__<codes>_final.mp4`. Empty/absent → **exactly today's name.**
3. `_otr_ledger.py::_published_obs_path` — accepts the third form.
4. **`scripts/otr_oneact_regression.py:115` — BREAKS SILENTLY OTHERWISE.** It
   globs obs with `*<native stem>*_final.mp4`; under a glossed name the native
   stem is not in the obs name, so **every leg reports NOT PUBLISHED.** Must glob
   on the identity tail.

Optional fifth: `_otr_ledger_cleanup._complete_prose:334` fills a missing title
late; call the same helper so a backfilled episode gets a readable name.

## Failure modes, ranked

1. **Mux and validator disagree — a third PBUG-06.** The only SILENT one, hence
   first. Guard: the shared module, plus extending
   `test_the_mux_writes_the_name_the_ledger_accepts`, which today only exercises
   the no-ledger path (`in_flight_ledger_path` monkeypatched to `None` at line 351).
2. **The regression script reports every leg unpublished.** Deterministic, day one.
3. **Model returns native script / a sentence / empty / the title verbatim.**
   Retry, then a visible native-title fallback. Costs a cosmetic name, never an episode.
4. **A plausible but wrong gloss.** Undetectable, not worth detecting — the ledger
   holds title and gloss side by side, so the audit is one grep.
5. **Ledger predates the field.** Absent → `""` → today's name, old binding form.
   No migration.
6. **Replay.** Id grows `_replay_<stamp>_<micro>`; the regex covers it; the gloss
   is deep-copied with `meta`.
7. **Two episodes in one second.** Impossible on one sequential box; across boxes
   the gloss differs and the pod bridge is add-only.
8. **Pod bridge.** Keys on `_final` and the `otr/obs` subfolder
   (`otr_pod_obs_bridge.py:57-59`) — both preserved. A pod on an older pack makes
   old-shape names into the same folder; mixed shapes, harmless.
9. **Hand rename.** Same as today — the pointer drops to the planned path and a
   re-queue republishes under the canonical name. Worth telling the operator that
   a hand rename orphans the ledger pointer.

## The 44 files already in obs — leave them

The standing rule forbids moving or renaming anything in `otr/obs`, and nothing
reads the name back except the bridge (`_final`) and the regression script (fixed
above). Two shapes in one folder; the four ugly names from tonight stay as the
receipt of the bug they document. Sorting by name was never chronological in
either shape. No migration script and no offer of one.

## Where the native title survives — everywhere it is the title

`meta.episode_title` (the one rule, `_otr_shared/episode_title.py`);
`episode_canon.json` `title`; the burnt-in hero card (`video_engine.py:2387`) and
the ASS caption copy; the announcer's mouth via `build_work_frame`
(`_otr_writer_tail.py:1092-1099`); the credits roll (`otr_credits_roll.py:341`);
the story treatment (`video_engine.py:1775`); the archival mp4 and episode
directory — `_slugify` stays frozen; the replay manifest
(`scene_sequencer.py:1500`).

**Cheap addition:** the obs publish already runs ffmpeg
(`otr_master_audio_mux.py:1599-1601`). Add `-metadata title=<native>` and
`-metadata comment=<gloss>` so the file carries the real title in Explorer's
Title column. Verify once on a CJK episode that the argument survives the Windows
subprocess boundary before trusting it.

## Notes

- Blast radius: `_obs_basename` and `_published_obs_path` are shared code; both
  boxes change identically and on purpose. No profile or variant is touched.
- User-visible shipped surface → a `2.1.x` bump when the operator says so (7A).
- Tests: `tests/test_obs_published_filename.py`,
  `tests/test_meta_paths.py::TestPublishedNameBindsTheEpisode`. No widget added,
  so no widget-count check.

---

# RESOLVED — Fable and cursor agree, 2026-09-18

Cursor verified the design against the code (staying on it, unlike its earlier
pass: it was given this FILE to read rather than a prose question, because
cursor-agent anchors on an artifact). It confirmed the load-bearing claims —
the three title branches merging at 843-956, `_published_obs_path:209-213`,
`work_title` unsafe at 880-885, `_obs_title("")` → `"episode"`, and that the
regression-script glob is a real silent break — and set four conditions.
Fable answered. Where they differed, Fable's answer is better and is what gets
built.

**1. The listen page is a second reader — and the fix is to stop reading names.**
`scripts/otr_build_obs_listen_page.py:48-61` parses obs filenames. Its
timestamp regex is anchored at the end of the stem, and names have carried
`__<codes>_final` after the timestamp since 2026-09-03 — so it has been failing
on every file for two weeks, showing raw names with no bank, duration or
voices. Do NOT teach it a third format. Walk `episodes/*/audio/*_ledger.json`
and read `meta.episode_title` (the native title, which is what a listen page
should show), `meta.obs_final_path`, and `meta.obs_title_gloss` for the short
label. Fixes old and new shapes at once and removes the reader that would
break on the next rename.

**2. THREE call sites, not two.** `_published_obs_path` is called from
`_build_meta_paths:284` (no `meta`, no gloss) and directly from
`save_ledger_safe:484` for the alias sync. Thread `obs_title_gloss` into
`_build_meta_paths` from the two save callers that hold `meta`
(`_otr_ledger.py:474`, `production_ledger.py:1770`), and supply it at 484 too —
miss that one and the alias sync silently keeps the planned path.

**3. THE ISO COMES FROM THE LEDGER, NOT FROM A REGEX.** Cursor proposed
narrowing `[a-z]{2}` to the eight admitted rows. That shrinks the hazard
without closing it: a title whose last word is a two-letter English word still
matches. `..._of_it_20260918_190917` parses `it` as the iso and the file reads
as an ITALIAN episode; `..._to_go_...` eats `go`. Both sides would agree on the
same wrong answer, so the binding still holds — the damage is a wrong language
in a name the operator reads. Instead take the iso from `meta.episode_language`
via the same `row_from_meta` that minted the id (`video_engine.py:98-119`).
Expected tail = `("_" + iso unless English) + "_" + timestamp(+replay)`;
confirm the id actually ends with it, else fall back to today's name. The "a
title slug cannot contain the pattern" argument only ever rested on the
`\d{8}_\d{6}` part, which is unchanged.

**4. Option B keeps the native forms; the STRENGTH CLAIM was wrong.**
`forms = [ep, ep_without_prefix, gloss + tail]` — the native forms only ever
match a file leading with this episode's own id, which is exactly the fallback
name, so keeping them costs nothing. But the accepted set is a SUPERSET of
today's, so "exactly as strong" is false. Correct claim: **as strong as today
except a same-second, same-gloss collision**, which the tail-only option could
not distinguish either.

## Corrections to the record

- **PBUG-09 was not "two modules spelling one rule differently."** PBUG-06 was
  (the mux stripped `SHOW_PREFIX`, the validator demanded it; the mux already
  imports it from the ledger at `otr_master_audio_mux.py:67`). PBUG-09 was the
  title going through `_obs_field`'s ASCII strip. Same failure MODE, different
  cause. The shared module is justified by the split THIS change creates —
  gloss sanitised in the mux, bound in the ledger — not by history.
- **`structured_call` needs `base_temperature=0.2` and
  `structural_retry_temperature=0.1`.** Dropped when this doc was condensed from
  the design; a transcription error, not a design one.
- **`_otr_ledger_cleanup.py:334` is the `def`; the fill is 351-374.**
- **The ffmpeg lines 1599-1601 are a docstring**; the publish command is the
  argv list at 1621-1624 via `otr_proc.run(..., encoding="utf-8")`. Adding
  `-metadata` as list argv is the right shape; verify a CJK value survives the
  Windows subprocess boundary before trusting it.
- **An English two-word title passing verbatim as its own gloss is FINE**, not
  a gap — a three-word-or-fewer English title *is* its own gloss.
- **The empty-gloss fallback is the mux's existing branch untouched** —
  `_obs_title(slug + iso + timestamp)`, never `_obs_title(meta.episode_title)`.
  Dropping the tail strips the very thing the binder matches.
