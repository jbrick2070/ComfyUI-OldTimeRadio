# Character Gender Ladder -- build SPEC (public_domain lane)

Date: 2026-08-05. Branch: v2.0-alpha. Class: CORRECTNESS defect (the story-quality
freeze does not apply -- "a character's gender/voice contradicting the source" is the
named carve-out). Prior entry: PBUG-20260805-01 (docs/PROD_BUG_LOG.md:3127-3152).
Sanction for the final tier: docs/GO_FORWARD_PLAN.md:725-730 -- "Operator is open to
a vendor-time LLM/web lookup as the final tier for stragglers, under its own
`gender_source` so it stays auditable."

Every path below is relative to the repo root
`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\` unless it
starts with `output\`, which is ComfyUI's output base. Every file:line was read from
the real Windows files on 2026-08-05.

---

## 1. The defect, measured live tonight

Published ledgers from 2026-08-05, prose lane WRONG, Shakespeare lane RIGHT:

| Episode (bank) | Row | Ledger says | Source says |
|---|---|---|---|
| signal_lost_frozen_chains (public_domain) | EBENEZER SCROOGE | `"gender": "female"` (ledger:43-45) | male |
| same | JACOB MARLEY | `"gender": "other"` (ledger:28-30) | male |
| same | BOB CRATCHIT | `"gender": "male"` (ledger:58-60) | male -- right by luck, same roll |
| signal_lost_wheel_of_wrath (public_domain) | HENRY HARTWICK OGLETHORPE | `"gender": "female"` (ledger:43-45) | male (protagonist of "The Water Ghost of Harrowby Hall", Bangs) |
| signal_lost_stormswept_prophecy (shakespeare) | MACBETH, BANQUO | male, male | correct |
| signal_lost_mercy_in_the_eye_of_the_storm (shakespeare) | PROSPERO, MIRANDA, ARIEL | correct | correct |

Ledgers: `output\otr\episodes\signal_lost_frozen_chains_20260805_195218\audio\
signal_lost_frozen_chains_20260805_195218_ledger.json` and
`output\otr\episodes\signal_lost_wheel_of_wrath_20260805_180143\audio\
signal_lost_wheel_of_wrath_20260805_180143_ledger.json`.

The blast radius is not the voice alone. The wheel_of_wrath ledger's own
`character_description` for Oglethorpe reads "...a prominent scar running down HER
left cheek" (ledger:44) -- the rolled gender reached the description LLM and the
portrait prompt. Per PBUG-20260805-01, `slot.gender` also feeds the outline prompt,
the dialogue cast block (`nodes/_otr_line_composer.py:446`) and the image prompt's
gender anchor (`nodes/otr_meta_brief_image_prompt.py:78-90`) [verify-at-build:
those two line numbers are quoted from the PBUG, not re-read tonight]. Moving
scripts and portraits is the downstream consequence of a correctness fix, not
story-quality work.

### 1a. Why the lanes split -- the whole insight

Both lanes run the IDENTICAL render path. The difference is what is on disk at
vendor time:

**Shakespeare ships a fact.** `scripts/otr_fetch_public_domain.py` (Folger branch)
parses the play's own dramatis personae BEFORE slicing the scene
(otr_fetch_public_domain.py:319-324, using `parse_character_roster` from
`nodes/_otr_character_roster.py:251-349`), scopes it to the scene's actual speakers
(:334-345, refusing to vendor a scene with zero parsed speakers), and writes
`characters` rows -- `{name, roster_name, description, gender, gender_source}` --
into the provenance sidecar (:346-370; sidecar path `<stem>.provenance.json`,
:244-246). Fourteen such sidecars exist under
`config\source_banks\shakespeare\sources\`.

**Prose ships nothing.** The same tool's Gutenberg branch leaves `extra = {}`
(otr_fetch_public_domain.py:307, 386-408) -- no `characters` key. And the bulk
vendor that actually built the 65-unit library,
`scripts/otr_vendor_public_domain_library.py`, writes ONLY the text file
(:909-910) and no sidecar at all -- despite its own docstring claiming it writes
"the unit body + a provenance sidecar ... via the shipped fetcher" (:16-18). That
docstring/behavior drift gets fixed by this build. Measured on disk: exactly ONE
public-domain sidecar exists (`config\source_banks\public_domain_story\sources\
time_machine__arrival.provenance.json`), and it carries no `characters` key
(lines 1-14).

**So the render path reads a fact on one lane and nothing on the other.** The
already-shipped chain, lane-neutral end to end:

1. Fetch loads the sidecar roster when present:
   `nodes/_otr_public_domain_sources.py:564-567` (`source_meta_from_unit`, called
   with `text_path` at :518); mirrored for Shakespeare at
   `nodes/_otr_shakespeare_sources.py:453-456`.
2. `source_meta` is copied into durable meta:
   `nodes/OTR_LedgerScriptWriter.py:3584` via
   `nodes/_otr_source_payload.py:195-219` (`_copy_sidecar` -- plain dicts only).
3. The adaptation-cast gate (`propagate_adaptation_cast: true`,
   `nodes/story_packs/banks.json:112` for public_domain, :150 for shakespeare)
   surfaces the source's names -- the briefs' LLM-extracted `character_names`
   first, manifest `cast_hints` as fallback
   (`OTR_LedgerScriptWriter.py:3834-3842`) -- then joins them to the roster:
   `_otr_roster_gender.gender_map_for_names` (`OTR_LedgerScriptWriter.py:
   3847-3859`), stamping `meta["_adaptation_character_genders"]`.
4. `lock_cast(source_character_genders=...)` (`OTR_LedgerScriptWriter.py:
   4141-4142`) derives the pin map (`nodes/_otr_casting.py:1635-1649`) and
   `precompute_ensemble_slots` overwrites the drawn gender at pinned,
   source-owned slots only (`_otr_casting.py:756-768`); only `male|female` may
   pin (`_PINNABLE_GENDERS`, :146).
5. The receipt lands in ledger meta as `cast_source_contract`
   (`_otr_casting.py:1855-1863`, copied key-by-key at
   `OTR_LedgerScriptWriter.py:4176-4179`).

With an empty roster, every join verdict is `no_roster`
(`nodes/_otr_roster_gender.py:170-171`), `gender_map_for_names` omits everything
(:310-337), the pin map is empty, and the 40/40/20 largest-remainder roll stands
(`_otr_casting.py:565-626`, weights :153-157). Scrooge = female and
Marley = "other" are exactly that roll. **The root fix is therefore at VENDOR
time: put the fact on disk for all 65 prose units. The render path already knows
how to read it.**

---

## 2. Operator rulings that bound this spec

1. **Shakespeare is NOT in scope.** Its roster works (proven tonight). Do not add
   a model or a lookup to a lane that already reads a parsed fact -- that would
   replace knowledge with a guess.
2. **Network is ALLOWED.** This repo is not offline-only: `scifi_news` /
   `scifi_news_pro` fetch `science_rss`, `media_archive` fetches
   `media_archive_rss`, and `nodes/_otr_feed_fetch.py` makes live HTTP at render
   time. A vendor-time lookup may hit the network.
3. **The ANNOUNCER's gender is deliberately random male/female.** By design, not
   a defect. This ladder never touches an announcer row (see 5.3 exclusion and
   Out of Scope).
4. **Invented lanes keep rolling.** See Out of Scope, section 10 -- stated there
   in full so a later reader cannot extend the ladder to them.
5. **Ask about the CHARACTER IN THE WORK, never the gender of a bare NAME.**
   "Li Wei" is unmarked in romanization, "Evelyn" was historically male, "Ariel"
   reads feminine as a name while being a spirit. The manifest requires non-empty
   `title` and `author` (`_otr_public_domain_sources.py:181-182`), so the strong
   question is always available.
6. **OPERATOR CORRECTION (2026-08-05, supersedes the earlier "unknown recorded
   honestly" tier 4): the ladder is TOTAL.** "Even if unknown we need to pick a
   voice ... it needs to do its best -- I realize it will not be perfect."
   Gender is consumed at cast time to pick a voice; a row resolving to `unknown`
   still gets spoken by somebody, so abstention just moves the decision to the
   blind roll that produced Scrooge = female. Therefore the DECISION is separated
   from the CONFIDENCE:
   - `gender` -- ALWAYS a usable value, `male` or `female`. Never `unknown`,
     never empty, never `other` (the pin vocabulary is `_PINNABLE_GENDERS`,
     `_otr_casting.py:146`; "other" remains an invention-lane ensemble label,
     :148-157, and is not something a source records about a named person).
   - `gender_source` -- which tier decided: `roster | pronouns | llm_web |
     name_frequency`.
   - `gender_confidence` -- `known` for tiers 1-2 (the source's own cast list or
     the author's own pronouns) vs `inferred` for tiers 3-4 (a lookup or a
     statistic). The ledger must distinguish "the source told us" from "a model
     decided". Tier 4 additionally stores its PERCENTAGE (ruling 7), which is
     that tier's fine-grained confidence -- no extra hand-waved label.
7. **OPERATOR REFINEMENT (2026-08-05, same session): tier 4 is NAME FREQUENCY.**
   "Ideally it searches the web and/or says what percentage of name YYY is male
   or female and we choose the highest percentage." The terminal tier does not
   make a free-form judgment call; it reports the male/female percentage for the
   character's GIVEN NAME and takes the higher side. A percentage is a
   confidence score for free: "Henry 97% male" carries strictly more than
   "male", it always returns a number, and therefore it always decides -- which
   is exactly the totality ruling 6 requires. Tier 3 (character-in-work) stays
   ABOVE it and is never skipped for a real published character, because for a
   real character gender is a FACT and the name statistic is only a correlation
   -- Scrooge and Marley resolve at tier 3, not tier 4.

   Two consequences, stated because this repo has shipped armed consumers with no
   producer before:
   - **The ladder always terminates in a usable value.** No downstream consumer
     needs an `unknown` branch for public-domain sidecar rows; writing one would
     be dead code from day one. (`_otr_character_roster`'s `unknown` vocabulary
     stays as-is for the Shakespeare lane, which is out of scope.)
   - **The error rate is accepted up front.** This spec does not chase
     perfection; it makes every call TRACEABLE so a wrong one is attributable to
     its tier and fixable at that tier. An auditable wrong answer is worth more
     here than an anonymous right one.

---

## 3. Design at a glance

```
VENDOR TIME (authoring tool, fail-loud, network allowed, runs once per unit)
  manifest unit (title, author, cast_hints)  +  vendored unit text
        |
        v
  candidate names = cast_hints  U  LLM name extraction from the unit text
        |                             (announcer forms excluded)
        v
  per name, first tier that DECIDES wins; every row stamped with its tier:
    T1 roster          parse_character_roster over the unit text     -> known
    T2 pronouns        deterministic pronoun/title scan, floor+ratio -> known
    T3 llm_web         character-in-work web question, cached        -> inferred
    T4 name_frequency  given-name percentage, higher side wins,
                       share stored, ALWAYS returns                  -> inferred
        |
        v
  <unit>.provenance.json  gains  "characters": [rows]  +  "gender_ladder": {...}

RENDER TIME (unchanged mechanism, degrade-honestly)
  fetch -> source_meta.characters -> gender_map_for_names join -> lock_cast pin
  -> ledger cast row + cast_source_contract receipt (now also carrying
     gender_source + gender_confidence per pinned name)
```

The answer is STATIC -- Scrooge's gender will not change before the next render --
so it is computed once at vendor time and cached in the sidecar, exactly as the
GO_FORWARD_PLAN sanctions. The render path stays local and offline for this
feature; no LLM or network call is added to any render.

---

## 4. Data model

### 4.1 Where the fact lives: the provenance sidecar, NOT the manifest

`_SOURCE_KEYS` / `_UNIT_KEYS` are CLOSED frozensets
(`nodes/_otr_public_domain_sources.py:53-68`); unknown keys are rejected loudly
(:117-123, applied at :160 and :178) and `schema_version` must equal `"v1"`
(:41, :234-237). Adding a manifest field therefore costs: a schema version bump,
a migration of `config\source_banks\public_domain_story\manifest.sample.json`
(65 units across ~60 sources -- `nodes/story_packs/banks.json:108` is the live
pointer), an emitter update in `write_manifest`
(`scripts/otr_vendor_public_domain_library.py:763-856`, whose key set must match
`_SOURCE_KEYS` EXACTLY per its own comment :766-770), and validator changes --
all to duplicate a channel that already exists.

The sidecar already carries `characters` end to end with ZERO schema change: the
Shakespeare lane proves the channel (section 1a chain), and the reader is
tolerant of extra row keys -- `load_roster_characters` passes rows through as
plain dicts (`nodes/_otr_roster_gender.py:85-104`). **Decision: the sidecar
carries it. No manifest schema change, no migration.**

### 4.2 The sidecar `characters` row (public_domain vocabulary)

```json
{
  "name": "Ebenezer Scrooge",
  "aliases": ["Scrooge"],
  "description": "",
  "gender": "male",
  "gender_source": "pronouns",
  "gender_confidence": "known",
  "evidence": "pronoun scan: he/him/his 41 vs she/her 2 across 17 mention windows"
}
```

- `name` -- the canonical form, matching how the manifest/cast_hints and the
  source text refer to the character. UPPER-cased on join, stored human-cased.
- `aliases` -- optional, additive. For a multi-token name the stamper writes the
  surname (and the leading-article-stripped form for role designations, e.g.
  "water ghost" for "the water ghost"). Needed because the existing join tiers
  (`resolve_roster_gender`, `_otr_roster_gender.py:151-201`) cannot join a
  render-extracted "SCROOGE" to a row named "EBENEZER SCROOGE": exact fails,
  honorific-strip does not remove EBENEZER, `qualified` needs the roster name to
  START with "SCROOGE ", `contains` needs the reverse. Shakespeare never hits
  this because both its `name` and `roster_name` come from the same
  speech-prefix vocabulary (otr_fetch_public_domain.py:346-362).
- `description` -- the roster description when T1 fired, else "".
- `gender` -- `male | female`. ALWAYS. (Ruling 6.)
- `gender_source` -- `roster | pronouns | llm_web | name_frequency`. Exactly one
  of these four on every public-domain row. (Shakespeare sidecars keep their
  existing finer vocabulary -- relation/title/group/back_reference/unknown --
  untouched; out of scope.)
- `gender_confidence` -- `known` (gender_source roster|pronouns) or `inferred`
  (llm_web|name_frequency). Redundant with gender_source by construction,
  stamped anyway so a ledger reader never needs the mapping table.
- `name_frequency` -- present ONLY on tier-4 rows (ruling 7): the winning side's
  share and where the statistic came from. This number IS tier 4's fine-grained
  confidence -- a stored 0.52 tells a later reader the call was a coin flip that
  was still decided, which is the whole value of a percentage over a label:

  ```json
  "name_frequency": {"given_name": "Henry", "share": 0.97,
                     "population": "England and Wales births, 19th century"}
  ```

- `evidence` -- non-empty quotable reason. For tiers 3-4: the model's one-line
  stated basis plus the resolved model id; tier 4's repeats the share, e.g.
  "name frequency: Henry 97% male (England and Wales, 19th c.); via <model-id>".

### 4.3 The sidecar-level ladder stamp

One additional top-level sidecar key, written by the stamper beside `characters`:

```json
"gender_ladder": {
  "version": "gender_ladder_v1",
  "ran_utc": "2026-08-05T23:41:00Z",
  "model_id": "openrouter/<resolved-slug-or-empty>",
  "prompt_version": "character_gender_v1",
  "tier_counts": {"roster": 0, "pronouns": 4, "llm_web": 1, "name_frequency": 0}
}
```

Auditability without schema cost: `load_roster_characters` reads only the
`characters` key, so extra top-level keys are inert to the render path.

### 4.4 Ownership (one owner per field, house rule)

- `characters[]` and `gender_ladder` in PUBLIC-DOMAIN sidecars: owned by the new
  stamper tool (section 6) -- exactly one writer.
- The base provenance fields (`schema_version`, `slug`, `unit`, hashes, ...)
  remain owned by `write_source` (otr_fetch_public_domain.py:222-277); the
  stamper MERGES and never rewrites them (section 6.2).
- The render path reads and never writes -- the boundary already documented in
  `nodes/_otr_roster_gender.py:19-25` stands.

---

## 5. The ladder -- exact algorithm and failure behaviour per tier

Runs at vendor time, per unit, over the candidate name set (5.0). First tier
that DECIDES a name wins and stamps `gender_source`. Later tiers never
re-litigate an earlier tier's answer.

### 5.0 Candidate names for a unit

Union of:
- the source's manifest `cast_hints` (required non-empty by schema,
  `_otr_public_domain_sources.py:200`; authored in the vendor table, e.g.
  `["Ebenezer Scrooge", "Jacob Marley", "Fred", "Bob Cratchit"]`,
  `scripts/otr_vendor_public_domain_library.py:196`), and
- an LLM name-extraction pass over the FULL unit text, same instruction shape as
  the render-time `character_names` extraction ("the story's OWN cast, taken
  straight from the source text ... a proper name when the source gives one ...
  OR the source's own role-designation", `_otr_public_domain_sources.py:676-680`,
  cleaned/deduped/capped like :592-609). This is what covers names the briefs
  will surface at render that the hints do not -- measured tonight: the briefs
  produced "HENRY HARTWICK OGLETHORPE" where cast_hints say "the heir of
  Harrowby" (vendor table :526).

Exclusion: any candidate whose normalized form is `ANNOUNCER` or
`THE ANNOUNCER` is dropped before the ladder runs (ruling 3; the announcer's
gender is random by design, and `cradle_protocol`'s cast_hints really do carry
"the Announcer", vendor table :79).

Plural/collective candidates ("the crew", "the twins", "armed callers") are NOT
special-cased: if such a name is cast, it is spoken by one voice, so it gets one
row like anything else -- decided by tiers 1-3 when they can, and otherwise by
tier 4's designation-population contract (5.4). Accepted-error territory
(ruling 6).

### 5.1 Tier 1 -- ROSTER (`gender_source: "roster"`, confidence `known`)

Run `parse_character_roster` (`nodes/_otr_character_roster.py:251-349`) over the
unit text. Prose without a cast block returns `()` (:256-265) -- normal absence,
fall through. If a block exists and `infer_gender` (:228-248) yields
`male|female` for a candidate (matched via `CharacterRecord.matches`, :149-155),
the row is decided: `gender_source="roster"`, evidence = the roster description
(the finer relation/title/back_reference sub-source goes into `evidence`, not
into `gender_source` -- the four-value vocabulary of 4.2 holds). Expected to
fire on approximately zero of the 65 prose units; kept because it is free,
deterministic, and makes the ladder uniform with the lane that works.

### 5.2 Tier 2 -- PRONOUNS IN THE SOURCE TEXT (`"pronouns"`, `known`)

Free, offline, deterministic, pure (same text -> same verdict).

- **Mention forms:** the candidate's full name; for multi-token names also the
  final token (surname) and the form minus a leading article ("the water ghost"
  -> "water ghost"). Case-insensitive, word-boundary regex. A mention form that
  is also a mention form of ANOTHER candidate in this unit (e.g. two Whites
  sharing "White", `monkeys_paw`) is counted for NEITHER -- shared surnames must
  not cross-contaminate.
- **Window:** the 240 characters following each mention's end, clipped at
  text end.
- **Signals, per window:**
  - pronouns: male `he, him, his, himself`; female `she, her, hers, herself`;
    weight 1 each;
  - a gendered TITLE or RELATION word directly prefixing the mention itself
    ("Mrs. Sappleton", "Aunt Em", "Sergeant-Major Morris"): weight 3. Reuse the
    word lists at `nodes/_otr_character_roster.py:78-105` verbatim -- they are
    period-tuned for exactly this corpus's era;
  - self-designation, counted once: a gendered title/relation word INSIDE the
    candidate name itself ("the Governess", "Mrs. White", "the shepherdess"),
    weight 3, same word lists. This is what lets the author's own naming of a
    role decide it deterministically before any model is consulted.
- **Decision:** sum scores over all windows. Decide iff
  `winner >= 4` AND `winner >= 3 * loser`. Otherwise DECLINE to tier 3 -- below
  the floor this tier never guesses. (4 and 3.0 are the starting calibration;
  the corpus report in 6.3 is the tuning harness. The floor SEMANTICS --
  decline, never guess -- are fixed and not tunable.)
- **Known limitation, accepted:** windows are not speaker-attributed, so
  dialogue cross-talk ("he said to her") adds noise; the dominance ratio is the
  absorber. Dickens-grade cases ("Scrooge ... he ... his ... he") clear the bar
  by an order of magnitude.

Evidence string: `"pronoun scan: he/him/his 41 vs she/her 2 across 17 mention
windows"`.

### 5.3 Tier 3 -- LLM + WEB SEARCH, character-in-work (`"llm_web"`, `inferred`)

For the stragglers tier 2 declines. Vendor-time only, cached into the sidecar --
the answer is static.

- **Eligibility:** only sources with `adapter_type` `project_gutenberg_text` or
  `standard_ebooks_epub` (`_ADAPTER_TYPES`, `_otr_public_domain_sources.py:
  47-51`) -- published works a web search can actually find.
  `local_text_fixture` sources (today: `cradle_protocol`, written for this show)
  SKIP straight to tier 4: the work is not on the web, so a character-in-work
  search cannot find a fact -- it can only surface real strangers who share the
  invented names, the same hazard that keeps the invented lanes out of scope
  entirely. (Tier 4's bare-name frequency question does not carry this hazard;
  see 5.4.) For a real published character this tier is never skipped: gender
  there is a FACT, and the name statistic below is only a correlation.
- **Transport:** the repo's OpenRouter backend
  (`nodes/_otr_openrouter_backend.py` -- env-key-only, per-call cost ceiling,
  bounded retry ladder, `_post_chat_completion` as the mockable seam, :9-26)
  with a web-search-enabled route (OpenRouter `:online` slug variant or the
  `web` plugin). [verify-at-build: whether the backend surfaces a web-search
  flag today; if not, the stamper -- an authoring-time script, not the render
  path -- may POST the OpenRouter chat endpoint directly using the same
  `OPENROUTER_API_KEY` env var and the same cost/retry discipline.]
- **Question form, FIXED (ruling 5 -- the character in the work, never the bare
  name):**

  ```
  In "{title}" by {author}, is the character "{name}" male or female?
  Answer ONE JSON object only:
  {"gender": "male" | "female",
   "evidence": "one sentence naming the textual or scholarly basis"}
  ```

  `{title}`/`{author}` from the manifest source row (required non-empty,
  `_otr_public_domain_sources.py:181-182`).
- **Accept iff** the JSON parses, `gender` is exactly `male` or `female`, and
  `evidence` is non-empty. Stamp `gender_source="llm_web"`, evidence = the
  model's sentence + the resolved model id.
- **On failure or ambiguity** (transport exhausted after the bounded retries,
  refusal, malformed JSON after one structural repair attempt, or any answer
  outside the vocabulary): fall THROUGH to tier 4. Never abstain, never write
  `unknown`, never leave the row absent.

### 5.4 Tier 4 -- NAME FREQUENCY, total and terminal (`"name_frequency"`, `inferred`)

The tier that makes the ladder total, in the operator's own mechanism
(ruling 7): report the percentage male vs female for the character's GIVEN
NAME, take the higher side, store the number.

- **Given names only.** Before the lookup, reduce the candidate to its given
  name: strip honorifics (reuse the `_HONORIFICS` sets,
  `_otr_character_roster.py:127-133` / `_otr_roster_gender.py:41-45`), strip a
  leading article, drop surname tokens -- "HENRY HARTWICK OGLETHORPE" ->
  "Henry", "Mrs. Sappleton" -> (no given name; see below), "Anne Shirley" ->
  "Anne". The full string would not appear in any name statistic, so it is
  never sent.
- **The question, via the same transport as tier 3** (web search allowed AND
  useful here -- frequency tables are exactly what search finds; the operator's
  wording is "searches the web and/or says what percentage"). Web use is
  permitted for ALL adapter types at this tier, including `local_text_fixture`:
  the question names only a bare given name and a population, never the work
  and never a person, so the living-person hazard of ruling 4 does not arise.

  ```
  For the given name "{given_name}", in {population_hint}, what share of
  people bearing it are male and what share female? Prefer a statistic
  matching the period and place; name the population it describes.
  Answer ONE JSON object only:
  {"share_male": 0.0-1.0, "population": "which population/era the number
   describes", "evidence": "one sentence naming the basis"}
  ```

  `{population_hint}` is built from the manifest row: the source `year`
  (`_otr_public_domain_sources.py:183` -- may be empty) and the author's
  country when the stamper's per-work table knows it, else "the work's period
  and setting". **Name statistics are culture- and era-specific** -- Andrea is
  predominantly male in Italy and female in the US; Evelyn flipped over the
  20th century -- so a matching population is PREFERRED and the population
  actually used is RECORDED. Modern US frequencies must never be silently
  applied to a Jacobean or Victorian cast; the stored `population` field is
  what makes that auditable.
- **Decision:** `gender` = the higher side of `share_male` (>= 0.5 -> male,
  else female). A near-50/50 split is still a decision -- take the higher side
  and ship it; the stored number tells a later reader it was a coin flip
  rather than a certainty. Stamp `gender_source="name_frequency"`,
  `gender_confidence="inferred"`, and the `name_frequency` object of 4.2 with
  `share` = the winning side's share (in [0.5, 1.0]).
- **Candidates with NO extractable given name** (collectives and bare role
  designations that survived tiers 1-3: "the crew", "armed callers", "the
  voice behind the door"): the same call is made over the DESIGNATION instead
  of a given name -- "for people referred to as {designation} in
  {population_hint}, what share are male/female" -- and the answer is handled
  identically, with `name_frequency.given_name` set to the designation. Still
  a percentage over a named population, still always a number, so the ladder
  stays total without a fifth tier.
- **Cached like every other tier:** the frequency for a given name does not
  change between renders; the row is written once at vendor time and re-read
  forever.
- **Structural exhaustion** (transport dead after the bounded retries,
  malformed JSON after one repair attempt): that is a TOOL failure, not a
  ladder answer -- the stamper reports the unit and character by name, exits
  non-zero, and leaves that unit's sidecar unwritten (fail-loud in
  authoring-time tools, house rule). `unknown` never enters the data; a
  sidecar either carries a complete total roster or does not exist yet.

### 5.5 Totality, restated as the contract

For every public-domain sidecar that exists after a successful stamper run:
every `characters` row has `gender` in `{male, female}`, `gender_source` in
`{roster, pronouns, llm_web, name_frequency}`, `gender_confidence` in
`{known, inferred}` consistent with the source, non-empty `evidence`, and --
on `name_frequency` rows only -- a `name_frequency` object whose `share` is in
[0.5, 1.0] with a non-empty `population`. No consumer of these rows may branch
on `unknown` -- there is nothing to catch. A wrong call is found by its
`gender_source` and fixed AT THAT TIER (re-run the stamper after tuning tier 2,
after a better tier-3 answer, or with a better-matched tier-4 population; the
merge rule in 6.2 protects `known` rows from `inferred` overwrites).

---

## 6. Vendor-time flow

### 6.1 The stamper tool: `scripts/otr_stamp_character_genders.py` (new)

Authoring-time script, same import discipline as the existing fetcher (direct
`nodes/` path insert, no ComfyUI import -- otr_fetch_public_domain.py:49-56).

```
python scripts/otr_stamp_character_genders.py            # all 65 units
python scripts/otr_stamp_character_genders.py --only christmas_carol_marley
python scripts/otr_stamp_character_genders.py --offline  # tiers 1-2 only; lists
                                                         # (does not stamp) the
                                                         # names needing 3-4
python scripts/otr_stamp_character_genders.py --report   # read-only tier census
```

Per unit: load the REAL manifest via `load_public_domain_manifest`
(`_otr_public_domain_sources.py:254-263`, path from
`nodes/story_packs/banks.json:108`), read the unit text, build candidates (5.0),
run the ladder (5.1-5.4), then MERGE into `<text stem>.provenance.json` beside
the text (`sidecar_path_for_text` shape, `_otr_roster_gender.py:74-83`) using
`atomic_write_json` (`_otr_public_domain_sources.py:770-787`).

Why a separate script rather than inside the library vendor's main loop: the
library vendor is a resumable FETCH tool, and the ladder must be re-runnable
WITHOUT refetching (tier-2 tuning, a model upgrade, a corrected tier-3 answer).
The library vendor gains exactly one call -- after writing a unit's text
(otr_vendor_public_domain_library.py:909-910) it invokes the stamper for that
unit -- which also makes its sidecar docstring claim (:16-18) true again.
[verify-at-build: hook point and whether it imports the stamper or shells out;
import preferred.]

### 6.2 Merge rules (the sidecar is shared property)

- 64 of 65 units have NO sidecar today: the stamper creates one carrying only
  what it can honestly state -- `schema_version: "otr_source_provenance_v1"`,
  `slug`, `unit`, `text_path`, `body_sha256`/`body_bytes`/`body_words` computed
  from the on-disk text (LF-normalized exactly as write_source does,
  otr_fetch_public_domain.py:248-269), plus `characters` + `gender_ladder`.
  Fields it cannot know (`fetched_utc`, `source_url` of the original crawl) are
  taken from the manifest source row where available (`source_url`,
  `work_title`=title, `author`) and otherwise omitted -- never invented.
- Where a sidecar EXISTS (time_machine__arrival today, all units after first
  run): the stamper replaces only `characters` and `gender_ladder`, preserving
  every other field byte-for-byte.
- Idempotent and monotonic in confidence: on re-run, an existing row whose
  `gender_confidence` is `known` is NEVER overwritten by an `inferred` result
  for the same name -- the mirror of the supplement merge rule ("a curated
  convenience file must never be able to overrule the source itself",
  `_otr_roster_gender.py:286-299`). A `known` row may be replaced only by
  another `known` row (fresh tier-1/2 evidence from the text).

### 6.3 The stamper's report

Every run prints a per-unit line (`OK <unit> roster:N pronouns:N llm_web:N
name_frequency:N`) and a corpus footer with tier totals -- the same
loud-receipt style as the vendor library (:917-919). `--report` emits it
without writing. This census is the tuning harness for the tier-2 floor and
the review surface for `inferred` rows; `name_frequency` rows print their
share, so the coin flips (shares near 0.5) are visible at a glance.

### 6.4 The curated supplement stays a Shakespeare device

`roster_gender_supplement.json` (`_otr_roster_gender.py:49, 229-273`) is keyed
by `play_code`, which public-domain source_meta does not carry
(`source_meta_from_unit`, `_otr_public_domain_sources.py:540-568` -- no
play_code; the writer passes `""` at `OTR_LedgerScriptWriter.py:3853`). With a
TOTAL vendor-time ladder the supplement is unnecessary on this lane: a curated
correction is made by re-running the stamper with the correction (or hand-editing
the sidecar row -- it carries its own evidence), not by adding a second lookup
file. No supplement for public_domain is introduced.

---

## 7. Render-time read

**The defect fix requires NO render-path change.** Once the sidecars exist, the
shipped chain (section 1a) pins public-domain casts exactly as it pinned
Shakespeare tonight. Explicitly: no node signature change, no `INPUT_TYPES`
change, no widget change, no `workflows/otr_canonical.json` change. If any of
those becomes necessary during build, STOP and say so up front -- none is
expected.

Two ADDITIVE code changes ride along for traceability and join coverage, both in
existing files, both inert to the Shakespeare lane:

1. **Carry the ladder provenance into the ledger receipt.**
   `_verdict_from` / `gender_map_for_names` (`nodes/_otr_roster_gender.py:
   127-148, 310-337`) currently emit `{gender, evidence, tier, roster_name}`,
   where `tier` is the JOIN tier (exact/short_form/qualified/contains). Add,
   from the matched sidecar row: `gender_source` (the LADDER tier) and
   `gender_confidence` (defaulting to `known` for a binary-gendered row that
   lacks the field -- which is every Shakespeare row). The receipt
   (`cast_source_contract.evidence`, `_otr_casting.py:1862`) then answers both
   questions a wrong published row raises: HOW the name matched, and WHO decided
   the gender. Free-form meta keys -- ledger schema unchanged.
2. **Alias-aware join.** `_candidate_names` (`_otr_roster_gender.py:117-124`)
   additionally reads an optional row `aliases` list (4.2). Absent key =
   byte-identical behavior; Shakespeare sidecars have no such key. This is what
   lets a render-extracted surname ("SCROOGE") join its full-name row.

**The residual roll, stated honestly:** a render-time name that joins NO sidecar
row still falls to today's 40/40/20 roll -- a usable gender, per the existing
design ("unresolved names are omitted, never recorded as 'unknown'",
`_otr_roster_gender.py:317-321`), visible in the ledger by its absence from
`cast_source_contract.gender_by_name`. This is the accepted-error path, bounded
by 5.0's name-extraction union and measured by acceptance test T3. It is NOT an
`unknown` branch -- nothing downstream distinguishes it from an invention-lane
row.

**Render never dies for this feature:** the only new render-path exception
surface remains the existing supplement-contradiction guard, which already
degrades to the roll with a logged receipt (`OTR_LedgerScriptWriter.py:
3860-3868`). Missing sidecar, malformed sidecar, empty roster: all already
degrade via `load_roster_characters`'s tolerant `()` (:91-104).

---

## 8. Failure behaviour summary

| Site | Failure | Behaviour |
|---|---|---|
| Stamper (authoring) | tier-4 structural exhaustion, unreadable unit text, manifest validation error | FAIL LOUD: name unit+character, exit non-zero, sidecar unwritten/unchanged |
| Stamper (authoring) | tier-3 transport/refusal/ambiguity | not a failure: fall through to tier 4 |
| Stamper `--offline` | names undecided by tiers 1-2 | listed by name, sidecar NOT written for that unit (a partial roster would read as total) |
| Render | sidecar missing/malformed/empty | degrade to today's roll; pin map empty; receipt shows no pins (`no_roster`, `_otr_roster_gender.py:170-171`) |
| Render | name joins no row / ambiguous join | that slot degrades to the roll; others still pin (:135, :317-321) |
| Render | supplement contradiction (shakespeare only) | existing catch + logged fallback (`OTR_LedgerScriptWriter.py:3860-3868`) |

---

## 9. Acceptance tests

Model: the Shakespeare corpus gate at `tests/test_roster_gender.py:134-150` and
the named-defect test at :153-169. New tests live in
`tests/test_character_gender_ladder.py` unless noted. All offline -- LLM tiers
are proven through the mockable transport seam
(`_otr_openrouter_backend._post_chat_completion`, backend docstring :21-23);
the conftest already forces `OTR_TEST_MODE` / no CUDA.

- **T1 corpus-wide gate (the 65-unit one).** Iterate every unit of the REAL
  manifest (`load_public_domain_manifest` on the banks.json:108 path -- 65
  `unit_id`s, matching the 65 `.txt` files on disk). Assert per unit: the
  sidecar exists; `characters` is non-empty; and EVERY row satisfies section
  5.5's totality contract (gender/male-female, four-value gender_source,
  consistent confidence, non-empty evidence) and no row's normalized name is
  ANNOUNCER. This test is the producer-side proof that no consumer ever needs
  an `unknown` branch.
- **T2 the named production defects resolve.** Via `gender_map_for_names`
  against the real sidecars: christmas_carol_marley -- EBENEZER SCROOGE male,
  JACOB MARLEY male, BOB CRATCHIT male, FRED male; water_ghost -- HENRY
  HARTWICK OGLETHORPE male (alias join from the sidecar roster), "the water
  ghost" female. Mirrors :153-169.
- **T3 join coverage.** For every unit, every manifest `cast_hint` reaches
  `is_pinned` through `resolve_roster_gender` against that unit's sidecar --
  the hint fallback path (`OTR_LedgerScriptWriter.py:3838`) can then always
  pin. Additionally: for multi-token roster names, the surname alias joins
  (exercises change 7.2).
- **T4 ladder mechanics (unit tests).** Tier-2 floor declines below
  `winner >= 4` and below the 3:1 ratio; tier-2 decides on a synthetic
  Dickens-shaped fixture; the self-designation signal genders "the Governess"
  without a model; shared-surname mentions count for neither candidate;
  tier-3 malformed/refused answers fall through to tier 4 (mocked transport);
  `local_text_fixture` sources never issue a tier-3 call (mock asserts zero
  character-in-work invocations for cradle_protocol). Tier 4: given-name
  reduction ("HENRY HARTWICK OGLETHORPE" -> "Henry", honorifics and articles
  stripped); the higher side wins and `share` is stored in [0.5, 1.0] with a
  non-empty `population`; a mocked 0.52/0.48 answer still DECIDES and stamps
  0.52; the no-given-name designation path returns a row of the same shape;
  nothing outside `male|female` is ever emitted.
- **T5 stamper merge discipline.** Creating a sidecar where none exists writes
  only honest fields; re-stamping time_machine__arrival preserves its existing
  provenance byte-for-byte outside `characters`/`gender_ladder`; a `known` row
  survives a re-run whose mock LLM disagrees (`inferred` never overwrites
  `known`); the same inputs re-stamp to identical output (idempotence).
- **T6 receipt provenance.** `gender_map_for_names` output for a PD row carries
  `gender_source` + `gender_confidence`; a Shakespeare-shaped row without the
  new fields derives confidence `known`; existing shakespeare assertions
  (`tests/test_roster_gender.py`, all of it) still pass UNMODIFIED -- the
  out-of-scope lane is provably untouched.
- **T7 live leg (admission bar for the PROD log).** One public_domain render
  through the REAL `workflows/otr_canonical.json` (rule 0) with
  `defaults.source_ref` pinned to `christmas_carol_marley:main` for the leg.
  Assert from the produced ledger: the SCROOGE/MARLEY/CRATCHIT cast rows carry
  the sidecar genders; `cast_source_contract.gender_by_name` is non-empty and
  each entry's evidence names its gender_source. RESULT SUCCESS + the asset at
  its canonical `otr\episodes\<ep>\` path, per house rules. A green unit suite
  does not close this item; the live leg does.
- **T8 regression.** Full suite + Bug Bible run
  (`C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide`,
  relative `tests\bug_bible_regression.py`), AST parse of touched files, widget
  count vs `widgets_values` audit (expected delta: none -- section 7).

---

## 10. OUT OF SCOPE -- read this before extending anything

- **Shakespeare.** Its roster is parsed from the play's own dramatis personae at
  vendor time and read as a fact at render (proven correct in tonight's
  ledgers). Do not add a model or a lookup to that lane -- it would replace
  knowledge with a guess. Its sidecar vocabulary
  (relation/title/group/back_reference/unknown) and its curated supplement stay
  exactly as they are.
- **The ANNOUNCER.** Deliberately random male/female, by design, on every lane.
  Not a defect; do not "fix" it. The stamper drops announcer-shaped candidate
  names (5.0) so this ladder can never pin one.
- **The invented lanes: `original`, `scifi_news`, `scifi_news_pro`,
  `media_archive`.** Their characters are MADE UP by the writer at run time.
  There is no fact to look up, and a name search there risks matching a real
  living person. Gender there is a free choice and ROLLING IS CORRECT. Do not
  extend this ladder, the stamper, or any web lookup to them -- this sentence
  exists so a later reader does not. (Mechanically they are already excluded:
  no `propagate_adaptation_cast`, no manifest, `supplement_dir_for_bank`
  returns None -- `_otr_roster_gender.py:207-222`.)
- **Story quality.** Closed by operator directive 2026-08-04. This spec fixes a
  correctness defect; no prose, prompt-craft, or model-quality work rides on it.
- **Word count.** Untouched.

---

## 11. Verify-at-build ledger

Claims marked for verification during the build, not asserted here:

1. `nodes/_otr_line_composer.py:446` and `nodes/otr_meta_brief_image_prompt.py:
   78-90` as gender consumers -- quoted from PBUG-20260805-01; re-read before
   citing in code comments.
2. Whether `_otr_openrouter_backend` exposes a web-search/route flag usable by
   tiers 3-4, or the stamper calls the OpenRouter endpoint directly (5.3, 5.4).
3. The exact hook point in `otr_vendor_public_domain_library.py` main loop
   (:909-911) for invoking the stamper after a fresh vendor, and fixing the
   :16-18 docstring.
4. Tier-2 floor numbers (4 / 3:1) against the stamper's corpus report; the
   decline-never-guess semantics are fixed regardless.
5. Whether `beleaguered_city` (vendor table :542-547, no `.txt` on disk) is
   absent from the manifest as expected -- reconcile 65 units / 65 texts during
   T1.
6. The exact additive diff in `_verdict_from`/`gender_map_for_names` for
   carrying `gender_source`/`gender_confidence` (7.1) without disturbing any
   existing test in `tests/test_roster_gender.py`.

## 12. Explicit no-change assertions

- No `INPUT_TYPES`, widget, `widgets_values`, or node-signature change.
- No `workflows/otr_canonical.json` change.
- No manifest schema change; `MANIFEST_SCHEMA_VERSION` stays `"v1"`.
- No new render-time LLM call and no render-time network use.
- No change to `_plan_gender_distribution`, the pin-overwrite mechanism, or any
  invention-lane behaviour (byte-identical when `gender_by_name` is None --
  `_otr_casting.py:745`).
