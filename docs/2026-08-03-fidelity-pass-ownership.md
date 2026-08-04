# Pass-to-field ownership for the verbatim play lane

**This table is the gate.** Nothing in the fidelity executor codes until every
field a bypassed pass used to write has exactly ONE new owner. The project rule is
hard: downstream consumers -- TTS, per-beat audio slicing, video/shot direction,
captions, credits, `obs_publish` -- read FIELDS, not intentions, and a ripped pass
with an unowned field is a broken render, not a simplification.

Scope: the VERBATIM path for play-form sources (`shakespeare` today; any
already-dialogue public-domain source tomorrow). Prose sources are a separate
design and are NOT covered here -- see the open question at the end.

Lane today: `legacy_many_pass_adapt`, ten passes, `executable: false` while the
writer treats it as inline (`nodes/_otr_lane_specs.py:118-123`).

## The table

| # | Pass (slot) | What it owns today | Verbatim-path owner | Why |
|---|---|---|---|---|
| 1 | `source_interpret` (technical) | `meta.news` -- `casting_brief`, `script_brief`, `key_terms`, `news_close_brief`, `source_hash`; `meta.source_interpreter`; `meta._adaptation_character_names` | **DETERMINISTIC** from manifest + provenance sidecar + the selected passage | Work, act, scene, place, author and speakers are source-owned facts already on disk. Asking an LLM to rediscover them is what let a paraphrase drift. `key_terms` derives from the passage text. |
| 2 | `outline` (creative) | beat/slot skeleton in `led.lines`, `setting`, `time_of_day`, provisional title | **THE PASSAGE** -- one speech, one voiced beat, in source order | The running order is the play's. Nothing to plan. |
| 3 | `dramatic_state` (creative) | `meta.dramatic_state`, `meta.arc_shape` | **GATED OFF** on `style_pool_class == "adaptation"`; both keys omitted | The arc is the source's own. `arc_shape=""` is an already-exercised path, the key is omitted rather than blank, and its only reader (`otr_credits_roll.py:440`) is behind a truthy guard. It printed "Arc: betrayal" over a courtship comedy. |
| 4 | `casting` (creative) | `led.cast`, `meta.cast_contract`, `cast_status`, voice assignment | **THE PASSAGE'S SPEAKERS**, in first-appearance order; voice assignment keeps its existing owner | The people in a scene are a fact OF the scene. Retires `cast_hints` as authority. Never pool-fill on a fidelity bank. |
| 5 | `line_compose` (creative) | the spoken text of every character row | **COMPILED** verbatim from the passage segment | The keystone. A pointer proves structure, not meaning (`PRODUCTION_SPRINT_LESSONS.md` lesson 11). No model authors source speech. |
| 6 | `exchange_compose` (creative) | grouped-exchange rewrite of the same rows | **NOT RUN** | It exists to author dialogue. There is no dialogue to author. |
| 7 | `announcer_intro_outro` (creative) | the two wrapper rows' text | **DETERMINISTIC TEMPLATE** from manifest + provenance | The announcer relocated Arden to Verona. Wrapper rows carry `row_origin=wrapper` and NO source pointer, so they can never be mistaken for source. |
| 8 | `announcer_coda` (creative) | the closing attribution line | **`_otr_provenance.py`** -- extend the EXISTING owner | It already owns `normalize_provenance` / `spoken_coda_line` / `printed_credit_line`. A second composer would split ownership of the same fields. Bind output to the verified body hash. |
| 9 | `title_regen` (creative) | `meta.episode_title`, `meta.title_source` | **DETERMINISTIC** from manifest: play, act, scene, scene label | A title is a fact here, not a creative act. `title_source` becomes `source_manifest`. |
| 10 | `story_brief_reflection` (technical) | `meta.story_brief`, `story_brief_terms` (setting/lighting/atmosphere), `visual_palette`, `music_mood_terms` | **KEEPS RUNNING, UNCHANGED** | It reads the DELIVERED script and was proven content-loyal. Video and music need these fields and nothing else produces them. It observes; it does not author dialogue. |

Also produced outside the pass list and still needed:
`meta.produced_story` (logline/subject, `llm_post_composition`) -- keep; it
describes what was delivered. `meta.source_bank` / `meta.visual_style` are stamped
before any lane dispatch and are untouched.

## Fields that must NOT survive on this lane

- `meta.style` / `story_contract` -- the invented sound world. A domestic
  "fire in the grate, a mantel clock, a teacup" was imposed on a forest scene and
  on Wells' Richmond parlour. Gate with the existing bank scaffold switch.
- `meta.arc_shape` -- see row 3.
- `meta._adaptation_character_names` from `cast_hints` -- superseded by parsed
  speakers. The manifest field stays until the schema migration (both runtime
  validators AND `public_domain_manifest_schema.json` still require it), but it is
  no longer authority.

## Overwrite paths that must be closed before wiring

These would silently mutate source-owned text after it is compiled:

1. **`run_post_script_spine` -> `strip_line_formatting`**
   (`nodes/_otr_ledger_scrub.py:191-230`) can rewrite spoken text at the tail.
   Either exempt rows carrying a source pointer, or register that exact cleanup as
   a named closed transformation and re-validate against the segment afterwards.
2. **`custom_premise`** (`OTR_LedgerScriptWriter.py:1738-1760`) bypasses the
   authenticated fetcher for source-contract banks -- unauthenticated caller prose
   can still reach a "fidelity" lane. Reject it for these banks, or demote it to a
   non-authoritative selector.
3. **The count-match invariant** (`OTR_LedgerScriptWriter.py:4061-4067`) hard-raises
   whenever locked != requested cast. The passage sets the cast, so this must
   accept `locked > requested` (never `<`) and stamp a reason code.
4. **`compute_episode_budget`** (`:4099-4104`) still receives the stale widget
   value rather than the true locked cast.

## Capacity, which is not optional

A verbatim passage is performed one speech per voiced beat, and beats come from the
act topology, not the word count (`_otr_episode_budget.ACT_COUNT_CONFIG`):

| target words | acts | voiced beats |
|---|---|---|
| 30-120 | 1 | **3** |
| 150-200 | 2 | 6 |
| 300-1200 | 3 | **14** |
| 1500+ | 5 | 17 (hard ceiling 19 x 80 = 1,520 words) |

Measured consequence on the real corpus: at 120 words a passage is a two-or-three
speech fragment (5-37 eligible windows per scene); at 300 it is an eleven-to-thirteen
speech exchange (50-385 windows). **The fidelity floor should be 300**, which is
what every manifest's `recommended_word_budget` already says. Below it, refuse with
a typed pre-model error naming required vs available beats -- do not render a
truncated exchange.

## Open, for the operator

**Prose sources need their own answer.** Wells' Chapter III is a first-person
narrator recounting a dinner party; it has no speech prefixes, so the passage
selector refuses it by design. The operator is weighing "public domain drama vs
public domain paraphrase". The honest split looks like source FORM rather than
accuracy: already-dialogue sources get verbatim passage performance; prose gets
either a narrator/reader speaking the author's own sentences, or an openly
adapted dramatization. Which of those two to build -- or both -- is not settled
here.
