# PROBLEM STATEMENT -- Item 8, voice/portrait consistency

**Date:** 2026-08-05
**Driver:** Claude (Cowork), CODER window, budget rung 4.
**Status:** EVIDENCE IS SOLID. MECHANISM IS OPEN -- two driver hypotheses are already
dead (section 4). This document goes to the panel to have its framing broken, per the
two-strikes rule, before any code is written.
**Gate:** full `kibitz-plugin:kibitz` four-round arc (operator directive 2026-08-04).

---

## 0. THE OPERATOR RULING THIS ITEM SERVES

> "I am fine with LLMs trying to choose the best gender as long as they choose one and
> pick a voice that matches and portrait matches." -- and -- "the voice and picture must
> match, that most important."

**CONSISTENCY BEATS ACCURACY.** A female Scrooge with a female voice and a female
portrait is coherent but unfaithful. A male Scrooge with a female voice is broken.

The ruling stands. **The diagnosis it was made on does not.**

---

## 1. THE PLAN'S PREMISE IS INVERTED -- MEASURED

`GO_FORWARD_PLAN.md` item 8 says "**`voice_ref_id` ignores the resolved gender** [...]
the preset follows the roll every time and the reference never does [...] AUDIBLE in
the finished episode. **Fix this first.**"

**Both halves are false.** Corpus: 1,595 ledgers / 5,123 cast rows under
`C:\Users\jeffr\Documents\ComfyUI\output\otr`, scored with the repo's own authorities
(`_otr_voice_bank.load_voice_bank`, `bark_preset_gender`).

| non-announcer rows, both genders resolvable | count |
|---|---:|
| row gender == **reference** gender | **1,169** |
| row gender != **reference** gender | **0** |
| row gender == **preset** gender | 1,334 |
| row gender != **preset** gender | **225** |

The only four rows whose reference crosses gender are `other`/`non-binary` -- genders
the bank cannot serve -- and all four carry the honest receipt
`voice_cast_fallback="gender_unservable"` (`cast_lock.py:645-667`).

**`voice_ref_id` follows the gender. `voice_preset` does not.** The plan named the one
field that is already right.

### It is live, not history

| ledger age | rows | preset wrong | rate |
|---|---:|---:|---:|
| > 30 d | 552 | 2 | **0.4 %** |
| 7-30 d | 770 | 130 | 16.9 % |
| 2-7 d | 124 | 27 | 21.8 % |
| **0-2 d** | 135 | 66 | **48.9 %** |

### And "audible" needs a correction

Each engine declares the field it speaks (`_otr_voice_node_common.py:454`):
`eng_bark.py:32` -> `voice_preset`; `eng_kokoro.py:117` -> `voice_ref_id`;
`eng_google_tts.py:355` / `eng_cloud_elevenlabs.py:140` -> `provider_voice_id`;
default -> `voice_ref_path`, the **indextts2** path (`:540`).

**1,147 of 1,559 recent comparable rows rendered on indextts2**, which speaks the
reference. So today the delivered voice is gender-correct and the wrong preset is
inaudible. It is still a real defect (section 5) -- but a fix aimed at the reference
would have been a fix aimed at nothing.

---

## 2. THERE ARE TWO DEFECTS, ON TWO LANE FAMILIES

Segmenting the last 30 days by whether an episode's preset-gender multiset matches its
row-gender multiset (>=2 comparable rows, 456 episodes):

| verdict | episodes |
|---|---:|
| clean | 307 |
| **NOT a permutation** (multisets differ) | **101** |
| **PERMUTATION** (same multiset, wrong pairing) | **48** |

And the split is almost perfectly by lane family:

| lane | verdict | episodes |
|---|---|---:|
| `scifi_news` | NOT a permutation | 49 |
| `scifi_codex` + `_v2/_v3/_v4` | NOT a permutation | 27 |
| `scifi_gemini` | NOT a permutation | 8 |
| **`shakespeare`** | **PERMUTATION** | **26** |
| **`public_domain`** | **PERMUTATION** | **16** |
| `public_domain_story*` / `shakespeare_v3` | PERMUTATION | 6 |
| `shakespeare` | NOT a permutation | 16 |

### Defect A -- the content-owned `scifi_*` family ships a FIXED ALL-MALE preset triple

Every sampled `scifi_*` episode carries the same three presets in the same order,
regardless of the row genders:

```
v2/en_speaker_6 (male), v2/en_speaker_3 (male), v2/en_speaker_0 (male)
```

`scifi_codex` (25.6 d), `scifi_codex_v2` (20.1 d), `_v3`, `_v4` (18.6 d) -- identical
triple each time, with row genders female/male/female, female/male/female,
male/female/male, male/female/female. **Gender is not consulted at all.**

This is consistent with the content-owned branch: `cast_lock._assign_bark_voices`
returns early when `delivery_mode_for_meta(meta) == CONTENT_OWNED` (`cast_lock.py:332-339`)
or when `meta.cast_contract` has no `cast_seed` (`:343-348`) -- "voice_preset preserved
(no writer replay)". The **lane** owns the preset there, and nothing reconciles it with
the row gender. Confirmed in the corpus: on the mismatching `scifi_news` rows,
`has_cast_seed` is `false`.

### Defect B -- the fidelity lanes ship a PERMUTATION

`shakespeare` and `public_domain` produce the right *set* of presets and attach them to
the wrong characters. Every sampled pair is a clean two-way swap:

| episode age | character | row gender | preset gender |
|---|---|---|---|
| 0.8 d | ENOCH SOAMES | female | male |
| 0.8 d | ROTHENSTEIN | male | female |
| 1.1 d | ELIZABETH BENNET | male | female |
| 1.1 d | MR. DARCY | female | male |
| 3.9 d | MACBETH | female | male |
| 3.9 d | BANQUO | male | female |

**They hold each other's presets.** A permutation is a MAPPING fault, not a
gender-source fault: the picker computed a correct gender-matched assignment and it was
attached to the wrong `char_id`.

---

## 3. WHERE THE MAPPING COULD COME APART (unconfirmed -- the panel's main question)

`cast_lock._assign_bark_voices` (`:306-370`) replays and stamps by `char_id`:

```python
voices = _OTRCAST.replay_voice_assignment(
    cast_seed=int(cast_seed), num_characters=num_characters, lemmy_hit=lemmy_hit)
for row in cast:
    if row.get("char_id") in voices:
        row["voice_preset"] = voices[row["char_id"]]
```

`replay_voice_assignment` (`_otr_casting.py:1053-1102`) reconstructs the cast from
`(cast_seed, num_characters, lemmy_hit)` alone:

```python
rng = random.Random(cast_seed)
pre_locked, open_slots, _hit = assemble_pre_locked_rows(
    num_characters=num_characters, rng=rng, force_lemmy=bool(lemmy_hit))
ensemble_slots = precompute_ensemble_slots(
    open_slots, prior_cast=prior_cast, rng=rng, cast_seed=cast_seed)
...
out[slot.char_id] = voice
```

**Candidate: on a source-owned lane the writer's real cast comes from the source roster
(names, order, genders), while the replay rebuilds a GENERIC roll from the seed. If the
reconstructed `char_id` -> slot ORDER differs from the writer's, correct presets land on
the wrong rows -- exactly a permutation.** Two specific suspects, both unverified:

* `precompute_ensemble_slots` is called by the replay **without `gender_by_name`**
  (`:1088-1090`), so its source-owned pin branch (`:765-768`) never fires. The
  replayed slot genders can therefore differ from the frozen rows even though the rng
  stream is identical -- the docstring at `:730-745` protects stream parity explicitly
  and says nothing about value parity.
* `_repair_ensemble_names` (`:777-780`, gated `repair_names=True`) is a
  "name<->gender coherence repair" that runs inside `precompute_ensemble_slots` and
  compares a name's gender tag to the slot gender (`:687`). Whether it reorders,
  renames, or reassigns is **not yet read** and is a prime suspect for a permutation.

**`replay_voice_assignment`'s docstring claims "The parity test pins
`replay == lock_cast` char-for-char" (`:1080`). It is green while production
permutes.** That test therefore exercises no lane that reproduces this -- **the third
instance of the vacuity class** the plan already records twice (the scene-coherence
gate that never read a populated field; the freeze test filtered on a retired prefix).
Whatever the mechanism turns out to be, that test needs a vacuity assertion.

---

## 4. TWO DRIVER HYPOTHESES ARE ALREADY DEAD -- do not re-derive them

Recorded so the panel spends its rounds on live ground.

1. **"The bark gender column exhausts."** DEAD. `config/cast_pools.VOICE_PROFILES` is
   6 male + 4 female = 10 presets (`:253-270`); the defect appears at cast size 2-3,
   which cannot exhaust a 4-deep column. Measured rate by cast size: 1 -> 1.8 %,
   2 -> 9.8 %, **3 -> 41.1 %**, 4-6 -> 0 % (small n).
2. **"Thread `meta.cast_source_contract.gender_by_name` into the replay."** DEAD as a
   fix for the observed rows. On the failing ledgers that field is **empty or absent**:
   of the mismatching rows in the last 14 days, 92 are on ledgers with no
   `cast_source_contract` at all, 19 have `gender_by_name: {}`, and only 26 have any
   pins. ENOCH SOAMES and ROTHENSTEIN (0.8 d, `public_domain`, `cast_seed` present)
   both show `gender_by_name: {}`. **Threading an empty map fixes nothing.** The
   thread may still be correct for *roster-pinned* rows, but it is not the cause here.

---

## 5. WHY THIS MATTERS WHEN IT IS INAUDIBLE TODAY

1. **`voice_preset` is read by 21 modules** (178 occurrences) including
   `otr_credits_roll.py` and `video_engine.py` -- it can reach a *visible* surface.
2. **`eng_bark` speaks it.** Any bark leg turns the contradiction into audio.
3. **The ledger asserts two different genders for one character.** Under the ledger
   law that is a fault regardless of who reads it.
4. **A claimed parity contract is false**, and anything built on it inherits the fault.

---

## 6. THE OTHER TWO SURFACES

### 6a. The portrait anchor is additive-only and pronoun-blind -- CONFIRMED

`_ensure_gender_anchor` (`otr_meta_brief_image_prompt.py:78-91`) prepends
`"adult woman, "` / `"adult man, "` only when the prompt contains none of
`_FEMALE_PROMPT_TERMS` / `_MALE_PROMPT_TERMS` (`:70-75`).

* **`he/him/his/she/her/hers` are in neither set.** "her left cheek" on a male row is
  not detected; the anchor is prepended and the prompt becomes
  `"adult man, ... her left cheek ..."` -- self-contradicting. That is the Wheel of
  Wrath symptom, mechanism confirmed.
* `gender not in ("female","male") -> return text` (`:82`): **`other` rows get no
  anchor at all** (253 rows in the corpus).
* The anchor never *corrects* -- a prompt already saying "woman" on a male row is left
  alone.

The description generator does receive the gender (`_otr_casting.py:830`), so this is
an LLM-compliance failure with a guard too weak to catch it.

### 6b. The announcer's label and its voice are independent by construction -- CONFIRMED

`announcer_voice_ref` -> `_seeded_preferred_announcer_voice_ref`
(`_otr_voice_bank.py:781-828`) picks the announcer gender from the **episode seed**
(`:805`) and never reads the row's `gender`. Measured: 229 rows labelled `female`
delivered a male announcer reference, against 233 labelled `male`.

The operator has ruled the announcer's randomness is BY DESIGN, so the voice is not the
thing to change -- the **label** is, or the announcer is excluded by name. Operator
decision, section 9.

---

## 7. THE GENDER VOCABULARY IS WIDER THAN THE PLAN SAYS

The plan's cite for "`scifi_news` emits `Male`/`Female`/`Non-binary`" is **wrong** --
the only "non-binary" in `_otr_line_composer.py` is a comment on `_PRONOUN_MAP`
(`:571`). The claim is right and understated. The corpus carries **18 distinct gender
strings**: `male` 2471, `female` 2220, `other` 253, `non-binary` 69, `unspecified` 53,
`neutral` 27, `woman` 7, `man` 4, `any` 3, `artificial` 3, `ai` 2, `synthetic` 2,
`genderfluid` 1, `n/a` 1, `unknown` 1, `various` 1, `child-like` 1, absent 4.

Last 7 days: `female` 113, `male` 106, **`Male` 20, `Female` 20** -- the capitalized
forms are live today.

`_VALID_GENDERS = {"male","female","other"}` (`_otr_casting.py:142`) is enforced only
on the `CastingResponse` path; other routes bypass it. Any check must normalize before
comparing and must not treat an unrecognized legacy string as a violation.

---

## 8. WHAT TO BUILD

**Chunk 0 (before any fix) -- name the mechanism.** Read `_repair_ensemble_names` and
`assemble_pre_locked_rows`, and reproduce a permutation in a test from a real
`shakespeare` ledger's `cast_contract`. The fix is not designed until a test fails for
the right reason. **This is the panel's first job: break or confirm section 3.**

**Chunk 1 -- fix defect B (the fidelity-lane permutation) at its root**, plus the
vacuity assertion the parity test is missing: a fixture whose pins/roster disagree with
a generic roll, asserting per-row `preset_gender == row.gender`, and an assertion that
the test examined a non-zero number of rows.

**Chunk 2 -- fix defect A (content-owned lanes).** Decide the OWNER: either the lane
assigns gender-aware presets, or CastLock reconciles them after the lane writes. One
owner, stated. Today neither does it.

**Chunk 3 -- make the portrait anchor corrective.** Add gendered pronouns to the term
sets; make a contradicting anchor a correction rather than a no-op; give `other` rows a
defined treatment.

**Chunk 4 -- `scripts/audit_voice_gender_consistency.py`**, shaped on
`scripts/audit_spoken_citations.py` (`os.walk` for `*_ledger.json`, per-ledger
findings, a `LEGACY_SCHEMA_VERSIONS` boundary so pre-fix ledgers are tolerated and
post-fix ones must be clean). Asserts per cast row that `gender`, the preset's implied
gender, the reference's implied gender and the portrait anchor agree.

**Not in scope:** which gender is *chosen*. That is item 7.

---

## 9. OPERATOR DECISIONS (flagged, not assumed)

1. **The announcer.** Voice randomly gendered by design. Should the row's `gender`
   label follow the seeded voice, or is the announcer excluded from the invariant?
   **Default if unruled: exclude by name and say so in the audit output.**
2. **`other` rows** (253). No bank voice, no bark column, no portrait anchor.
   **Default if unruled: derive one presentation gender from the cast seed, stamp it,
   and have all three surfaces consume it** while the row keeps its `other` label.
3. **Item 7 may outrank item 8 after all.** Item 8's headline defect is largely already
   satisfied -- the delivered voice follows the ledger. What the scan surfaced instead
   is that **the ledger's gender is wrong on named characters**: `ELIZABETH BENNET`
   shipped male and `MR. DARCY` female within 24 hours, `MACBETH` female at 3.9 days,
   `ANTIPHOLUS` male in one leg and female in another. Those episodes are *coherent* --
   voice and portrait follow the label faithfully -- so they satisfy the consistency
   rule and still sound wrong. **Recommendation: land chunks 3-4 (small, and the audit
   is the standing receipt), fix defect B, then move item 7 to the front.**

---

## 10. RECEIPTS THIS CAN AND CANNOT CLAIM

* **`code-complete + suite-green`** -- achievable this session.
* **`corpus-proven`** -- the audit re-scans all 1,595 ledgers; post-fix ledgers clean.
* **`production-qualified`** -- needs one live leg on `shakespeare` or `public_domain`
  with `RESULT SUCCESS`, `obs_publish OK`, the asset on disk, and a clean audit on the
  ledger it produced.
