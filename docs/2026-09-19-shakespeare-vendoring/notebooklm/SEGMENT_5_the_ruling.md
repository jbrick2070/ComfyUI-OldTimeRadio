# The ruling: a mangled native text beats a clean AI translation

## The one thing to take away (3-5 sentences)

This radio project does not treat "faithful" as "pretty." It treats faithful as *whose words are on the page* and *whose mouth speaks them*. Scanner damage, abbreviated Spanish cues, and labels the roster cannot bind are imperfections the operator accepts; putting Hermia's line in Demetrius's mouth is not. Rights dates and credit-roll prose inform listeners but refuse nothing, because the operator is not selling episodes commercially and does not want research to shrink the corpus. Generated performances may carry the source's own rough language; the repository's code, comments, and logs stay clean anyway. What remains unsettled is mostly engineering on the scan lane—how to emit unbound speakers—not a reopening of these fidelity lines.

## The decisions (one short section per ruling: what was decided, the quoted words, the reason, one example)

### Native mangling ships; AI substitution does not

**Decided:** Gates refuse misattribution, never imperfection. A legitimate translation on paper, even with OCR damage, beats a machine rewrite of the same page.

**Quoted:** *"im noit pa perfetrcuioniust so any natiev even tough the chaters may be mushged is beter tahn a ai atrasnation thast my standing call dto asking me"* [docs/OTR_STANDING_RULINGS.md:28-32]

**Plainly:** The operator is not a perfectionist; native text with mangled characters still wins over an AI translation—and that call is final, not a question for later windows.

**Reason:** Jaime Clark's Spanish Tempest can show `Mır` for Miranda and `alguenı` for *alguien*; that is still Clark's work, whereas a clean model translation is not [docs/OTR_STANDING_RULINGS.md:36-38]. An unresolvable cue becomes its own speaker, left **unbound** (no voice cast), never merged into the previous character—that merge is chosen behavior and is ruled indefensible [docs/OTR_STANDING_RULINGS.md:43-52, docs/GO_FORWARD_PLAN.md:255-261].

**Example:** Macpherson's *Lear* 1.1 prints damaged forms like `LENT.` for Kent; the fold table keeps Kent's physician speech on Kent after ink review, while `REQ` at p.254 stays **unbound** when the glyph does not safely settle on Regan [GROK_FOLD_TABLES_measured.md:11-14, 79].

### One model's OCR is verbatim enough

**Decided:** A single model may transcribe a scan; consensus is not required.

**Quoted:** *"im fin eitwh one model ocr"* [docs/OTR_STANDING_RULINGS.md:67-69]

**Reason:** Twenty-three corpus cells exist only as 1890s–1910s page scans with no text layer; closing them on "human only" capped coverage near fifty-six percent [docs/OTR_STANDING_RULINGS.md:71-75]. OCR changes **how** words are captured, not **which** edition qualifies.

**Measured check:** Three models read the same page of Tsubouchi's Japanese *Macbeth*. Gemini 3.8 Flash and GPT-5.6 Luna matched all thirteen speaker labels; Grok's first pass had the dialogue right but **four of thirteen labels wrong**, fixed only on a second, higher-magnification read—content-right-labels-wrong is the dangerous success shape [docs/OTR_STANDING_RULINGS.md:84-92].

### Renamed cast and intermediary translations still fail

**Decided:** Fidelity on **source choice** is unchanged by the scan rulings.

**Quoted (intermediary):** Castilho "worked from a French text" and such rows stay refused [docs/OTR_STANDING_RULINGS.md:77-79]. **Quoted (adaptation):** A version that "renames the cast is an ADAPTATION and still refused" [docs/OTR_STANDING_RULINGS.md:79-80].

**Reason:** OCR does not rescue a text that is not Shakespeare-via-that translator's own direct work, or that rewrites who the characters are.

**Example:** Nine Hindi cells closed on 2026-09-19 because the cast was renamed—not for lack of a scan [docs/OTR_STANDING_RULINGS.md:80-81]. GO_FORWARD repeats that fidelity still governs intermediary work while rights do not [docs/GO_FORWARD_PLAN.md:238-239].

### A wrong fold is worse than no fold

**Decided:** Unbound costs a **voice** (no cast assignment); a wrong fold costs **dialogue in the wrong mouth**, which the 2026-09-19 ruling still refuses [PROMPT7_grok_a_wrong_fold_is_worse_than_no_fold.md:12-19, docs/OTR_STANDING_RULINGS.md:48-52].

**Reason:** Speaker maps must key on `(sha256, scene_stem)`, not language or volume alone—`SEB.` in one Clark hash names two different Sebastians, so a global Spanish alias would misattribute [GROK_FOLD_TABLES_measured.md:41-46].

**Example:** On Macpherson *Midsummer* 3.2 p.424, ink reads `HER.` for Hermia asking Demetrius about Lysander; folding `Per` to Demetrius would put her speech in his mouth. The table demotes `Per` → HERMIA and ships `Hek` and `DER` **unbound** when inference would risk wrong-mouth [GROK_FOLD_TABLES_measured.md:8-15, 28-29].

### Rights refuse nothing; dates stay required

**Decided:** No lead, manifest row, or render is refused on copyright arithmetic.

**Quoted:** *"I don't want to waste anything in rights I'm not publishing these commercially"* [docs/OTR_STANDING_RULINGS.md:173-174, docs/GO_FORWARD_PLAN.md:234-236]. Code comment: *"rights refuse nothing now"* (operator: same commercial posture) [nodes/_otr_verbatim_corpus.py:61-63, 651-664].

**Reason:** A boolean gate in `load_manifest` would have decided renders at authoring time; the operator wants translator death year and first-publication year as **data for the credit roll**, not a lock [nodes/_otr_verbatim_corpus.py:663-664, docs/OTR_STANDING_RULINGS.md:177-180]. `publication_reasons` writes explanatory sentences into reports; it does not refuse [nodes/_otr_verbatim_corpus.py:71-73, 76-79].

**Fields:** `translator_death_date` and `translation_first_published` remain in `REQUIRED_MANIFEST_FIELDS` [nodes/_otr_verbatim_corpus.py:577-581].

### Generated episodes vs clean repository speech

**Decided:** Two layers—performance content vs project authorship.

**Quoted (episodes):** *"we [have] too [many] guardrails, no violence or swearing guardrails, they just cause problems"* and *"I've given up chasing profanity"*—do not filter generated episode content or forbid the source's own language on adaptation lanes [CLAUDE.md:11-19].

**Quoted (repo):** *"authoring style stays clean -- no curse words in CODE, COMMENTS, LOGS or commit messages"* [CLAUDE.md:20-22].

**Reason:** Filtering Macbeth's violence in prompts was a **fidelity defect** (forbidding what the author wrote), not a safety win [CLAUDE.md:16-18]. That is not permission to dirty the codebase.

## Why it matters for a performance

Text-to-speech and casting read **speaker labels** and roster bindings, not intentions. When the extractor merged orphan cues into the previous speaker, listeners heard the right words from the wrong character—a silent rewrite of the translator's drama [docs/OTR_STANDING_RULINGS.md:48-52]. Unbound labels honestly cost a voice in the roll but preserve every line [docs/GO_FORWARD_PLAN.md:293-300]. Wrong OCR on a **label** while dialogue is correct still sounds like success until someone counts edition-marked speeches against the transcript [docs/OTR_STANDING_RULINGS.md:94-99]. The asymmetry is performative: lose a cast slot, not a soliloquy.

## For the hosts: three hooks (one line each) and two open questions

**Hooks**

1. The operator typed, typos and all, that mangled native Shakespeare beats a spotless AI translation—and meant it as law, not a poll [docs/OTR_STANDING_RULINGS.md:28-32].
2. One model may OCR a 1910 page into the corpus, but the same day's Macbeth bake-off showed four wrong speaker names on a first glance—verbatim text, wrong mouth [docs/OTR_STANDING_RULINGS.md:67-69, 87-88].
3. They demoted a fold because the ink said `HER.` and the table had almost given Hermia's plea to Demetrius [GROK_FOLD_TABLES_measured.md:8-10].

**Open questions (the record, not this segment's lane)**

1. **Speaker emission:** Boundary `--pages` work is measured on eight Spanish windows; the standing ruling requires unbound speakers, but README states that behavior is *"Not yet built"* [docs/2026-09-19-shakespeare-vendoring/README.md:10-13, docs/GO_FORWARD_PLAN.md:255-266].
2. **Golden counts vs live extractor:** Fold tables publish expected speech totals and unbound label lists for the emitter to hit; today's path still discards-and-merges [GROK_FOLD_TABLES_measured.md:25-33, docs/GO_FORWARD_PLAN.md:256-258].

## Claims register

| Oddity or claim | Edition (translator, year) | Evidence | Status |
|---|---|---|---|
| Gates invert: refuse wrong mouth, accept glyph damage | Spanish scan lane (Clark / Macpherson volumes, 1910s print) | Operator quote + `Mır`/`alguenı` example [docs/OTR_STANDING_RULINGS.md:25-52] | RULING |
| Unbound cue, never merge | Macpherson *Lear* 1.1 | `REQ` demoted unbound p.254 [GROK_FOLD_TABLES_measured.md:11-14] | MEASURED |
| One-model OCR opens 23 scan-only cells | Corpus-wide | [docs/OTR_STANDING_RULINGS.md:71-75] | RULING |
| Tsubouchi *Macbeth* page: 13 labels, Grok 4 wrong first pass | Tsubouchi, Japanese (edition page same-day read) | [docs/OTR_STANDING_RULINGS.md:85-88] | MEASURED |
| Hindi nine cells: renamed cast, refused | Hindi leads (nine cells) | [docs/OTR_STANDING_RULINGS.md:79-81] | RULING |
| Castilho refused (French intermediary) | Castilho Portuguese line | [docs/OTR_STANDING_RULINGS.md:78-79] | RULING |
| `Per` fold would wrong-mouth Hermia | Macpherson *Midsummer* 3.2, p.424 | Ink `HER.` [GROK_FOLD_TABLES_measured.md:8-10] | MEASURED |
| `SEB.` one hash, two Sebastians | Clark Tempest / *Noche de Reyes* (`b79223db…`) | [GROK_FOLD_TABLES_measured.md:41-46] | MEASURED |
| Rights dates required, refuse nothing | Manifest rows (all vendored scenes) | `REQUIRED_MANIFEST_FIELDS` + `load_manifest` comment [nodes/_otr_verbatim_corpus.py:577-581, 651-664] | RULING |
| No episode profanity/violence filter; clean code | Pipeline vs CLAUDE.md | [CLAUDE.md:11-22] | RULING |
| Speaker half open (unbound emit) | Eight Spanish windows | [README.md:10-13] | MEASURED |

SOURCES READ: 7
