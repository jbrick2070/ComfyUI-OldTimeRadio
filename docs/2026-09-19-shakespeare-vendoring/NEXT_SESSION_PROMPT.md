# Kickoff prompt — Shakespeare vendoring, coordinator session

Paste the block below into a fresh Claude Code window in the OTR repo.

---

You are the coordinator for the Shakespeare translation vendoring work. Read
`docs/2026-09-19-shakespeare-vendoring/` first — it is the whole state of play,
committed 2026-09-19 in `6b13b100`.

**First action, before reading or editing anything:** `git fetch origin main`,
report what came down, `git pull --rebase origin main`. Two boxes push to this
repo.

## Where the corpus stands

- `config/source_banks/shakespeare/translations/manifest.json` — **16 scenes
  vendored: fr 12, es 3, it 1.** `tests/test_verbatim_corpus.py` is green at 128.
  **ja, zh and pt vendor ZERO**, even though the Japanese and Portuguese
  extractor rules ship and are tested — those rows are separately held. A
  shipped rule is not a vendored scene; quote the manifest, not the rule
  inventory. This exact conflation went into the plan on 2026-09-19 and had to
  be corrected.
- `config/source_banks/shakespeare/translations/leads.json` — **95 rows**, of
  which **18 carry a `hold` note** saying why that cell is not vendored. Every
  unvendored cell has a named next step. There are no unhunted cells left.
- Extractor: `scripts/otr_vendor_shakespeare.py`. Per-edition rules live in
  `_SPEAKER_SPANS`, `_DIRECTION_BLOCK`, `_TIINHERIT_BLOCK`, `_AOZORA_BUSINESS`,
  `_AOZORA_SPEAKER`. Scene labels live in `EDITION_LABELS`, keyed
  `(iso, play, scene) -> (play_anchor, act_label, scene_label)`.

## The one lesson that governs this work

**Three hold notes in a row were wrong about their own edition** — Japanese,
Portuguese, Chinese. Each note said the edition was unmarked or unparseable;
each edition marks cleanly in a vocabulary no rule knew. The note is why nobody
looked again. Memory has this as
`shakespeare-hold-notes-lie-about-their-editions`.

So: **read the markup before believing any note about the markup, including one
written by me.** Fetch through the repo's own cached `V.fetch`, dump the class
inventory, look.

And: **count what the EDITION marks, then compare** — never count what survived
parsing. A character swallowed whole never appears as a speaker to look for. The
Italian set showed a healthy-looking 142 speeches / 13 speakers while listing
stage directions and Feste's Latin joke among its cast.

## Three tracks, in the order I'd take them

### 1. The Chinese extractor rule — shortest path, unlocks Midsummer

Fully specified in `chinese_edition_diagnosis.md`. Speaker is 1–4 CJK chars plus
U+3000 at a paragraph head; block business is a centred div opening with a
fullwidth black bracket; headings match `^第.*[幕场場]$` and must be exempted or
the scene anchors break.

**The trap, confirmed independently by two models:** `_marked_name` rejects
`len(name) < 2`, and most of this cast is a single character (波, 衮, 蒂, 勃,
弗). The rule can mark every speaker correctly and still drop them all
downstream. Relax that guard for CJK.

Measurement to beat: the page marks **65 speeches across 14 characters** in
3.1; `mark_speakers` currently claims **0** and the heuristic tail returns 56.

This touches shared extractor code that every edition runs through. Blast-radius
check before pushing: re-run `--check` across all 16 vendored scenes and show
the speech counts unmoved. A harness that has never reproduced a known-good
result cannot be trusted to detect a change from one.

### 2. The Italian Rusconi speaker rule — 12 cells

Diagnosed in `italian_rusconi_diagnosis.md`. Abbreviated marks (`Orl.`, `Ber.`,
`Ob.`) followed by a period at the head of a paragraph. The generic name shapes
claim these correctly and *also* claim stage directions and in-dialogue text.

**The obvious fix is proven harmful:** anchoring on `<p>` breaks two shipped
scenes — tempest went 142 speeches to 6. Do not re-derive that; it is measured.

All 12 act-page URLs are verified and sitting in `italian_act_urls.json`. Note
the Twelfth Night wrinkle: standard 2.5 is printed as SCENA VI because that act
skips SCENA IV.

### 3. Production OCR — 23 scan cells

Unblocked by the 2026-09-19 ruling in `docs/OTR_STANDING_RULINGS.md`: one
model's OCR counts as verbatim. The Spanish *Tempest* 1.2 prompt is already
written in `url_hunt_prompt.md`. This is the track to hand out to Grok / ChatGPT
/ Antigravity in parallel while you code tracks 1 and 2.

## Also open, smaller

- `pt/hamlet 1.1` extracts 25 of the 56 speaker blocks the page marks, and the
  tail runs past the scene into wiki chrome.
- `es/hamlet 1.1` returns 923 chars for a scene that runs ~9k elsewhere. URL and
  labels are right; the scene-end cut fires early on a single-file play text.
- `fr/midsummer 3.1` and one sibling: Hugo numbers scenes continuously, so the
  recorded label spans several Folger scenes. Re-derive the number from the
  text, not from the researched table.
- Four Japanese Aozora landing pages and four Chinese later-輯 volumes still
  need a source hunt. The 1947 World Book Company scan is at 0% OCR.

## How to run the work

Per `CLAUDE.md`: this is a design change with more than one defensible answer,
so track 1 and track 2 each get a contrarian on the diff before the push —
briefed to refute, given the real Windows files, from a different model family
than whatever drives. Do not push first and review after. Commit and push each
green chunk without asking; named files only, never `git add .`.

Test runner is `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe` with
`PYTHONUTF8=1`, `pytest -q -p no:cacheprovider`.

Start with track 1 and tell me the two counts before you write any rule.
