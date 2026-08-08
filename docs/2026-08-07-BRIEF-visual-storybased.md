# CAMPAIGN BRIEF -- `visual_storybased` (GO_FORWARD queue item 2)

Date 2026-08-07. Branch `v2.0-alpha`, HEAD `1d106603`.
**Status: NO SPEC EXISTS.** GO_FORWARD's queue row says "Spec below" and no spec
section was ever written; ROADMAP says the spec "lives there". This brief is the
r1 input, not a spec -- producing one is what this campaign's early rounds owe.

## 1. THE ASK (operator, verbatim from ROADMAP.md:148-155)

> a dynamic visual style whose parameters the LLM decides from the story -- no
> presets, no seeds -- shipped as a TENTH dropdown entry peer to `anime` and
> `paper_origami`, with the existing nine packs and the seeded roll kept as the
> fail-closed floor

That is the entire recorded requirement.

## 2. WHAT EXISTS TODAY (grounded, real files)

* **Nine authored style packs**, `nodes/visual_styles/*.json`, each a v2 pack of
  ~25 validated leaves. Validation is `_otr_visual_styles._validate_row`, which
  enforces real contracts -- notably `announcer_subject_ltx_mouth` must carry
  mouth-prominence vocabulary (`:234-238`), because the ia2v announcer lane
  lip-syncs to whatever reads as a mouth. Templates must contain their
  placeholders exactly once; `motion_registers` is char-budgeted (BUG-LOCAL-112).
* **ONE resolver seam:** `get_visual_style(meta)` at `_otr_visual_styles.py:345`.
  Every consumer funnels through it -- image composer, render driver, still-word
  cards, credits.
* **A seeded roll** over the registry: `_otr_rolls.eligible_style_ids()`
  (`:264`). `_otr_rolls.py:20` states **"ZERO LLM CALLS. A roll reads a
  registry, filters, and draws."** `eligible_style_ids` documents that it
  applies **"NO filter, by design and not by omission"**.
* **A command-row precedent:** `rolls.STYLE_SENTINEL` sits at `choices[0]` of
  the widget and is deliberately NOT in `list_style_ids()`. Pinned by
  `tests/test_visual_style_widget_3c.py:72-74`.
* **A no-fallback run() gate:** `OTR_LedgerScriptWriter.py:3555-3560` resolves
  the requested style and raises `UnknownVisualStyleError` before any story work.
* **Some story-fitting already happens:** the brief's `visual_palette` and
  `atmosphere_line` already reach every prompt via `get_era_tail`
  (`_otr_story_brief_helpers.py:263`), so per-episode COLOUR is not new. What
  nine packs cannot vary per episode is medium, texture, lighting character,
  linework and typography.

## 3. HARD CONSTRAINTS (violating these makes a proposal useless)

* 100% local and offline-first by DEFAULT; 16 GB VRAM ceiling.
* **Story/prose quality is CLOSED by operator directive** -- this campaign is
  visual only. A proposal that improves scripts is out of scope by rule.
* Reproducible receipts: the ledger records what was actually used, and replay
  must reconstruct it.
* Fail-closed: the nine packs and the seeded roll remain the floor.
* `widgets_values` is POSITIONAL. A new dropdown ENTRY is cheap; a new WIDGET is
  not. The canonical workflow JSON must not need to change.

## 4. AN UNVALIDATED FIRST PROPOSAL -- attack this

From a cold first-read pass. It is NOT decided and NOT driver policy; it is here
so the panel has something concrete to break rather than a blank page.

1. The LLM emits a small **STYLE CARD** (~6-8 structured fields: medium,
   palette, lighting character, texture, linework/grain, typography voice,
   motion temperament) in ONE grammar-constrained call on the already-resident
   local writer -- claimed zero VRAM delta.
2. **Deterministic Python expands the card into a full v2 pack** through fixed
   templates, so the mouth contract, placeholders, char budgets and genre keys
   are satisfied BY CONSTRUCTION rather than by hoping the model complies.
3. The expanded pack is gated by the **same `_validate_row`** that gates
   authored packs.
4. The validated pack is **EMBEDDED IN THE LEDGER** beside the style id, and
   `get_visual_style` grows exactly one arm.
5. `visual_storybased` ships as a **COMMAND ROW** (like `STYLE_SENTINEL`), NOT a
   registry pack, and never enters the roll's draw pool.
6. One card per EPISODE, frozen after the brief. No per-beat dynamism.
7. Floor triggers on exactly two conditions -- malformed card after bounded
   attempts, or expanded pack fails validation -- then hands to the existing
   seeded roll, with a receipt naming which path ran.

## 5. THE OPEN QUESTIONS THE PANEL SHOULD ANSWER

**Q1.** "TENTH DROPDOWN ENTRY PEER TO `anime`" vs the roll's zero-LLM law. A
registry pack is automatically in the roll's draw pool and `eligible_style_ids`
has no filter to stop it. Is the command-row reading right, or is there a better
reconciliation that keeps registry membership?

**Q2.** "No seeds" vs "the seeded roll is the fail-closed floor" -- the floor IS
a seed. What does "no seeds" actually mean, in words a builder cannot misread?

**Q3.** "No presets" vs reproducible receipts -- a frozen emitted pack in the
ledger is functionally a one-episode preset. Is that the intended reading?

**Q4.** Is card-then-compose right, or should the LLM emit more (or less)? What
is the smallest emission that still produces a visually DISTINCT episode?

**Q5.** What is the replay contract for a ledger that carries the id but no
embedded pack (e.g. a crash between skeleton save and the LLM phase)? And for
OLD ledgers written before this feature exists?

**Q6.** What does the operator SEE that tells them the dynamic path ran versus
the floor? Where does that receipt live?

**Q7.** What should be CUT from this campaign entirely?
