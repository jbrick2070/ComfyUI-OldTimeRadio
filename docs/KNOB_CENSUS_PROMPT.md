# Knob census — which widgets earn a place in front of a first-time user?

**This is NOT a dead-code hunt.** Inert widgets are already being removed by a
separate campaign. This census is about widgets that WORK but may not deserve
to face the end user: a stranger downloading the shipped template sees every
knob as a question to answer, and most of those questions have exactly one
right answer we already know. Your job is to produce the EVIDENCE TABLE the
operator rules from — you recommend, he decides. Nothing is removed by this
pass.

**READ-ONLY. No edits, no renders, no server** — a GPU queue is often active.
CPU scans, AST reads and corpus scripts are all encouraged.

    REPO:   C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
    BRANCH: v2.0-alpha
    PYTHON: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe  (set PYTHONUTF8=1)
    GRAPHS: workflows\otr_canonical.json, workflows\otr_story_only.json,
            workflows\variants\*.json  (61 generated variants -- per-machine
            configuration lives HERE by operator ruling, never in canonical)
    CORPUS: C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\*\audio\*_ledger.json
            (~2000 frozen production ledgers -- what was ACTUALLY run)

## The method

**Step 1 — inventory.** For every registered node in the canonical workflow,
list its widgets in INPUT_TYPES order: name, type, default, tooltip (note if
the tooltip is missing or unhelpful — that is a finding in itself). The big
one is OTR_LedgerScriptWriter (~32 widgets); do them all, every node.

**Step 2 — the usage census, from evidence.** For each widget answer, with
real numbers:

  a. CANONICAL: what value does the shipped canonical carry? Is it the
     INPUT_TYPES default or an operator-chosen override?
  b. VARIANTS: across all 61 variants, how many distinct values appear? A
     widget that only varies across variants is a PER-MACHINE fact, already
     owned by the variant system.
  c. CORPUS: where the ledger records the resolved value (many widgets stamp
     into meta/receipts -- trace which do), how many distinct values appear
     across ~2000 real episodes? A widget at its default in 100% of recorded
     episodes was never turned by anyone.
     Where the ledger does NOT record it, say so -- absence of evidence is a
     finding, not a guess.
  d. WHO SETS IT headlessly: is it on the CREATIVE_WHITELIST
     (nodes/_otr_workflow_apply.py, scripts/otr_api.py)? Does any profile,
     launcher or soak script patch it? A widget patched by harnesses but
     never by a human is infrastructure, not UX.

**Step 3 — classify.** One row per widget:

  KEEP-SURFACED    varies per episode by real human/creative choice
  VARIANT-OWNED    per-machine; correct value ships in each variant already
  TEMPLATE-PINNED  one right answer for a newcomer; the shipped
                   example_workflows template should pin it and the knob can
                   stay for experts (no migration cost)
  DEMOTE-CANDIDATE never varied anywhere, no plausible first-user reason to
                   touch it; removing it would cost a full 4-part migration
                   (widgets_values, inputs descriptors, links, Python kwargs
                   -- see CLAUDE.md "REMOVING A WIDGET TOUCHES THREE THINGS",
                   plus the kwarg fallout) -- say whether the payoff justifies
                   that cost, honestly
  TOOLTIP-FIX      works, stays, but the tooltip fails a newcomer (missing,
                   jargon, or describes retired behaviour)

**Step 3b — THINK LIKE A HUMAN, per row (operator instruction, added
2026-08-28).** The counts are evidence, not the verdict. For every widget,
before you classify it, answer these as a person would:

  * **Would a real user ever touch this?** Not "could" -- WOULD. Picture the
    two actual users: the operator running dailies on his 5080, and a
    stranger who downloaded the template for a 4060. Name which of them
    would plausibly reach for this knob, and when. If the honest answer is
    "neither, ever", say that in plain words.
  * **Is it really worth touching?** Some knobs vary in the corpus only
    because a harness swept them, not because turning them ever made an
    episode better. A knob that was turned and never mattered is weaker than
    its count suggests.
  * **What does the user GAIN by seeing it?** A knob earns its place by
    giving a real choice with a understandable consequence. "Exposes an
    internal tuning number whose right value we already know" is not a gain,
    it is homework we assigned the user.
  * **Is the migration worth it AS A HUMAN CALL?** A DEMOTE-CANDIDATE that
    saves a newcomer one confusing decision but costs a 63-file migration
    may still be worth it before a v2 ship -- and may not. Say which way you
    would call it and why, in one sentence a person would actually say, not
    a hedge.

Rows where the numbers and the human judgment DISAGREE are the most valuable
rows in the whole census -- mark them.

**Step 4 — the newcomer walk.** Imagine the 4060 user who downloaded the
template: open the canonical in your mind node by node, in graph order, and
list the widgets they would encounter BEFORE their first successful render.
For each: what happens if they touch it ignorantly? A knob whose wrong value
silently ruins a 2-hour render ranks differently from one that fails loud in
10 seconds. Flag the silent-ruin knobs loudest -- those are the ease-of-use
hazards, whatever class they fall in.

## Ground rules

* **Per-machine dropdowns are settled** (operator ruling 2026-08-25):
  canonical stays put, per-machine picks live in variants. Do not propose
  moving those; classify them VARIANT-OWNED and move on.
* **No content guardrails** -- do not propose hiding a knob because its values
  could produce edgy output; that is not what this pass is for.
* **A working widget is a behavior surface.** You classify and recommend; the
  operator rules per row. Never present a DEMOTE-CANDIDATE as a done deal.
* **Numbers over adjectives.** "Varied in 3 of 2001 episodes (values: X, Y)"
  beats "rarely used". Show the scan logic so it can be re-run.
* Where a widget's value is resolved through env vars or profiles BEFORE the
  widget is read (the resolver pattern), say so -- the widget may already be
  subordinate to configuration, which strengthens TEMPLATE-PINNED.

## Output

1. **The full table**, one row per widget: node, name, default, canonical
   value, variant spread, corpus spread, headless patcher, CLASS,
   one-line reason.
2. **The newcomer walk** as a short ordered list with the silent-ruin flags.
3. **Top 10 recommendations** ranked by ease-of-use payoff per unit of
   migration cost, each with its class and the single sentence you would put
   in front of the operator.
4. **WHAT I COULD NOT CHECK** -- honest and unpadded.

The deliverable is the table. A ruling session with the operator walks it row
by row; your numbers are what make that session short.
