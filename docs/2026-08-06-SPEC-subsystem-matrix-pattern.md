# SPEC -- one matrix per plug-and-play module, starting with video

**Date:** 2026-08-06. **HEAD:** `d58eb0e9` on `v2.0-alpha`.
**Driver:** Claude (Cowork), CODER window. **Design:** Fable, grounded pass.
**Status:** DESIGN ACCEPTED, NOT BUILT. Owes a `kibitz-plugin:kibitz` arc
(Codex + Antigravity) before any code.

## 0. THE OPERATOR'S ASK, ASSEMBLED FROM ONE SESSION

1. ONE matrix doc as the source of truth per module, not three overlapping ones.
2. Better format, more columns -- **so a reader understands how each video model
   path DIFFERS**. Comparison first.
3. Registered in `CLAUDE.md` as a standing source of truth that binds any turn
   touching video models or maths.
4. Built for CHURN: Flux3 open weights and a lowest-lift MiniMax are the
   near-term adds.
5. **Each plug-and-play module gets its own inventory doc** -- video, still, TTS,
   possibly LLM recipes, maybe source banks and visual styles.
6. **Still logic and clip maths are the payload** -- they are what a user adding
   their own visual model gets wrong, and what "extending OTR" must teach.

## 1. THE CORE DECISION -- templated prose fragments

The tension: a generated doc cannot hold judgment; a hand-written doc cannot be
trusted with numbers (proven twice this week). The operator wants ONE doc.

**Resolution: the doc stays 100% GENERATED, and the generator gains a SECOND
INPUT** -- hand-authored prose fragments (`tools/engine_matrix_prose.md`, one
file of `<!-- section: ... -->` blocks) in which **every load-bearing number is a
placeholder** resolved from the live registry at generation time:

```
A {ref_beat_seconds} s beat becomes {humo.seg_count} clips on `humo` but
{wan_ti2v.seg_count} chained clips on `wan_8gb`, because humo's per-call
ceiling is {humo.max_seconds} s and it cannot chain.
```

* Judgment stays HUMAN -- the fork narrative, "what this lane is bad at", the
  still-logic analysis. A generator never pretends to produce it.
* Numbers CANNOT ROT -- they are the registry's, resolved at build. The exact
  failure that retired the 08-02 doc (hand-typed 3 and 10 segments against a
  live 5) becomes structurally impossible.
* The drift gate stays BYTE-EXACT and WHOLE-FILE. No protected-marker parsing,
  no partial comparison.

**Rejected, with reason:** hand-owned regions inside the generated file, because
its header swears "GENERATED FILE. Do not edit by hand" (enforced by
`tests/test_engine_matrix_doc.py`) -- half the file would become a lie. Also
rejected: a "no digits in the prose" test, which would forbid the exact sentence
the reader came for.

### 1.1 Three generator guards

1. `REQUIRED_PROSE_SECTIONS` -- a missing or empty fragment fails `render()`.
2. **Digit-lint on the FRAGMENT SOURCE, not the rendered page**: a literal
   integer >= 3 digits, or a token matching `\d+\s*(frames|fps|s\b|MB|GB|x\d)`,
   fails the suite unless the line carries `<!-- literal-ok: reason -->`.
3. An unresolvable placeholder KeyErrors at build. **This is the churn property:
   retiring an engine breaks every sentence that mentions it, by name, in the
   same commit.**

**Header must state loudly:** a hand edit to the generated doc is not merged on
regeneration, it is silently DESTROYED (`DOC.write_text(fresh)`). Prose goes in
the fragments; numbers go in the adapters.

### 1.2 The one judgment surface that moves onto the engine class

A required one-line `doc_purpose` attribute ("Talking-head lane; the only local
lip-sync"), reviewed whenever the adapter is edited, emitted as the "what it is
for" cell. Long-form judgment stays in fragments.

## 2. THE READER-FIRST PAGE

Section order -- comparison first, receipts last:

1. **How to pick a lane** (prose fragment) -- the five forks below.
2. **The comparison table**, grouped by FAMILY, seconds-first.
3. **Per-family detail** -- ladders, quantum, still plans, canvas authority.
4. **What each lane costs** -- VRAM admission enforced/unenforced, cloud
   delivered-fps and cost basis (today: honest "unmeasured/undeclared").
5. **Caps and the evidence behind them** -- the existing honesty receipts. This
   section is the doc's immune system; keep it.
6. **Adding a new engine** -- the checklist-as-columns (section 4).
7. **Unfilled cells** -- the grandfathered-holes allowlist. It only shrinks.
8. Counts + generation stamp.

### 2.1 The five forks, each with its cost stated

* **CHAIN vs JUMP.** Chain = the next clip's first frame IS the previous clip's
  real last frame; motion continues through the cut. Earned via
  `strict_first_frame`, never assumed. Jump = an edit -- and on a
  still-consuming cloud lane, a freshly minted still per cut, **so a character's
  face can change mid-beat**. Local lanes never re-mint.
* **AUDIO-DRIVEN vs NOT.** Only `audio_driven_face` lip-syncs and therefore
  cares about per-segment audio slicing; `audio_conditioned_video` reacts
  without lip-sync; everything else is silent motion.
* **BOUNDED vs UNBOUNDED.** Bounded engines split beats; unbounded lanes
  (stills, viz, mesh) never do.
* **LOCAL vs CLOUD.** VRAM and minutes, versus dollars, a still-continuity
  defect, and cannot run on this box today (offline-first).
* **PORTRAIT vs WIDE, AND WHO PICKS THE CANVAS.** Self-sizing
  (`_native_dims`), declared, or driver default -- the `wan_i2v` unclaimed-
  landscape lesson.

### 2.2 The primary table shape (real values, live registry)

The headline column is **"a 17.7 s beat becomes"** -- it answers the operator's
"why 3 clips here and 5 there" in-row. It is the existing multi-clip computation
promoted from a back section into the comparison table.

**TALKING HEADS -- audio-driven face. The only lip-synced lanes. One shared
portrait per beat: identity holds, pose resets at cuts.**

| menu name | runs on | one call | a 17.7 s beat becomes | at the seam | canvas |
|---|---|---|---|---|---|
| `humo` | local, heavy (14B) | 1.3-3.9 s | 5 clips, jump cuts | pose snaps back; same face | 480x832 portrait, self-sized |
| `humo_1.7B` | local, light | 1.3-7.1 s | 3 clips, jump cuts | pose snaps back; same face | 480x832 portrait, self-sized |

**SCENE MOVERS -- image-to-video b-roll. Silent; animate the beat's still.**

| menu name | runs on | one call | a 17.7 s beat becomes | at the seam | canvas |
|---|---|---|---|---|---|
| `wan_8gb` | local, 8GB tier | 0.7-7.1 s | 3 clips, **chained** | motion continues through the cut | 832x480, declared |
| `ltx_8gb` | local, 8GB tier | 0.4-6.4 s | 3 clips, **chained** | motion continues through the cut | 512x288, declared |

**AUDIO-REACTIVE -- moves with the sound, no lip-sync.**

| menu name | runs on | one call | a 17.7 s beat becomes | at the seam | canvas |
|---|---|---|---|---|---|
| `ltx23_16gb_audio_in` | local, 16GB | 0.4-**19.9 s** | **1 clip -- no seams** | n/a | 832x480 / 512x288 |

**CLOUD -- provider-side; dollars not VRAM; re-mints a still at every cut.**

| menu name | runs on | one call | a 17.7 s beat becomes | at the seam | resolution |
|---|---|---|---|---|---|
| `google_veo_video` | cloud, BYO key | menu: 4/6/8 s | 3 clips, jump, **2 re-minted stills** | cut; face may change per cut | provider 720p |

## 3. STILL LOGIC AND CLIP MATHS -- the teaching spine

The operator: these are what a user adding their own visual model gets wrong.
The guide TEACHES them, the matrix TABULATES them per lane, the preflight GATES
them. A reader must be able to answer, for their own new model: **what still
does it need, does it re-mint, and how will my beat be cut up?**

**STILL LOGIC.** Does the lane consume a scene still or a portrait? Does it
re-mint PER SEGMENT or share one across the beat? Locally nothing re-mints --
chain engines overwrite `asset_refs["init_image"]` with the predecessor's real
terminal frame, and the humo variants are JUMP but consume no scene still, so
identity holds by construction. Eleven of twelve CLOUD engines re-mint per cut,
via a dispatcher clone that deliberately drops the fixed seed. **That reasoning
was written for a SCENE bookend and now governs CHARACTER beats.** A new visual
model that consumes a still and jump-cuts inherits that defect on day one.

**CLIP MATHS.** The frame ladder (min/max/quantum) and what falls off it; how a
beat is partitioned; chain vs jump on screen; the seam `drop_head` -- exactly 1
per chained successor, so **a beat RENDERS MORE THAN IT DELIVERS**; tail trims;
and that "renders 507 to show 442" is NORMAL, not padding. A new author
declaring the wrong contract gets silently wrong segmentation.

## 4. THE NEW-ENGINE CHECKLIST -- columns are gates

A blank cell must fail the suite BY NAME. Today most do not.

| # | column | declaration surface | today |
|---|---|---|---|
| 1 | registered + CAPABILITIES row | `@register` + import + registry row | LOUD |
| 2 | family | `family` attr | assert at import |
| 3 | inputs / role fit | `required_inputs` | **SILENT** -- falls back to legacy roles; with `roles` also empty the engine vanishes from every role listing while staying in the dropdown |
| 4 | frame contract | `frame_contract()` | LOUD by name in the roster test |
| 5 | canvas | `render_canvas` | **SILENT** -- inherits 1472x832 "unclaimed" |
| 6 | still plan | `still_plan` | SILENT -- prints "none" |
| 7 | VRAM cost row + qualification | `FRAME_COST_MODEL` + `QUALIFIED_COST_ROWS` | silent-with-receipt; **`QUALIFIED_COST_ROWS` is an EMPTY frozenset today**, so admission is unenforced for every engine |
| 8 | planning-cap decision | `PLANNING_CAP_ENGINES` membership | new cell; "n/a" allowed WITH a reason |
| 9 | public menu id + label | `public_engines.py` | new cell |
| 10 | purpose one-liner | new `doc_purpose` attr | generator errors if absent |
| 11 | audio-sync class | **no surface -- must be added** | -- |
| 12 | provider side / delivered fps / cost basis | **no surface -- must be added** | -- |
| 13 | license identity | only a `commercial_clean` bool | insufficient |

### 4.1 What breaks today if someone registers an engine and fills nothing in

**Verified 2026-08-06.** Three loud gates, two fully silent holes, one
silent-with-note, and -- worst -- two generator surfaces that produce a WRONG
cell for a new cloud engine rather than a blank one:

* **Frame contract has FOUR silent paths to `SINGLE_ONLY`** (driver correction
  to the design, which said three): engine is None, declaration is None, the
  declaration RAISES, or it returns the wrong type. All four return the same
  value with no signal. The roster test catches it by name; production without
  CI plans an open ladder and fails after weights load, or OOMs.
* **Canvas silently inherits 1472x832** -- the same dead channel that cost
  `wan_8gb` a 268-minute leg.
* **Role fit silently vanishes** the engine from per-role listings.
* Cloud `resolution` is a name-prefix if-chain and `side` is detected by name
  prefix, so a new provider gets a wrong cell.

## 5. CHURN -- Flux3 and MiniMax

**Flux3 (local, open weights)** fills columns 1-10 with surfaces that exist
today. Missing: `doc_purpose` (trivial), a license identity richer than a
`commercial_clean` bool, and a QUALIFIED cost row -- the surface exists but entry
requires a real lifecycle measurement that no bench may substitute for. **Flux3
therefore ships VRAM-admission-unenforced and its row says so, honestly.**

**MiniMax (cloud, lowest-lift)** needs FOUR declaration surfaces built first:
`provider_resolution`, `provider_side`, `delivered_fps` (or an explicit
`unverified` sentinel), and an informational `cost_basis` string. It also joins
the per-cut face-drift class on day one, which its row must say at birth. And
the seconds-for-frames trap is armed: its menu must be declared in FRAMES at its
native fps, with `allow_tail_trim=True` mandatory for any menu.

## 6. THE PATTERN, AND WHICH MODULES EARN ONE

**Do not extract a shared `tools/_matrix_lib.py` yet** -- one instance is not a
pattern, and extracting from one is guessing. Extract when the SECOND lands.
Roughly 35-40% of the current generator is reusable scaffolding (`--check`,
diff, header convention, the drift-gate test quartet, the evidence-citation
scanner); ~60% is video semantics.

Ranking by value-per-effort:

1. **Image / still engines -- BUILD NEXT.** `nodes/_otr_image_engines/registry.py`
   exists and shares `nodes/_otr_shared/engine_registry_base.py` with video
   (both verified), so the registry walk generalizes for free. Highest
   defect-adjacency: the unresolved face-drift dispute lives at the image
   dispatcher.
2. **Voice / TTS -- SECOND, but it forces a prerequisite.** Live correctness
   defects here (the operator's own carve-out). But the catalogue is scattered
   across four modules -- **the scatter IS the hazard** -- so it needs
   consolidating into one walkable registry first. That is worth doing on its
   own merits; the matrix is its receipt.
3. **LLM recipes -- DEFER.** Story quality is done by directive; a recipes
   matrix mostly documents routing and cost, which changes rarely.
4. **Visual styles -- DEFER.** Already data-driven; a matrix would restate the
   data files. (No defect history found -- UNVERIFIED, not swept.)
5. **Source banks -- DO NOT BUILD.** They already have the right mechanism: a
   JSON manifest schema, which is their drift gate. There is no "which bank do I
   pick and why" comparison a matrix would answer.

## 7. THE AUTHORING LANE -- clone the one that works

Source banks already have a complete plug-and-play lane, 1,238 lines:
`EXTENDING_OTR.md` (contract, 322) + `SOURCE_BANK_GUIDE.md` (playbook, 478) +
`SOURCE_BANK_PREFLIGHT.md` (gated acceptance with per-item evidence, a hashed
receipt, and a teardown protocol, 438).

**Video, still, TTS, LLM recipes and visual styles have NO equivalent.**
`EXTENDING_OTR.md` mentions engines three times in 322 lines.

Build the video lane as the SECOND INSTANCE of that proven triad, reusing its
naming so a reader who has added a bank recognises the shape:

* **matrix** = the inventory (what exists, how the paths differ)
* **guide** = the how (still logic + clip maths -- section 3)
* **preflight** = the proof a new arrival is admissible

**The symmetry to exploit:** the matrix's columns and the preflight's gates are
the same list seen from two directions -- **a blank matrix cell IS a failed
preflight gate.** Write the list once; enforce it from both ends.

## 8. FILENAME -- and a cost correction

Design proposed `docs/VIDEO_ENGINE_MATRIX.md`: undated, because it is a living
record, with the `VIDEO_` prefix load-bearing once siblings exist
(`IMAGE_ENGINE_MATRIX.md`, `TTS_MATRIX.md`).

**Driver correction: the rename is more expensive than the design estimated.**
It said 28 referencing files; the real count is **92** (excluding stale
worktrees). Recommendation: **defer the rename** to its own mechanical commit,
or accept the churn deliberately -- do not bundle it with the generator change,
because a 92-file rename inside a behaviour change is how a review stops being
readable. The evidence scanner is a partial safety net: an adapter still citing
the old path surfaces as MISSING at the next generation.

## 9. THE STANDING RULE FOR `CLAUDE.md` (proposed text, not yet added)

```
## 0B. SUBSYSTEM MATRICES ARE THE ONLY NUMBERS DOCS (operator directive
##     2026-08-06 -- hard)
A pluggable subsystem with a generated matrix has ONE source of truth for its
per-model numbers and comparisons. Today that is the video engine matrix
(generated by `tools/engine_matrix.py`, drift-gated by `--check` +
`tests/test_engine_matrix_doc.py`); siblings (image, TTS) adopt the same shape
when they land, and this rule covers them the day they do.
- ANY turn that touches a matrixed subsystem -- a video engine, a frame
  contract, coverage maths, a canvas, a still plan, a cost row, a cloud lane --
  MUST read that matrix FIRST and MUST update it IN THE SAME COMMIT. "Update"
  means exactly: edit the adapters (numbers) or the prose fragments under
  tools/ (judgment), rerun the generator, and leave `--check` green. NEVER a
  hand edit to the generated doc -- regeneration DESTROYS hand edits without
  warning, by design.
- A hand-maintained doc must NEVER re-type a number a matrix owns. Cite the row.
- A NEW engine ships with every matrix column filled in the same change; a
  blank cell fails the suite by name. The Unfilled-cells allowlist grandfathers
  only the holes listed in it today, and it only SHRINKS.
- Where a dated doc and a matrix disagree, the matrix is right and the dated
  doc gets a SUPERSEDED stamp in that same session.
```

## 10. BUILD ORDER (proposed; owes a kibitz arc first)

1. `doc_purpose` + the four missing declaration surfaces; convert the two silent
   defaults (role-vanish, unclaimed canvas) into named CI failures, grandfathering
   today's holes.
2. Prose fragments + placeholder substitution + the three guards.
3. Re-order the page reader-first; promote the "a 17.7 s beat becomes" column.
4. The `CLAUDE.md` 0B rule.
5. The video guide + preflight, cloned from the source-bank triad.
6. Image-engine matrix as the second instance; extract `_matrix_lib` then.
7. Rename, as its own mechanical commit, if taken at all.

## 11. RECEIPTS OWED

* `tools/engine_matrix.py --check` green; the drift-gate suite test green.
* Full suite (baseline **8836 passed / 131 skipped / 1 xfailed** at `e499b7fc`)
  + Bug Bible 17.
* `workflows/otr_canonical.json` byte-unchanged -- nothing here touches
  `INPUT_TYPES`, widgets, node classes or links.
* STATIC throughout: nothing in this spec may enter `PROD_BUG_LOG.md`.
