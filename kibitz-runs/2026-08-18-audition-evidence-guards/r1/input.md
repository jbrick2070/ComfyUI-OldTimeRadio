# DRIVER ANCHOR -- audition evidence guards (Bible 12.111 verify step 3)

Written 2026-08-18 by the Cowork driver (Claude) BEFORE any panel round, from the
real Windows files at HEAD `345ea230` on `v2.0-alpha`. Every line number and hash
below was read off disk in this window, not recalled. The panel proposes; the
driver disposes, and every panel claim gets checked against these same files.

---

## 1. WHAT THE QUEUE ASKED FOR, AND WHY IT IS BIGGER THAN THAT

`docs/GO_FORWARD_PLAN.md` (queue row):

> cross-engine audition overwrite guard | OPEN -- `otr_lemmy_cross_engine_audition.py`
> has no guard and its manifest is cited by sha256 in THREE provisional records.
> Bible 12.111 verify step 3. Needs an `--out-dir` flag plus a citation check; it
> is deliberately resumable, so do NOT copy the production audition's blanket
> refusal.

Bible `12.111` verify step **3** is the instruction that widened this:

> Grep the repo for every writer of the cited filename and check each one has the
> guard. The defect class travels: fixing the instrument that bit you while its
> sibling stays unguarded is how the same outage recurs under a different name.

I ran that grep. **There are three evidence-writing instruments, not one, and two
of them are defective.** That is the finding this anchor exists to put in front of
the panel.

---

## 2. GROUND TRUTH -- THE THREE INSTRUMENTS, READ OFF DISK

| instrument | `--out-dir` | guard today | verdict |
|---|---|---|---|
| `scripts/otr_lemmy_production_audition.py` | yes (`:223`) | refuses ANY non-empty output dir **and** its `_KEY` sibling (`:246-254`) | **CORRECT** -- this is the post-12.111 reference implementation |
| `scripts/otr_g1_lemmy_audition.py` | yes (`:231`) | refuses **only when `MANIFEST.json` exists** (`:253-254`), with an `--overwrite` escape; `_KEY_DIR` is created `exist_ok=True` at `:156` and is **not guarded at all** | **PARTIAL GUARD -- the exact trap 12.111 names** |
| `scripts/otr_lemmy_cross_engine_audition.py` | **no** -- `_OUT_DIR` is a module constant (`:57-59`) | **none whatsoever** | **UNGUARDED** |

### 2A. What the cross-engine script actually does today

* `_OUT_DIR` is hard-coded to `…\output\otr\episodes\lemmy_cross_engine` at
  module scope (`:57-59`). `main()` (`:399-425`) offers `--render`, `--engine`,
  `--allow-resident-server`. **No output-path flag exists.**
* `render()` (`:281`) calls `os.makedirs(_OUT_DIR, exist_ok=True)` and proceeds.
* `_write_clip()` (`:216`) writes `path + ".part"` then `os.replace(part, path)`
  -- atomic, and an **unconditional overwrite** of any clip already there.
* `_save_manifest()` (`:267-277`) rewrites `MANIFEST.json` after **every clip**.

So a second `--render` silently destroys all eight clips and the manifest.

### 2B. THE RESUMABILITY IS REAL AND DESIGNED -- this is the crux

The blanket refusal that is correct for the production audition would **break a
documented recovery path** here. Evidence, all from the file itself:

* `_load_manifest()` (`:238-265`) deliberately **reads and merges** an existing
  manifest when `lines_version` matches, and only starts fresh when it does not.
* `_save_manifest()`'s own docstring: *"Called after EVERY clip, so a batch that
  dies at engine three keeps what engines one and two earned."*
* `render()` (`:333-337`) exits **1** printing
  `"INCOMPLETE: %s -- re-run those with --engine <name>"`.
* The module docstring advertises
  `python scripts/otr_lemmy_cross_engine_audition.py --render --engine dia`.

**The script's own recovery instruction is "re-run into this same directory."** A
guard that refuses any non-empty directory makes the script tell the operator to
do something it then refuses to do. That is the design fork below.

### 2C. What is cited, and by whom -- ALL VERIFIED INTACT TODAY

`config/cast_pools.py` carries **three** provisional receipts citing this
campaign -- kokoro (`:1084`), chatterbox (`:1114`), dia (`:1140`) -- each naming
`otr/episodes/lemmy_cross_engine/MANIFEST.json` plus its own two clips. I
re-hashed all seven artifacts (12.111 verify step 4):

```
MANIFEST.json             ac55c90ce8325705862d6f8fbdaaadaf4153681444363dca4572f5583d4b2762  MATCH
kokoro_neutral.wav        996d9e005e49ce6fc217c5df6b964e733c20a0cb2c43763240aff1ce9c6d230d  MATCH
kokoro_emotional.wav      3902f5354d96fc75f61709122bb1ce11c64e530af3abe36a51cff726847e46ab  MATCH
chatterbox_neutral.wav    4ac0a455825b77c4f4026b8ce0b03faa84935fefe5fb84d0a1dba1c454d64a8d  MATCH
chatterbox_emotional.wav  fbd9a72962160b71fc479dd8fda3dfc772506ab5b0ffc320c15973aceb576437  MATCH
dia_neutral.wav           b576a561c2fb9c97d2cd8774f066537962582eccd9bf9fbbd334649ca74ba355  MATCH
dia_emotional.wav         840acfde18b31e1b8fc0cc78e07c564d891a9a990621619e4a87742216830a72  MATCH
```

And G1's, cited at `config/cast_pools.py:995` inside `superseded_native_routes`:

```
g1_lemmy_test_a/MANIFEST.json  34dd4c9d8b3404814d1d7d0703d8f0e8f71893a62455169eae67b8199c90da67  MATCH
```

**Nothing has rotted yet.** This work is PREVENTIVE. Say so plainly in any
write-up -- an honest "we caught it before it bit" is worth more than an implied
save.

### 2D. Why G1's partial guard is worse than it looks

G1's manifest is cited inside `superseded_native_routes`, whose own comment
(`config/cast_pools.py:985-988`) reads: *"a superseded qualification must remain
auditable without ever being selectable. Its cited manifest still hashes to the
value below, so the August audition can still be re-verified byte for byte."*
That is a **permanent archival promise**, and the guard protecting it has three
holes:

1. It checks `MANIFEST.json` only. A directory holding the six WAVs but no
   manifest passes -- and the WAVs are the audio a verdict was formed on.
2. `_KEY_DIR` (`:68`, `…/g1_lemmy_test_a_KEY`, currently holding `KEY.json`) is
   created `exist_ok=True` at `:156` and written unconditionally. **The KEY is
   what says which blinded arm the operator approved.** Losing it makes the
   verdict unattributable while every other file still looks fine.
3. `--overwrite` (`:236`) turns the whole guard off in one flag, on a directory
   the script's own comment calls cited evidence.

12.111's `cause` section describes hole 2 almost verbatim. G1 is the sibling that
kept the defect.

---

## 3. THE DESIGN FORK -- this is what the panel is for

**Question: what exactly should the cross-engine guard refuse, given the script is
deliberately resumable?** More than one answer is defensible, which is why this
gets an arc rather than a straight edit.

* **A -- Cite-aware refusal.** Import the provisional receipts from
  `config/cast_pools.py`; refuse to write any artifact whose path is cited.
  Resuming into an *uncited* partial directory still works.
  *Against:* the instrument now imports the record it is evidence for; a receipt
  written after a render retroactively locks the directory (arguably correct, but
  it is a coupling and it makes the refusal depend on import success).
* **B -- Per-file refusal.** Refuse to overwrite any clip that already exists;
  permit writing new ones. Resumption is naturally safe -- a failed engine left
  no clip.
  *Against:* does not protect `MANIFEST.json`, which is rewritten on every save
  **by design** -- and the manifest is the artifact all three receipts cite.
* **C -- Explicit resume flag.** Default to the production shape (refuse
  non-empty); `--resume` opts into merging.
  *Against:* makes the script's own documented recovery path opt-in. In its
  favour: the polarity is right -- the *destructive* case becomes the one you ask
  for, which is what 12.111's "no `--force` that defaults on" is really about.
* **D -- Combination.** `--out-dir`, plus a cite-aware refusal that is **never**
  overridable, plus `--resume` for the uncited-partial case.

**My going-in position, for the panel to break:** **D**, with the cite check
reading the receipts and failing CLOSED if it cannot read them. Rationale: the
cited-vs-uncited distinction is the one that actually maps onto the harm, and
resumability only ever needs to apply to a directory no record has claimed yet.
An unguarded import failure that silently degrades to "nothing is cited" would
reintroduce the whole bug, so it must raise.

**Second question for the panel:** does G1 get fixed in this same change? My
position is **yes** -- 12.111 verify step 3 exists precisely to stop a one-sibling
fix, and G1 guards the more valuable evidence. The counter-argument worth hearing
is scope: G1 is a *frozen* instrument and touching it risks the thing it protects.

**Third question:** `--overwrite` on G1 -- delete it, or keep it and widen the
guard beneath it? 12.111 forbids "a `--force` that defaults on"; G1's defaults
off, so it is not literally forbidden. But a single flag that disables protection
on evidence carrying a permanent archival promise is hard to justify when
`--out-dir` already gives the legitimate use a safe path.

---

## 4. WHAT IS EXPLICITLY OUT OF SCOPE

* **No re-render.** The evidence is intact; re-rendering to test a guard would
  destroy the thing under discussion. Guard tests use temp dirs.
* **No verdict, no qualification.** The three provisional routes stay
  `rendered_pending_listen`. Nothing here promotes them.
* **`LISTEN.html` / `DECISIONS.json`.** `scripts/otr_lemmy_listen_page.py:334-336`
  already leaves `DECISIONS.json` untouched when it exists; `LISTEN.html` is a
  derived view and is cited by no hash. Neither is in scope, but flag it if you
  disagree.
* **Not a workflow change.** These are standalone scripts; nothing here touches
  `workflows/otr_canonical.json`. State it explicitly at r3 rather than assuming.

---

## 5. ACCEPTANCE -- what "done" must prove (12.111 verify steps 1-4)

1. Running an instrument twice against the SAME directory: second run exits
   non-zero and writes nothing. **Assert on exit code and file mtimes, not on
   stdout text.**
2. Refusal covers a directory holding only siblings (clips, no manifest), and
   separately a `_KEY` directory existing while the primary does not.
3. Every writer of a cited filename has the guard -- all three instruments, with
   a test that fails if a fourth appears unguarded.
4. A standing re-hash check over every artifact a record cites, so a rotted
   citation is caught cheaply rather than discovered.

Step 5 (byte-identical-under-changed-code) is **not applicable without a
re-render**, which section 4 rules out. Say that rather than claiming coverage.

Plus the house gates: full suite green, Bug Bible regression green, Sonnet 5 QA
on the finished diff BEFORE the push.

---

## 6. PROVENANCE OF THIS CAMPAIGN

Codex is **not installed on this box** (`codex` is not a recognised command);
Antigravity answers `agy models` rc=0 and is live. Per the operator's 2026-08-17
substitution directive a missing lane never blocks an arc -- the seat gets filled
and the roster is stated honestly. The roster actually used will be recorded at
the close of this campaign, and a campaign a reviewer short will be described as
a campaign a reviewer short, never as a full arc.
