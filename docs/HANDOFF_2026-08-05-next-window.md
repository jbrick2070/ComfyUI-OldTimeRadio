# Handoff -- next coder window (written 2026-08-05)

**Branch:** `v2.0-alpha`. **HEAD at handoff:** `f240e835`.
**Suite:** 8700 passed / 131 skipped / 1 xfailed. **Bug Bible:** 17 passed.
Everything below is pushed and HEAD == origin.

Do NOT re-derive today's work. The four-round kibitz arc is DONE and its
artifacts are the record: `kibitz-runs/2026-08-05-next-queue/` (r1..r4, each
with `driver_anchor.md`, `codex.md`, `antigravity.md`; r1 also has an
independent Fable pass folded into `final.md`).

---

## START HERE -- the two things owed

### 1. LIVE PROOF (highest value, needs the GPU)

**Nothing shipped today has run a live render.** Six commits, suite-only.

The one that matters: `scifi_news` was measured **0-for-4** on batch v2 while
shakespeare went 12/12, public_domain 9/9, original 2/2 -- one failure burned
**45.1 minutes**. The fix is `016ad146`. It is unproven because the batch server
booted BEFORE the commit and held the old module in memory.

**Do:** reset per `CLAUDE.md` section 4 (selective CIM kill by CommandLine --
never a blanket python kill, it severs the MCP tooling), confirm port 8000 is
free and VRAM is at desktop baseline, boot fresh, run ONE `scifi_news` leg at
180 words / 2 characters through `workflows/otr_canonical.json`. Then update
PBUG-20260805-03, whose last line currently reads `live receipt: OWED`.

### 2. agy QA verdict

An agy-only QA pass was launched at handoff over the whole shipped set:
anchor at `kibitz-runs/2026-08-05-next-queue/qa/driver_anchor.md`, output will
land in `kibitz-runs/2026-08-05-shipped-qa/r4/antigravity.md`. Read it and
judge it -- ground every claim before folding anything in.

---

## What shipped today (do not re-litigate)

| commit | change |
|---|---|
| `adfd10a0` | captions show performance direction ON PURPOSE (operator ruling); TTS strips it independently; pinned by tests |
| `aeb6c227` | content-guardrail rip part 1 -- prompt clauses, brief re-rolls, episode-title check, `_repair_safety`, scrub blocking, **G9**, `validate_sfw` |
| `952c59b4` | rip part 2 -- freeze-cascade inline cleanup, codex + fable2 passes; `_otr_content_safety` at ZERO production refs |
| `016ad146` | P3 `cast_coverage` recoverable draft error + prompt invariant; 7 more SFW clauses out of the story packs |
| `7696222f` | r3/r4 must-fixes + PBUG-20260805-03 |
| `f240e835` | path guard -> whole-string classification; BOTH producer sanitizers removed |

**Operator rulings that are CLOSED:**
- Stage direction STAYS in `lines[].text`. It is load-bearing on the visual
  path -- still prompts (`otr_meta_brief_image_prompt.py:1313`), the i2v MOTION
  CLAUSE (`_otr_motion_clause.py:135`, under the standing directive at `:47`
  "the line drives the motion"), and the HUD print. Captions show it
  deliberately. Do not "fix" this; two tests pin it.
- Content guardrails are GONE on every lane, by directive. ~22 tests were
  INVERTED rather than deleted so a guardrail cannot creep back. If one fails,
  someone re-armed content enforcement -- that is an operator decision, not a bug.
- Five ledger fields keep a defined value on every path rather than vanishing
  with the passes that wrote them: `meta.ledger_cleanup.safety`,
  `ScrubResult.safety_violations`, `meta.freeze_block_class`,
  `meta.same_story_safety_cleanup`, `lane_meta.safety_cleanup`.

---

## UNBUILT -- three items, specs already settled by the panel

### Item 7 -- source citations leak into spoken dialogue (recommend next)
Two shipped episodes SPEAK a Project Gutenberg citation; one is the full
`pg1342.txt` URL. Because `lines[].text` also feeds the still prompt and the
motion clause, that URL is handed to two LLM prompt builders as if it were
stage direction -- it pollutes the VISUAL path, not just the screen.

**Root:** the interpreters ASK for attribution and hand the model the URL in the
same payload (`_otr_public_domain_sources.py:448` puts `source_url` in;
`:669` requests `"news_close_brief": "source attribution note"`). The writer and
composer then append that model-authored text unchanged AS DIALOGUE
(`OTR_LedgerScriptWriter.py:4892`, `:5489`; `_otr_line_composer.py:1240-1287`).
Shakespeare has the same shape (`_otr_shakespeare_sources.py:583-629`).

**Trap, driver-verified:** `meta["provenance_coda_line"]` -- the deterministic
alternative -- is WRITTEN at `OTR_LedgerScriptWriter.py:3595` but has ZERO
readers. Grep confirms exactly two mentions in the tree: that write, and the
module docstring at `_otr_provenance.py:19`. Nothing consumes it.

**CORRECTED 2026-08-05 PM (the original text of this paragraph was wrong, and
building to it would have produced a no-op change).** It claimed the write is
"gated behind `defaults.provenance_normalize` which is False for every bank"
and "has never run". Both halves are false:

- `provenance_normalize` is **`true`** for `public_domain` and `shakespeare`
  (`nodes/story_packs/banks.json`) -- the exact two lanes that leak the
  citation. They opted in 2026-08-04.
- It is **pinned by a test**: `tests/test_provenance_v4.py:119`
  `_PROVENANCE_OPTED_IN = frozenset({"shakespeare", "public_domain"})`, asserted
  per-bank at `:128-135` so the flag cannot move silently.

So the coda IS composed and stamped on both affected lanes today. **The flag
does not need enabling -- only a CONSUMER is missing.** The fix is a wire-up,
not a wire-up plus a switch.

Two comments still assert the old state and should be corrected in the same
change, since they are what misled this spec: `OTR_LedgerScriptWriter.py:3584-3585`
("Default False -> key absent -> inert for every current bank") and
`_otr_ledger_freeze.py:713` ("inert for every current bank").

**Do NOT** bump `NORMALIZATION_VERSION` -- wrong boundary, source bytes are
unchanged (Codex r4 overruled agy r2 here). Bump the interpreter PROMPT versions
instead (`_otr_public_domain_sources.py:42`, `_otr_shakespeare_sources.py:42`).
Add an end-to-end regression: no URL or licence identifier in `lines[].text`,
printed credits unchanged.

### Item 1 -- 1,090 cast rows claim a non-commercial model is commercially clean
`eng_indextts2.py:55` says `commercial_clean = False` (bilibili non-commercial);
all 40 bank rows say `true`; `cast_lock.py` trusts the bank row.

The row flag is the CLIP's licence, the engine flag is the MODEL's -- genuinely
different facts, both already in the right layers. Stamp the JOIN. **Do NOT edit
the 40 bank rows** (`otr_dl_indextts2_refs.py:11-17` documents them as clip
provenance; the ingest mints three rows across three engines from one PD clip).

**Must heal ATOMICALLY** or it creates the defect it fixes: the stamp
(`cast_lock.py:742`), the `gated` counter (`:575/:614/:661/:670`), AND the three
report strings (`:578/:618/:673`) -- otherwise the report prints `clean=True`
beside a ledger saying `False`. Resolve ONE profile by `(role, engine)` --
role-scoped, not engine-name-scoped. **Enforcement stays OFF.** Prospective-only
for the 1,090 frozen ledgers.

### Item 2 -- a terminal freeze gate that has never read a populated field
`find_scene_coherence_issues` reads `lines[].scene_id`; the `scifi_news` lane
writes `beats[].scene_id`. 55 ledgers assert the check, 0 carry the field, 55
pass. Nothing in `nodes/` writes `lines[].scene_id` on ANY lane -- the check
never had a producer.

Join per line: `beat_id` -> beat -> `scene_id` -> declared scene. Add a VACUITY
refusal (an armed gate that examined zero linkages FAILS -- that is how this
survived 55 episodes). **Split request from verdict:** keep a
configuration-derived `scene_coherence_required`, and write
`{required, checked, verdict, issues}` into `report.info` -- `run_gap_audit` is
READ-ONLY (`_otr_ledger_freeze.py:664-698`), so the gate must not mutate the
ledger; the phase wrappers already persist the report. Measure OFFLINE over the
published corpus first, then arm in ONE change -- no intermediate flag-off ship.
Replace the stale hard-coded bank list at `tests/test_scene_guard_v4.py:89-99`
with registry-derived coverage (it omits `scifi_news`, the one bank that enables
the flag).

---

## Open questions for the operator
- A cycle cap on the fresh-candidate loop: agy wants one, Codex argued against.
  I ruled with Codex because the `cast_coverage` invariant is provably
  satisfiable (cast <= 7, beats <= 12) and the unbounded loop pre-dates it.
  Reopen only on evidence.
- Carried from the prior arc: ARIEL/PUCK roster supplement (40/42 vs 42/42),
  tier floor 2 vs 3, the reference A/B eyeball.

## Gotchas that cost time today
- The conftest known-fail guard SUPPRESSES tracebacks and hard-exits 2. To see a
  real failure, reproduce it in a temp script under `tmp/` and run it directly.
- A test file can pass only because a SIBLING test module put `nodes/` on
  `sys.path` first. `tests/test_otr_captions.py` had that; fixed there, may exist
  elsewhere.
- Editing a name out of a `from X import (a, b, c)` block can leave an EMPTY
  import block that fails to parse. Bit me twice in the flat-load fallbacks.
