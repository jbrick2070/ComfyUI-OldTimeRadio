# r2 JUDGMENT -- coding plan / implementability

Reviewer seats this round: **two Cowork subagents (Opus, Sonnet)**, substituted per
the operator's 2026-08-17 directive after Antigravity returned
`RESOURCE_EXHAUSTED (429)` on r2 -- a confirmed provider quota block, not the
`KIBITZ_AGY_PRINT_TIMEOUT` failure mode. Codex is not installed on this box.
Driver (Claude) remains sole judge; every claim below was checked against the
real files before acceptance.

---

## ACCEPTED -- verified

### B1. The guard is in the wrong place and costs four engine loads. (Opus MUST-FIX 3)

**CONFIRMED.** `main()` calls `preflight(engines)` at `:377`, which calls
`adapter.load()` for **every** engine at `:193` -- and preflight's own docstring
(`:170-174`) says that is the expensive path: *"it spawns the sidecar worker,
opens the venv, reads the weights."* A guard placed beside
`os.makedirs(_OUT_DIR, exist_ok=True)` at `:282`, inside `render()`, fires only
**after** all four engines have loaded. The operator waits through the whole
expensive path to be told the directory is protected.

Both correct siblings already guard before the expensive work: G1 at `:253-254`
before `preflight()` at `:266`; production at `:246-254`.

**Accepted:** the guard runs in `main()` **before** `preflight()`.

### B2. The manifest carries a build-provenance field that is structurally always empty. (Opus MUST-FIX 2)

**CONFIRMED, and it is worse than "not populated" -- it is not populatable.**

* Driver read the live cited manifest: **all four engine rows carry
  `engine_impl_version: ''`.**
* `scripts/otr_lemmy_cross_engine_audition.py:292` reads
  `getattr(adapter, "impl_version", "")`.
* `grep -rn "impl_version" nodes/_otr_audio_engines/` returns **nothing**. No
  adapter defines it. The field has never held a value and never will.
* `RUNTIME_FINGERPRINT_SOURCES` (`nodes/_otr_voice_route.py:158-165`) has exactly
  **one** key, `indextts2`. The other four engines have no recipe, and
  `live_engine_impl_version` returns `""` for them by design (`:201-220`) --
  *"silence, not a guess"*, pinned by
  `tests/test_voice_identity_fix.py:666-673`.

This is an **evidence-shaped field that does nothing** -- exactly what this repo
refuses everywhere else (the 08-18 window deliberately declined to add
`emo_mass_cap` to a profile for this precise reason). Bible 12.111's `fix`
section demands *"Put the runtime INSIDE the artifact."*

**Accepted, but SCOPED DOWN, and the reasoning is stated so it is not re-litigated:**
Opus's full remedy -- author fingerprint recipes for bark/kokoro/chatterbox/dia
and refuse when empty -- is real design work (which source files constitute each
engine's recipe is a judgment call), and Opus itself labelled that scoping
UNVERIFIED. Building it here would also **brick the instrument for all four
engines** until the recipes exist.

The proportionate fix is to **stop writing a field that cannot be filled and make
the gap visible**: ask `ROUTE.live_engine_impl_version(engine)`, record the real
fingerprint when a recipe exists, and record an explicit
*"no fingerprint recipe registered for this engine"* when it does not. That
touches neither the function nor its bark test, makes the manifest honest, and
turns an invisible hole into a visible one. **The four missing recipes are filed
as a separate queue row** -- they deserve their own consideration, not a silent
ride on a guard change.

**Load-bearing note:** because `--resume` is withdrawn (see B5), nothing *decides*
anything from this field. It is an honesty fix, not a safety-critical one.

### B3. `--overwrite` has ZERO callers -- remove it, do not narrow it. (Opus Q3)

**CONFIRMED by independent grep.** The only hits for `--overwrite` are its own
definition (`scripts/otr_g1_lemmy_audition.py:237`) and its own help text
(`:260`). The `scripts/otr_h3_mime_runner.py` hits (`:288`, `:459`, `:539`,
`:543`) and `docs/evidence/lane_receipts/lane21-h3_low_mime.md:127` are a
**different script's** identically-named flag.

**This overrides the driver's own r1-addendum compromise.** The driver proposed
*narrowing* `--overwrite` so it could not override a citation. With zero callers,
narrowing preserves a flag nobody uses whose only function is to overwrite cited
output. Opus's argument stands: `--out-dir` (`:231-236`) already serves every
legitimate use, and an old command line dies with `unrecognized arguments` and
exit 2 -- loud, not silent. **Remove it**, and move the escape route into
`--out-dir`'s help text so the removal is self-documenting.

### B4. G1's manifest and KEY writes are non-atomic. (Opus SHOULD-FIX 10, Fable concurring)

**CONFIRMED.** `scripts/otr_g1_lemmy_audition.py:210-211` opens `MANIFEST.json`
and `json.dump`s straight into it; `:221-222` does the same for `KEY.json`. The
cross-engine sibling already does `.part` + `os.replace` (`:270-274`). A crash
mid-write truncates the cited manifest -- a different road to the same outage this
campaign exists to prevent. Two lines each. Accepted.

### B5. `--resume` stays withdrawn, and r2 supplies a second independent reason.

The r1 Fable addendum already withdrew the idempotent-`--resume` machinery as
unnecessary under a cite-aware guard. Opus, reviewing the older plan, independently
found two ways the same mechanism turns destructive:

* **Resuming ONE engine rots the OTHER TWO records.** The manifest is a single
  shared file cited identically by kokoro (`config/cast_pools.py:1094-1097`),
  chatterbox (`:1124-1127`) and dia (`:1149-1152`) --
  `tests/test_lemmy_provisional_tier.py:800-806` pins that sharing on purpose
  (*"One audition, one manifest"*). Any render restamps `generated_utc` (`:268`),
  so resuming dia alone rots citations on two engines nobody touched.
* **`--resume` plus a `lines_version` bump is a silent full overwrite.**
  `_load_manifest` (`:247-249`) discards the existing manifest and starts fresh on
  a version mismatch, so nothing hash-matches, nothing is skipped, and all eight
  clips are re-rendered over cited evidence -- on the flag whose name promises
  safety.

Both are fully answered by the cite-aware guard, which refuses on the citation
regardless of which engine was requested or which lines version is loaded.
**Confirmation that the smaller design is the right one.**

### B6. The roster pin needs a stated membership predicate. (Opus SHOULD-FIX 5)

**ACCEPTED.** Other writers into cited-or-citable locations exist and must be
explicitly allowlisted with reasons rather than left for a future reader to guess:
`scripts/otr_lemmy_listen_page.py:374` (`LISTEN.html`) and its `write_decisions`
(`:334`) both write **into the cited campaign directory**;
`scripts/otr_g1_listen_page.py` writes `G1-LISTEN.html` and `LISTEN-ME-FIRST.md`
into `g1_lemmy_test_a/`; `scripts/bark_preset_audition.py:38` writes a
`manifest.json` nothing cites. Predicate: *writes an artifact whose sha256 could
be cited by a record*. Pin with a per-entry reason string.

### B7. Operator-facing text must be fixed in the same change. (Opus SHOULD-FIX 7, 8)

**ACCEPTED.** The refusal message must branch -- incomplete rows present says
"resume is not available, name a new `--out-dir`"; all-complete says the same for
a different reason -- and the module docstring (`:34-36`) currently teaches
exactly the two invocations that will now refuse. Both fixed in this commit.

---

## REJECTED / DEFERRED

### R4. Opus's "open-campaign marker" alternative. REJECTED, with reasons.

Opus proposed a self-contained seal: write a marker while any requested engine is
incomplete, remove it when all complete, permit `--resume` only into an open
campaign, and treat marker-absence as closed (so the existing cited directory is
protected without the instrument knowing about citations). It is a genuinely
elegant design and it fails closed.

**Rejected because it is a second mechanism for a case the cite-guard already
covers**, and it protects the *directory* while Fable established -- verified at
`scripts/otr_lemmy_listen_page.py:374-375` -- that the directory is a legitimate
shared workspace. The cite-guard protects the **bytes a record actually names**,
which is what 12.111 is about. Building both is over-engineering; building the
marker instead means adding new on-disk state whose absence is load-bearing.

**Opus's accompanying warning is accepted, though:** the guard must not depend on
an engine having a manifest row. A new engine added to `TARGETS` (`:74-79`) would
have no row and hash-match nothing. The cite-guard is immune -- it hashes the
**files about to be written**, not rows -- and that is now stated explicitly in
the design.

### R5. Opus SHOULD-FIX 6 (`DECISIONS.json` marks a resumed engine `settled` forever). DEFERRED, recorded.

**CONFIRMED as real:** `scripts/otr_lemmy_listen_page.py:173` sets state
`"missing"` with no clips, `:180` sets `"decidable": bool(clips)`, `:354` writes
`"decision": "settled"` when not decidable, and `write_decisions` (`:335-337`)
never overwrites. So an engine that renders *after* a page build stays `settled`
and is never listened to.

**Deferred deliberately.** This is a listen-page correctness bug, not an evidence
-destruction one; the change here adds `--campaign-dir` to that script but does not
otherwise touch its decision logic, and widening into decision reconciliation
inside a guard commit is how a focused change becomes an unreviewable one. Filed
as its own row.

### R6. Opus's `CUT` on the phrase "a true no-op". ACCEPTED as a documentation fix.

The sentence was correct only in the all-complete case. It is moot now that
`--resume` is withdrawn, and it does not appear in the final plan.

---

## THE SETTLED DESIGN

1. **`scripts/_otr_evidence_citations.py`** -- a shared helper. Walks the **live**
   `LEMMY_VOICE_POLICY` recursively, collecting every value matching 64 hex chars
   into a set (schema-agnostic: covers approved, provisional, superseded and any
   tier invented later; commented-out hashes cannot leak in because it reads the
   loaded object). Given paths about to be written, it hashes each that exists and
   refuses if any digest is in the set, naming the file. **Fails CLOSED** if the
   policy cannot be read. **No flag may override a citation.**
   Testable via the repo's own established pattern -- tests reach scripts through
   `importlib.util.spec_from_file_location` (`tests/test_audit_voice_gender_consistency.py:16-17`).
2. **`scripts/otr_lemmy_cross_engine_audition.py`** -- add `--out-dir` (bare name
   resolves under `otr/episodes/`, matching production `:234-236`); call the guard
   in `main()` **before** `preflight()`; record an honest fingerprint or an
   explicit "no recipe registered"; fix the refusal message and the docstring.
3. **`scripts/otr_g1_lemmy_audition.py`** -- call the same guard; **remove**
   `--overwrite`; make the manifest and KEY writes atomic.
4. **`scripts/otr_lemmy_listen_page.py`** -- add `--campaign-dir` so a new campaign
   can be listened to. Decision reconciliation deferred (R5).
5. **Detection, the highest-value half (Fable's r1 finding):** extend the on-disk
   verifier to the **qualified** route's manifest and the **superseded** G1
   manifest -- neither has any today -- and fix `_output_root`
   (`tests/test_lemmy_provisional_tier.py:727-747`) so that a resolved output root
   with a **missing** cited artifact FAILS instead of silently skipping.
6. **Tests** per 12.111 verify steps 1-4, plus the roster pin with its predicate.

**Out of scope, stated:** no re-render (the evidence is intact and re-rendering to
test a guard would destroy the subject); no verdict or qualification; verify step 5
is not provable without a re-render and will not be claimed.
