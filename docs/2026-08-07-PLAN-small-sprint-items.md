# PLAN -- the three small sprint items (queue item 1, 2026-08-07)

Driver: Claude (Cowork), coder + judge. Every cite below was re-read against the
real Windows tree at `76a5208a` before this document was written; **the source
documents' line numbers had drifted and are re-pinned here.** Where a claim is
reproduced by execution rather than by reading, that is stated.

Scope: GO_FORWARD queue item 1 -- THE CODING SPRINT items 2, 3 and 4 (the
latter scoped by the queue to **B4 + B6**).

## R1 -- CONVERGED (Fable as senior architect, operator ruling 2026-08-07)

Fable reviewed COLD, before this document existed, so its architecture is not an
echo of mine. It reproduced item 3's failure on the real venv and probed the
class-identity mechanism directly. Its architecture GOVERNS below; every claim
was re-grounded by the driver against the Windows tree and all of the following
survived.

**Three findings the driver did NOT have, all verified:**

1. **`news_interpreter.py:74`'s `SCHEMA_VERSION` is the SAME STRING for a
   DIFFERENT CONCEPT and must NOT be touched.** Its module docstring legend at
   **`:45`** says it "bumps when `meta.news` shape changes", and its inline
   comment at **`:70-73`** says it "participates in the cache key" (an earlier
   draft attributed both quotes to `:70-74`; they are two different places --
   Sonnet QA). It is read at `:132` and `:533` to stamp `brief.schema_version`.
   Bumping it forces news-brief cache regeneration for nothing. This resolves
   the driver's open question -- it is a separate versioning surface.
2. **THE QUIET HAZARD: `save_ledger_safe` restamps `schema_version` on EVERY
   write** (`_otr_ledger.py:289-291`, unconditional, first statements in the
   `try`). Post-hoc tools call it on existing episodes
   (`scripts/audit_otr_full_run.py`, `audio_enhance.py`,
   `otr_master_audio_mux.py`, `scene_sequencer.py`, `otr_video_render_batch.py`,
   `otr_post_upscale_procgen_blend.py`). After the bump, any of those touching a
   LEGACY episode silently promotes it to the new version; the citation audit
   then sees a non-legacy provenance-owned ledger with no receipt and reports a
   **false regression on history**. Nothing fails loudly.
   **r1 RULING: accept and DOCUMENT it; do not "fix" the restamp.**
   **>>> OVERTURNED IN SECTION 3-BIS -- DO NOT FOLLOW THIS LINE. <<<** r1's
   reasoning (that preserving old versions would contradict the "always sets"
   contract and `tests/test_meta_paths.py:172-173`) did not survive grounding:
   that test saves a ledger with NO `schema_version`, so a preserve-policy
   passes it unchanged. r3 overturned the remedy and a corpus measurement
   settled it. **The ACTIVE policy is section 3-BIS.** This paragraph is kept
   only as the r1 record.
3. **`LEGACY_SCHEMA_VERSIONS` carries a typo:** `"l2-2026-05-02"` matches no
   version in the lineage (`_otr_ledger.py:63-99` has `l2-2026-04-25` and
   `l3-2026-05-02`) -- it is those two mashed together. Harmless today because
   only `l3-2026-05-14` ledgers can carry `meta.provenance`. Fix while in the
   file.

**Version string: `l4-<date>`.** A newly REQUIRED field is more than additive,
and the audit's own tests already speak l4
(`tests/test_spoken_citation_audit.py:156-169` use `l4-2999-01-01`).

## 0. FIRST: none of this is already built (checked, because this repo has
## shipped "finished work" twice from a stale plan -- `d548ac54`)

| Item | Status | How it was checked |
|---|---|---|
| 2 -- notice reaches no surface | OPEN | `grep noncommercial_notice` over the tree: stamped at `OTR_LedgerScriptWriter.py:3632-3634`, read by the provenance module and tests only. **`otr_credits_roll.py` has zero references.** |
| 3 -- test-ordering pollution | OPEN | REPRODUCED: `pytest tests/test_public_domain_sources.py tests/test_public_domain_interpreter.py` fails `test_empty_cast_is_rejected_and_retried_to_failure`; the interpreter file alone is **11 passed**. |
| 4 / B4 -- coda helper | OPEN | no orchestration helper exists in the writer. `_otr_line_composer.compose_news_coda` is the pre-existing LLM call, a different thing. |
| 4 / B6 -- schema bump | OPEN | `_otr_ledger.py:58` is still `"l3-2026-05-14"` and `audit_spoken_citations.py:44` still lists it as legacy. |

**Re-pinned cites (the plan and the r4 final are both stale on these):**

| Claim | Document says | Actually at |
|---|---|---|
| writer stamps the notice | `:3590` | `OTR_LedgerScriptWriter.py:3632-3634` |
| `news_meta` defined | `:5463` | `OTR_LedgerScriptWriter.py:5526` |
| caller reads `news_meta.get("key_terms")` | `:5596` | `OTR_LedgerScriptWriter.py:5736` |

The `news_meta` TRAP is still real -- definition inside the candidate range,
read far below it -- only the numbers moved.

## 1. ITEM 2 -- the non-commercial notice reaches no human surface

**Truth today.** `_otr_provenance.noncommercial_notice()` (`:131`) returns a
plain-language warning beginning `NON-COMMERCIAL SOURCE: this episode adapts
...`, empty unless the source forbids commercial use. The writer stamps it into
`meta["noncommercial_notice"]` at `OTR_LedgerScriptWriter.py:3634`, **only when
non-empty**. Nothing renders it. `otr_credits_roll.py:519-520` renders the
sibling source line as `>> SOURCE: %s` from `meta["credits_source_line"]`.

**Build.** Add ONE flow entry beside `:519-520`, same three-line shape, reading
`meta["noncommercial_notice"]`.

**r2 CORRECTION 6 -- the exact prefix, which the plan demanded and never gave
(both lanes).** The notice string ALREADY begins `NON-COMMERCIAL SOURCE:`
(`_otr_provenance.py:151-155`), so a `>> NON-COMMERCIAL NOTICE: %s` wrapper
STUTTERS (`>> NON-COMMERCIAL NOTICE: NON-COMMERCIAL SOURCE: ...`). Antigravity's
`>> NOTICE:` is rejected for inventing a second label in front of the notice's
own.

**Use `">> %s" % _nc`, rendering exactly `>> NON-COMMERCIAL SOURCE: ...`** --
consistent with `>> SOURCE:` at `:520`, no stutter, no new vocabulary. Test that
exact prefix.

**r4 CORRECTION -- STRIP AND GUARD ON THE STRIPPED VALUE.** This was accepted in
r3 and then never written down; r4 caught its absence. Read
`_nc = str(meta.get("noncommercial_notice") or "").strip()` and guard on `_nc`,
matching how `credits_source_line` is read at `:519`. Without the strip a
whitespace-only field emits a bare `>>` intercept. **Add a whitespace-only
regression** so that cannot come back.

**The two conditions that are easy to get wrong, and are the acceptance:**

1. **The notice must render even when `credits_source_line` is absent.** It is a
   SEPARATE `if`, never an `elif` or a nested branch under the source line. A
   malformed legacy ledger can carry the notice and no source line, and the
   rights warning is the one that must not be lost.
2. **Adjacency applies only when both exist** -- source line immediately
   followed by the notice, each its OWN `("intercept", {...})` entry. Appending
   the notice directly after the source-line block gives this for free.

**No new wrapping helper.** `render_scroll_canvas` already measures and wraps
every `intercept` entry through `_wrap` (`otr_credits_roll.py:1133-1135`), so a
long notice wraps like any other footer line.

**Test the ORDERED list, not a dict.** `col3_flow` (`:544`) is a list of
`(kind, block)` tuples and there are already three `"intercept"` entries
(`:504`, `:509`, `:520`) -- converting to a dict collapses duplicate keys and
would silently pass. Assert on the ordered sequence.

Legibility on canvas is EYEBALLED on a future permitted render. This plan does
not claim it from a test.

## 2. ITEM 3 -- the test-ordering pollution

**Truth today, by execution.** Adjacent run fails; interpreter file alone is
11/11. Not inferred -- run twice, both ways, before this was written.

**Mechanism.** `tests/test_public_domain_sources.py:223-233`
(`test_module_import_is_lazy`) calls `importlib.reload(pd)` -- twice, once in
the body and once in the `finally`. Reload REPLACES the module's class objects.
`tests/test_public_domain_interpreter.py` imported its exception classes at
COLLECTION time, so after the reload `except OldClass` no longer matches
instances raised by the reloaded module. **This is module-identity breakage, not
leaked state, and no cleanup fixture can restore class identity** -- which is
why the r1 "cleanup fixture" idea was withdrawn.

**Build.** Run the lazy-import assertion in a SUBPROCESS: `sys.executable`,
repo-root `cwd`, `check=True`. The parent process then never reloads the module,
so no class identity moves.

**r2 CORRECTION 7 -- the read guard cannot come from `monkeypatch` (Antigravity).**
pytest's `monkeypatch` fixture is IN-PROCESS and cannot patch `Path.read_text`
inside a child. The plan's "fresh import with the read guard installed" was
hand-waving. **The child payload must install the guard itself**, inside the
`-c` program, BEFORE importing `nodes._otr_public_domain_sources`. Give the
child a `timeout` and surface its stdout/stderr on failure -- `check=True` alone
leaves a hung import unbounded and a failure undiagnosable.

**r2 CORRECTION 8 / r3 SETTLED -- no permutation harness.** A test that invokes
pytest on a file containing itself can recurse. The driver argued in r3 that once
the in-process reload is deleted the ordering bug is **unreachable**, so a
permutation harness pins a property that can no longer fail. **Both panel lanes
independently agreed and both filed the third-file harness under CUT** --
three-way convergence.

**r4 CORRECTION -- the assertion the driver adopted in r3 was BROKEN.**
`tests/test_public_domain_sources.py:12-13` imports only
`_otr_public_domain_sources as pd` and `_otr_source_payload as osp`, so the r3
form (`assert PublicDomainInterpreterError is mod.PublicDomainInterpreterError`)
references two names that do not exist in that file -- a `NameError`.
`PublicDomainInterpreterError` is defined in
`nodes/_otr_public_domain_sources.py:93`, i.e. the module aliased `pd`.

**The regression is:**

```python
before = pd.PublicDomainInterpreterError
# ... run the child probe ...
assert pd.PublicDomainInterpreterError is before
```

-- the parent's exception-class identity across the child probe. That is the
actual invariant; the class object must not be rebound.

**The two file ORDERS are external build commands in the VERIFY-AT-BUILD list,
NOT a test.** (An earlier draft said "pin both order permutations" in this
section while section 4 said no harness -- r4 caught the contradiction.)

**r3 CORRECTION -- the child needs `sys.path` (Antigravity).** `cwd` is not
guaranteed to be `sys.path[0]` under `python -c`, so the payload must
`sys.path.insert(0, ...)` the repo root itself before importing. Catch BOTH
`CalledProcessError` and `TimeoutExpired` and surface the child's captured
stdout/stderr in the pytest failure -- `check=True` alone shows neither.

**CUT, and stays cut:** renaming to a private module name. It risks exercising
fallback import paths instead of the production
`nodes._otr_public_domain_sources` package identity -- i.e. it would make the
test pass by testing something else.

**Note for the panel:** this repo has a KNOWN-FAIL-GUARD conftest
(`EXPECTED_FAILED_NODEIDS` + `docs/known-failures.md`) that printed a REGRESSION
banner during the reproduction. The fix must make the test pass, **not** be
added to the known-fail list.

## 3. ITEM 4 -- B4 (helper boundary) and B6 (schema bump)

### B4 -- extract the coda orchestration so tests exercise the production reader

**Why it exists:** the 2026-08-04 attempt at this defect edited
`spoken_coda_line()`, which had ZERO readers, and 30 episodes leaked after it
"landed". A fix applied to a function with no callers is not a fix.

**Boundary (r1 Fable, CORRECTED TWICE IN r2).** Extract **`:5533-5727`**:
everything inside the `if` guard, from the `intro_text` readback through the
`news_coda_spoken_reduction` stamp, into ONE helper taking

```
(led, meta, *, first_announcer_id, last_announcer_id, provenance_owned,
 style_grammar_on, effective_spoken_fact, nc_brief, script_brief, premise,
 resolved, slot_scheduler, creative_generate_fn)
```

**`premise` is REQUIRED keyword-only, not `premise=""` (r4).** A default would
let missing wiring pass silently, which is this item's whole failure mode. The
production caller always has an outline. Coerce INSIDE the helper with
`str(premise or "")`.

**r3 CORRECTION -- the body's REAL names (Codex).** The extracted body reads
`_style_grammar_on` (`:5560`, `:5669`, `:5689`), `creative_generate_fn`
(`:5571`, `:5660`) and `outline` (`:5573`), NOT the plan's earlier
`style_grammar_on` / `creative_fn` / `premise`. Note `:5571` reads
`creative_fn=creative_generate_fn` -- `creative_fn` is `compose_news_coda`'s
KEYWORD, `creative_generate_fn` is the local being passed. Rebind these
explicitly during extraction and exercise BOTH the `compose_news_coda` route and
the fictional `compose_announcer_outro` route, or the helper raises `NameError`
on only one branch.

`premise` replaces `str(getattr(outline, "premise", "") or "")` at `:5573`;
coerce inside the helper with `str(premise or "")` so a test passing `None`
cannot stringify `"None"` into the prompt (Antigravity).

**r2 CORRECTION 1 -- `script_brief` (found INDEPENDENTLY by both panel lanes).**
`compose_announcer_outro` at `:5661` passes `script_brief=script_brief`, and
`script_brief` is bound OUTSIDE the range. Omitting it is a `NameError` on the
fictional-outro branch -- the same bug class as the `news_meta` trap, at a third
location. It is a PARAMETER.

**Cite corrected by Sonnet QA:** an earlier draft cited `:5301-5302` and `:5469`
as the binding sites. Those are a USE and a hardcoded `script_brief=""` KEYWORD
ARGUMENT, neither of which binds the outer variable. The real bindings are
**`:3825`** (`script_brief = briefs.script_brief`) and **`:3949`**
(`script_brief = ""`), both far outside the range. Conclusion unchanged; the
evidence now actually shows it.

**r2 CORRECTION 2 -- the helper must RETURN (driver anchor, seconded by Codex).**
`led.save()` is at `:5728` and the `log.info` at `:5729-5732` reads
**`outro_res.compose_flags`** -- both INSIDE the guard, AFTER a `:5727` cut. A
helper that stops at `:5727` strands `outro_res` in the caller: `NameError` on
every episode that composes a coda.

Resolution keeps the r4 spec's intent (helper stays disk-free and testable):
**the helper returns `_OTRLC.LineResult`**, and the caller assigns it before
`led.save()` and the log line. Do NOT move `led.save()` into the helper.

**The return contract is safe on every path -- and the FIRST version of this
proof was FALSE. Corrected by Sonnet QA.**

The earlier wording claimed "**zero** early exits
(`return`/`continue`/`break`/`raise`) anywhere in `:5533-5733`". That came from a
probe using `\s` in an `awk` regex, which this `awk` does not support -- so the
pattern matched NOTHING and reported zero for a range that actually contains
THREE. The POSIX-safe `grep -E '^[[:space:]]*(return|continue|break|raise)\b'`
finds them all. **A grep that silently matches nothing is not a proof, and it
was presented as one.**

The real picture, re-verified line by line:

| Line | Statement | Effect on the contract |
|---|---|---|
| `:5537` | `break` | Loop-local -- exits `for _ln in led.data...` while reading `intro_text`, BEFORE `outro_res` exists. Harmless. |
| `:5657` | `break` | Loop-local -- exits `for _ln in reversed(...)` finding the final character line, before the `:5659` compose call binds `outro_res`. Harmless. |
| `:5680` | `raise ValueError` | A REAL early exit, in the closed-vocabulary check. But all five binding sites precede it, and an exception propagates out of a helper exactly as it does out of inline code. |

**Corrected claim:** five `outro_res` bindings (`:5570`, `:5602`, `:5636`,
`:5659`, `:5672`) and **zero early exits that SKIP BINDING `outro_res`**. Every
path that REACHES the tail reaches it with `outro_res` bound, so returning the
`LineResult` is safe -- including on the three `"none"` downgrades.

**r2 CORRECTION 3 -- B4 PRESERVES `news_coda_spoken_reduction`.** The B5
restatement below says "do NOT write" it, and the extracted range writes/pops it
at `:5714-5727`. Those contradicted. **B4 is byte-identical behaviour: it keeps
the stamp exactly as-is inside the helper.** Removing that chain is the parked
worktree's separate item and must not ride this commit.

The grounds for exactly this cut, not a smaller one:

* Every local born inside the range (`intro_text`, `_spoken_fact_for_coda`,
  `_spoken_coda_source`, `outro_res`, `_deferred_to_credits`, `_coda_action`)
  DIES inside it -- nothing after `:5733` reads them.
* Inputs born outside must be PARAMETERS, not moved. `nc_brief`,
  `provenance_owned` and `effective_spoken_fact` are defined at `:4941-4959`,
  and **`nc_brief` is also read at `:5327`** inside the composer loop, so it
  cannot migrate into the helper.
* The guard itself (`:5527-5530`) stays in the caller, **alongside `news_meta`
  (`:5526`)**, which the caller still reads at `:5736`. Extracting `news_meta`
  with the block is a `NameError` on every episode -- both lanes caught this in
  the original arc and it is still true at `76a5208a`.
* **Do NOT go smaller.** A "routing only" helper over `:5558-5567` misses the
  three `"none"` downgrades at `:5611`, `:5620` and `:5645`, which are
  interleaved with the composer calls -- and those, plus the closed-vocabulary
  check (`:5679-5685`) and the `news_coda_emitted` stamp (`:5689-5693`), are
  precisely the production reader the routing tests must exercise.
* With `creative_fn` INJECTED, a routing test passes a stub and asserts
  `meta["spoken_coda_source"]`, `meta["news_coda_emitted"]` and the patched
  last-line text -- no full writer boot.
* Sequencing: the parked worktree's rip of `news_coda_spoken_reduction` targets
  `:5704-5727`, which lands INSIDE this helper. Re-ground that rip against the
  new boundary afterwards.

**Receipts that must not lie (B5, already ruled, restated so the helper honours
them):** `meta["spoken_coda_source"]` validated at WRITE time against
`_SPOKEN_CODA_SOURCES` (`OTR_LedgerScriptWriter.py:5679-5685`); `provenance`
only when the owned deterministic fact was appended, `news_close_brief` only
when that fact was deterministically appended on an unowned lane, else `none`.
`news_coda_emitted` is stamped OUTSIDE the `_style_grammar_on` conditional
because owned routes bypass that gate (`:5558-5563`). Do NOT write
`news_coda_spoken_reduction` -- dead chain, separate item. **SCOPE CORRECTION
(independent QA):** that B5 sentence is about not writing a NEW reduction
receipt in future work. It does NOT license B4 to delete the existing stamp:
**B4 is byte-identical and KEEPS `news_coda_spoken_reduction` exactly as-is
inside the helper** (see the B4 correction above). Removing that chain is the
parked worktree's separate item.

### B6 -- bump `CURRENT_SCHEMA_VERSION`

**Why.** `audit_spoken_citations.py:43-45` lists `"l3-2026-05-14"` -- the
CURRENT version -- inside `LEGACY_SCHEMA_VERSIONS`, and `:176` reads
`is_legacy = (not schema) or schema in LEGACY_SCHEMA_VERSIONS`. So **every live
ledger is currently treated as legacy and the `spoken_coda_source` requirement
never fires.** The audit is already written to expect the bump; bumping the
version is what ACTIVATES it.

**r2 CORRECTION 4 -- "no edit to the audit is needed" was too strong (Codex).**
No audit edit is needed to ACTIVATE l4 -- that part holds. But the legacy set is
independently WRONG and must be corrected in the same commit: it is missing FOUR
real lineage versions and carries one that never existed. Real lineage
(`_otr_ledger.py:58-105`): `l1-2026-04-24`, `l2-2026-04-25`, `l3-2026-04-28`,
`l3-2026-05-02`, `l3-2026-05-08`, `l3-2026-05-14`. Make the set exhaustive and
**keep `l3-2026-05-14` in it** -- that string staying is what preserves the
boundary.

Antigravity argued an `l2-2026-04-25` ledger would trigger false violations
today. **Rejected as reasoning, accepted as fix:** `audit_spoken_citations.py:177`
requires `isinstance(meta.get("provenance"), dict)` before enforcing, and pre-l3
ledgers predate the provenance block, so the gap is LATENT, not active. Fix it
because it is wrong, not because it is burning.

Antigravity argued in r4 that the bump is unnecessary ("purely additive,
consumers degrade gracefully"). **Overruled and staying overruled:** graceful
degradation IS the failure mode here -- it makes a dropped receipt
indistinguishable from history, which is exactly how the zero-reader coda and
the vacuous G9 test survived.

**Blast radius -- smaller than a grep suggests, because it DERIVES.**

```
_otr_ledger.CURRENT_SCHEMA_VERSION        (:58)   <- the one edit
  -> production_ledger.Ledger.SCHEMA_VERSION      (production_ledger.py:700)
       -> _otr_ledger_freeze.EXPECTED_SCHEMA_VERSION (_otr_ledger_freeze.py:89)
```

The hardcoded literals at `production_ledger.py:703` and
`_otr_ledger_freeze.py:92` are **defensive `except` fallbacks**, reached only
if the import chain breaks. They must still be updated (a fallback that lies is
worse than none), but they are not the live path.

`nodes/news_interpreter.py:74` has its OWN `SCHEMA_VERSION = "l3-2026-05-14"`
that does NOT derive from the ledger constant. **RESOLVED in r1 and NOT an open
question: it is a SEPARATE versioning surface (the `meta.news` shape / cache
key, per its docstring legend at `:45` and its inline comment at `:70-73`, and
its uses at `:132` and `:533`). DO NOT EDIT THAT CONSTANT.** (An earlier draft resolved this in the r1 section while
still calling it open here; r2 caught the contradiction. This is now
authoritative.)

**Pinned version: `l4-2026-08-07`.** Not a placeholder -- `l4-<date>` is not an
implementable value and the sweep needs the literal.

**Tests that must move, and the one that must NOT.**

* `tests/test_lfc_phase_0_10_gap_audit.py:600` --
  `assert _LFC.EXPECTED_SCHEMA_VERSION == "l3-2026-05-14"` asserts the CURRENT
  version. Must move to the new value.
* `tests/test_workflow_json_guardrails.py` --
  `TestVintageLedgerSchemaCompat::test_schema_version_pinned` at **`:1250`**,
  its two asserts at **`:1253-1254`** (re-pinned by Sonnet QA; an earlier draft
  said `:1252-1255`). It is DESIGNED to
  fail loud on a bump ("a SILENT schema bump is forbidden"). Its first assertion
  moves to the new version. **Its second assertion is the subtle one:**
  `assert self._vintage_l3_ledger()["schema_version"] == EXPECTED_SCHEMA_VERSION`
  -- after the bump the vintage fixture must NOT equal the current version;
  that is the whole point of a vintage fixture. That assertion must be rewritten
  to prove the vintage ledger still READS cleanly while being a different
  version, not deleted. Deleting it removes the l3 compatibility coverage the
  r4 ruling explicitly said to keep.
* **r2 CORRECTION 5 (Codex) -- `tests/test_production_ledger.py:104` asserts
  `led.data["schema_version"].startswith("l3-")` and WILL FAIL on l4.** Change
  it to compare against `_otr_ledger.CURRENT_SCHEMA_VERSION`, which is what its
  own comment already says it means ("live-pulled from
  `_otr_ledger.CURRENT_SCHEMA_VERSION` so both write paths stay in lockstep").
* The ~18 fixture ledgers hardcoding `"schema_version": "l3-2026-05-14"` become
  LEGACY fixtures. They should mostly be left alone -- that is now their
  correct role -- but each must be checked for a path that requires the current
  version.

**SWEEP BY PATTERN CLASS, NOT BY LITERAL.** Correction 5 exposed a hole in the
driver's own method: grepping the literal `l3-2026-05-14` and the constant name
CANNOT see a `startswith("l3-")`. The re-sweep for that class found one more
site, reported here for completeness:

* `tests/_helpers.py:34`, inside `_looks_like_l3_ledger`, is a **discovery
  filter** (`load_all_ledger_fixtures`, `:66-118`) rather than an assertion -- an
  l4 ledger would stop "looking like a ledger" and be SILENTLY dropped from
  auditing, since that function skips non-matching files by design rather than
  failing. **Blast radius today is ZERO** -- the function has no callers anywhere
  in the tree and none of the 5 JSON files in `tests/fixtures/` carry an `l3-`
  schema_version -- so this is a NOTE, not a must-fix. Generalize the prefix
  check only if something ever starts using it.

**Not touched:** `.claude/worktrees/awesome-brahmagupta-a509b4/` is another
session's parked uncommitted work and is out of scope here.

## 3-BIS. THE SAVE-VERSION POLICY -- r3 OVERTURNS r1, SETTLED BY MEASUREMENT

r1 (Fable) ruled: ACCEPT the `save_ledger_safe` restamp and document it. r3
(Codex) ruled: that is an operational promise, not an engineering control.
**r3 wins**, and not on seniority:

* **Fable's stated blocker does not block.** It said the restamp cannot change
  because of `save_ledger_safe`'s "always sets" contract and
  `tests/test_meta_paths.py:172-173`. That test saves
  `{"episode_id": ..., "lines": []}` -- a ledger with **no `schema_version` at
  all** -- so a *"stamp when absent or already current, else preserve"* policy
  passes it unchanged.
* **The exposure is reachable:** `scripts/audit_otr_full_run.py:363` really calls
  `save_ledger_safe` over whatever episode it audits, historical included.
* **MEASURED** (`tmp/_schema_exposure_probe.py`, read-only, 1,599 ledgers):
  **48 carry `meta.provenance`; 43 of them have NO `spoken_coda_source`.** Each
  of those 43 is a landmine -- one routine write-back promotes it to l4 and the
  citation audit reports a FALSE REGRESSION on history. A lying receipt is the
  exact failure the item-7 arc exists to end.
* The same histogram shows **10 ledgers at `l3-2026-05-08`**, a version missing
  from `LEGACY_SCHEMA_VERSIONS` today -- the set's incompleteness is live, not
  cosmetic.

**THE POLICY (r4-final).** In **`save_ledger_safe` ONLY**: stamp
`CURRENT_SCHEMA_VERSION` when the ledger's `schema_version` is ABSENT or ALREADY
EQUAL to current; otherwise PRESERVE the existing value, mirror it into
`meta.schema_version` so the two cannot disagree, and log a WARNING so
non-migration is observable.

**`production_ledger.Ledger.save()` is NOT changed.** The driver went into r4
believing the policy needed a shared helper both stampers call. **That was
wrong, and r4 settled it on evidence:** every one of the six post-hoc writers
reaches history through `save_ledger_safe` --

```
audit_otr_full_run.py  audio_enhance.py  otr_master_audio_mux.py
scene_sequencer.py     otr_video_render_batch.py  otr_post_upscale_procgen_blend.py
```

-- so protecting that one function is not partial, it is SUFFICIENT. And
`Ledger.__init__` (`production_ledger.py:705-714`) builds `self.data` with
`"schema_version": self.SCHEMA_VERSION`, so a live ledger is born current and
never enters the preserve branch. Changing the active producer's core path would
add unsupported resume semantics for no gain.

**RESIDUAL, recorded not hidden -- and RE-CITED after Sonnet QA.** An earlier
draft blamed `_merge_with_disk` (`:1444`). **That was wrong and backwards:**
`_merge_with_disk`'s `TOP_PRESERVE` list (`production_ledger.py:1530-1536`)
INCLUDES `"schema_version"` and copies the disk value in only when the in-memory
ledger lacks one -- it PRESERVES, it never promotes.

The actual unconditional promotion is a SEPARATE block that runs after the merge
returns: **`production_ledger.py:1461-1462`**, which sets both
`_meta["schema_version"]` and `merged["schema_version"]` to
`CURRENT_SCHEMA_VERSION` with no condition. Tree-wide, that plus
`_otr_ledger.py:289,291` are the **only two** production sites that assign
`schema_version` -- so this is the genuine "seventh writer".

It stays DORMANT because no caller constructs a `Ledger` over an existing
on-disk file (`new_ledger()` always starts fresh,
`production_ledger.py:486-488`), and `Ledger.__init__` births `self.data`
already current. **Add a regression pinning that** -- no caller resumes a
`Ledger()` over a pre-existing on-disk ledger carrying a foreign
`schema_version` -- so the residual cannot silently start mattering.

**Transition window:** an episode started pre-bump and frozen post-bump would
hold l3 and fail the exact-equality freeze gate. A live ledger is born with no
`schema_version` and gets l4 on its first save, so this is only reachable by
resuming a pre-bump episode -- the unsupported flow above. Box verified idle
(no server, GPU at desktop baseline), so the window is empty in fact as well.

Test both save paths with: absent, current, legacy, and disagreeing mirrored
versions.

## 4. SHIP ORDER AND GATES

**FOUR commits, in this order (r1 Fable, ADOPTED):**

1. **Item 3** -- the subprocess lazy-import test. FIRST, because it repairs the
   very signal used to validate everything after it, and it is fully
   independent. **The two file ORDERS are external build commands, not a test**
   -- an earlier draft said "both order permutations pinned" here while section
   2 had already CUT the harness (caught by independent QA).
2. **Item 2** -- the credits item + integration test. Small, isolated, no schema
   interaction.
3. **B4** -- the extraction + writer-level routing tests. Behaviour
   byte-identical.
4. **B6** -- the bump + both fallback literals + the pin/fixture sweep + the
   legacy-set typo.

**Do NOT fold B4 and B6 together.** One is a provably-neutral refactor, the other
changes what the corpus audit ENFORCES. B4-before-B6 also puts the production
reader under test *before* enforcement turns on, and keeps the schema event
bisectable on its own.

B6 must land with its test updates in the SAME commit or the suite is red
between commits.

Every chunk: focused tests -> full Windows suite (baseline **9081 passed / 111
skipped / 1 xfailed** at `76a5208a`) -> Bug Bible **17** -> commit -> push ->
`HEAD == origin/v2.0-alpha`, AST parse, no BOM, no zero-byte.

**r2 CORRECTION 9 / r3 REFINED -- the Bug Bible is a DIFFERENT REPO, and its
landing is ATOMIC (Codex).** Bug Bible coverage is MANDATORY per CLAUDE.md, but
`BUG_BIBLE.yaml` lives in
`C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide`, so it
**cannot be one of the four OTR commits**. Its regression
(`tests/bug_bible_regression.py:857-943`) requires `BUG_BIBLE.yaml`, the README
entry COUNT and executable coverage to stay synchronized, so all three land
together in ONE commit in that repo, pushed, immediately after B4 -- then rerun
the Bible gate before B6. Record BOTH repo SHAs in the handoff, and **stop
quoting "17" as a fixed expectation** once the entry lands.

### Two test additions r3 demands, both guarding known failure modes

* **A REACHABILITY regression for B4 (Codex).** Direct helper tests do NOT prove
  the helper is connected -- that is exactly the zero-reader failure this whole
  extraction exists to prevent (the 2026-08-04 attempt edited a function with no
  callers and 30 episodes leaked). Assert that `OTR_LedgerScriptWriter.run`
  calls the helper EXACTLY ONCE under the guard, passes the complete keyword
  contract, assigns its `LineResult`, then performs `led.save()` and the flag
  log. Keep the direct routing tests for behaviour.
* **THE ROUTING ACCEPTANCE MATRIX -- restored (r4).** This was already specified
  at `GO_FORWARD_PLAN.md:597-601` and the driver LOST it when rewriting the plan.
  Restored verbatim in substance: both fidelity banks (public_domain and
  shakespeare) x {non-empty, empty} provenance, PLUS an owned/non-empty case with
  `_style_grammar_on == False`. **Assert the coda is PRESENT, not merely that the
  URL is absent.** Control is **`media_archive`, NEVER `scifi_news`** -- that lane
  dispatches to `scifi_news_circuit` and returns before this block, so it would
  prove nothing. Assert the coda in `lines[].text` and in a `Dialogue:` cue from
  `build_ass_from_ledger`.
* **Wire the audit tests to the REAL constant (Codex).**
  `tests/test_spoken_citation_audit.py:156-175` uses an invented
  `l4-2999-01-01`, so it stays GREEN even if the real `l4-2026-08-07` were
  wrongly added to `LEGACY_SCHEMA_VERSIONS` -- the #1 trap, undetected by its own
  guard. Add assertions using `nodes._otr_ledger.CURRENT_SCHEMA_VERSION`:
  current must NOT be legacy and MUST require the receipt; all six historical
  versions must be legacy.

**WHICH lesson is promoted, stated explicitly (Sonnet QA).** The Bug Bible entry
documents **B4's lesson ONLY**: *a fix applied to a function with no callers is
not a fix.* That one clears CLAUDE.md's admission bar -- it is a live production
defect (`GO_FORWARD_PLAN.md:602-604`: the 2026-08-04 attempt edited
`spoken_coda_line()`, which had zero readers, and **30 episodes leaked after it
"landed"**).

**Items 2 and 3 do NOT get Bible entries.** Item 3 is pytest-verified only, and
item 2's own acceptance says canvas legibility is eyeballed on a future render
and "not claimed from a test" -- neither is a bug verified by a live production
artifact, so neither may create a Bible rule. That is the admission rule, not a
judgment call.

**Suite runs are BACKGROUNDED and polled** -- a foreground 9,000-test run
exceeds the ~60 s MCP command ceiling.

## 5. THE KNOWN TRAPS -- named so nobody rediscovers them the expensive way

**THE #1 TRAP (r1): the completionist find-replace on `"l3-2026-05-14"`.** Grep
returns 30+ hits and two of them INVERT the fix if "completed":

* `nodes/news_interpreter.py:74` -- different concept (above). Bumping it forces
  news-brief regeneration for nothing.
* **Worst: `scripts/audit_spoken_citations.py:44`.** Replacing the string
  *inside* `LEGACY_SCHEMA_VERSIONS` with the new value simultaneously
  un-legacies 1,587 historical ledgers and legacies every post-fix one --
  inverting the entire boundary -- **while the audit's unit tests stay green**,
  because they hardcode their own `l4-2999` strings. That string must STAY.

**Runner-up trap:** "fixing" the l3 fixtures that MODEL legacy ledgers
(`tests/fixtures/fable2/legacy_reference_ledger.json`, the vintage fixture at
`test_workflow_json_guardrails.py:1236`, `test_cue_manifest.py:246`) to the new
constant during the sweep -- that silently deletes legacy-compat coverage.
Fixtures modelling CURRENT ledgers switch to `OTRL.CURRENT_SCHEMA_VERSION`;
fixtures modelling LEGACY keep the literal. Sort by ROLE, not by grep hit.

**Fixture-fed gap audits to check individually:** `test_provenance_v4.py:204`,
`test_scene_guard_v4.py:110`, `test_freeze_cascade_g6.py:29`,
`test_g8_line_id_uniqueness.py:21/:134` feed hardcoded-l3 ledgers into
`run_gap_audit`; post-bump each report gains a schema error. Tests filtering
errors by substring survive; any asserting a CLEAN report break.

**Item 3's tempting wrong fix:** re-importing the exception inside the failing
test. It passes both orders and leaves the reload landmine armed for every other
collection-time importer. Removing the in-process reload is the root fix.

## 6. WHAT I WANT THE r2-r4 PANEL TO ATTACK

1. Does the subprocess fix actually leave class identity untouched in the
   parent, or does merely IMPORTING the module at collection already move it?
2. Is there a consumer of `schema_version` that COMPARES rather than records --
   one that would silently change branch on a bump instead of failing loud?
   (r1 found the `save_ledger_safe` restamp; is there a second?)
3. Is the B4 boundary at `:5533-5727` drawable without moving `led.save()`, and
   does any local born inside it actually escape?
4. Does the notice's literal prefix collide with any existing credits parsing?
5. Is `l4-<date>` right, or does some consumer parse the `l<N>` prefix
   numerically and change behaviour at l4?

## 7. VERIFY-AT-BUILD -- the acceptance gate (adopted wholesale from r4)

This list IS the build's definition of done. Nothing ships on "the suite is
green" alone.

- [ ] No `_otr_public_domain_sources` reload remains anywhere; then run BOTH
      external orders as build commands: sources->interpreter and
      interpreter->sources. (These are commands, not a test -- no nested pytest.)
- [ ] Child probe installs its OWN `Path.read_text` guard before import, inserts
      the repo root into `sys.path`, has a timeout, surfaces stdout/stderr for
      both `CalledProcessError` and `TimeoutExpired`, and preserves
      `pd.PublicDomainInterpreterError` identity in the parent.
- [ ] Credits: exact `>> NON-COMMERCIAL SOURCE:` output; renders independently
      WITHOUT `credits_source_line`; adjacency when both exist; exactly once;
      whitespace-only suppressed; wraps on canvas. Re-run the parser/search
      audit for prefix collisions (the driver's standing UNVERIFIABLE).
- [ ] B4: routing matrix passes; `run()` reachability test proves the helper is
      called exactly once with the full keyword contract; helper returns
      `_OTRLC.LineResult`; caller saves AFTER; both composer branches execute
      without `NameError`.
- [ ] `save_ledger_safe` cases cover absent / current / legacy / disagreeing
      mirrored versions; a legacy write-back does NOT promote history.
- [ ] A fresh `Ledger` writes `l4-2026-08-07` and passes the exact-equality
      freeze gate at `_otr_ledger_freeze.py:608-613`.
- [ ] `CURRENT_SCHEMA_VERSION` is NOT legacy and DOES require the receipt; all
      six historical versions ARE legacy; `l3-2026-05-14` REMAINS in the set.
- [ ] Sweep literal, prefix, fallback, fixture-fed gap-audit and
      vintage-compat sites. **Do NOT edit `nodes/news_interpreter.py:74`.**
- [ ] `workflows/otr_canonical.json` byte-identical; workflow validator, JSON
      round-trip, link/input and widget-vector audits all run.
- [ ] Focused tests, full Windows suite (backgrounded + polled), Bug Bible gate
      re-run after its own commit, AST / BOM / zero-byte checks, BOTH repo SHAs
      recorded, `HEAD == origin/v2.0-alpha`.
- [ ] Other windows' dirty paths untouched; every commit path-scoped.
- [ ] POST-BUILD (first real render, not claimed from tests): the first
      provenance-owned production ledger is l4, carries `spoken_coda_source`,
      passes the citation audit, and freezes successfully.
