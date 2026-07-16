# Randomizer Rolls -- CODING PLAN (Design A: source-bank roll)

> **STATUS 2026-07-15 (baseline): STALE -- re-ground before code; NOT "coder slot only".**
> (1) Section A1's `_otr_lane_specs` is ABSORBED by
> `docs/2026-07-12-user-source-lanes-architecture.md` (that build lands first and CREATES the
> authority; this build shrinks to `_otr_bank_roll` + eligibility on top). That plan awaits its
> r5 pass + section-16 ratification; this doc gets its delta note after. (2) The roster changed
> under section 0: 24 runnable banks (8 base + 8 _v2 + 8 _v3), scifi_gemini +
> original_codex56sol lanes DELETED @ 3312aec7, the three _v3 dispatched runners are
> factory-wrapped closures (`_make_v3_runner`) that strain the NAMES-ONLY LaneSpec design, and
> the 11f6214a line pins drifted ~+300 lines. Design A itself (sentinel, receipt, sorted
> eligible_order, two-filter pool) is unchallenged.

**Date:** 2026-07-12. **Stage:** CODE-READY -- converged through the full /kibitz
arc **r2 (coding) -> r3 (wiring) -> r4 (convergence)**; panel = codex
`gpt-5.6-sol` @ ultra + antigravity `gemini-3.5-pro`; Claude = anchor panelist +
sole judge. Judgment logs:
`kibitz-runs/2026-07-12-randomizer-rolls/{r2,r3,r4}/final.md`.
**Contract:** `docs/2026-07-11-randomizer-rolls-r1.md` -- R1 decisions are not
re-litigated.
**Scope:** Design A ONLY. Design B (visual pack roll) is parked -- no placeholder
widget, value, env var, or code ships in this change.
**Binding:** `CLAUDE.md`, `docs/PRODUCTION_SPRINT_LESSONS.md`.

## D-1 RESOLVED -- NO RIGHTS GATE (operator ruling, 2026-07-12)

**The roll does NOT filter on rights, and no `rights_class` field ships.** Operator:
the LLM may write in whatever style it wants; understanding the terms under which AI
output is used is the END USER's responsibility, not a filter inside the writer. This
OVERRIDES R1 s2's rights-class exclusion and the r2 panel's affirmative-allowlist
argument (operator directives win over any doc, panel, or memory that disagrees).

Consequences -- the plan gets SMALLER:

- `shakespeare` is roll-eligible like every other runnable bank.
- **`banks.json` is NOT touched.** No new registry field, no parser change, no
  synthetic-row-builder migration, no schema-doc churn. The registry-collision hazard
  that dominated r2/r3 is GONE, and the old sub-commit 2 is deleted.
- Eligibility is now exactly TWO filters: **runnable** + **request-compatible**.
- The disclosure surfaces the repo already ships (credits source line, HUD origin
  label, machine-generated disclosure) are unchanged and remain the honest signal.

## PRECONDITION (one, before the first edit)

**Coder slot + a re-grounded base.** All line pins below were taken at `11f6214a`.
The base has moved TWICE during this planning window alone (`11f6214a` -> `9d8265c0`
-> `efb6b6ad`), and the writer's gates have shifted with it. Standing instruction:
**claim the slot in `docs/GO_FORWARD_PLAN.md`, record the actual HEAD, re-read EVERY
pin below at that HEAD, and only then edit or compute qualification seeds.** The
collision surface is now just `OTR_LedgerScriptWriter.py` (the lane-spec rip), not
the registry.

---

## 0. Ground truth (re-read 2026-07-12; several R1 pins had moved)

- **ELEVEN banks: 10 runnable + `custom_source_bank`.** Five runnable lanes are
  DISPATCHED custom pipelines; the rest are inline. Contender rows keep landing.
  **Nothing here may hard-code a bank list, count, or name** (r1 CUT2).
- Writer entry: `require_runnable_bank` is the FIRST statement (:3333); visual
  style (:3339); fable2 word gate hardcoded by pipeline name (:3351-3356);
  dispatched lanes reject refine (:3357-3363).
- `_RUNNER_BY_PIPELINE` (:1656) is the ONLY pipeline->runner authority and it lives
  in the WRITER. Its runners are ALREADY lazily imported by `_run_*_lane` wrappers
  (:1632-1647). `_LEGACY_INLINE_PIPELINES` (:1665-1667) + `_resolve_lane_runner`
  (:1718-1735, no fallback) are the inline authority. Four consumers total:
  telemetry membership (:1684), lane resolution (:1724), the refine rejection
  (:3357), the dispatch site (:3721-3743). `pipeline.executable` is NOT a proxy for
  "dispatched" -- `original_multi_pass` is executable AND inline.
- **Lane word constraints (audited):** fable2 declares a ceiling + a `<120`
  one-draft gate (`_otr_scifi_fable2.py:236-264`) and is the ONLY lane gated at
  writer entry. `scifi_codex` (:810-835), `scifi_gemini` (:235-258) and
  `scifi_sonnet` (:386-404) each reject `target_words` outside **30-900** -- but
  from INSIDE the runner, after source work. `original_codex56sol` treats
  target_words as ADVISORY. **So "no hook = compatible" is FALSE today.**
- **No rights field exists, and none is added** (D-1). `runnable` remains the ONLY
  curation surface, exactly as R1 s4 wanted.
- **The validator is a LENGTH check** (`widget_vector_drift`,
  `_otr_workflow_validator.py:158-172`); the guardrail pins `wv[23] ==
  "science_news"` and `wv[23] in list_bank_ids()`
  (`tests/test_workflow_json_guardrails.py:702-714`). **A prepended CHOICE produces
  ZERO canonical-JSON diff.**
- **The headless path validates combo membership** against the LIVE `/object_info`
  schema (`scripts/otr_api.py:198-216`); `source_bank` is whitelisted (:766). The
  sentinel is accepted only because the live schema carries it -- so it cannot be
  smoked against a stale server.
- `IS_CHANGED` returns `time()` (:3024-3028) -- caching is a non-issue.
- The refine loop rebuilds `_core` from `locals()` filtered against the run()
  SIGNATURE (:3399-3410) and calls `self.run(**_core, ...)` (:3085-3091);
  `TestRefineCoreCapture` (tests/test_source_bank_widget_2c.py:133-172) pins it.
- **Refine's effective state is NOT the widget.** `OTR_STORY_REFINE_BAR` can enable
  refine with the widget `Off`; pass-count/bar/provider clamps can disable it with a
  non-Off widget (`_otr_story_select.py:229-286`). `effective_passes >= 2` is the
  truth.

---

## 1. Ownership rows (Lesson 1)

**`meta["bank_roll"]`** -- writer: `_otr_bank_roll.resolve_bank_selection()`, called
from ONE site (the first statement of `run()`, before `require_runnable_bank`).
Consumers: the refine re-entry (reads the carried receipt, never re-rolls) and the
episode ledger meta. HUD/credits do NOT read it -- they read the SELECTED bank's
defaults, exactly as on a manual pick. (Replay tooling is FUTURE work, not a wired
consumer.) DERIVED (registry x request x seed); zero LLM calls. Fixed at submission,
carried through every refine pass, frozen with the ledger. **Absent -- not null, not
a stub -- on a manual pick.**

**`_otr_lane_specs.LANE_SPECS`** -- the ONE lane authority: dispatched-lane runners
(lazy), their compatibility policy, and the inline-pipeline set. Consumers: the
writer's dispatch + telemetry + refine rejection, and `_otr_bank_roll`'s filter.
**It REPLACES `_RUNNER_BY_PIPELINE` -- there is no second table and no view.**

`meta["style_roll"]` belongs to Design B and is NOT created here.

---

## 2. Receipt schema

```json
"bank_roll": {
  "receipt_version": 1,
  "requested": "roll (any eligible bank)",
  "selected": "media_archive",
  "seed": 2894187203,
  "seed_source": "OTR_BANK_SEED override" | "OS entropy",
  "eligible_order": ["media_archive", "original_radio", "public_domain_story", "science_news"]
}
```

`eligible_order` is the exact ordered list handed to the draw -- **sorted by bank
id**, so the result depends on registry CONTENT, never on banks.json row order (a
contender window reordering rows must not change what a replayed seed selects).
`to_meta()` emits it as a JSON list, not a tuple. `seed` is a 32-bit int, always
present -- matching the shipped cast-seed receipt idiom (writer :1093-1104,
:4030-4031). One receipt convention, not two.

---

## 3. The change

### A1 -- `nodes/_otr_lane_specs.py` (NEW; the lane authority)

```python
@dataclass(frozen=True)
class RollRequest:                  # lives HERE, not in _otr_bank_roll (see below)
    target_words: int
    refine_active: bool             # the EFFECTIVE state, not the widget string
    source_ref: str

@dataclass(frozen=True)
class LaneSpec:                     # NAMES ONLY -- never callables, never classes
    module: str                     # e.g. "_otr_scifi_codex"
    runner_attr: str                # e.g. "run_scifi_codex_episode"
    preflight_attr: str             # target-only compat predicate; "" = unconstrained
    compat_error_attrs: tuple[str, ...]   # e.g. ("CodexTargetRangeError",)

LANE_SPECS: dict[str, LaneSpec]     # dispatched lanes only; the key IS the pipeline id
INLINE_PIPELINES: frozenset[str]    # moved from the writer (:1665-1667)

def runner_for(pipeline_id) -> Callable | None    # None ONLY for a known inline lane;
                                                  # unknown pipeline -> RAISE (no fallback)
def assert_supported(bank, req) -> None           # writer gate: NATIVE error identity
def is_roll_compatible(bank, req) -> bool         # roll filter: bool, no new exception type
```

**LaneSpec stores NAMES, resolved lazily inside the three functions.** Holding
callables or exception CLASSES would require importing every runner to BUILD
`LANE_SPECS` -- i.e. at ComfyUI startup, since the writer imports this module at top
level. That defeats the whole lazy-import contract (all three reviewers hit this
independently at r4).

**Inline and unknown pipelines are specified, not left to the implementor:** a
pipeline in `INLINE_PIPELINES` -> `assert_supported` is a no-op and
`is_roll_compatible` is True. A pipeline in NEITHER map -> `assert_supported` raises
`UnknownPipelineError` and `is_roll_compatible` is False.

**Import order is acyclic and pinned by a pure-import test:**
`_otr_story_routing <- _otr_lane_specs <- _otr_bank_roll <- OTR_LedgerScriptWriter`.
`_otr_lane_specs` imports routing for the `SourceBank` type and NEVER imports the
writer. `RollRequest` lives here (not in `_otr_bank_roll`) precisely to keep that
edge one-way.

**Two entry points, one policy** (the r3 headline: one function CANNOT both preserve
native error identity and raise a single catchable type):

- `assert_supported` -> the writer's entry gate. Re-raises the runner's OWN
  exception, unwrapped, unreworded. A direct `scifi_fable2` pick at 720 words still
  produces exactly today's `Fable2ScriptError`.
- `is_roll_compatible` -> the roll's filter. Resolves that lane's declared
  `compat_error_attrs` to real classes, catches ONLY those, and returns False. An
  `ImportError`, `AttributeError`, or runner defect PROPAGATES (a bare `except` would
  silently turn a bug into "ineligible"). **No `LaneIncompatible` type exists** -- a
  bool-returning predicate needs no roll-only exception vocabulary (r4 CUT, both
  panelists).

**Every dispatched lane declares a policy. Absence is not a yes.** Shipped:
`fable2_multipass` -> the existing `assert_supported_target_words`;
`scifi_codex_circuit` / `scifi_gemini_multipass` / `sonnet_archive_multipass` ->
target-only preflight functions HOISTED from each runner's 30-900 band (each runner
keeps its own defensive re-assert); `acoustic_puzzle_v1` -> `unconstrained`,
declared explicitly (advisory target_words -- re-verify at the build HEAD). Plus the
cross-lane law that belongs to the dispatch, not to any runner: **every dispatched
lane rejects refine re-entry.**

**`_RUNNER_BY_PIPELINE` is DELETED, not viewed.** All four consumers move to
`LANE_SPECS` membership / `runner_for()`. The telemetry branch at :1684 gets its own
test -- a wrong branch there silently mislabels the model receipt of every custom
lane.

### A2 -- `nodes/_otr_bank_roll.py` (NEW; pure, no ComfyUI import)

```python
SENTINEL = "roll (any eligible bank)"       # a UI command, never a registry row
RECEIPT_VERSION = 1

class BankRollError(StoryRoutingError): ...  # empty pool / sentinel+source_ref = loud

@dataclass(frozen=True)
class BankRollReceipt: ...                   # -> to_meta(), shape s2

def is_sentinel(value) -> bool
def eligible_banks(req: RollRequest) -> tuple[SourceBank, ...]
def resolve_seed(env: Mapping[str, str]) -> tuple[int, str]   # reads the SUPPLIED mapping
def draw(eligible_order: Sequence[str], seed: int,
         rng_factory=random.Random) -> str                    # pure; the test seam
def resolve_bank_selection(requested, request_factory, carried=None,
                           env=os.environ, rng_factory=random.Random)
        -> tuple[str, BankRollReceipt | None]
```

`resolve_bank_selection` is the ONE writer of the receipt:

1. `carried is not None` -> `(carried.selected, carried)`. **No roll, no RNG.**
2. `not is_sentinel(requested)` -> `(requested, None)`. Byte-identical legacy path.
   **The request is NOT built on this path** -- otherwise a malformed `target_words`
   (or a refine probe) would raise BEFORE the unknown/non-runnable bank gate and
   break the gate-first ordering pinned at tests/test_source_bank_widget_2c.py:106-127.
3. sentinel -> build the request (this is where `RefineConfig` is finally evaluated),
   **reject a nonblank `source_ref` LOUD** (a pinned reference belongs to a
   specific bank: pin the bank), filter, draw. Empty pool -> `BankRollError` naming
   every filter that removed candidates and how many each removed.

**Eligibility (TWO filters, each counted for the error message):**
`bank.runnable is True` -> `is_roll_compatible(bank, req)`. That is the whole pool
(D-1: no rights filter). Filtering BEFORE the draw is legal; a bank that fails AFTER
the draw fails LOUD -- never a silent re-roll.

**Seed:**

```python
raw = env.get("OTR_BANK_SEED", "").strip()
if raw:  seed, source = _parse_seed(raw), "OTR_BANK_SEED override"   # int; 0 <= seed < 2**32; else LOUD
else:    seed, source = random.SystemRandom().getrandbits(32), "OS entropy"
selected = draw(eligible_order, seed, rng_factory)                   # sorted order
```

Both paths end in a SEEDED `random.Random`, so `seed` always replays -- SystemRandom
only MINTS, it never draws. A malformed or out-of-range `OTR_BANK_SEED` raises: an
operator who typed a seed asserted replay intent, and ignoring it is the
silent-degrade class this repo bans.

### A3 -- the writer seam

```python
from . import _otr_bank_roll as _BANKROLL      # (the r2 draft called these without
from . import _otr_lane_specs as _LANES        #  importing them -- agy r3 MF1)

# ... first statements of run():
#
# ONE memoized refine config, built LAZILY (never on a manual pick, so the
# runnable gate still fails first) and REUSED by the refine-loop branch below --
# today that branch builds its own at :3393-3420. The widget string is NOT the
# effective state: OTR_STORY_REFINE_BAR can force refine ON with the widget at
# "Off", and pass-count/bar/provider clamps can force it OFF with a non-Off widget.
# And _refine_active is authoritative on re-entry.
_refine_cfg_cache = []
def _get_refine_cfg():
    if not _refine_cfg_cache:
        from . import _otr_story_select as _OTRSEL_GATE
        _refine_cfg_cache.append(_OTRSEL_GATE.resolve_refine_passes(
            creative_writing_model, widget_target=refine_target_grade))
    return _refine_cfg_cache[0]

source_bank, _bank_roll = _BANKROLL.resolve_bank_selection(
    source_bank,
    request_factory=lambda: _LANES.RollRequest(
        target_words=int(target_words),
        refine_active=bool(
            _refine_active or _get_refine_cfg().effective_passes >= 2),
        source_ref=str(source_ref or ""),
    ),
    carried=_bank_roll_receipt,
)
# CRITICAL (kibitz r2 -- found INDEPENDENTLY by BOTH panelists): the refine loop
# rebuilds _core from locals() filtered against the run() SIGNATURE, so it carries
# the PARAMETER name. Without this rebind the refine pass sees carried=None + a
# concrete bank, takes the manual path, and meta["bank_roll"] never reaches the
# shipped ledger.
_bank_roll_receipt = _bank_roll

_source_bank_row = _otr_story_routing.require_runnable_bank(source_bank)   # unchanged
_LANES.assert_supported(_source_bank_row, ...)   # REPLACES the hardcoded :3351-3363 block
```

- **`source_bank` (the local) is REBOUND to the concrete id**, so everything
  downstream -- `require_runnable_bank`, `_resolve_inputs`, `resolved["source_bank"]`,
  `meta["source_bank"]`, pack routing, HUD/credits -- sees a concrete bank and needs
  no change. That is why this change stays small.
- New keyword-only run() param `_bank_roll_receipt=None`, NOT added to the `_core`
  exclusion tuple, so it rides into every refine pass -> step 1 short-circuits ->
  **one submission = one roll.**
- Receipt stamped beside `meta["source_bank"]` (:3634) FROM `_bank_roll_receipt`.
- INPUT_TYPES (:2847-2862): choices become `[SENTINEL] + list(list_bank_ids())`;
  default stays `"science_news"`; the tooltip is REWRITTEN (the current one claims
  other banks "are not yet runnable" -- ten of them are) and documents the roll and
  `OTR_BANK_SEED`. **No new widget, no positional shift** -- `widgets_values` law
  (BUG-LOCAL-097) is not engaged, and the commit message says so explicitly.

### A4 -- registry: NOTHING (D-1)

`banks.json`, `_otr_story_routing`, and the synthetic row builders are **not touched**.
No `rights_class`, no parser change, no schema-doc churn. `runnable` remains the only
curation surface. This is the operator ruling, and it also removes the registry
collision hazard that made the slot requirement so heavy.

### A5 -- build order (two green, PUSHED sub-commits + a closeout; do not interleave)

Each sub-commit: focused tests -> full Windows suite -> Bug Bible -> commit -> push ->
verify `HEAD == origin`. If a sub-commit is not green, STOP; do not stack on it.

1. **`_otr_lane_specs`** + the writer's dispatch/gate swap (`_RUNNER_BY_PIPELINE`,
   `_LEGACY_INLINE_PIPELINES`, `_resolve_lane_runner` DELETED from the writer; all
   four consumers repointed). **Migrate the three EXISTING tests that reference the
   deleted names in this same sub-commit** --
   `tests/test_custom_runner_truthfulness.py:35-45`,
   `tests/test_fable2_runner_ladders.py:471-484`,
   `tests/test_original_codex56sol_registry.py:12-26` -- or the suite is red before
   step 2 exists. Also update `docs/SOURCE_BANK_PREFLIGHT.md:217-220`, whose normative
   checklist still tells a builder to register runners in the table being deleted.
   **This step is valid-request/success-path neutral** -- NOT "behavior-neutral":
   unsupported codex/gemini/sonnet requests now fail EARLIER (at the gate rather than
   inside the runner), with the identical native type and message.
2. **`_otr_bank_roll`** + the writer seam + INPUT_TYPES + the new tests.
3. **Closeout commit** (after qualification): evidence, `docs/PROD_BUG_LOG.md` entries
   for anything that actually failed a LIVE leg, the sprint receipt, GO_FORWARD_PLAN,
   handoff. Pushed. (PRODUCTION_SPRINT_LESSONS:138-139.)

---

## 4. No LLM call exists anywhere above

The roll reads the registry, reads three request fields, filters, and draws. No
prompt, no schema, no repair ladder, no context budget. Lesson 3: selection under a
declared uniform policy over an enumerated pool is MECHANICAL. **If a later draft
introduces a model call into the roll, that draft is wrong.**

---

## 5. Test surface (deterministic only -- no statistical N-trial, no live server)

New `tests/test_bank_roll_design_a.py`:

- **a. widget** -- choices == `[SENTINEL] + list(routing.list_bank_ids())`;
  `SENTINEL not in list_bank_ids()`; default still `science_news`; `order[23] ==
  "source_bank"`. **Update the existing exact-choice assertion** at
  tests/test_source_bank_widget_2c.py:76-83.
- **b. error identity** -- a DIRECT `scifi_fable2` pick at an unsupported
  `target_words` raises fable2's OWN exception type and message; likewise the
  codex/gemini/sonnet 30-900 band on a direct pick. The gate never wraps.
- **c. the filter catches ONLY `compat_errors`** -- a lane whose hook raises
  `ImportError` PROPAGATES; it does not silently become ineligible.
- **d. deterministic draw** -- same seed -> same `selected`, over a SYNTHETIC
  candidate tuple via the pure `draw()` (r2 CUT: no assertion that a live-registry
  seed lands on a NAMED bank -- that breaks every time a bank lands).
- **e. unbiased choice over the FULL tuple** -- inject `rng_factory`; assert the draw
  received the complete sorted `eligible_order` (no truncation, no re-draw loop).
- **f. receipt** -- shape s2, `receipt_version`, `selected` is a real registry row,
  `eligible_order` sorted + JSON list, both `seed_source` values; **ABSENT (not null)
  on a manual pick.**
- **g. refine re-entry reuses the receipt** -- extend `TestRefineCoreCapture` to run
  the SENTINEL: the captured `_core` carries BOTH the concrete bank id AND the
  receipt; the RNG is never called twice (inject a raising `rng_factory`); the final
  meta still carries `bank_roll` on a refine-enabled run. **The defect both panelists
  found gets a test, not just a fix.**
- **h. refine truth** -- with the widget at `Off` but `OTR_STORY_REFINE_BAR` forcing
  `effective_passes >= 2`, dispatched lanes are excluded from the pool (the widget
  string alone would have admitted them).
- **i. eligibility** -- `custom_source_bank` excluded (runnable); **`shakespeare` IS
  eligible** (D-1: no rights filter); with refine effective the pool is exactly the
  **runnable banks on INLINE pipelines**, derived from `INLINE_PIPELINES` (never a
  hard-coded list); at 1000 words codex/gemini/sonnet are excluded while a DIRECT pick
  still fails loud.
- **j. sentinel + nonblank `source_ref` raises** before any RNG call or fetch.
- **k. empty pool raises**, naming every filter that emptied it and its count.
- **l. seed hygiene** -- malformed / negative / >= 2**32 `OTR_BANK_SEED` raises
  `BankRollError` (one loud type); `resolve_seed` reads the SUPPLIED mapping.
- **m. no registry change** -- `banks.json` and `_otr_story_routing` are byte-identical
  after this chunk (guarded by the same no-diff discipline as the workflow JSON).
- **n. lane authority** -- every `LANE_SPECS` row resolves to a callable;
  `runner_for()` returns None for a known inline pipeline and RAISES for an unknown
  one; the telemetry membership branch (:1684) still classifies custom vs legacy
  correctly; importing the writer does NOT import any runner module (pure-import
  test), and the module graph is acyclic.

Then: focused subset -> full Windows suite -> Bug Bible, before any commit.

---

## 6. Workflow JSON obligations

**Expected diff: NONE.** A choice is not persisted; `widgets_values` is. Slot 23 keeps
shipping `science_news`; the vector stays 34.

CLAUDE.md s0 is discharged and RECORDED in the commit message: (1)
`OTR_WorkflowValidator` -> OK, `widget_vector_drift=0`; (2) JSON round-trip; (3)
widget audit (count vs live INPUT_TYPES, every wired input name in INPUT_TYPES, link
referential integrity); (4) **`git diff --exit-code HEAD -- workflows/otr_canonical.json`
plus a pre/post SHA-256** -- `git diff --stat <path>` alone sees only UNSTAGED changes
and would FALSE-PASS after staging. Canonical SHA-256 at `9d8265c0`:
`fb5c75801a5013e189c685dd9d1fbdf069ff22b3843d7ce9adf727efe3c5a830` (re-verify at build).
A non-empty diff means something else moved: **stop and report, do not fix forward.**

---

## 7. Qualification (Lesson 6/7, adapted; runs AFTER r4 and after the code is green)

Box reset per CLAUDE.md s4 before EVERY leg (selective CIM kill by CommandLine, port
8000 clear, VRAM back to the ~1.5-2.0 GB desktop baseline). 30-word canonical smokes,
sentinel selected through the headless whitelist, **against a server booted from the
BUILT code** (the sentinel only validates against the LIVE `/object_info` schema).

**`OTR_BANK_SEED` is set in the LAUNCHER ENV BEFORE the server boots** for legs 1-2
and **CLEARED before leg 3** -- the resident ComfyUI process snapshots its
environment, so exporting a seed afterwards cannot reach it. One boot per leg.

1. **Leg 1 -- seeded onto a LEGACY INLINE bank** (pre-gate resolution, inline shape).
2. **Leg 2 -- seeded onto a DISPATCHED CUSTOM lane** (resolution survives LANE_SPECS
   dispatch).
3. **Leg 3 -- unseeded** (OS entropy; the receipt records what it drew).

Leg 1/2 seeds are **COMPUTED offline** from the shipped resolver against the live
registry at the build HEAD (throwaway script, deleted after) -- never guessed, never
"re-rolled until it lands right."

**Evidence must be CORRELATED, not inferred** (the wrapper reuses one server log and
the launcher truncates it per boot; the API runner stops at status): unique
server/leg log per leg; record the prompt id + start time; select only a NEWLY-created
ledger. Then, per leg, ALL of -- an API SUCCESS is not proof:

- `meta.bank_roll` present, `receipt_version` correct, and
  `meta.source_bank == meta.bank_roll.selected`;
- **legs 1-2:** `selected` == the seed's computed expectation.
  **Leg 3 proves itself by REPLAY** (there is no prior expectation for an entropy
  draw): `draw(receipt.eligible_order, receipt.seed) == receipt.selected` and
  `seed_source == "OS entropy"`;
- the episode asset under `otr\episodes\<ep>\`;
- `obs_publish OK` in that leg's server log;
- **`Test-Path` on BOTH of the ledger's OWN final paths** -- `final_video_path` AND
  `meta.obs_final_path`, both nonblank, both passing (the mux stamps both; "either"
  is how a half-published episode sneaks through). **Never** a path reconstructed from
  `episode_id` -- the muxer derives a sibling dir from the input stem. Output root:
  `C:\Users\jeffr\Documents\ComfyUI\output`.

**The render watchdog is NOT used for these legs.** `scripts/otr_render_watchdog.ps1`
only recognizes `[soak] t=...` heartbeats and cannot read a canonical `RESULT`, so it
would declare a healthy >300s canonical render dead. That is a real harness defect, and
it belongs to `docs/GO_FORWARD_PLAN.md` as a queued item -- **not** to
`docs/PROD_BUG_LOG.md`, which per PRODUCTION_SPRINT_LESSONS s9 is reserved for bugs that
actually failed a live smoke/soak/published episode. It is a review catch. The 30-word
legs are polled directly.

---

## 8. Sprint receipt (filled at close)

```text
SPRINT RECEIPT: PASS | FAIL
scope: randomizer rolls Design A (source-bank roll)
authoritative_writers: _otr_bank_roll.resolve_bank_selection (meta.bank_roll);
                       _otr_lane_specs.LANE_SPECS (dispatch + compat)
durable_artifacts: meta.bank_roll in the episode ledger
canonical_workflow_hash:
focused_tests:
full_suite:
bug_bible:
model_pairings:
30_word_receipts:
120_word_receipts: n/a (the roll is model-free; the rolled LANES are separately qualified)
720_word_receipts: n/a
live_ledgers:
published_assets:
prod_bug_entries:
head:
origin:
remaining_risks:
```
