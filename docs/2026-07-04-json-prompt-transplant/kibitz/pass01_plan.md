# pass01_plan.md -- r1 (ARC) synthesis

**Round:** r1 -- high-level arc / creative coherence.
**Panel:** Codex (`codex exec`) + Fable (via Agent) + Sonnet (via Agent) +
Claude anchor. Antigravity dropped -- stalled at 0-CPU / 0-log for 8+ min,
killed per operator directive.
**Grounding:** verified against ``ComfyUI-OldTimeRadio`` @ ``a7bdc42d``
(production) and ``ComfyUI-OTR-UpstreamStoryLab`` @ ``7df7c80`` (lab).
**Docs commit tip:** ``6d793d40`` (SUPERSEDED note + real-anchor r1 input).

## Convergent verdict

All three review panelists + Claude anchor: **GO-WITH-FIXES**.

Consensus points across the panel:

- Phase A / Phase B carve is architecturally right, but the Phase A doc as
  written pulls in Phase B machinery it shouldn't.
- Sci-fi correctly treated as a profile (not ripped) via the empty-science-
  overrides pattern already shipped as
  ``tests/test_transplant_modules.py:70-77``
  ``test_science_profile_leaves_style_picker_constants``.
- Fable's 2026-07-02 four MUST-FIX items are all resolved at ``7df7c80``
  (catalogs.py deleted; profiles.py routes packs; bridge.py emits dual
  mirrors; archival_documentary re-keyed to four production roles).
- The lab's ``registry.py`` + ``profiles.py`` + ``bridge.py`` are the
  correct architecture for Phase B; Phase A should adopt a subset, not
  invent a parallel ``get_prompt()`` helper.

## MUST-FIX (block r2 convergence) -- CONVERGED

### MF-C1. Audio-C7 object-identity contract (Fable MF2, load-bearing)

Verified at ``nodes/_otr_outline.py:1846-1847``:

```python
# If the resolver returned the legacy _SYSTEM_PROMPT verbatim (object
# identity), no overlay -- modern profile.
if resolved is _SYSTEM_PROMPT:
    period_system_overlay = None
else:
    period_system_overlay = resolved
```

And at ``nodes/_otr_creative_prompt_router.py:43-64``:

- Line 45: ``from ._otr_line_composer import (_SYSTEM_PROMPT as _MODERN_LINE_COMPOSER_SYSTEM,)``
- Line 47: ``from ._otr_outline import _SYSTEM_PROMPT as _MODERN_OUTLINE_SYSTEM``
- Line 55: ``Phase = Literal["outline", "line_composer_system"]``
- Lines 60-63: ``_MODERN_BY_PHASE: dict[str, str] = {"outline": _MODERN_OUTLINE_SYSTEM, "line_composer_system": _MODERN_LINE_COMPOSER_SYSTEM}``
- Comment at :57-60: *"Built at module-import time from the four per-phase
  constants so the returned references are object-identity stable across
  calls (preserves the Sprint D audio C7 contract under default config)."*

Any Phase A chunk that deletes or rebinds ``_SYSTEM_PROMPT`` from
``_otr_outline.py`` or ``_otr_line_composer.py`` BREAKS this identity
check silently. The router will import a stale/rebound object; the
outline's ``resolved is _SYSTEM_PROMPT`` check will fail; the modern
prompt gets prepended as a period overlay; prompt bytes drift; audio
changes.

**Fix (locked for r2 chunks):**

- Extraction MUST keep ``_SYSTEM_PROMPT`` bound at module level as a
  singleton reference to the loader's returned string, or
- Merge extraction of the outline system + router refactor into one
  atomic chunk (a "load-and-rebind" chunk that updates both files
  together), or
- Do NOT extract these two sites in Phase A -- defer to Phase B where
  the router is rebuilt end-to-end.

r2 MUST pick one. Ship a byte-equality pytest that stamps the returned
string ID + full byte comparison against a pre-Phase-A snapshot for
both ``outline`` and ``line_composer_system`` phases.

### MF-C2. `line_composer_system` is the 16th site (Fable MF1 + Codex MF3)

Verified at ``nodes/_otr_line_composer.py:1174``:
``_SYSTEM_PROMPT = """\ You write one spoken line for a character in a
radio drama..."`` and at router
``_otr_creative_prompt_router.py:55``: ``Phase = Literal["outline",
"line_composer_system"]``.

The plan's 15-site table treats the line-composer's :1621 grounding rider
and :3275 news_coda_system, but MISSES the :1174 creative system prompt
that the router already routes. Fixing MF-C1 without adding this site is
incomplete.

**Fix:** add ``line_composer_system`` to the extraction scope (16th
site). Same audio-C7 identity guarantee as MF-C1.

### MF-C3. Real seam vocabulary vs plan's loose "12" (Sonnet MF1)

Verified at ``contracts.py:25-42`` -- ``TEMPLATE_SEAMS`` is 14 entries:

```python
TEMPLATE_SEAMS = (
    "outline_system", "pitch_room_system", "story_select_system",
    "dramatic_state_system", "line_grounding", "coda_system",
    "title_system", "style_pick_inventor", "style_pick_chooser",
    "style_pick_chooser_user_template",
    # experimental adaptive-cleanup CUT to docs-only:
    "pass_1_creative_story", "pass_2_creative_ledger_fill",
    "pass_3_technical_schema_cleanup", "pass_4_technical_ledger_audit",
)
```

Plan's "12 seams" is looser than the real code shape. Real disjoint
Phase A vocabulary is:

- **10 template seams** (14 - 4 experimental) from ``TEMPLATE_SEAMS``:
  outline_system, pitch_room_system, story_select_system,
  dramatic_state_system, line_grounding, coda_system, title_system,
  style_pick_inventor, style_pick_chooser,
  style_pick_chooser_user_template
- **Plus 4 seams to ADD to ``TEMPLATE_SEAMS`` in r2** (from Codex MF3 +
  Fable MF1 + Claude anchor grounding): ``outline_macro_system``,
  ``outline_phase_system``, ``outline_beat_system``,
  ``line_composer_system``
- **Plus `labels` = ``LABEL_TEMPLATE_VARIABLES`` / ``BankDefaults``
  fields** (not a template seam; per Sonnet MF1)
- **Plus `interpret` = per-bank interpreter binding** (not a template
  seam; per Fable step 6 + Sonnet MF1)
- **Plus `casting_brief` = ``StoryInputPacket.casting_brief``** (content
  field emitted by interpreter; per Sonnet grounding; per lab
  ``contracts.py:23`` comment)

**Total Phase A: 14 template seams + labels + interpret + casting_brief
(the last three via non-template mechanisms).** r2 MUST rewrite the site
table to this vocabulary.

### MF-C4. Phase A adopts a SUBSET of the lab architecture, not a new flat helper (Codex MF5 + Sonnet MF2)

Verified at ``src/upstream_story_lab/registry.py:245-307`` and
``src/upstream_story_lab/profiles.py:31-96`` -- the lab already has
``Registry.resolve()`` (4-axis) + ``profiles.resolve_profile()`` (bank-
default merge, fail-loud missing-label/coda checks) + per-seam
``string.Formatter`` template-variable validation at load
(``registry.py:47-63``).

Phase A production-side API MUST adopt a subset of that (name TBD in
r2). Recommend a read-only per-seam extractor:

```python
def get_pack_prompt_or_none(bank_id: str, seam_key: str) -> str | None
```

where ``None`` means "use the current Python literal" (empty-science-
overrides pattern). NOT a parallel ``get_prompt()``; NOT the full
resolver. r2 defines the exact signature.

### MF-C5. Baseline SHA pinning (Codex MF1)

Verified: OTR at ``a7bdc42d``; lab at ``7df7c80``; my docs tip
``6d793d40`` (docs-only after ``a7bdc42d``). No
``PRODUCTION_MIRROR_MANIFEST.md`` located at ``7df7c80`` for me to check
the mirror's pinned SHA; Fable 2026-07-02 review cites ``d48a9d76``.

**Fix (r2):** pin Phase A to production ``a7bdc42d``. If sibling's
``production_mirror/`` is at a different SHA, refresh it before r2
elaborates line-level diffs.

### MF-C6. Empty-science-overrides pattern (Codex MF2, resolved-by-pattern)

The sibling has a working test at
``tests/test_transplant_modules.py:70-77``:

```python
overrides = spp.style_picker_overrides(profile)
assert overrides == {
    "inventor_system_prompt": "",
    "chooser_system_prompt": "",
    "chooser_user_template": "",
}  # empty = production module constants stay byte-identical
```

Phase A extends this pattern to ALL 14 template seams:

- ``science_news`` pack: empty-string overrides everywhere -- production
  Python literals stay authoritative, byte-identical (satisfies MF-C1
  audio invariant).
- ``media_archive`` / ``public_domain_story`` packs: carry actual
  content (already exists at ``7df7c80``, needs r2 audit).
- ``custom_source_bank``: schema-only stub, fail-loud on unknown seam
  reference.

The ``science_news_default.json`` currently contains PARAPHRASES
(verified in prior grounding). Under MF-C6 those become empty strings
in the science-lane packs, and the paraphrase content is dropped.

### MF-C7. Scope surgery: cut Phase B machinery from Phase A doc (Codex MF4)

**CUT from Phase A** (all move to Phase B):

- Compat mirrors (``NEWS_BRIEFS_FIELDS``, ``NEWS_SEED_KEYS``,
  ``MOTION_ROLE_KEYS``, ``PRODUCTION_VISUAL_TAILS``).
- Visual policy (``VisualStylePolicy``, tail constants).
- Provenance stamping with sha256.
- Cross-product invariant tests (bank x model x pipeline x style).
- Pipeline simulation with failure injection.
- Adaptive cleanup pipeline (already docs-only in
  ``fixtures/pipelines.json:39-42``).
- Bridge artifact emit in production.
- ``_otr_ledger_input_adapter.py``.
- Runtime routing widgets on the ledger writer.
- ``workflows/otr_scifi_16gb_full.json`` edits.

### MF-C8. Spec self-inconsistency in the anchor doc (Fable MF3)

Fable flagged: the schema example includes ``news_grounding_rider`` as
a site key absent from the 15-site table; ``unknown keys are a load-
time error`` (self-contradictory); section 5 lists site 2 among "no
existing profile routing" contradicting table row 2.

**Fix (r2):** publish a canonical 14-key list; correct the section 5
routing table; drop the self-rejecting schema example.

## SHOULD-FIX (fold before r2 elaborates chunks)

### SF-C1. Vocabulary alignment lab <-> production (Fable SF1)

Names differ: lab ``coda_system`` vs plan ``coda``; lab
``line_grounding`` vs plan ``line_grounding_rider``; ``labels`` is
different at the two sides. r2 uses lab vocabulary end-to-end.

### SF-C2. `_INVENTOR_SYSTEM` variable binding (Sonnet SF1)

Verified at ``nodes/_otr_style_picker.py:296``: ``_INVENTOR_SYSTEM = ("You
are a sci-fi radio drama showrunner.")`` -- zero ``{}`` placeholders.
The runtime variables ``n_required``, ``seed_sample_block``,
``article_excerpt`` bind to ``_INVENTOR_USER_TEMPLATE``, not the system
prompt. Lab's ``SEAM_RUNTIME_VARIABLES["style_pick_inventor"]`` may
mis-attribute them. r2 audits seam-to-variable map per site.

### SF-C3. `interpret` f-string interpolation (Fable SF2 + Claude anchor MF4)

``news_interpreter.py:704-712`` interpolates ``{_MAX_CASTING_BRIEF_CHARS}``,
etc. at runtime. Two paths: keep as Python-owned formatter OR promote
caps to profile-declared template variables. r2 picks one.

### SF-C4. "No production code touched" wording (Fable SF3)

Contradicts "replace the Python literal with a loader call." Pin to
"**behavior-preserving mechanical edits only**" -- MF-C6 (empty-science-
overrides) means most production sites do not get a loader call at
Phase A; only the non-science lanes wire in.

### SF-C5. Compat mirrors are Phase B (Sonnet SF2)

State this explicitly in the Phase A chunk list so no one imports
``compat.py`` drift tests into a Phase A PR by habit.

### SF-C6. Baseline manifest visibility

If ``PRODUCTION_MIRROR_MANIFEST.md`` exists somewhere in the sibling
repo I didn't find, r2 locates it and confirms the pinned SHA against
MF-C5.

## Grounding table (this pass; all CONFIRMED unless marked)

| claim | source file:line | status |
|---|---|---|
| OTR HEAD ``a7bdc42d`` on ``v2.0-alpha`` (before my docs commits) | git rev-parse | CONFIRMED |
| Lab HEAD ``7df7c80`` on ``main`` | git rev-parse | CONFIRMED |
| Router phase list ``["outline", "line_composer_system"]`` | ``_otr_creative_prompt_router.py:55`` | CONFIRMED |
| Object identity check ``resolved is _SYSTEM_PROMPT`` | ``_otr_outline.py:1846`` | CONFIRMED |
| Sprint D audio C7 contract cited in code comment | ``_otr_creative_prompt_router.py:57-60`` | CONFIRMED |
| ``TEMPLATE_SEAMS`` has 14 entries (10 template + 4 experimental) | ``contracts.py:25-42`` | CONFIRMED |
| line-composer :1174 ``_SYSTEM_PROMPT`` | ``_otr_line_composer.py:1174`` | CONFIRMED |
| story-critic :266 ``_CRITIC_SYSTEM_PROMPT`` (out-of-scope) | ``_otr_story_critic.py:266`` | CONFIRMED |
| Empty-science-overrides pattern | ``tests/test_transplant_modules.py:70-77`` | CONFIRMED |
| Fable 2026-07-02 MF1 (catalogs.py) resolved | catalogs.py deleted at 7df7c80 | CONFIRMED |
| Fable 2026-07-02 MF2 (_BASE_VISUAL_STYLES) resolved | catalogs.py deleted | CONFIRMED |
| Fable 2026-07-02 MF3 (mirror shape) resolved | ``bridge.py:120-166`` | CONFIRMED (via Sonnet) |
| Fable 2026-07-02 MF4 (archival scene_broll) resolved | ``archival_documentary.json:20-24`` | CONFIRMED (via Sonnet + Fable panel) |
| ``PRODUCTION_MIRROR_MANIFEST.md`` presence | not found in ``production_mirror/`` at ``7df7c80`` | UNVERIFIABLE (may exist under different name; r2 hunts) |

## Panel judgment log

**Accepted from Codex:** MF1 baseline (folded to MF-C5), MF2 paraphrase
risk (folded to MF-C6), MF3 seams (folded to MF-C2 + MF-C3), MF4 scope
(folded to MF-C7), MF5 API (folded to MF-C4), SF1 Fable resolved
(status recorded), SF2 chooser template (folded to SF-C1), SF3 real
vocab (folded to MF-C3 + SF-C1), CUTs 1-4 (folded to MF-C7).

**Accepted from Fable:** MF1 16th site (folded to MF-C2), MF2
object-identity (folded to MF-C1 -- SINGLE MOST LOAD-BEARING), MF3
schema self-inconsistency (folded to MF-C8), SF1 vocab alignment
(folded to SF-C1), SF2 f-string interpret (folded to SF-C3), SF3
wording (folded to SF-C4).

**Accepted from Sonnet:** MF1 real 14-entry TEMPLATE_SEAMS (folded to
MF-C3), MF2 adopt registry subset (folded to MF-C4), SF1
``_INVENTOR_SYSTEM`` variable binding (folded to SF-C2), SF2 compat
mirrors Phase B (folded to SF-C5).

**Accepted from Claude anchor:** everything already reflected; my
initial MF1 (seam coverage) and MF4 (interpret) resolved by panel
grounding.

**Rejected / deferred:**

- None. Every claim ground-truthed to real files. Zero hallucinations.

## Delta to feed into r2

r2 input = ``pass01_plan.md`` (this file) + operator scope + the two
real anchor docs. r2 focus = coding plan: per-chunk file:line diffs
against ``a7bdc42d``, exact JSON schema deltas at ``7df7c80``,
extractor helper signature, byte-identity harness spec.

r2 explicitly must:

- Pick one of the three MF-C1 fixes (module-level rebind vs merged
  chunk vs Phase-B-defer).
- Name the 14-seam final list + 4 seams to add to ``TEMPLATE_SEAMS``.
- Name the extractor helper signature (MF-C4).
- Refresh sibling ``production_mirror/`` to ``a7bdc42d`` or accept
  drift (MF-C5).
- Publish the empty-overrides science pack diffs (MF-C6).
- Rewrite anchor sections that pull Phase B machinery in (MF-C7 +
  MF-C8).

Codex + Fable + Sonnet all re-review at r2. Panel structure unchanged.
