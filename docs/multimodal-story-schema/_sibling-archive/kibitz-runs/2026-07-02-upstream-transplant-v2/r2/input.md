# r1 Final - Upstream Multi-Source Architecture (synthesized)

Inputs: driver anchor + Codex + Antigravity + Claude Code r1 reviews, all
grounded against production_mirror @ d48a9d76; inherits locked decisions from
ComfyUI-OldTimeRadio/kibitz-runs/2026-07-01-source-bank-visual-style-transplant/r4/final.md
(pre-SFX arc). Judgment log at bottom.

## Locked vocabulary (do not reopen)

- Three selectors: `source_bank`, `story_model`, `visual_style` (+
  `story_pipeline` as the pass-structure axis). `story_model` NAME IS LOCKED
  (prior r4 lock; production stamps `meta.story_model`). To kill the
  LLM-vs-story "model" collision: LLM engines are always called SLOT MODELS
  (creative/technical); the spec carries a `slot_plan` (pass -> slot role),
  never engine ids. Engine ids are resolved at runtime from the production
  node widgets and stamped by production, not by the lab.
- `media_archive` is a source bank; `archival_documentary` is its default
  visual style. Never conflate.

## Seams: templates vs generated fields vs bindings (r1 fix)

PROMPT TEMPLATE SEAMS (pack-ownable prose; label-substitution only, variable
allowlist derived from the resolved profile model fields, never a hardcoded
Python set): outline_system, pitch_room_system, story_select_system,
dramatic_state_system, line_grounding, coda_system (+ coda_mode +
coda_examples), title_system, style_pick_inventor, style_pick_chooser (+
chooser_user_template), labels (story_form_label, source_material_label,
source_develop_verb, source_grounding_label, key_terms_label,
close_brief_label, title_form_label).

GENERATED CONTENT FIELDS (interpreter output, never pack templates):
casting_brief, script_brief, close_brief, key_terms.

PYTHON BEHAVIOR BINDINGS (named, allowlisted, declared in banks.json):
`interpreter` and (science-only, v1) `fetcher`. Declared protocol:

```python
def interpret_source(packet: SourceMaterialPacket, bank: SourceBankSpec,
                     llm_fns: LlmFns | None) -> StoryInputPacket: ...
def fetch_source(bank: SourceBankSpec, request: FetchRequest) -> SourceMaterialPacket: ...
```

Science's production interpreter wraps `build_news_briefs` verbatim and maps
NewsBriefs -> StoryInputPacket. Lab ships deterministic fixture interpreters.
Adding a bank that reuses existing bindings = JSON-only; new behavior = one
new binding + allowlist entry (extensibility claim narrowed accordingly).

## Non-science runtime scope (v1, explicit)

media_archive and public_domain_story are PACKET-DRIVEN in the first
transplant (validated source packets / PD manifests; no live archive RSS -
prior r4 cut list holds). Only science_news fetches at runtime
(`_fetch_rss_seed_or_die`, gated to that bank). A live media-archive fetcher
is a later declared binding, not v1.

## Missing production pass folded in (Codex r1)

`story_brief_reflection` (writer :5438, technical slot) joins the descriptive
pass list of `legacy_many_pass`. It is NOT pack-overridable in v1; a test
asserts the bridge path does not require it. pipelines.json is DESCRIPTIVE
for legacy_many_pass (stamping/audit metadata only - the writer owns the real
sequence) and EXECUTABLE-in-lab only for simple_4_prompt_experimental via a
minimal FakeLLM runner (4 passes; per-pass loud-failure tests; no
generalized simulation framework). Adaptive-cleanup: documentation only, no
schema, no allowlist code (Codex+Antigravity cut, accepted).

## Fallback taxonomy (Codex r1, confirmed in mirror)

Two different things must never be conflated:
- LANE/CONTENT SUBSTITUTION (bank/model/style swapped silently): forbidden
  everywhere, forever. This is what "no fallbacks" means here.
- DETERMINISTIC QUALITY FLOORS inside a lane: production ships three,
  operator-directed, grandfathered for the science baseline: style-picker
  candidate PADDING (_otr_style_picker.py:109-115, :681-690, operator
  directive 2026-06-18), title regen -> outline.title (writer :5201-5208),
  announcer outro resolved fallback (_otr_line_composer.py:3595-3601).
  v1 policy: new lanes FAIL LOUD at these sites (no padding, no silent
  floor); each site enters a FALLBACK_INVENTORY table in the plan with a
  per-lane decision and a test at transplant time.

## Compatibility mirrors (three shapes, canonical-JSON hashed)

- `meta.news` = NewsBriefs fields, all 13, exactly (news_interpreter.py:151-186).
- `meta.news_seed` = {headline, source, url, date, body_chars, style,
  selected_at}; REQUIRED subset = {headline, source} (the keys the adapter
  actually reads, _otr_legacy_to_stage1_adapter.py:95-102); others optional -
  drift test does required-subset matching (Claude Code r1).
- `news_article` (writer internals / news_used socket shape) = {headline,
  summary, full_text, source, date, link, seed_text} (writer :1126-1133).
Drift-proofing stays AST-ONLY against production_mirror (never import
production modules - the mirror is deliberately dependency-incomplete;
Codex+driver confirmed; Antigravity's import-and-introspect proposal
rejected for that reason). Scope: exactly four extractions (NewsBriefs
fields, news_seed docstring dict keys, _LTX_MOTION_PROMPT_BY_ROLE keys, the
four tail constants). Provenance hashes = sha256 over
json.dumps(sort_keys=True, separators=(',',':')) for JSON, raw bytes for .py.

## Visual constants group renamed

`PRODUCTION_VISUAL_TAILS` (not SCI_FI_TAILS - the constants are genre-neutral
cinema tails; `sci_fi_radio` is the policy that must REPRODUCE them
byte-identically: _otr_story_brief_helpers.py:228-251). Motion keys pinned to
{announcer, music_open, music_close, music_inter}; policy validator rejects
unknown keys; dead roles stay dead.

## Inherited locks from the pre-SFX r4 (apply verbatim at transplant)

style-picker override kwargs (inventor_system_prompt, chooser_system_prompt,
chooser_user_template; empty = science default); outline injection via
OutlineRequest.outline_system_prompt with empty -> existing
resolve_creative_system_prompt path; profile holds persona overrides while
OutlineRequest holds labels/verbs/rules; compose_source_coda returns
LineResult; auto story_model resolution in production catalog module;
append-only widgets; no production import from the lab (grep gate); no PD
workflow selector v1; no deep render-driver edits in the source/story stage.
SUPERSEDED by verification: the old "mirror includes title+headline" line -
meta.news has neither (NewsBriefs is authoritative); title/headline live in
news_seed/news_article shapes.

## Tonight's build (revised C-chunks)

C1 contracts.py / compat.py / registry.py / interpreters.py / profiles.py /
   bridge.py / preview.py - as planned, minus adaptive-cleanup schema, plus
   SourceBankSpec bindings + slot_plan + three-mirror emit. banks.json gains
   default_story_pipeline + bank-level label/coda defaults (single-level
   pack override; no inheritance trees).
C2 fixtures - 12 packs ALREADY RECOVERED from git 41c6512 (extraction
   verified, 42 files) and extended to seam-complete; diverged v1
   Python-vs-JSON content resolved by per-seam CHOSEN WINNER with a one-line
   reason in a reconciliation table (union only for forbidden-term metadata);
   PD packet shape documented (file-backed: source_text_ref + manifest with
   text_files/image_files, rights_status=public_domain enforced).
C3 tests - declared-matrix invariants (valid pack pairs x styles, ~60
   combos + one negative per axis, <10s budget), leakage (story + visual),
   AST drift (4 shapes), tail byte-pin, PD path safety, registry fail-loud,
   simple_4 runner loud-failure per pass.
C4 nodes.py (choices registry-discovered, never module lists) +
   scripts/validate_lab.py (KEPT: proven CI/sandbox use without ComfyUI -
   ran headless today; rejection of cut recorded).
C5 production-SHAPED modules tested in lab, deployed NOWHERE tonight:
   _otr_source_interpreter.py, _otr_story_prompt_profile.py,
   _otr_visual_style_policy.py, _otr_ledger_input_adapter.py + module-level
   change DESCRIPTIONS for the writer/composer/picker/brief-helpers edits.
   Line-cited hunks CUT (production HEAD moves nightly - hunks are generated
   at transplant against live HEAD; Claude Code r1 cut accepted). C5 starts
   only after C1-C4 green (translator-head-first, TRANSPLANT_MANIFEST:49-76).

## Judgment log

Accepted: Codex M1 (reflection pass), M3 (fetcher protocol + narrowed
extensibility claim), M4 (fallback taxonomy/inventory), S1 (visual scope
trim), S2 (chosen-winner reconciliation), S3 (AST-only explicit); Antigravity
M1-partial (slot_plan + terminology note), M2 (packet-driven non-science v1),
S1 (derived variable allowlist), S2 (default_story_pipeline in banks.json);
Claude Code M1 (news_seed optionality), M2 (seam vs generated fields), M3
(facade signature), M5 (phasing statement), S1 (tails rename), S3 (combo
budget), S4 (interpret not a seam), S5 (PD shape), O1 (canonical hashing);
cuts: adaptive-cleanup schema, line-cited hunks, full Cartesian.
Rejected: story_model rename (locked vocabulary, cross-repo stamps);
import-the-mirror introspection (mirror is dependency-incomplete by design);
validate_lab.py cut (headless CI use is real); FakeLLM runner cut (kept
minimal, simple_4-only - without it the experimental pipeline is fiction).
Verified-done: v1 pack recovery from 41c6512 (Claude Code M4 was a fair
challenge; extraction already performed and read).
