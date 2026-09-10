# My Story + native App Mode: implementation and documentation scope

Date: 2026-09-10

Status: SCOPE ONLY. No implementation, workflow edit, render, or qualification is
authorized by this document. The user asked for scoping without coding, then
explicitly included currentizing the README and its linked source-bank documents.
This document scopes that documentation refresh; it does not claim those guides
have already been updated.

Read-only baseline: `v2.0-alpha` at
`1f27e4123a9567400ab39ef698d2a8a394601284`.
Canonical SHA-256:
`9ab0abe6f03f2da845983888f4a1e269066d1d915982453469b6e581da96bcf2`.
Existing unrelated working-tree changes were present and are outside this scope.

## 1. Intended result

A person opens the existing workflow in graph view or optional native App Mode, selects
**My Story**, describes an idea in their own words, chooses the story size and
character count, optionally adds character notes, plot ideas and setting,
chooses a visual style, and runs it. The system generates the story, voices, music, and
finished video. On the supported deployment, where published output is inside
ComfyUI's served output directory, the final playable movie appears in the same
interface. External publishing paths remain supported with the preview
limitation in D4 stated honestly.

Keep this in the existing repository and in `workflows/otr_canonical.json`.
Every authoring field, saved-input draft and published-delivery rule works in
ordinary graph execution. App Mode is an optional presentation; opening its
view/sidebar is never a prerequisite. Graph stays the default.
The source-bank ID proposed here is `my_story`; its user-facing label is
**My Story**. The native dropdown currently displays raw bank IDs, so v1
instructions must say select `my_story` (My Story); setting `BankRow.label`
does not create a friendly combo alias. The precise pack/pipeline IDs are
design deliverables. A custom dropdown-label extension is outside this slice.

The input can be one rough sentence, a plot, fragments, or messy notes. The LLM
interprets that input, fills the gaps, and produces the complete script ledger
needed by the existing production pipeline. The person supplies no ledger,
schema, required outline, or detailed questionnaire. Optional fields provide
useful guidance without requiring every field to be completed. At least one
creative text field must contain input; the numeric controls alone are not an idea.

My Story success means a real published episode. An archival file or status
report is not an alternate successful deliverable. Save the submitted input
as a durable draft on Run for later editing/retry; draft preservation does not
replace generation, rendering or publication. Failures remain explicit failures.

The user's intended story is the creative authority. Preserve explicit cast,
relationships, setting and outcome requirements; distinguish requirements from
incidental mentions, examples, typos and brainstorming alternatives. Unspecified
details are creative choices. Retain the exact raw text as evidence while using
a structured interpretation for generation. This is idea-to-story generation;
arbitrary file/media uploads and verbatim script import are separate features.

The first version uses the existing audio/video presentation choices. Removing
every radio-themed visual, credit, or bookend from the product is a separate
presentation extension and is not required to prove this first creator path.
The new bank may own neutral spoken framing and need not force a sci-fi genre,
vintage announcer or house coda. Explain that the available visual treatments
and presentation are still those of the current OTR production workflow.

## 2. What the current code actually supports

| Finding | Evidence | Scope consequence |
|---|---|---|
| Five runnable banks plus one non-runnable `custom_source_bank` signpost | `nodes/story_packs/banks.json` | Refresh prose saying six runnable banks or calling a new bank the seventh. |
| Original-story input is an operator hint alongside a random spark | `nodes/_otr_writer_inputs.py:266-354`; `nodes/_otr_original_radio.py:263-265` | A form alone does not make the user's idea authoritative. |
| Empty fetcher/interpreter on a runnable bank selects the original-bank initialization shape | `nodes/_otr_writer_inputs.py:_bank_has_no_source_contract` | Give `my_story` explicit initialization; never accidentally inherit the original runner. |
| Every runnable bank is currently eligible for a random roll | `nodes/_otr_rolls.py:eligible_bank_ids` | Required-input banks need an explicit automatic-selection policy. |
| The existing dispatched lane hands a completed ledger to the shared writer tail | `nodes/OTR_LedgerScriptWriter.py:3434-3480` | Add a new dispatched bank at this boundary; the current original runner is inline and is not the template. |
| `LaneSpec` has only `module` and `runner_attr` | `nodes/_otr_lane_specs.py:61-84` | Do not recreate deleted `compat_attr` or word-budget compatibility hooks. |
| No current `story_rules` files or runner argument | Live dispatch above; current `nodes/` file inventory | Remove these stale instructions from the extension guides rather than restoring retired infrastructure. |
| Story size is `act_count`, 1-6 | `nodes/OTR_LedgerScriptWriter.py:2163-2196`; `nodes/_otr_episode_budget.py` | One size authority; no new minute or word target. |
| Canonical defaults to bank/style rolls | `workflows/otr_canonical.json:530-531` | App Mode must not silently change existing saved creative defaults. |
| Installed frontend package is 1.49.6 | Windows venv package METADATA | Native App Mode is a viable integration target; running-server overrides remain unverified. |
| No App Mode layout is saved on the canonical graph | Canonical `extra` currently contains `ds` and `info` | Save the layout on this exact graph. |
| Final mux returns strings, not a UI media result | `nodes/otr_master_audio_mux.py:1069`, `:1520` | A final-player integration is required. |
| Upstream video preview is an intermediate | `nodes/video_engine.py:2365` | It cannot stand in for the finished movie. |

The installed frontend's source maps confirm `extra.linearData` for selected
inputs/outputs and `extra.linearMode` for the default view. Input bindings use
graph/widget identities. Prefer the native builder's saved metadata and verify
save/reload; do not hand-invent cross-version IDs.

Official references:
- [App Mode guide](https://docs.comfy.org/interface/app-mode)
- [App Mode and App Builder announcement](https://blog.comfy.org/p/from-workflow-to-app-introducing)

## 3. Workstream D0: currentize the README and source-bank documentation

Do this as a documentation chunk before implementing the new bank. Describe
current behavior as current and the proposed `my_story` bank as proposed until
it exists and has live proof. Preserve dated historical evidence as history.

| Document | Required refresh |
|---|---|
| `README.md`, Story sources and Preflight guides sections | State the live runnable roster and non-runnable signpost; explain premise semantics per current bank; link directly to the authoring contract, implementation guide, and acceptance checklist; distinguish creating a story from developing a bank. Add App Mode instructions only after the layout and final output are proven. |
| `docs/EXTENDING_OTR.md` | Recheck the activated `user_packs/source_banks/` bundle contract against its live loader/checker. Distinguish that developer extension route from adding a first-party dispatched bank and from an end user typing an idea. Correct roster counts and obsolete claims about cleanup/content filtering. Preserve valid activation, signature-binding, provenance, and quarantine behavior. |
| `docs/SOURCE_BANK_GUIDE.md` | Update the exact runner arguments, registry/pack surfaces, current casting/freeze/tail ownership, and act-based size control. Remove `story_rules`, retired compatibility requirements, and the retired `science_news` default. Keep independent-design and complete-ledger requirements. |
| `docs/SOURCE_BANK_PREFLIGHT.md` | Keep six evidence-based gates, but remove checks for deleted interfaces and superseded content/word policies. Update addition, teardown, and restoration sections consistently. Add input-authority, manual-versus-random availability, App Mode round-trip, and final-player checks with applicability stated explicitly. |
| `docs/PRODUCTION_SPRINT_LESSONS.md`, current source-bank contract summaries only | Reconcile current-use instructions for retired word/cleanup surfaces; keep dated incident narratives intact. Freeze the named source-bank link inventory before editing. No recursive documentation crawl or unrelated video/hardware refresh. |
| `docs/SOAK_LEG_GUIDE.md`, sections 3-5; `docs/README.md`, source-bank navigation only | Correct the directly README-linked source/style widget indices (currently 21/22), distinguish dated slot-proof claims from current paths, and distinguish three media roles from voiced-beat counts. Add navigation to the refreshed source-bank trio. |
| `nodes/story_packs/banks.json`, `custom_source_bank.guide_ref` copy | Replace fragile numbered-roster guidance with accurate count-neutral wording in the later registration chunk; runtime JSON is outside initial Markdown-only D0. |

Specific stale instructions already verified:

1. Six shipped runnable banks: current truth is five runnable plus the signpost.
2. `story_rules/<bank>.json` and a `story_rules` runner argument: absent from the
   live implementation.
3. `LaneSpec.compat_attr`: explicitly removed with the word authority.
4. A shipped `science_news` default: current saved source selection is
   `roll (any eligible bank)`.
5. `target_words` and 30/120/720-word qualification: obsolete controls. Replace
   with explicit current act-count shapes while retaining staged progression,
   model diversity, and ledger/published-asset evidence.
6. Blanket generated-content restrictions: superseded by the 2026-08-03
   operator directive in `CLAUDE.md`. Do not reinstate them in prompts or tests.
7. "Shared writer owns every ledger write" applies to the documented client
   fetch/interpreter path, not every first-party dispatched runner. Explain
   the two supported ownership routes accurately.
8. "Readonly freeze" is not evidence that the earlier shared tail cannot
   transform text. Document the actual final text/TTS hash boundary.

Make the loader/checker boundary precise: static inspection does not import a
bundle; activation probes declared self entry points and fixtures in a bounded
child process before digest/receipt verification. Trusted in-process client
Python is not a security sandbox. Document creative ledger clean, technical
cleanup, reconciliation, freeze policy and final hash/TTS projection in their
actual order. Do not copy obsolete cleanup claims from old docstrings.

The bounded lessons inventory is sections 5-6 (word controls/ladder), 25
(`story_rules`), 34-36 (current applicability of historical band/liveness
guidance), and sprint receipt keys. Prefer a current supersession note beside
dated material. Other linked historical plans remain evidence. The companion
LLM preflight guide has no verified source-bank correction in this audit.

D0 must land before D1 prompt/schema authoring. Explicitly clarify Gate 1's
independence evidence: shared-interface study is necessary integration work;
copying another bank's pass graph or prompts is not independent design. Disclose
the earlier original-lane exposure, retain the six-dimension comparison, and
assess the actual new design against the revised documented criterion. Never
backdate a before-comparison receipt or count this scope as a Gate 1 pass.

Documentation verification is a read-only comparison with live definitions,
paths, registry data, and runnable examples. Check relative links and anchors,
UTF-8/no-BOM, and cross-document agreement. Do not resurrect removed runtime
features merely to make an old checklist pass. Do not record these static
documentation findings as new production bugs.

## 4. Workstream D1: independent bank design before implementation

`SOURCE_BANK_GUIDE.md` and preflight Gate 1 require an independent design, not
the existing original lane with a different prompt pack. The formal design
must specify its source/input authority, pass graph and two-slot assignments,
roles, artifacts, retry boundaries, and ledger-write strategy.

This scope is not that locked design. Earlier discussion inspected the current
original-bank input behavior. Do not retrospectively claim a pristine
before-comparison design receipt. Record that exposure, author the independent
design before further implementation comparison, and evaluate Gate 1 honestly.

Derive the design from these requirements:

- Exact field-level user input is retained before stripping/normalization, with
  a versioned bundle digest and honest user-input provenance. Keep the general
  idea, character notes, plot and setting distinct in that bundle.
- User-authored requirements outrank generated elaboration; random original-bank
  spark material and RSS retrieval do not enter this bank.
- Rough text is interpreted by the LLM into a complete story and ledger without
  a mandatory interview or script-writing step. There is no minimum detail form.
- Missing details are filled creatively. State a bounded ambiguity policy:
  interpret ordinary ambiguity, record material assumptions, and explain an
  irreconcilable explicit constraint instead of silently dropping it. Do not
  literalize every noun into a required speaker or turn semantic taste into a gate.
- All creative fields blank gives a helpful error before model-heavy work; D2 also covers
  upstream downloads for the canonical literal-input route.
- Size limits derive from effective context and output capacity, including
  system prompts and repairs. Reject an input that cannot fit with an actionable
  message; never silently truncate or invent a fixed character/word quota.
- Creative writing and repair use the supplied creative slot; extraction and
  structured verification use the supplied technical slot. No third model,
  direct provider route, or new credential is introduced.
- Tone and visual appearance remain separate: changing the look cannot change
  the story instructions or silently replace the user's setting.

Required design outputs: exact artifact schemas and valid examples; prompt,
schema, parser and repair parity; explicit context/output budgets; the concrete
voice-selection API; cast/cameo policy; field ownership and final text/hash
boundaries; the bank-owned ledger closure proof; and the six-dimension
independence comparison required by the preflight.

Premise fidelity needs actual carried meaning, not just a matching input hash
or an echoed requirement ID. Preserve explicit names/constraints in typed
artifacts and check their owned downstream use. Distinguish mechanically
provable retention from semantic judgment; do not claim a general semantic
guarantee from string matching or turn taste into a runtime gate.

Both `act_count` and `num_characters` remain visible. Act count, 1-6, is the
only length control. Character count keeps its existing meaning: requested
speaking cast size, not an absolute cap. Preserve requested and actual counts
separately; do not overwrite the request to hide a difference. Explicit named
cast/only-these-people instructions outrank an incompatible numeric request,
and the result should explain the difference. The LLM fills unnamed roles when
appropriate. The new dispatched runner owns this policy; incidental names need
not all become speaking roles. Preserve supplied identity/voice attributes, without
inferring gender solely from a name. Specify unsupported structural cast
requirements honestly before expensive production. State how the existing
`include_act_breaks` request is handled; do not inherit inline-original behavior.

The existing cameo-policy matrix must include the new bank. Decide how user
cast restrictions interact with the house cameo explicitly; if a new policy
is needed, name its behavior and receipt rather than inheriting a random cast
change by accident. This is part of the design review, not a global cast reset.

D1 is a hard dependency for D2: until its schemas, pass graph, exact prompt
seams, parser/repair contracts, voice-stock API/cast policy and receipts are
reviewed, the D2 table is an integration inventory, not executable instructions.
Use actual available distinct voices as the capacity boundary; do not invent
a fixed six-character cap or derive it from the six-act limit.
Use existing `_otr_generation_budget.fit_output_tokens` and backend context
resolution: HF tokenized chat-template length, GGUF resolved context cap and
estimate. The GGUF 4096 setting is not every backend's limit. Size artifacts
and repairs explicitly; preserve contract markers and treat no-room as terminal,
not a reason to retry the same impossible request.

## 5. Workstream D2: new bank and safe input routing

| File/surface | Implementation scope |
|---|---|
| `nodes/story_packs/banks.json` | Add the bank row, its own coordinates, truthful defaults, and a declared input/selection policy. |
| `nodes/story_packs/pipelines.json` | Add the executable pipeline, slots, and declared prompt seams. |
| `nodes/story_packs/my_story/<story_model_id>.json` (new) | Store the new bank's own prompts under the current pack schema. |
| `nodes/_otr_my_story.py` (proposed new module) | Implement the independently designed runner and typed artifacts; use the provided ledger and shared tail. |
| `nodes/_otr_lane_specs.py` | Register module and runner by name, with lazy loading. |
| `nodes/_otr_writer_inputs.py` | Handle admitted `user_fields_v1` directly before legacy source-snapshot loading, the empty-fetcher/interpreter original branch and generic `elif custom`. Existing banks retain their source route. Append optional keyword-only field arguments after the existing `*`, with empty defaults. Retain the seven-string payload and versioned raw-field metadata sidecar. |
| `nodes/_otr_story_routing.py` and `nodes/_otr_rolls.py` | Declare and validate manual availability separately from automatic selection. Existing banks keep their current roll eligibility. A required-premise bank cannot be selected by a blank automatic run. No ad hoc special-case fallback. |
| `nodes/OTR_LedgerScriptWriter.py` | Reuse `custom_premise`, `num_characters` and `act_count` positions. Append optional `story_characters`, `story_plot`, `story_setting` string widgets after `replay_from`, with empty defaults and matching trailing run arguments. Forward them through resolution to the new runner. |
| `nodes/_otr_story_input.py` (proposed new shared helper) | Pure typed raw-field intake/admission using supplied policy and resolved source authority. No file/model/runner I/O. D1 locks signatures/examples; distinguish strings, numeric/COMBO values, deferred links and omitted new empty strings. |
| `nodes/_otr_story_drafts.py` (proposed storage/admission coordinator) | Own atomic draft persistence; bind the draft to a queued writer/run and reuse that identity at writer execution. No saved-episode import or snapshot-restoration service. |
| `nodes/_otr_workflow_validator.py` | Inspect the actual queued prompt, scoped to writers depending on this validator, before `ensure_prompt_visual_assets` in both branches. Reuse/factor existing reachability; the saved graph is only the structural authority. |
| Writer metadata, terminal mux and canonical link | Carry publication requirement and stable delivery token through existing writer `script_json` output and a new trailing mux input. This functional wiring lands before My Story is runnable. |
| Casting policy and `tests/test_cast_lock_policy_repin.py` | Record the design's cameo decision and verify user-defined cast requirements. |

The integration policy is explicit: add optional scalar bank defaults at
`defaults.story_input_mode` (`legacy` when absent; `user_fields_v1` for My Story)
and `defaults.auto_select` (boolean, true when absent; false for My Story). Validate known
values and reject `user_fields_v1` plus `auto_select=true`. Do not add unknown
top-level registry keys or require new fields on existing/client rows. The
roll predicate becomes `bank.runnable and effective_auto_select(bank)` with
the existing sorted order. Keep manual `runnable=true`; malformed client policy
uses existing activation/quarantine handling. Registry owns declarations;
the pure input helper consumes resolved policy without importing the writer.

For v1, My Story is manual-only even when a rolled run has nonblank text. Keep
existing banks' premise semantics and roll pool unchanged. Do not globally
auto-route text to My Story, fail existing `custom_premise`-plus-roll runs, or
draw twice. Nonblank dedicated `story_characters`, `story_plot` or `story_setting`
on a non-My-Story selection is a new-input mismatch: fail before roll/downloads
with guidance to select My Story or clear those fields. Empty new defaults
preserve existing workflows. My Story is a fresh-generation path.
In writer `run`, perform this dedicated-field mismatch check before
`resolve_bank_selection`; it is distinct from the evaluated My Story admission
after binding `_source_bank_row`. The validator applies both applicable checks
before assets. Capture the raw request before rolls, persist admitted input
from that capture, and never make `_resolve_inputs` a second storage owner.

Admission ordering is explicit. Canonical link 279 orders validator 63 before
writer 1. The validator currently downloads visual assets before the writer
runs, including with `validate_anyway=False`; validate literal My Story field bundles
before either download path. Use the queued PROMPT and validator dependency
reachability, not saved widget values or unrelated writers. If any applicable
reachable writer has invalid input, raise before downloads; do not merely skip
assets and return success. Unresolved linked values
are not blank: recognize real Comfy links separately from malformed values,
defer them to the writer's evaluated-input check, and state that
the no-download promise covers the canonical/App literal-input route. In the
writer, first reject incompatible saved-episode settings for My Story as below,
then use selection/bank admission and the
new evaluated check immediately after `_source_bank_row` is bound and before
slot preflight, environment changes, input resolution or inference. Reuse one
policy contract for both checks. New creative fields affect only My Story;
other banks keep their existing `custom_premise` semantics. Reject unsupported
nonblank `source_ref` on this bank with a specific message, not a hidden fetch.
Capture raw values before rolls without I/O; later admission/persistence uses
that request, never rebound roll values. Preserve raw `num_characters` before
its legacy clamp in a separate My Story requested-count field. Keep legacy
normalization for other consumers; do not silently rewrite the saved request.

The operator did not request a replay system. Saved-episode rerendering,
source-snapshot restoration and old-bundle compatibility are OUT OF SCOPE.
My Story starts from the submitted fields. Native saved workflow/history reuse
means restoring those controls and generating a fresh episode.
Guard the existing bypasses only: if My Story/new dedicated fields are selected
with nonblank `replay_from`, or My Story has an active source-snapshot manifest,
fail early with clear guidance to clear that setting/use the current canonical
workflow. Check before the writer's existing saved-episode shortcut and before
validator assets; do not silently ignore user input or import saved assets.
Do not extend the snapshot loader, synthesize old metadata, migrate old bundles,
or qualify My Story against old harness graphs. Existing unrelated bank paths
stay outside this new feature. Tests prove these cheap rejection boundaries,
not a new saved-episode rerendering feature.

The raw four-field bundle travels `run -> _resolve_inputs -> resolved/source_meta
-> runner -> final ledger`. Its deterministic seed/full-text projection must
fit the existing seven-string payload; it does not replace the raw evidence.
Do not concatenate and discard field boundaries, add an eighth payload key,
or generate prose in the UI. The LLM interprets the bundle and authors the ledger.

Drafts are input artifacts, stored durably under
`otr_state_dir()/story_drafts/<draft_digest>/input.json`, resolving under
`<output>/otr/episodes/_shared/state/story_drafts/`. The existing output contract
permits only episodes and obs at top level; do not add `otr/story_drafts`.
Use persistent state, never swept tmp or disposable cache.
Include exact creative fields, requested counts, source/style selections and
schema version; D1 specifies deterministic serialization, timestamps outside
the content identity, atomic writes and idempotent reuse for identical input.
For canonical literal inputs, save the admitted draft before asset downloads;
for evaluated linked input, save before writer generation. One coordinator does
idempotent create/verify through either boundary, with same-directory atomic
writes and matching-digest verification for existing entries. A storage error
prevents new generation instead of claiming input was saved. Later ledger
creation records the draft identity.
Draft persistence begins when admission executes, before new generation. Invalid
blank submissions and queued jobs canceled before any admission node executes
are not promised a new draft file; native saved workflow/history is the separate
recovery route. Test cancellation after admission retains the saved draft.
No keystroke autosave, draft-approval wizard, or draft-browser UI is required;
native saved-workflow/history reuse instructions belong in the finished guide;
people need not author JSON. A field-only draft-library restore UI is not claimed. Rendered
assets still go directly to episodes, and the published movie goes to OBS.
Capture submitted controls before bank/style rolls so drafts retain requested
values; resolved choices belong in episode receipts and must not change the
draft digest between validator and writer. D1 specifies this draft schema.
The installed `comfy_execution.utils.get_executing_context()` supplies prompt
and node identity. Coordinate validator/target writer by prompt ID and target
writer, with mapped-call identity if needed; separate writers must not collide.
Direct calls outside ComfyUI need an explicit caller/test identity or a
writer-owned submission. Do not create an episode merely to save input, and
never persist from `IS_CHANGED`.
Resolve core execution context lazily so importing the new helper in standalone
tests does not require ComfyUI on `sys.path`. The installed context is a
NamedTuple with `prompt_id`, `node_id`, `list_index`, not a dict. Prefer explicit
caller context when supplied; do not invent shared `test_standalone`/node1
identities on missing context. Test independent writers and direct callers.

Every authored field has one owner. The runner supplies the existing cast,
scenes, shots, beats, lines and music structures, and leaves downstream-owned
clips/timing to their producers. It returns the current tail parts; the writer
constructs `WriterTailContext`. Preserve source/title provenance through that
tail, and prove final accepted text, TTS projection, and hashes agree.
Run through existing freeze, the sole publication-eligibility receipt writer.
Verify its same-episode receipt reaches the terminal mux. Do not hand-stamp
`publishable=True`, invent a user license, or mistake absent bibliographic
rights provenance for a publication failure on an original story.

Match live runner keyword arguments exactly: `payload`, `pack`, `resolved`,
`led`, `meta`, `creative_fn`, `technical_fn`, `slot_scheduler`,
`source_bank_row`, `episode_root`, `episode_id`. The runner returns its own
duck-typed tail parts with `outline_view`, `canon`, `final_title_override`,
`run_story_spine` and any supported finalizer; the writer builds the shared
context. Do not import another bank's tail-parts class or duplicate context
construction. D1 specifies the full shapes those attributes require.
At minimum `outline_view.title` and `.premise` are strings; `canon` supports
shared title binding and `_OTRC.write_canon`. These minimum interfaces do not
substitute for the complete independent bank design.

Register only in dispatched `LANE_SPECS`, not `INLINE_PIPELINES`. Pipeline
passes use existing creative/technical slots; custom declared seams and
production `required_seams` obey the live registry/pack validation rules.
Explicit graph `episode_title` retains shared-tail precedence; blank generates
a title. Record the final title and its authority after the tail rather than
claiming the runner title always wins.

No new source bank is marked runnable before its pack, runner, registration,
input route, and tests land together.

## 6. Workstream D3: native App Mode on the canonical graph

Use one native form with optional text guidance; expose node-1 controls:

- `source_bank`: selects `my_story` (My Story) or an existing source. Keep this visible in
  the first native version so selecting the creator path is explicit.
- `custom_premise`: use a bank-honest shared label such as **Story input**, with
  a roomy multiline field and App description **Your story idea for My Story**.
  Describe it as authoritative only after selecting My Story; other banks
  retain their existing hint/source-override behavior. It is not inert elsewhere.
- `num_characters`: **Character count**, preserving its existing requested-size
  semantics and allowed values. No second cast-size control is added.
- `act_count`: label it **Story size (acts)**; explain 1 as a single scene,
  3 as a short arc, and 6 as a fuller arc. Keep the existing 1-6 values.
- `story_characters` (new optional): **Character names and notes**; allow names,
  relationships, roles or voice/identity details in ordinary text.
- `story_plot` (new optional): **Plot ideas**; events, conflicts or an ending.
- `story_setting` (new optional): **Setting**; place, era, atmosphere or world.
- `visual_style`: label it **Look** and use the existing selectable styles.

Use native Run/Cancel/queue/results. Native configuration supports reordering,
descriptions, multiline size, and field-label changes. It does not establish
custom button text, renamed individual combo values, thumbnail pickers, a
draft-approval wizard, or detailed per-production-stage progress.
Native rename changes the shared widget/input label, including graph view;
only the App description is isolated layout help. Raw combo option IDs remain
visible, and My Story must be clearly distinguished from `custom_source_bank`
(the developer extension signpost).
Save labels with native rename (shared widget/input `label`), not by renaming
programmatic input keys or changing `localized_name`. `linearData.inputs` uses
`[widgetId, widgetName, {height?, description?}]`; it has no label key. Arrange
the existing core controls before optional detail fields using App Builder.
Any one creative field is enough. Native conditional hiding/grouped wizard
behavior is unverified and not required; descriptions identify My Story-only fields.

App Mode and graph mode edit the SAME widget values. There is no verified
separate creator-only set of defaults. Preserve the current saved bank/style
rolls and hardware values in the minimum change. A later decision to default
the canonical to My Story must deliberately account for existing dailies; it
is not a harmless display preference. Preserve graph mode as the initial view
in v1; save the native App layout so it is available by switching views. Keep
Source first with visible guidance to choose My Story before typing/running.
The first native flow therefore includes that explicit selection; it is not an
independent app with its own hidden defaults.

All layout metadata and any changed widget/input descriptors go into the real
canonical JSON in the same implementation chunk. Keep its graph identity and
links intact. Verify with the workflow validator, JSON round-trip, link/input
audits and live widget counts. Append any genuinely new optional widget.
Here the three optional fields are genuinely new: append after current widget
32 (`replay_from`), preserving `num_characters`1, `custom_premise`4, `act_count`6
and bank/style21/22. Update descriptors, tests and the canonical together.
Widget indices are not socket indices: `gate_in` stays writer input slot32 and
link279 keeps that destination. Append after the existing replay input; never
insert ahead of the gate or retarget it when widgets grow from33 to36.
Regenerate derived platform artifacts with the existing variant builder and
run its check when schema/canonical inputs change. These remain derivatives;
every headless run still loads the real canonical.
Existing saved graph copies need the updated canonical or an explicit resave
procedure; do not weaken strict widget-vector checks to hide shorter vectors.

## 7. Workstream D4: show the terminal movie

Select terminal node 85 (`OTR_MasterAudioMux`) as the App Mode output. Extend
its successful return with supported ComfyUI media metadata and keep the two
wire values/types under `result`. A UI dictionary changes the direct Python
return shape: migrate tuple-unpacking callers and tests explicitly, including
`test_publication_eligibility.py` and `test_video_render_path_cw4.py`. Do not
claim direct-call tuple compatibility or add a shape-hiding shim.
There are four active successful tuple unpackers (publication tests at
582/621/679 and video-render test375) and a raising call at video-render142.
Keep the latter raising. Preserve `(final, report_text)` under `result`; the
second wire is not replaced with a report pathname.

Publication enforcement works without App Mode. Append optional force-input
`script_json` to mux85 (Python default `None` for legacy absent direct calls),
and wire existing writer1 output slot1 to it with a fresh link ID and both
endpoints updated. No new writer output or node. Present empty/malformed input
is an error, never equivalent to legacy absence.
Append at mux inputs[10], preserving video_policy_json input8/link287 and
foley_receipts_json input9/link288; allocate above current last_link_id 290.

Before serialization, the writer stamps a typed delivery intent containing
schema version, source-bank identity, required-publication boolean, draft digest
and a fresh stable delivery token. Stamp this envelope for EVERY new bank run:
My Story has required-publication true; existing banks false, with absent draft
provenance represented explicitly by the D1 schema. My Story requires complete intent with
publication true; missing/contradictory intent fails. The existing full ledger
JSON carries it to the mux, which treats story content as opaque. This wire
expresses required delivery, never permission to publish.

Bind wire/live ledger by delivery token plus bank identity. A draft digest is
not a unique run. The writer's pending root episode ID is later renamed by
video_engine; do not compare that stale ID with the terminal ID. The token
survives rename. This is fresh-run delivery identity, not a saved-episode
replay system. Do not extend replay import, synthesize old intent, migrate old
bundles or reconstruct old metadata. Test every existing bank's NEW runs with
the always-connected wire. Use the terminal ledger's current identity for freeze
receipt/asset checks. Missing/unrelated ledger cannot erase the wired requirement.
Legacy absent-wire direct calls keep their behavior; the supported My Story
canonical must have this link and valid intent before becoming runnable.

Include the new input/intent in terminal cache identity. A required published
file that has disappeared must invalidate success: force revalidation or use
verified output existence/identity in cache validity. Test successful publish,
deleted/moved OBS output, and unchanged-input rerun. Cache checks stay read-only.

The player must use the actual published, browser-compatible OBS MP4 after
successful mux/publication and terminal-path stamping: `obs_copy` / stamped
`meta.obs_final_path`, never archival `final` / `final_video_path`. The archival
PCM-in-MP4 path remains result[0]; the published AAC copy supplies `ui.gifs`.
Resolve filename and subfolder from ComfyUI's real output root and that OBS path.
Do not reconstruct filenames or display the upstream intermediate as final.

Use the actual server root from `folder_paths.get_output_directory()`, not the
assumption that `_otr_paths.comfy_output_dir()` always equals it. Both
`OTR_OUTPUT_DIR` and `OTR_OBS_DIR` can override defaults. The current canonical
launcher aligns the roots, but external OBS configuration is legitimate.
For an external/unservable terminal path, preserve publication and show its
truthful episode report with an explicit preview-unavailable status; emit no dead
or traversal descriptor. Automatic viewing copies/endpoints are deferred in
v1. This limitation must appear in setup and acceptance instructions.

My Story must reach successful publication, with a real OBS file, to count as
success. Its delivery contract is recorded and enforced at the terminal owner;
an unexpected blocked/missing/malformed eligibility receipt is a failed My
Story delivery, never successful archival-only output. Keep freeze as the
eligibility authority and add no content filters or invented permission stamp.
Preserve existing other-bank publication-withheld behavior. D1 must specify
the exact delivery-intent metadata/owner and compatibility tests before code.

A report may accompany a published episode; it never replaces it. Persist it
directly under the authoritative episode directory with an episode-specific
name and expose it through native `ui.files` with a real `.txt` descriptor.
Core SaveText is the supported pattern; a STRING output or `ui.text` alone
does not appear in App results. Include episode identity, publication status,
actual paths and draft identity. Do not use an overwritten shared report name.
Keep the movie as the ordinary successful preview. A native text report is
needed when inline playback is unavailable or an existing bank withholds
publication; it is optional beside a working player. Do not make auxiliary
report failure erase an already verified publication, and never swallow actual
production/publication exceptions.
Emit descriptors only for actual files inside the server's served root. For
external OBS but in-tree episodes, the published movie exists and the report
explains unavailable inline playback. If episode storage itself is external,
both native outputs are unsupported: diagnose during App setup, retain wire/log
reports and existing headless behavior, and state the limitation. A real
production exception still raises; failure/cancel retains the input draft.

Current mux behavior for existing banks has a publication-withheld branch. Respect
that distinction: a successful archival file is not proof of OBS publication,
and the UI must not falsely label that case as a published story. Existing
failure and cancellation behavior must remain truthful.

Installed frontend parsing supports MP4 media descriptors in principle. Prove
actual audio/video playback, download/open behavior, and history/reload with
this custom node before advertising the player as working. No copied media or
new public web endpoint is required for the aligned-root minimum. Use the
installed frontend's supported MP4 media envelope (existing `ui.gifs` pattern);
use `{filename, subfolder, type:"output"}` for a real `.mp4`;
`format:"video/mp4"` is supported but optional in the inspected frontend.
Keep `final_video_path`'s archival meaning intact. Associate results with the
current episode/job; deliberately selected history may remain visible during
a new run, so do not claim every old video always clears on cancel or failure.

## 8. Verification and acceptance scope

### Offline and structural checks

Extend the relevant existing suites, deriving the live roster rather than
copying old counts:

- `test_story_routing_stage2.py`, `test_story_pack_stage1.py`,
  `test_bank_variants.py`, `test_lane_specs.py`;
- `test_source_payload_chunk3.py`, `test_source_bank_widget_2c.py`,
  `test_rolls_source_bank_and_visual_style.py`;
- `test_freeze_policy_readonly.py`, `test_cast_lock_policy_repin.py`;
- new bank-owned tests for premise admission/retention, retries, authorship,
  voice/graph closure and shared-tail handoff;
- focused terminal-media metadata and App Mode binding checks.

Essential cases: blank input fails early; no RSS/spark call for My Story;
named characters and explicit premise constraints survive accepted artifacts
and all repair routes; changed look leaves the user-story authoring contract
unchanged (visual-reflection messages may differ); existing
banks keep their input/roll behavior; the new bank remains manually selectable
while blank automatic runs retain their existing eligible pool; accepted text
and final provenance survive the tail; cancelled/failed runs show no false
finished result.

Cover each creative field alone, combinations, all-blank/whitespace bundles,
explicit names plus count requests, exact raw-field preservation, and empty
defaults on old direct-call signatures. Cover queued literal admission before asset calls in both validator
modes; unrelated and multiple writers; unresolved linked values;
direct evaluated writer calls; one-sentence and messy-note input; explicit cast
larger than the hidden default; context exhaustion without truncation; repeated
generation from reused input controls; same-episode freeze receipt; aligned,
external OBS, and mismatched OTR/server roots; and publication withheld. Keep
structural checks separate from semantic assertions and human spot checks.
Prove incompatible saved-episode/source-snapshot settings fail early for My
Story. No saved-episode system or old-harness qualification is required.
Update widget-count fixtures for the three appended fields;
derive roster/automatic eligibility without assuming every runnable bank rolls.
Prove draft persistence before generation and on later failure/cancel, immutable
identity on repeats, and draft-to-episode linkage. My Story green requires
published-file proof; an unexpected publication block fails its delivery.
Add stable delivery-token checks across episode rename and fresh generation;
missing wire/intent, false intent, missing or unrelated ledger, malformed/wrong
eligibility and vanished published output must not yield My Story success.
Prove both graph and App execution with the same controls. Test wrong-bank
dedicated fields without changing legacy custom-premise/roll behavior.

### Live qualification after implementation

Replace the obsolete word ladder in the operational docs with a recorded
act-based ladder: one-act canonical smokes first, then three-act runs with the
same selected model pairings, then the supported six-act boundary. Retain the
preflight's two materially different local model families and one configured
frontier/cloud creative lane, independent technical-slot evidence, and real
ledger/publication receipts. This is compatibility proof, not a story-quality
contest or a reason to change the accepted writer-model defaults.
Use an already configured cloud lane and introduce no credential. If it is
unavailable, record that qualification as pending/N/A with its reason and do
not claim an unconditional six-gate pass; do not silently remove the requirement.

Run the full Windows regression suite and Bug Bible after code changes, as
required. Reset selectively before every headless run; load the real canonical
workflow; use existing sanctioned machine settings and output paths. Verify
both episode assets and the published file under `otr/obs/`. Record exact
model labels, requests, workflow hash, ledgers, and assets; never count a
resident server or API success alone as completion.

Browser acceptance: save/reload and switch App/Graph; confirm the submitted
values; play the final movie with its final master audio; reload/history and
play again; check clear errors/cancellation. Establish the frontend version
actually tested, rather than assuming the documented App Mode minimum proves
compatibility with this custom output.
Prove saved workflow/history Reuse parameters restores all fields and counts;
history restores a whole graph, not a field-only draft. Test new generation
after reuse, current versus deliberately selected
historical output, and the default playable result. Run the full creator path
in graph view with App Mode never opened. Declare the tested core/context API
as well as frontend version; optional App layout does not create an alternate backend.

Shared-code changes must demonstrate that existing bank and machine paths
remain intact. Hardware-specific qualification stays with the existing
machine-profile process; this feature does not reopen the video-engine matrix.

## 9. Evidence and delivery sequence

### Priority #1 and ownership handoff

The operator's September 10 instruction makes this plan and the already
reviewed [Opus cleanup handoff](2026-09-10-cleanup-opus/HANDOFF.md) /
[implementation plan](2026-09-10-cleanup-opus/PLAN.md) one priority #1
programme in [GO_FORWARD_PLAN](GO_FORWARD_PLAN.md). Production implementation
is still pending. Run these five sprints in order:

1. Opus P1 (C4 + C1): remove automatic roomtone/tape hiss, set clean code and
   canonical defaults together, and delete the seven unused assignments.
2. Opus P2 (C2), then P3 (C3), as separate reviewed/qualified/pushed chunks:
   one G8 duplicate-ID diagnostic owner, then removal of unused freeze model
   acquisition while preserving validation, unloading and recovery contracts.
3. My Story D0 documentation currentization against that cleaned-up HEAD,
   followed by D1 independent bank design and its grounded review.
4. Complete the creator backend and normal graph route, including durable
   drafts and required publication, with all functional canonical wiring in
   the same changes and before the bank is runnable.
5. Ship optional App presentation and native player/report integration,
   complete canonical/model/browser qualification and final user docs.

Opus owns the cleanup files until its final qualified push. My Story then
re-reads the resulting HEAD before touching shared writer/validator/freeze or
canonical contracts. Do not interleave these baselines. Cleanup qualifications
use the linked plan's measured existing-failure comparison; its 54 existing
unexpected failures do not mean a green full suite. No new GPU/server run is
part of cleanup. Later creator qualification follows this plan's canonical-run
requirements. Hardware asset-recovery obligations remain separate. The older
GO_FORWARD queue resumes after this programme; no release promotion is implied.

The operator confirmed coding ownership: **Opus owns sprints 1-2; Codex owns
sprints 3-5.** Codex starts dependent work after verifying the cleanup pushes.
A different reviewer checks each finished code change under the standing rule.

Cleanup already has its own completed R1-R4 campaign and receipts. This scope's
R4 checks the combined dependency handoff; it does not claim that R1-R3 of this
campaign re-reviewed cleanup implementation. The next coder must still perform
the required finished-change review and qualification in each linked plan.

### My Story internal delivery order

1. Currentize the scoped documentation against live code and operator rulings.
2. Write and review the independent My Story design and exact integration plan.
3. Implement bank/ingress/drafts/selection plus required-publication metadata,
   terminal contract and direct canonical wire. In this SAME
   chunk append writer widgets/input descriptors/defaults, synchronize canonical
   vectors and links, regenerate/check derived variants and update fixtures.
   Mark the bank runnable only when normal graph execution can publish fully.
   Keep mux's direct return as the existing `(final, report_text)` tuple in
   this chunk; new delivery enforcement alone does not require a UI envelope.
4. Add optional App layout/labels and terminal native player/report output;
   update the canonical metadata and direct-call return consumers atomically
   with those changes. Functional delivery never depends on enabling App Mode.
5. Run required tests and canonical/live/browser qualification; finish the
   preflight item-by-item evidence matrix and hashes.
6. Update the README with proven user instructions and accurately stated
   limitations. Commit and push each green chunk to `v2.0-alpha`; registry
   publication, tags and promotion remain separate operator decisions.

This document completed the user-requested R1-R4 scope campaign, grounded by
the driver. The [review receipt](2026-09-10-my-story-app-review/README.md)
records actual rounds, reviewers, judgments and final operator scope cuts.
That hardens scope and integration obligations; D1's exact bank schemas/pass
graph still need a grounded design review before code. Neither scope convergence
nor a panel verdict passes a bank preflight gate. The required local CLI review
of the finished code change still applies.

Relevant production lessons consulted: source meaning can disappear despite
valid IDs (PBUG-20260712-04 / Bible 11.39); retakes must use the same guarded
acceptance boundary (PBUG-20260712-15 / Bible 11.44); shared defaults can corrupt
lane provenance (PBUG-20260712-05 / Bible 12.49); the terminal publisher owns
observed output paths (PBUG-20260721-03 / Bible 12.56); offered size values must
fit real capacity (PBUG-20260825-01). Older word-policy entries in the lessons
and Bible are historical and do not supersede the current no-word-chasing law.

Read-only contributors to this scope: the primary Codex agent; a source-bank
contract/documentation reviewer; and an App Mode/frontend/output reviewer.
All inspected the real Windows files. No test, render, code edit, source-bank
registration, or App Mode workflow edit was performed during scoping.

## 10. Explicitly later

Speaker placement is a separate optional follow-on, not part of clean-audio
P1 or a blocker for the creator flow. The preferred listening-test starting
point is gentle, balanced placement assigned once per episode by stable
speaker identity, narration centered, every supported cast size handled, and
the same position retained between lines. Bounded randomness is an alternative
to evaluate only if listening evidence warrants it. No numeric pan law or DSP
algorithm is approved by this scope.

Design this at the per-speaker assembly stage while identity is available;
whole-mix enhancement cannot independently place already mixed voices. Define
channel/sample shape and propagation through concatenation, enhancement,
crossfades, export and OBS. Persist the placement map and revision with the
episode; prove levels, mono compatibility, timing and cache identity.
Applying a new placement map requires fresh assembly. Include small and
larger cast listening checks, and review
this independent design before implementation. It does not reopen the clean
audio decision or add roomtone/hiss.

Bring-your-own audio, verbatim scripts, image uploads, timeline editing,
multi-step draft approval, a separate website/repository, bespoke progress UI,
new model engines, a broad rebrand, and a universal one-click installer are
outside this first slice. The installation/weight-readiness work already in
the release queue remains necessary; hiding nodes does not remove it.

Relative effort: documentation refresh and native form configuration are the
smaller pieces; final-player integration is bounded but needs browser proof;
the independent bank and its qualification are the substantial work. No
calendar estimate is justified before the independent design is reviewed.
