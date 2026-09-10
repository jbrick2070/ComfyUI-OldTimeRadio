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

A person opens the existing workflow in native ComfyUI App Mode, selects
**My Story**, describes an idea in their own words, chooses the story size and
visual style, and runs it. The system generates the story, voices, music, and
finished video. The final playable movie appears in the same interface.

Keep this in the existing repository and in `workflows/otr_canonical.json`.
The source-bank ID proposed here is `my_story`; its user-facing label is
**My Story**. The precise pack and pipeline IDs are design deliverables.

The user's supplied premise is the creative authority for this bank. The model
may elaborate unspecified details, but must preserve supplied characters,
relationships, setting, and explicitly requested outcomes. This is generation
from an idea, not verbatim script import or adaptation of arbitrary documents.

The first version uses the existing audio/video presentation choices. Removing
every radio-themed visual, credit, or bookend from the product is a separate
presentation extension and is not required to prove this first creator path.

## 2. What the current code actually supports

| Finding | Evidence | Scope consequence |
|---|---|---|
| Five runnable banks plus one non-runnable `custom_source_bank` signpost | `nodes/story_packs/banks.json` | Refresh prose saying six runnable banks or calling a new bank the seventh. |
| Original-story input is an operator hint alongside a random spark | `nodes/_otr_writer_inputs.py:266-354`; `nodes/_otr_original_radio.py:263-265` | A form alone does not make the user's idea authoritative. |
| Empty fetcher/interpreter on a runnable bank selects the original-bank initialization shape | `nodes/_otr_writer_inputs.py:_bank_has_no_source_contract` | Give `my_story` explicit initialization; never accidentally inherit the original runner. |
| Every runnable bank is currently eligible for a random roll | `nodes/_otr_rolls.py:eligible_bank_ids` | Required-input banks need an explicit automatic-selection policy. |
| Dispatched runners already hand a completed ledger to the shared writer tail | `nodes/OTR_LedgerScriptWriter.py:3434-3480` | Reuse this production boundary. |
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
| `docs/PRODUCTION_SPRINT_LESSONS.md` and other directly linked operational guidance | Reconcile current-use references to the old word ladder and removed surfaces. Keep incident narratives intact. Follow links from the four main documents and correct live-contract claims within this source-bank subject; do not turn this into unrelated video/hardware documentation work. |

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

- Exact user input is retained with a digest and honest user-premise provenance.
- User-authored requirements outrank generated elaboration; random original-bank
  spark material and RSS retrieval do not enter this bank.
- A short idea works without a mandatory interview or script-writing step.
- Empty input gives a helpful error before any LLM/model-heavy story work.
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

The existing cameo-policy matrix must include the new bank. Decide how user
cast restrictions interact with the house cameo explicitly; if a new policy
is needed, name its behavior and receipt rather than inheriting a random cast
change by accident. This is part of the design review, not a global cast reset.

## 5. Workstream D2: new bank and safe input routing

| File/surface | Implementation scope |
|---|---|
| `nodes/story_packs/banks.json` | Add the bank row, its own coordinates, truthful defaults, and a declared input/selection policy. |
| `nodes/story_packs/pipelines.json` | Add the executable pipeline, slots, and declared prompt seams. |
| `nodes/story_packs/my_story/<story_model_id>.json` (new) | Store the new bank's own prompts under the current pack schema. |
| `nodes/_otr_my_story.py` (proposed new module) | Implement the independently designed runner and typed artifacts; use the provided ledger and shared tail. |
| `nodes/_otr_lane_specs.py` | Register module and runner by name, with lazy loading. |
| `nodes/_otr_writer_inputs.py` | Add explicit premise-led ingress and early blank-input validation; avoid the broad empty-fetcher/interpreter original-bank branch. Retain the shared seven-string payload envelope where possible and use provenance sidecars. |
| `nodes/_otr_story_routing.py` and `nodes/_otr_rolls.py` | Declare and validate manual availability separately from automatic selection. Existing banks keep their current roll eligibility. A required-premise bank cannot be selected by a blank automatic run. No ad hoc special-case fallback. |
| `nodes/OTR_LedgerScriptWriter.py` | Reuse the existing `custom_premise` widget position; correct help text and place validation before expensive work. Reuse dispatched-runner handoff. |
| Casting policy and `tests/test_cast_lock_policy_repin.py` | Record the design's cameo decision and verify user-defined cast requirements. |

A scalar bank default is a possible place for an explicitly validated
input/roll policy because the live bank schema already supports scalar
defaults. The exact field design is not locked here. Do not add required
fields that break existing banks or activated client bundles. Manual runnable
status must not be set false merely to hide a usable bank from random rolls.

Every authored field has one owner. The runner supplies the existing cast,
scenes, shots, beats, lines and music structures, and leaves downstream-owned
clips/timing to their producers. It returns the current tail parts; the writer
constructs `WriterTailContext`. Preserve source/title provenance through that
tail, and prove final accepted text, TTS projection, and hashes agree.

No new source bank is marked runnable before its pack, runner, registration,
input route, and tests land together.

## 6. Workstream D3: native App Mode on the canonical graph

Expose the existing node-1 widgets:

- `source_bank`: selects My Story or an existing source. Keep this visible in
  the first native version so selecting the creator path is explicit.
- `custom_premise`: label it **Your story idea**, with a roomy multiline field.
- `act_count`: label it **Story size (acts)**; explain 1 as a single scene,
  3 as a short arc, and 6 as a fuller arc. Keep the existing 1-6 values.
- `visual_style`: label it **Look** and use the existing selectable styles.

Use native Run/Cancel/queue/results. Native configuration supports reordering,
descriptions, multiline size, and field-label changes. It does not establish
custom button text, renamed individual combo values, thumbnail pickers, a
draft-approval wizard, or detailed per-production-stage progress.

App Mode and graph mode edit the SAME widget values. There is no verified
separate creator-only set of defaults. Preserve the current saved bank/style
rolls and hardware values in the minimum change. A later decision to default
the canonical to My Story must deliberately account for existing dailies; it
is not a harmless display preference. Saving App Mode as the initial view is
a separate layout choice and must survive reopening the graph.

All layout metadata and any changed widget/input descriptors go into the real
canonical JSON in the same implementation chunk. Keep its graph identity and
links intact. Verify with the workflow validator, JSON round-trip, link/input
audits and live widget counts. Append any genuinely new optional widget.

## 7. Workstream D4: show the terminal movie

Select terminal node 85 (`OTR_MasterAudioMux`) as the App Mode output. Extend
its successful return with supported ComfyUI media metadata while preserving
the current result tuple and output types for existing consumers.

The player must use the actual published, browser-compatible OBS MP4 after
successful mux/publication and terminal-path stamping. Resolve filename and
subfolder from ComfyUI's real output root and the observed terminal path.
Do not reconstruct filenames or display the upstream intermediate as final.

Current mux behavior also has an explicit publication-withheld branch. Respect
that distinction: a successful archival file is not proof of OBS publication,
and the UI must not falsely label that case as a published story. Existing
failure and cancellation behavior must remain truthful.

Installed frontend parsing supports MP4 media descriptors in principle. Prove
actual audio/video playback, download/open behavior, and history/reload with
this custom node before advertising the player as working. No copied media or
new public web endpoint is necessary for the minimum design.

## 8. Verification and acceptance scope

### Offline and structural checks

Extend the relevant existing suites, deriving the live roster rather than
copying old counts:

- `test_story_routing_stage2.py`, `test_story_pack_stage1.py`,
  `test_bank_variants.py`, `test_lane_specs.py`;
- `test_source_payload_chunk3.py`, `test_source_bank_widget_2c.py`,
  `test_rolls_source_bank_and_visual_style.py`, `test_source_snapshot.py`;
- `test_freeze_policy_readonly.py`, `test_cast_lock_policy_repin.py`;
- new bank-owned tests for premise admission/retention, retries, authorship,
  voice/graph closure and shared-tail handoff;
- focused terminal-media metadata and App Mode binding checks.

Essential cases: blank input fails early; no RSS/spark call for My Story;
named characters and explicit premise constraints survive accepted artifacts
and all repair routes; changed look leaves story messages unchanged; existing
banks keep their input/roll behavior; the new bank remains manually selectable
while blank automatic runs retain their existing eligible pool; accepted text
and final provenance survive the tail; cancelled/failed runs show no false
finished result.

### Live qualification after implementation

Replace the obsolete word ladder in the operational docs with a recorded
act-based ladder: one-act canonical smokes first, then three-act runs with the
same selected model pairings, then the supported six-act boundary. Retain the
preflight's two materially different local model families and one configured
frontier/cloud creative lane, independent technical-slot evidence, and real
ledger/publication receipts. This is compatibility proof, not a story-quality
contest or a reason to change the accepted writer-model defaults.

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

Shared-code changes must demonstrate that existing bank and machine paths
remain intact. Hardware-specific qualification stays with the existing
machine-profile process; this feature does not reopen the video-engine matrix.

## 9. Evidence and delivery sequence

1. Currentize the scoped documentation against live code and operator rulings.
2. Write and review the independent My Story design and exact integration plan.
3. Implement bank/ingress/selection behavior with its complete registration.
4. Add App Mode configuration and terminal-player output on the canonical.
5. Run required tests and canonical/live/browser qualification; finish the
   preflight item-by-item evidence matrix and hashes.
6. Update the README with proven user instructions and accurately stated
   limitations. Commit and push each green chunk to `v2.0-alpha`; registry
   publication, tags and promotion remain separate operator decisions.

Before coding, the standing design-review rule applies to the new capability:
full four-round review, grounded by the driver. The required local CLI review
of the finished code change still applies. This read-only scope is not that
campaign and does not claim any preflight gate passed.

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

Bring-your-own audio, verbatim scripts, image uploads, timeline editing,
multi-step draft approval, a separate website/repository, bespoke progress UI,
new model engines, a broad rebrand, and a universal one-click installer are
outside this first slice. The installation/weight-readiness work already in
the release queue remains necessary; hiding nodes does not remove it.

Relative effort: documentation refresh and native form configuration are the
smaller pieces; final-player integration is bounded but needs browser proof;
the independent bank and its qualification are the substantial work. No
calendar estimate is justified before the independent design is reviewed.
