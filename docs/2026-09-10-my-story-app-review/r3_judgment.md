# R3 judgment - wiring and state

Codex driver; Gemini3.8 Flash High and Cursor Grok4.6 High completed. Advance
revised scope to R4. User clarified App features are required in the scope but
using App view/sidebar is optional; graph has the same complete creator flow.
No implementation, live graph editing, tests, downloads or render.

## Gemini dispositions

| Claim | Grounding / decision |
|---|---|
| M1 sequence separates widgets/JSON | CONFIRMED ambiguous step3/4 despite same-change rule elsewhere. D2 now includes canonical fields/vectors/links/derived variants, plus functional publication enforcement before runnable. D3/D4 add optional presentation. |
| M2 widgetId should be nodeId | MISREAD explicitly assumed from minified type. Installed builder uses widget identities, including graph/node/widget identity; actual tuple widgetId/widgetName/config. Native builder is authority; reject invented numeric-node replacement. |
| M3 snapshot before blank | CONFIRMED requirement already stated. Clarify file-backed coordinator resolves/pins authority, pure helper receives it. Do not make pure helper perform I/O or bypass on mere file existence. |
| M4 new dedicated fields ignored | CONFIRMED new usability gap. Reject new story_* fields on non-My-Story selection before roll/download. Preserve old custom_premise-plus-roll behavior and empty new defaults. |
| M5 publication refusal success | CONFIRMED current _publication_decision returns refusal, mux archives successfully. Ledger-only source_bank check is insufficient if ledger missing. Add independent direct writer snapshot wire with stable delivery token; eligibility still live freeze-owned. |
| M6 resolver signature | CONFIRMED forwarding obligation, already intended; explicitly require optional keyword-only additions and raw sidecar/resolved propagation. |
| S1 source_ref | CONFIRMED already covered early admission; sentinel roll check alone is insufficient. |
| S2 test unpackers | Four active success calls confirmed; helper details add no new caller. Migrate direct results atomically; raising test remains raising. |
| S3 subfolder | CONFIRMED existing containment requirement; compute relative to real served root and normalize separators, never an absolute directory. |
| S4 tail shape | CONFIRMED minimum title/premise and canon write compatibility added; full D1 schema still prerequisite. |
| S5 draft race | CONFIRMED atomic/idempotent requirement already stated; one coordinator verifies existing content and writes atomically. |
| Optional tooltip/format | Bank-honest tooltip retained. format supported optional, not a requirement for inspected frontend. |
| Cut validator persistence | REJECTED: active form is not guaranteed durable; native persistence is setting-dependent. Preserve before-download draft storage via one idempotent owner, not duplicate storage logic. |
| Cut report | Partly accepted: movie is primary, text optional alongside playable output. A real ui.files report is still needed for native no-preview cases; second STRING wire/ui.text alone is not App-visible. Report never replaces My Story publication. |

## Cursor dispositions

| Claim | Grounding / decision |
|---|---|
| M1 PCM vs AAC | CONFIRMED already explicit. Player uses actual OBS AAC, result0 retains archival PCM. No intermediate substitute. |
| M2 widget32 vs socket32 | CONFIRMED critical distinction. Replay widget32; gate_in input32/link279. Append descriptors after replay; preserve gate link exactly. |
| M3 _literal / types | CONFIRMED. Reuse reachability only; separate string/int/combo parsing, omitted new strings empty, real links defer. Missing actual queued PROMPT retains structural failure. |
| M4 draft before rolls | Raw capture before rolls CONFIRMED. Reject mandatory persistence of unvalidated/irrelevant requests; store admitted input using preserved raw values, validator before assets and direct writer before generation. |
| M5 count clamp | CONFIRMED resolver clamp would lose request. Keep raw count sidecar without globally changing legacy normalization; requested versus actual separate. |
| M6 policy defaults | CONFIRMED already locked optional scalars/absent legacy+true, no ID-only roll exception. |
| M7 pipeline/seams | CONFIRMED already scoped dispatch-before-inline and exact coordinates/custom-seam parity. |
| M8 draft path | Reviewer proposed path violates live _otr_paths._validate_contract: only episodes/obs allowed. Use existing otr_state_dir()/story_drafts in persistent _shared/state. Neither cwd-relative nor new top-level directory. |
| M9 variants | CONFIRMED builder --all/check and named widget/link fixtures in the same schema/canonical chunk. Generated variants do not replace canonical for runs. |
| S1 hidden replay | CONFIRMED shared-state hazard. Expose existing advanced replay control, preserve frozen-authority first path; do not fail legitimate replay to force generation. |
| S2 path splitting | CONFIRMED no /output/ string split; use actual root containment. |
| S3 strict snapshot manifest | CONFIRMED known behavior, document entry/allow-partial need and preserve other banks. Pin accepted authority once per queued writer. |
| S4 cameo | CONFIRMED D1 needs actual behavior/receipt, not just exhaustive matrix row. No retrospective gate pass. |
| S5 act breaks | CONFIRMED live request; D1 declares handling, no assumption it is off. |
| S6 new kwargs | CONFIRMED add optional keyword-only defaults after existing *, retain seven payload keys. |
| S7 tail duck type | CONFIRMED own tail parts/minimum attributes; no imported bank class. |
| Optional / cuts | Agree no retired hooks, eighth payload key, inline template, second context builder or gate reindex. Native real-file report needed only for applicable UI status. |
| Assumptions | Actual launcher snapshot env is verify-at-build, not claimed absent. Installed maps ground labels/layout; browser integration still pending. |

## Driver wiring corrections and evidence

Draft store uses _otr_paths.otr_state_dir:595, persistent state allowed by output
contract at437. Existing source snapshot hash excludes metadata and loader has
no save/restore service. The coordinator pins fields and validates their own
digest. Installed comfy_execution/utils.py:4-19 provides execution context;
validator PROMPT/UNIQUE_ID scopes queued target writers. Direct calls need an
explicit identity or writer-only submission. No episode created for a draft.

Publication wiring: existing writer script_json slot1 fans to a new trailing
optional mux input. Presence is validated; no use of opaque audio_done or retired
clip_manifest_json, no new writer output. Stable delivery token is necessary:
writer starts pending ID and tail serializes at1541; video_engine:2296 renames
episode later. Token survives rename; replay import refreshes it. Draft digest
is not a run ID. Terminal compares bank/token against current ledger, then live
freeze receipt/current episode and actual published file. Missing ledger cannot
drop wired requirement. New input participates in cache identity; lost output
must invalidate success. These tests and exact D1 schema remain before-code gates.

Normal graph mode is first-class and default. App adds layout and native output
presentation, not a second backend or a publication prerequisite. Future schema
changes synchronize canonical immediately; UI-only work is a later chunk.
LATE OPERATOR STEERING BEFORE R4
The operator additionally asked to put both this scope and the already reviewed
Opus cleanup plan at priority #1 in GO_FORWARD_PLAN, with logical sprint order
and 1-2 helpers for ideas. Two existing read-only helpers independently checked
the sequencing. Accepted: cleanup P1, then separate P2/P3 chunks, then My Story
D0/D1, complete creator backend/canonical delivery, optional App presentation
and qualification. Speaker placement stays an optional separate design follow-on.
The driver verified the real cleanup HANDOFF/PLAN and updated GO_FORWARD's top,
pickup section and attack order. These are new user requirements, not findings
claimed to have been reviewed during R1-R3. R4 will review this combined handoff.
