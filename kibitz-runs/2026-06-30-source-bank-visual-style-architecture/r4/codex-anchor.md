VERDICT: converge-with-minor-guards. R3 is now a buildable coding plan. I do
not see a remaining architecture inversion, but there are several implementation
guardrails that should be explicit in the final doc.

MUST-FIX BEFORE BUILD:

1. [C2/C3 branch ownership] The final plan must pick a concrete first behavior
   for non-science pitch selection. My recommendation: in the first media
   archive sprint, bypass `_otr_pitch_room.py` and `_otr_story_select.py` for
   `source_bank != science_news`, then parameterize those modules in the next
   prompt-profile chunk if we want multi-candidate archive selection. That keeps
   C3 small and avoids half-science prompts.

2. [C4/C6 policy parse failures] Invalid `visual_style_policy_json` must not
   silently become cinematic in tests. Runtime can fall back to default with a
   warning/report, but unit tests should assert malformed policy JSON emits a
   visible warning. Otherwise a broken style link becomes too easy to miss.

3. [C6 workflow edit] The plan should require a code-driven workflow patch or a
   link-by-input-name audit. The current slot estimates are correct for the
   inspected graph, but hand-editing socket slot numbers in the single-line JSON
   is still a known failure mode here.

SHOULD-FIX:

1. [C0 packet defaults] Avoid mutable literal defaults in the Pydantic models
   unless the repo's Pydantic version is confirmed to copy them safely. Prefer
   `Field(default_factory=list)` and `Field(default_factory=dict)` for
   `key_terms`, `adaptation_trace`, and `forbidden_terms`.

2. [C1 meta.source compatibility] The strict-meta risk appears lower than R3's
   wording implied: my direct search found `extra=forbid` concentrated in
   image/video engine request schemas, not a full-ledger `meta` schema. Keep the
   verification gate, but do not let it block C1 unless a real consumer rejects
   `meta.source`.

3. [C6 method call compatibility] Existing `OTRShotLock().lock(...)` tests use
   a positional ledger and keyword arguments for the optional inputs I checked.
   Appending `visual_style_policy_json` at the end remains the correct
   compatibility move. Add one explicit test that old keyword call sites still
   run without supplying the new input.

4. [C3 archive fetch determinism] The archive source list should have fixture
   tests and a timeout/failure path from the start. For workflow reliability,
   a dead archive site must fail closed with a clear message, not hang the
   writer.

OPTIONAL / NICE-TO-HAVE:

1. [C5 node naming] Keep class/file naming consistent with the existing repo:
   file `otr_visual_style_director.py`, class `OTRVisualStyleDirector`, exported
   node key/display name `OTR_VisualStyleDirector` if that matches current
   registration style.

CUT:

1. Do not add a report output to `OTR_VisualStyleDirector` in V1. Warnings can
   live in downstream reports that parse the policy.

2. Do not implement public-domain fidelity verification in the source selector
   chunk. C7 starts with source text -> blueprint -> outline, with trace stamps;
   hard verifier follows only after that path is green.

CONVERGENCE CHECK:

R3 correctly preserved the two-axis architecture:

- source axis before story/ledger generation
- visual style/model axis after the story ledger exists, feeding prompt
  composition

It also correctly avoids the earlier overbuild: no standalone source director
node in V1, no arbitrary public-domain search, no custom policy JSON, and no
visual-style-only model selection that fails to rewrite prompts.
