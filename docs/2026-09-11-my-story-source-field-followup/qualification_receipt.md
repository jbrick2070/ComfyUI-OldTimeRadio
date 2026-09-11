# Source-field conformance: qualified for the next live measurement

The existing spoken source editor now gets the same four-field namespace in
native grammar that its application validator already enforces. Source/draft
quotes and candidate-line offsets are clarified in its actual instruction.
No extra correction calls, source gate, node/widget/wire or semantic certificate.
The exact final production/test diff and hashes are in final_state.md. This
does not claim the remaining source omissions or added child are fixed live.

## Verification

- Focused real owner/input/runner/native-decoder tests:186 passed.
- First full run:14410 passed,52 failed,183 skipped,1 xfailed.51 failure IDs and
  normalized payloads match the prior baseline. The extra failure found the
  manual README qualification note inside its generated dropdown table.
  Moved it outside and regenerated via the owner; --check passes.
- Final full run:14411 passed, the same51 inherited failures with identical
  normalized payloads,183 skipped,1 xfailed. No introduced failure and no
  quarantine edits. The known-fail guard exits2; this is not an all-green suite.
  Normalization ignores addresses, temporary paths/truncated fixture reprs and
  pytest's order of unequal dictionary entries; all actual key/value findings
  remain equal. Raw XML is preserved for inspection.
- Final controlled Bible:38 passed,10 inherited failures,11 skipped,3 xfailed.
  Same final Bible against untouched18569a3b:36 passed,12 failures,11 skipped,
  3 xfailed. New schema guard and coverage catalog fail baseline/pass candidate;
  the ten common failure payloads match after path/repr/address normalization.
- Canonical unchanged, SHA256
  d586a286aaee4c039e410ae9a10014c5c7f4ab82d00eac0e9e1cc0564415057c.
  Shipped validator, JSON round-trip, live widget counts and link audit pass:
  23 nodes,63 links. Touched Python parses with Python3.10 grammar; UTF8/noBOM
  and nonempty-file checks pass. Bible retains344 entries; existing11.39 extended.

Actual LMFE excludes the live invalid alias and admits all four valid fields.
The separate public two-call test proves applied replacement and conservation,
but already passed before fix; it is not semantic or fix-discriminating proof.
All raw logs/XML, including the first README failure and before-fix result,
are preserved under tests/. Controlled worktrees separate current probes from
the portable scan. No whole-suite failures were hidden or marked newly known.

## Review and next action

Opus5 and Sonnet5 each reviewed twice, including Sonnet after the last test-code
revision. Both found no production change needed. Root grounded their remaining
test/artifact concerns against the final files; see review_judgment.md and the
authoritative final_state.md. Historical duplicate Bible hunks in a review
packet are not duplicate real code. Cursor follow-up yielded no review after
two prior timeouts; no Cursor/unanimous consensus is claimed. API cost USD0.485079.

Independent read-only wiring review found no must-fix. Independent05 evidence
review checked hashes and corrected one receipt claim: age bands come from
accepted treatment, not numerical ages supplied in the raw story.

Commit/push this qualified chunk before one fresh full canonical5080 attempt06
with the same source, Mistral pair, profile and controls as05. Verify submitted
graphs match. Inspect actual applied edits, spoken source, voice/credits and
pixels, retaining every failure. No retry reset and no claim an alias exclusion
guarantees semantic recovery. Mac/4060 stay held until operator release; RunPod
remains unauthenticated and no rental has started. GO_FORWARD is the sole queue.
