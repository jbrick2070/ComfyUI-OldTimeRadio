# EOS code qualification receipt -- fresh canonical recovery remains open

Scope: native decoder, grammar and completion accounting now share effective
model-plus-chat EOS IDs. Configured multiple stops, scalar padding, owner
nonmutation and fresh LMFE parser histories are preserved. The writer records
actual final-token evidence before halt classification and does not print a
truncation warning for an EOS at the exact output allowance. Sampling, model,
profile, source, retry policy and canonical interface are unchanged.

Final focused: 292 passed. Full: 14,379 passed / 51 inherited failures /
183 skipped / one xfailed, versus 14,361/51/183/1 at 2c9d47f2. Failure IDs and
normalized payloads match; no new failure or quarantine. The initial EOS full
run had two integration failures (ordinary-path eager indexing and a line-distance
seed pin); both were corrected and their failed receipt remains preserved.
A second full run matched baseline; the final full run includes the subsequent
clean-EOS diagnostic correction. See eos_*_comparison.json.

Controlled Bug Bible: 34 passed / ten unchanged failures / 11 skipped /
three xfailed versus clean baseline 33/11/11/3. Extended rule 12.100's new
resolver guard fails on baseline and passes on candidate. No new rule ID;
343 entries, 146 inherited metadata issues unchanged. Bible commit/push:
283fb360fd875332c1a93a0290646f9bca3b0fcf, HEAD equals origin/main.

All six final Python files match roundtable/snapshot_final.json, AST parse,
nonempty UTF-8/no-BOM checks. The real canonical JSON remains SHA256
 d586a286aaee4c039e410ae9a10014c5c7f4ab82d00eac0e9e1cc0564415057c
with 23 nodes, 63 links and 37 writer widget slots. Production validator,
JSON round-trip and live schema/link/widget audit pass. No new node/widget/wire.
The native EOS fix reaches the existing scheduler/generate factories already
used by the canonical writer; it is not an unwired helper.

Production code/tests and authored documentation pass whitespace checks. Raw
review source excerpts, the archived queue and original server log intentionally
retain quoted whitespace. Windows file hashes identify the observed local bytes;
Git applies the repository's existing LF normalization. The attempt manifest also
records SHA256 of the archived Git bytes so other machines can verify that copy.

Actual Cursor plus Opus consensus and Sonnet finished-code/delta QA are grounded
in consensus_judgment.md. Root rejected incorrect model-default and zero-output
claims against installed source, accepted the test/diagnostic improvements,
and retained unresolved live questions. Final Sonnet delta has no remaining
must-fix. Reported successful API spend totals $0.315459; the failed Opus API
attempt's usage and local CLI dollar usage were not returned.

Live denominator: one full canonical 5080 attempt on prior commit 2c9d47f2
failed at 400.67 seconds during P1 treatment, no media/publication. One optional
source correction spent two attempts; complete JSON and 82/87 trailing newlines
preceded repetition halts. P1 separately repeated inside open strings across
three attempts. All five halts were verbatim_cycle, not the string-size cap.
Configuration readback proves differing native/chat EOS; exact hidden sampled
EOS and any causal contribution to padding were not saved. No EOS/P1 live cure
or new model-matrix proof is claimed by this code qualification.

Next: ONE fresh full canonical 5080 pair-lock retry of this qualified commit,
with identical pairlock_01 source/model/profile/sampling controls and preserved
failure history. Check actual EOS/fit, accepted source/ledger, final dialogue,
scene prompts and image pixels. Mac/4060 stay held; RunPod is currently blocked
by missing authenticated access and no rental is running.
