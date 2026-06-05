<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The design is not build-ready because the proposed unattended auto-fix loop lacks exclusive lifecycle control, reproducible failure capture, atomic git/revert mechanics, and an actual audio byte-identity gate.

MUST-FIX BEFORE BUILD:

1. [2 Triage + auto-fix agent] Scheduled triage can race the running soak, edit `.py` files while ComfyUI has old code loaded, restart ComfyUI mid-render, or run tests against a moving working tree. The context explicitly says `.py` edits need a ComfyUI restart. Concrete fix: add one exclusive supervisor/lock protocol before any auto-fix:
   - soak owns a lock while an episode is rendering;
   - triage requests PAUSE and waits for IDLE, or safely cancels and confirms ComfyUI is idle;
   - triage takes an exclusive repo/ComfyUI lock;
   - after any kept `.py` change, ComfyUI is restarted before soak resumes;
   - if safe restart cannot be performed, do not apply the fix; flag it.

2. [1 Soak loop] The described “continuous unattended loop” is not what the grounded driver does. `_otr_soak_matrix.py` runs one finite matrix and writes `_otr_matrix_<tag>.json`; with the same tag it overwrites the same file each cycle. There is no append-only episode log for “NEW failures.” Concrete fix: build a wrapper that invokes the matrix repeatedly with unique episode/run IDs, or change the driver to write append-only JSONL/NDJSON records with `episode_id`, `cycle`, `combo`, `start/end`, and status. Writes must be atomic enough for the triage reader, e.g. write complete lines only or temp-file + `os.replace`.

3. [2 Triage + auto-fix agent] The reader/writer contract for the soak log is unsafe. The current driver writes JSON by `json.dump(results, open(out, "w"...))`; a scheduled reader can observe a partially written JSON file and misclassify or crash. Concrete fix: use append-only JSONL, or write to `<file>.tmp` and `os.replace()` only after the JSON document is complete. The triage task must tolerate malformed/partial last records.

4. [5 Morning report] “repro seed/episode” is not available as written. The context says each episode draws fresh OS-entropy RNG; the grounded driver does not log a seed. A morning report cannot provide a reproducible seed unless the system controls or captures one. Concrete fix: either:
   - add explicit seed injection/logging to the episode generation path [ASSUMPTION: if supported by the OTR code], or
   - capture enough artifacts to replay/debug: API prompt JSON after pruning, combo, model names, ComfyUI prompt ID, generated story/intermediate JSON, raw LLM output if available, log slice bounded by run start/end, and output paths.

5. [1 Soak loop] The proposed statuses do not match the grounded driver. The script currently returns statuses such as `SUCCESS`, `TIMEOUT`, `REJECT`, and `EXC`; it does not emit `PASS`, `HALT(freeze)`, `CRASH(exc)`, or `QUALITY(flagged)`, and it does not record “failing phase.” Concrete fix: define a status mapping and implement instrumentation. Example:
   - `SUCCESS + valid audio_done` => PASS;
   - known freeze-halt marker => HALT;
   - exception/traceback/reject => CRASH/REJECT with phase;
   - explicit validator failure => QUALITY.
   Without this, the auto-fix classifier has no reliable input.

6. [1 Soak loop] `bypass_freeze_halt=OFF` is assumed but not implemented in the grounded `_otr_soak_matrix.py`. The shown script patches `target_words`, LLM models, and engines only. Concrete fix: add an explicit patch for the freeze/halt widget if it exists in the live schema; otherwise remove this guarantee from the design and treat BUG-276 as a known expected failure. [ASSUMPTION] Verify the actual workflow node/widget name before coding.

7. [3 Auto-fix whitelist] The whitelist is too broad for unattended code modification. “Clearly-bounded guard/known-pattern with an obvious fix” is not a hard guardrail; it lets the agent decide what is obvious. Concrete fix: replace it with mechanical constraints:
   - allowed categories only: schema boundary coercions for count/length/type where the maximum/minimum is already declared in code/schema;
   - mandatory regression test reproducing the bad input;
   - no edits to audio render code, workflow JSON, model loading, ComfyUI integration, VRAM/loader code, Bark/Kokoro/SA3 paths, or BUG-276-related code;
   - path allowlist for writer/schema validation files [ASSUMPTION: exact paths must be verified from repo];
   - diff size cap and no broad refactors.

8. [4 Guardrails] “Never break audio byte-identity” is stated but not enforced by the proposed keep/revert gate. Tests + Bug Bible may pass while output changes. Concrete fix: add an explicit deterministic audio identity gate before keeping any fix:
   - run a fixed-input audio-only fixture before and after the candidate change;
   - compare byte hash of the produced audio or the exact serialized audio payload expected by the existing discipline;
   - if deterministic rendering is not currently possible, then unattended auto-fix must be limited to files proven not to affect audio output, or disabled.

9. [2 Triage + auto-fix agent] The git keep/revert mechanic is underspecified and not atomic. Generated soak logs, report files, and edited tests can contaminate commits or be destroyed by reset/clean. Concrete fix: define the exact transaction:
   - require clean `v2.0-alpha` at start, excluding ignored runtime artifacts;
   - create/reset `soak/auto-YYYYMMDD` from `v2.0-alpha`;
   - store soak logs/reports outside tracked paths or ensure they are ignored;
   - apply exactly one candidate fix;
   - inspect `git diff --name-only` against the allowlist;
   - run full gates;
   - if green, commit once with the test and BUG_LOG entry;
   - if red, `git reset --hard` to the pre-fix commit and clean only known generated paths, not logs/reports;
   - never stash unknown human changes.

10. [2 Triage + auto-fix agent] Scheduled task re-entry is not handled. A 30–60 minute task can overlap a previous triage run if tests hang, ComfyUI restart hangs, or Cowork takes longer than expected. Concrete fix: add a triage lock file/mutex with PID, start time, heartbeat, and stale-lock policy. If another triage is active, the new invocation exits.

11. [1 Soak loop] The grounded driver’s `last_audio_done()` and `err_tail()` read global ComfyUI logs and can attribute stale markers/errors to the wrong episode. `last_audio_done()` just scans backward for the latest marker; `err_tail()` scans the last 400 lines. Concrete fix: record the ComfyUI log offset or timestamp at episode start and only search the bounded slice for `audio_done` and errors. Include prompt ID in the record where possible.

12. [1 Soak loop] Timeout handling may not be sufficient for a truly hung render. The grounded script calls `otr_api.cancel_queue()` on `TIMEOUT`, but the design does not prove that this stops an executing node or frees VRAM. Concrete fix: verify `cancel_queue()` behavior for an executing prompt. If it does not reliably kill the active job, add a hard watchdog: stop/restart ComfyUI after timeout/no-progress, verify port health, verify queue empty, then resume.

13. [1 Soak loop] BUG-276 is both expected and excluded from auto-fix, but the soak plan can still repeatedly surface it and waste auto-fix cycles. Concrete fix: add known-failure classification/deduplication for BUG-276 before the whitelist classifier. It should be counted and reported, but never proposed as an auto-fix and not repeatedly re-queued as “new.”

14. [2 Triage + auto-fix agent] The plan says it “logs a BUG_LOG entry” but does not define the file, format, or failure behavior. Concrete fix: specify the exact BUG_LOG path/format and make a missing or invalid BUG_LOG update fail the candidate fix before commit. [ASSUMPTION] Verify the real repo’s bug log convention.

15. [1 Soak loop] Model/runtime preflight is missing. The context mentions an Ollama sidecar for gemma-4-12b, while the proposed combo uses `gemma-2-2b`. The grounded script will submit whatever model string is in combos; unavailable sidecar/model failures will look like soak failures. Concrete fix: before the overnight run, preflight ComfyUI health, Ollama sidecar health if used, all model names in combos, all audio engines, and output/log directories. Classify preflight failure as infrastructure, not code bug.

SHOULD-FIX:

1. [4 Guardrails] Add a baseline gate before the overnight starts: clean tree, current branch is `v2.0-alpha`, full tests green, Bug Bible green, ComfyUI healthy, audio-only smoke green. Otherwise do not start auto-fix mode.

2. [4 Guardrails] Replace vague “cap N auto-fixes” with explicit limits: e.g. max 2 kept fixes/night, max 3 attempted fixes/night, max 1 red revert then disable auto-fix, max consecutive infrastructure failures before stopping soak.

3. [2 Triage + auto-fix agent] Add a “test must fail before fix” rule for auto-generated regression tests where practical. For schema coercion bugs, the agent should first add the test and confirm it fails, then apply the fix and confirm it passes. This reduces false tests that do not cover the bug.

4. [3 Auto-fix whitelist] Require all auto-fixes to be local, monotonic validation changes: clamp/truncate/retry/normalize at schema boundaries only. Reject fixes that change prompts, creative policy, model selection, audio generation parameters, node wiring, or retry budgets unless explicitly reviewed.

5. [5 Morning report] Add distinct IDs and dedupe keys for bugs: category, combo, model, engine tuple, exception signature, failing phase, and first episode ID. Otherwise repeated instances of the same bug will inflate the report.

6. [1 Soak loop] Define QUALITY detectors or remove the status. As written, `QUALITY(flagged)` has no concrete mechanism. If kept, specify exact machine-checkable validators, not subjective Cowork judgment.

7. [4 Guardrails] Preserve generated failure artifacts before any git reset/clean. Store them under an ignored `artifacts/soak/<date>/<episode_id>/` or outside the repo. The morning report should link to these.

8. [2 Triage + auto-fix agent] Make push behavior explicit. The context says “one git push attempt then hand a block,” but the plan only says commit to a branch. Decide whether the overnight agent pushes. If yes, one push attempt only; on failure, keep the local branch and report the block.

9. [1 Soak loop] Add heartbeat/progress to the soak itself, not only per-run polling output. The supervisor/triage needs to know whether the soak is alive, hung, paused, or fixing.

10. [1 Soak loop] For VRAM peak, define fallback behavior if `smoke_watcher.vram_gb_from_lhm()` returns unavailable. Do not classify missing VRAM telemetry as a code failure.

OPTIONAL / NICE-TO-HAVE:

- Add a dry-run mode for the auto-fix agent that classifies and proposes actions but never edits. Run this for the first night before enabling writes.
- Add a nightly summary CSV/JSON plus human-readable Markdown report.
- Add per-combo quotas so one flaky combo does not consume the entire night.
- Add an allowlisted “infra restart only” remediation path separate from code auto-fix.

CUT THESE (over-engineering):

1. [2 Triage + auto-fix agent] Cut the separate scheduled Cowork task if you can replace it with a single long-running supervisor. A single supervisor is safer because it owns soak state, triage state, repo lock, ComfyUI restart, and reporting without cross-process races.

2. [1 Soak loop] Cut `QUALITY(flagged)` until there are concrete validators. A subjective or undefined quality classifier is unsafe unattended and will either do nothing or produce noisy false positives.

3. [3 Auto-fix whitelist] Cut the broad “known-pattern with an obvious fix” category. Keep only mechanical schema-boundary coercions with regression tests. This still covers the stated low-risk class like BUG-264/307-style over-cap fields while removing agent discretion.

4. [2 Triage + auto-fix agent] Cut automatic fixing on the first overnight run. First run should collect failures and validate logging/classification. Enable write-mode only after the report proves the classifier correctly distinguishes known BUG-276, infra failures, schema coercions, and real crashes.