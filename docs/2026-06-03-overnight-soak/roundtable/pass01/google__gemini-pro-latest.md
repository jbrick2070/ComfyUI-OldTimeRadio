<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The asynchronous scheduled agent and continuous soak loop will inevitably collide, causing VRAM OOMs, mid-render crashes, and corrupted tests.

MUST-FIX BEFORE BUILD:
1. [Section 2 & 5] **Concurrency / VRAM Collision:** A scheduled cron agent waking up to run full regressions (which use ComfyUI/VRAM) while the soak loop is continuously submitting jobs to the same 16GB RTX 5080 will cause CUDA OOMs or job failures. *Fix:* Abandon the cron schedule. The soak driver must *synchronously* pause its loop, invoke the agent on failure, wait for the keep/revert cycle to finish, and then resume.
2. [Section 4] **Restart Sequencing:** The plan states `.py` edits need a ComfyUI restart, but the keep/revert sequence in Section 2 does not mandate a restart *before* the regression test. If the agent edits a file and runs tests, it will test the old in-memory code. *Fix:* The agent must hard-restart ComfyUI immediately after applying the fix, and again if it reverts, before running any tests.
3. [Section 1 & Grounding] **Log Overwrite:** `_otr_soak_matrix.py` writes results via `json.dump(results, open(out, "w", encoding="utf-8"))`. If the soak loop re-runs this script with a static `<tag>`, it overwrites the JSON every time, destroying the history the agent needs to read. *Fix:* Append a timestamp to `<tag>` in the outer shell loop, or change the script to append to a JSONL file.

SHOULD-FIX:
1. [Grounding `err_tail`] **Stale Error Tails:** `err_tail()` blindly searches the last 400 lines of `comfyui.log` for error keywords. If a combo fails without emitting a recognized keyword (e.g., a silent hang or timeout), it will return the error tail from a *previous* run, tricking the agent into fixing the wrong bug. *Fix:* Record the file byte-size of `comfyui.log` at `t0` in `run_combo` and only read lines written after that offset.
2. [Section 2 & 3] **Revert Mechanics:** Do not use `git stash` for the trial fix, as it handles untracked files poorly and complicates the working tree. *Fix:* The agent should `git commit -am "trial fix"` -> restart ComfyUI -> run tests -> if red, `git reset --hard HEAD~1`.
3. [Section 1] **Connection Error Handling:** If the agent restarts ComfyUI, the soak script's `requests` calls (inside `otr_api`) will throw `ConnectionError`. The script catches `Exception` and logs "EXC", which will rapidly burn through the remaining matrix combos. *Fix:* Add a connection-retry loop in `otr_api.submit_prompt` or pause the matrix if the server is unreachable.

OPTIONAL / NICE-TO-HAVE:
- Pass the failing RNG seed/episode directly to the agent. If it auto-fixes a bug, it should verify against that specific seed, not just rely on the static Bug Bible regressions.

CUT THESE (over-engineering):
1. [Section 2] **The scheduled Cowork task (cron).** Cut it entirely. It introduces race conditions, requires complex locking, and risks stepping on mid-render episodes. A synchronous call from the soak driver is vastly simpler and 100% safe.
2. [Section 2] **"Classifies each... for a WHITELISTED EASY category only".** Cut complex LLM classification. Just regex-match the `err_tail` against a hardcoded list of known whitelist exception strings (e.g., `ValidationError.*String should have at most`). If it doesn't match the regex, skip auto-fix.

[ASSUMPTION] I am assuming `otr_api.submit_prompt` and `poll_history` do not have built-in server-wait logic, meaning they will instantly fail if ComfyUI is down during a restart.