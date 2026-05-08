# Round B -- Gemini (gemini-3.1-pro-preview-customtools) elapsed=27.6s

**1. Core Verdict**
PARTIALLY AGREE: The additive minor bump strategy is correct, but the proposed implementation for Proposal 10 actively defeats your stated goal, and the answer completely misses a critical file-corruption risk inherent to your VRAM constraints.

**2. Factual Errors in the ChatGPT Answer**
*   **Proposal 10 (Process ID) destroys the chain:** You explicitly requested that a multi-batch resume sequence be "provable from any single ledger as a chain of restarts." ChatGPT recommended scalar fields (`meta.process_id`, `meta.process_started_at`). If you resume, the new process overwrites the scalar fields, destroying the history. You need an append-only list (e.g., `meta.process_history: list[{pid, started_at}]`).
*   **Dangerous Default Types:** For `cast_contract_version` (Proposal 6 & 8), ChatGPT recommends `None` as the default. If your downstream nodes use string methods (e.g., `if line.get("cast_contract_version", None).startswith("sha:")`), `None` will throw a `TypeError` and crash the run. If downstream expects a string, default to `"unversioned"` or `"legacy"`, not `None`.

**3. Important Things That Were Omitted**
*   **Atomic Writes during Violent OOMs:** You are running HuMo/LTX on a 16 GB laptop with a 14.5 GB ceiling. CUDA OOMs and hard resets (`_hard_reset_cuda_context()`) are violent. If `nodes/_otr_ledger.py` writes directly to `<episode_id>_ledger.json` and the process dies mid-write during a BUG-126 recovery, your *single source of truth is truncated and destroyed*. 
*   **C7 Hash Contamination:** You mandate C7 byte-identity for audio. If your audio nodes (BatchBark, KokoroAnnouncer) calculate their cache/tripwire hashes by hashing the *entire* `line` dictionary, injecting `lines[].oom_recovery_count` or `lines[].fallback_kind` will alter the hash, trigger a re-render, and violate C7. Telemetry must be explicitly excluded from audio hashing.

**4. My Short Recommendation**
*   **Ship the minor bump (`l3-2026-05-08`)** with all 11 proposals, but make them strictly additive. Do not delete the stale fields yet; schema bloat is a future problem, breaking a solo dev's pipeline today is a present problem.
*   **Change Proposal 10 to a list:** Use `meta.process_history: list[{pid: int, ts: str}]` and append to it on startup in `story_orchestrator.py`.
*   **Rename Proposal 2:** I agree with ChatGPT's distaste for `fallback_kind`. Use `lines[].render_method`. It's accurate and requires no mental gymnastics.
*   **Enforce Atomic Writes:** In `nodes/_otr_ledger.py` (wherever the JSON dump happens), you *must* write to `<episode_id>_ledger.json.tmp` and then `os.replace()` to the final filename. This is non-negotiable for a system designed to soak through OOMs.
*   **Audit your C7 Tripwires:** Ensure the sha256 generation in your `audio_gates[]` only hashes creative fields (`text`, `voice_spec`, `tts_engine`), not the whole `lines[]` dict.

**5. Uncertainties to Verify**
*   **How does `_otr_ledger.py` currently save to disk?** I cannot see the file, but if it's doing a standard `with open(filepath, 'w') as f: json.dump(...)`, you are at extreme risk of file corruption during a hard CUDA crash.
*   **How are the `audio_gates[]` sha256 tripwires calculated?** I need to verify that adding keys to the `lines[]` dictionaries won't accidentally change the inputs to your audio hash functions.
