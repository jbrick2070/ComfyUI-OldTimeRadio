### 1. Does PASS actually mean what it claims?

**Mostly, but not fully.**  
A green run here proves the basic Bridge→Poll→Renderer chain is wired and working for the happy path (READY status, stub motion, audio tensor contract, MP4 output).  
**However, it does *not* guarantee:**
- That the nodes work when called as ComfyUI nodes (with ComfyUI's async, graph, and tensor lifecycles).
- That all contract invariants (e.g., C7 audio byte-identity, temp file cleanup, sidecar process cleanup) are met.
- That error/fallback paths are robust (e.g., worker crash, OOM, parse errors).
- That the system is robust to concurrency or multiple jobs in flight.
- That the temp WAV is actually deleted (only implied, not asserted).
- That the output MP4’s audio is truly byte-identical to the input tensor (not checked).

### 2. Coverage gaps worth adding

**Blockers (should add before calling "stable"):**
- **C7 Audio Byte-Identity:**  
  - *What to add:* After render, extract the audio track from the MP4 and compare its PCM bytes to the original tensor (after round-tripping through WAV).  
  - *Why:* Contract says audio must be byte-identical (C7 guarantee). Not currently checked.
- **Temp WAV Cleanup:**  
  - *What to add:* Assert that the temp WAV file (e.g., `hyworld_audio_*.wav`) is deleted after render.  
  - *Why:* Contract requires temp cleanup; not asserted.
- **Sidecar Worker Cleanup:**  
  - *What to add:* Assert that the worker process is dead and `sidecar_pid.txt` is deleted after render.  
  - *Why:* Prevents orphaned processes; not checked.
- **Renderer FALLBACK Path:**  
  - *What to add:* At least one test that forces a FALLBACK (e.g., by corrupting assets or simulating a worker crash).  
  - *Why:* Ensures fallback logic and cleanup are exercised.

**Nice-to-haves:**
- **Multiple Job IDs:**  
  - *What to add:* Run two jobs in sequence to check for job ID collisions or state bleed.
  - *Why:* Ensures isolation.
- **Non-canonical Script Edge Cases:**  
  - *What to add:* Try a script with missing/extra fields to check Bridge validation.
  - *Why:* Robustness.
- **Output File Integrity:**  
  - *What to add:* Check that the MP4 can be opened and played (e.g., via ffprobe).
  - *Why:* Surface-level smoke, but not a core contract.

### 3. FALLBACK branch not exercised — is that a problem?

**For a smoke test, not a blocker, but...**  
- The main smoke test should pass the happy path, but a *minimal* separate test should force FALLBACK to ensure that path is exercised and cleanup is correct.
- If FALLBACK is never tested, regressions in fallback handling (e.g., temp files left, wrong status, orphaned processes) could go unnoticed.

### 4. Contract invariants not currently asserted

- **C7 Audio Byte-Identity:**  
  - *Not asserted.* Should compare original tensor→WAV→MP4 audio bytes.
- **Temp WAV Cleanup:**  
  - *Not asserted.* Should check temp WAV is deleted after render.
- **Worker Process/sidecar_pid.txt:**  
  - *Not asserted.* Should check worker process is dead and `sidecar_pid.txt` is gone after completion (both success and fallback).
- **No orphaned temp files or processes:**  
  - *Not asserted.* Should check for lingering temp files or processes after test.
- **MP4 output is valid and playable:**  
  - *Not asserted.* Should at least check file opens with ffprobe.

### 5. Honest verdict

**NEEDS-ITERATION**  
- The test is a solid happy-path smoke, but does *not* fully guarantee contract invariants or robust cleanup.  
- Add at least: C7 audio byte check, temp WAV/sidecar cleanup assertions, and a forced FALLBACK test before calling the HyWorld wiring "stable".
