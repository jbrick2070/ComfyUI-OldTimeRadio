# ChatGPT consult -- RUN 245 TIMEOUT wedge
- model: gpt-4.1
- elapsed: 17.9s
- prompt size: 56,000 chars

---

Let's go straight to the root causes, with brutal specificity.

---

## 1. **Top 3 Wedge Candidates (in order of likelihood)**

### **1. HyworldPoll infinite wait (likely in poll.py:HyworldPoll.execute)**
- **File/Function:** `otr_v2/hyworld/poll.py`, `HyworldPoll.execute`
- **Bug class:** Infinite polling loop (waiting for a status file or process that never completes or dies).
- **Evidence:** 
    - ComfyUI queue is stuck: `running=1, pending=0` (main loop is blocked).
    - No error surfaced, no treatment file written, VRAM is not maxed, and the soak log shows the workflow is stuck after dialogue generation.
    - The poll node's `while True` loop (lines ~77–147) only exits on terminal status, PID death, or timeout (600s), but your soak timeout is 1800s, so it's possible this loop is holding the queue open for the full soak timeout.

### **2. HyworldBridge sidecar spawn (bridge.py:HyworldBridge._spawn_sidecar)**
- **File/Function:** `otr_v2/hyworld/bridge.py`, `HyworldBridge._spawn_sidecar`
- **Bug class:** Subprocess spawn hangs or never returns (e.g., waiting on a process that never starts, or atomic write deadlock).
- **Evidence:** 
    - If the sidecar worker never starts or hangs at process creation, the poll node will wait forever for a status that never arrives.
    - If atomic writes block (e.g., on a locked file), the contract files may never be written, and the poll node will never see a status.

### **3. EpisodeAssembler or downstream node deadlock (scene_sequencer.py:EpisodeAssembler.assemble)**
- **File/Function:** `nodes/scene_sequencer.py`, `EpisodeAssembler.assemble`
- **Bug class:** Deadlock or infinite loop during audio assembly, possibly due to malformed input or resource starvation.
- **Evidence:** 
    - If the audio pipeline (EpisodeAssembler) never returns, the workflow would hang before even reaching HyWorld.
    - However, your log shows dialogue lines generated and script/director succeeded, so it's less likely, but still possible if the hang is in the transition to the video pipeline.

---

## 2. **Minimum Instrumentation (ONE log/print per candidate)**

### **1. poll.py:HyworldPoll.execute**
- **Insert at:** Just inside the `while True:` loop, right after `poll_count += 1` (line ~87).
- **Line:**
    ```python
    log.info(f"[HyworldPoll] Polling job_id={hyworld_job_id} poll_count={poll_count} sidecar_pid={sidecar_pid}")
    ```
- **Purpose:** Confirms the poll loop is running and not stuck on a file read or deadlock. If this log repeats for hundreds of cycles, you know the poll node is the wedge.

---

### **2. bridge.py:HyworldBridge._spawn_sidecar**
- **Insert at:** Immediately before the `subprocess.Popen` call (line ~191).
- **Line:**
    ```python
    log.info(f"[HyworldBridge] Spawning sidecar: {sidecar_python} {str(_WORKER_SCRIPT)} {str(job_dir)}")
    ```
- **Purpose:** Confirms the sidecar spawn is attempted. If this log appears but no subsequent `[HyworldBridge] Sidecar spawned PID=...` log, the wedge is in Popen or the child process.

---

### **3. scene_sequencer.py:EpisodeAssembler.assemble**
- **Insert at:** First line of `assemble` (line ~1870).
- **Line:**
    ```python
    log.info(f"[EpisodeAssembler] assemble() called with title={episode_title}")
    ```
- **Purpose:** Confirms the assembler is entered. If this log is missing, the hang is upstream; if present but no return, the hang is in audio assembly.

---

## 3. **Kill-switch Recommendation**

**Do NOT kill ComfyUI immediately.**  
Let the in-flight run finish or timeout, unless you see the poll log repeating for >10 minutes with no progress.

- **Justification:**  
    - If the wedge is in the poll node, forcibly killing ComfyUI may leave orphaned processes, temp files, or partial outputs, making post-mortem harder.
    - The poll node has a hard-coded 10-minute timeout (`poll_count > 1200`), after which it should return "TIMEOUT" and let the workflow proceed/fail gracefully.
    - If the process is still chewing GPU/CPU after 10–15 minutes, and your new logs confirm the poll loop is spinning, then a controlled shutdown is safer than a hard kill.

---

## 4. **Treatment-write Path**

**Single call site:**  
- **File:** `nodes/scene_sequencer.py`
- **Function:** (Not shown in your snippet, but by convention and prior code, likely in the main workflow script after director/scriptwriter.)
- **Exception/return path that skips write:**  
    - If the script/director nodes succeed but a downstream node (e.g., BatchBarkGenerator, SceneSequencer, or any HyWorld node) hangs or throws an unhandled exception, the treatment file write is skipped.
    - **Most likely culprit:** If `otr_v2/hyworld/poll.py:HyworldPoll.execute` never returns (infinite loop), the workflow never reaches the treatment write call (which is after poll/render).
    - **Reference:** The treatment file is written after successful completion of the full pipeline, so any infinite wait or deadlock in the video pipeline (especially HyWorldPoll) will silently skip the write.  
      **See:** Your soak log shows "Treatment file written: NO" and no error, matching this scenario.

---

## **Summary Table**

| Rank | File:Function:Line                | Bug Class                  | Instrumentation Line                                                                 |
|------|-----------------------------------|----------------------------|--------------------------------------------------------------------------------------|
| 1    | poll.py:HyworldPoll.execute:87    | Infinite poll loop         | `log.info(f"[HyworldPoll] Polling job_id={hyworld_job_id} poll_count={poll_count} sidecar_pid={sidecar_pid}")` |
| 2    | bridge.py:HyworldBridge._spawn_sidecar:191 | Subprocess spawn hang      | `log.info(f"[HyworldBridge] Spawning sidecar: {sidecar_python} {str(_WORKER_SCRIPT)} {str(job_dir)}")`           |
| 3    | scene_sequencer.py:EpisodeAssembler.assemble:1870 | Audio deadlock             | `log.info(f"[EpisodeAssembler] assemble() called with title={episode_title}")`        |

---

**TL;DR:**  
The most probable wedge is an infinite wait in `HyworldPoll.execute`. Instrument the poll loop to confirm. If the log repeats for >10 min, kill ComfyUI only after confirming no progress. The treatment file is skipped because the workflow never reaches its write call due to the hang.