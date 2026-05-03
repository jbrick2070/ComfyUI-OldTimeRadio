# Round C -- NVIDIA (nvidia/llama-3.3-nemotron-super-49b-v1.5) elapsed=66.7s

### 1. Direct Answers to Q1-Q7  

**Q1: Retry semantics**  
**Agree with Gemini's correction of ChatGPT.**  
- Use **3 attempts × 0.5s** inline (no helper function).  
- **Enhance error message** to include: *"Check for manually locked files (e.g., open in editors/players)."*  
- **Why:** Defender/indexer locks clear quickly, but human-held locks (e.g., Notepad, VLC) require user intervention. Retries won’t resolve these, so the error message must guide the user.  

**Q2: Hard-fail vs split-state recovery**  
**Agree with both.** Hard-fail is correct.  
- **Add explicit check:** If `new_ep_dir` exists, raise `RuntimeError` immediately.  
- **Why:** Split state leads to downstream chaos. Immediate failure is safer.  

**Q3: Treatment rename ordering**  
**Agree: Ledger before treatment.**  
- **Why:** Ledger is the source of truth. A failed treatment rename is recoverable; a failed ledger rename is not.  

**Q4: Other `pending_<ts>_*` files**  
**Audit required.**  
- **Recommend:** Use `old_id_*.txt` glob (not `pending_*`).  
- **Why:** Avoid accidental renames of unrelated files. If `old_id` is the original episode ID, this ensures only owned files are renamed.  

**Q5: Concurrent writer interleaving**  
**No risk in current model.**  
- **Why:** ComfyUI is strictly sequential. No background threads or parallel processes. `_otr_ledger.save_ledger_safe()` cannot interleave during `rename_episode()` because it’s part of the same sequential workflow.  

**Q6: Hardening tests**  
**Add these tests:**  
1. **Happy path:** All files renamed, state updated.  
2. **Dir move fails 2/3 attempts then succeeds.**  
3. **Dir move fails all 3 attempts:** `RuntimeError`, state unchanged.  
4. **Destination exists:** Hard-fail.  
5. **Old missing, new exists:** Treat as already moved.  
6. **Treatment rename fails after ledger succeeds:** Warning only.  
7. **Re-rename idempotency:** Second call to `rename_episode` with same `new_id` is no-op.  

**Q7: Windows-specific footguns**  
**Key risks:**  
- **Destination directory exists:** Check before retrying.  
- **Human-held file locks:** Error message must warn about this.  
- **Case-insensitive paths:** Normalize with `os.path.normcase()` when comparing paths.  

---

### 2. Factual Errors in Previous Answers  

**ChatGPT’s `os.replace` claim:**  
- **Error:** Stated that `os.replace` can replace an empty destination directory on Windows.  
- **Correction (Gemini):** On Windows, `os.replace` **always fails** if the destination directory exists, even if empty.  

**ChatGPT’s helper function suggestion:**  
- **Issue:** Extracting a helper (`_replace_with_retry`) increases diff size.  
- **Fix:** Keep retry logic inline to minimize changes.  

---

### 3. Important Omissions by Both  

1. **Slugification consistency:**  
   - Must ensure `new_id` is pre-slugified **or** apply the same slug function used for the ledger to treatment/sidecars.  
   - **Risk:** Mismatched names (e.g., `My_Episode_ledger.json` vs `my-episode_treatment.txt`).  

2. **Open file handles in nodes:**  
   - If audio/text nodes (e.g., `AudioGen`, `OTR_LLMScriptWriter`) leave files open, directory moves will fail on Windows.  
   - **Action:** Audit nodes for proper file closure (e.g., `with open(...)`).  

3. **Idempotency of `rename_episode`:**  
   - Calling `rename_episode(new_id)` twice should either no-op or fail.  
   - **Test:** Ensure second call detects the directory is already renamed.  

---

### 4. Prioritized Recommendations  

1. **Hard-fail directory move with retry:**  
   - Inline 3 attempts × 0.5s.  
   - **Error message:** Include advice to check for manually locked files.  

2. **Strict directory state handling:**  
   - If `new_ep_dir` exists → `RuntimeError`.  
   - If `old_ep_dir` missing but `new_ep_dir` exists → treat as already moved.  

3. **Sequence operations:**  
   - **Order:** Move dir → Update state → Rename ledger → Rename sidecars.  

4. **Precise sidecar renaming:**  
   - Use `old_id_*.txt` glob.  
   - Avoid `pending_*` to prevent accidental renames.  

5. **Slug consistency:**  
   - Ensure `new_id` is slugified **before** `rename_episode` is called, or apply the same slug function to both ledger and treatment.  

---

### 5. Uncertainties Needing Verification  

1. **Slugification workflow:**  
   - Is `new_id` pre-slugified by the caller, or does `rename_episode` need to slugify it?  

2. **Open file handles:**  
   - Do nodes like `AudioGen` or `OTR_LLMScriptWriter` properly close files?  

3. **Idempotency:**  
   - Does `rename_episode(final_ep)` called twice behave correctly?  

4. **Path normalization:**  
   - Are paths compared case-insensitively using `os.path.normcase()`?  

---

### Final Recommendation Summary  

- **Adopt hard-fail with retry** (inline, 3×0.5s).  
- **Enforce strict directory state checks** (existing dirs = error).  
- **Rename ledger before treatment.**  
- **Use `old_id_*.txt` for sidecars.**  
- **Verify slug consistency and file handle closure in nodes.**  
- **Add tests for edge cases (existing dirs, idempotency).**
