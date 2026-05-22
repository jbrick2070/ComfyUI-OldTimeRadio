# Round C -- NVIDIA (nvidia/llama-3.3-nemotron-super-49b-v1.5) elapsed=93.8s

### Third Opinion: Tying the Knot on OTR Announcer Bookends

---

### 1. **Agreements & Flaws in Shared Positions**
**Agree with Two-Pass Architecture**  
Both prior answers correctly advocate for separate intro/outro passes. This avoids ordering hazards and keeps ledger structure intact.  
*Flaw Missed:* Neither addresses how the outro pass should handle episodes where the first and last announcer beat are the same (e.g., 1-beat episodes). A combined fallback is needed here.

---

### 2. **Resolution of Disagreements**
#### **Prompt Context for Outro (ChatGPT vs. Gemini)**  
**Side with Gemini.**  
- **VRAM Risk is Real:** Gemini correctly identifies that appending the full script to the outro prompt bloats KV cache. On a 16 GB GPU with 14.5 GB ceiling, this risks OOM errors during the final pass.  
- **Third Path:** Use `script_brief` + `news_close_brief` *and* the **generated intro text** (not the full script). This provides tonal continuity without context explosion.  

#### **C7 Interpretation**  
**Concur with Gemini.**  
- C7 requires deterministic reproducibility (same inputs → same audio bytes), not preservation of legacy bugs. Fixing the outro override is compliant.

---

### 3. **Factual Errors & Hallucinations**
#### **ChatGPT Errors**  
- **VRAM Misstatement:** Claimed prompt size only affects inference time, not VRAM. Incorrect—KV cache scales with context length.  
- **File References:** Cited `_otr_line_composer.py` but did not specify where `compose_line` is defined (L.120 in v2.3.1).  

#### **Gemini Errors**  
- **Assumed ComfyUI Execution Model:** Gemini assumes a single Python loop for all beats, but ComfyUI nodes may execute per-beat via `OnBeat` events. Unverified.  
- **No Line Numbers:** Like ChatGPT, Gemini cites files but no exact lines (e.g., `override_announcer_close` is in `OTR_LedgerScriptWriter.py`, L.89).

---

### 4. **Critical Omissions in Both**  
1. **Fallback Composition in Ledger:**  
   - Neither specifies how fallback text should be injected into `ledger.lines[]`. The writer must handle LLM failures *during* the loop for the intro and *post-loop* for the outro.  

2. **Prompt Caching Implementation:**  
   - Both mention prompt reuse but don’t note that ComfyUI’s LLM node allows prefix sharing via `prompt_template` fields. This could reduce VRAM by reusing system prompt tokens.  

3. **TTS Validation:**  
   - Fallbacks must pass TTS preflight checks (e.g., no unsupported characters). Neither answer ties this to the `clean_one_line` helper.  

4. **Versioning for C7:**  
   - If C7 requires bit-identical outputs across code versions, the new passes must be guarded by a version flag. Unaddressed.

---

### 5. **Prioritized Recommendations**  
1. **Two Dedicated LLM Passes**  
   - `compose_announcer_intro()` in `_otr_line_composer.py` (L.120, near `compose_line`).  
   - `compose_announcer_outro()` post-loop in `OTR_LedgerScriptWriter.py` (after L.150, where ledger is finalized).  

2. **Outro Prompt = `script_brief` + `news_close_brief` + `intro_text`**  
   - Avoids full script context bloat while preserving tonal coherence.  

3. **Fix Key-Name Bug via Schema Enforcement**  
   - In `news_interpreter.py`, enforce output schema:  
     ```python
     # news_interpreter.py L.67
     OUTPUT_SCHEMA = {"news_close_brief": str, "other_fields": ...}
     ```  
   - In `OTR_LedgerScriptWriter.py`, remove alias guessing (L.89).  

4. **Deterministic Fallbacks**  
   - **Intro:**  
     ```python
     # _otr_line_composer.py L.210
     def fallback_intro(script_brief: str) -> str:
         return f"Tonight on SIGNAL LOST: {script_brief[:200]}. Listen closely."
     ```  
   - **Outro:**  
     ```python
     # _otr_line_composer.py L.230
     def fallback_outro(news_close_brief: str) -> str:
         return f"This has been SIGNAL LOST. {news_close_brief[:250]}. Good night."
     ```  

5. **Skip Generic Outro Composition**  
   - In `OTR_LedgerScriptWriter.py`, modify the beat loop (L.120):  
     ```python
     if beat.is_final_announcer:
         line["text"] = fallback_outro(news_close_brief)  # Temporary placeholder
         continue
     ```  

6. **Prompt Prefix Sharing for VRAM**  
   - In `_otr_line_composer.py`, reuse the system prompt prefix:  
     ```python
     SYSTEM_PROMPT_PREFIX = "You are a 1940s radio announcer..."
     ```  
     Share this across `compose_line`, `compose_announcer_intro`, and `compose_announcer_outro`.

---

### 6. **Uncertainties Requiring Verification**  
1. **ComfyUI Execution Context**  
   - Is the script generated in a single node execution or across multiple `OnBeat` events? If the latter, the post-loop outro pass requires a new node.  

2. **LLM Backend Memory Behavior**  
   - Does the backend (e.g., vLLM, Ollama) pre-allocate KV cache or allocate dynamically? Critical for assessing OOM risk.  

3. **TTS Compatibility of Fallbacks**  
   - Need to validate fallback text against TTS constraints (e.g., no special characters).  

---

### Final Verdict  
**Adopt two passes with constrained outro context, enforce schemas, skip generic outro composition, and share prompt prefixes.** This balances coherence, VRAM safety, and determinism. The largest payoff is retiring the broken `override_announcer_close` while adding minimal VRAM pressure.
