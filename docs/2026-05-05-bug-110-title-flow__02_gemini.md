# Round B -- Gemini (gemini-3.1-pro-preview-customtools) elapsed=31.8s

**1. Agreement Statement**
PARTIALLY AGREE; ChatGPT's string manipulation logic is sound, but it severely underestimates the ComfyUI execution graph risks (specifically node caching) and cross-run state leakage inherent in the Layer 3 singleton read.

**2. Factual Errors in ChatGPT's Answer**
*   **Origin of the double underscore:** ChatGPT states, "punctuation is currently removed... The doubled underscore is more likely coming from... existing trailing underscore". This is incorrect. If a title is `"Signal Lost - The Crystal!"`, the generator `c.isalnum() or c in "_ "` strips the `-` and `!`, leaving `"Signal Lost  The Crystal"` (note the double space). The subsequent `.replace(" ", "_")` turns that double space into the double underscore. 
*   **ComfyUI Graph Execution / Race Conditions:** ChatGPT claims, "If the video engine already consumes script/audio/scene outputs... you are fine." This ignores ComfyUI's aggressive node caching. If the inputs to `story_orchestrator.py` haven't changed between queue runs, ComfyUI will *skip* executing it. If your `production_ledger` singleton is re-initialized per run, `video_engine.py` will read an empty title because the orchestrator never ran to populate it.

**3. Important Omissions**
*   **Rule C7 (Byte-identical audio):** ChatGPT failed to explicitly clear your C7 constraint. I will clear it: This patch is 100% safe for Rule C7. You are strictly mutating downstream filename strings and ledger metadata. No RNG seeds, VRAM allocations, or audio tensors are touched.
*   **Singleton State Leakage:** Because ComfyUI keeps the Python process alive, if Run A succeeds (Title A), and Run B fails upstream of the orchestrator but somehow triggers the video node (e.g., via a bypassed/cached graph), `video_engine.py` might read Title A for Run B's output. 
*   **Ledger Persistence:** If `get_ledger()` loads state from disk based on an `episode_id` passed through the graph, the caching issue is mitigated. If it's purely an in-memory dict, you are highly vulnerable to ComfyUI caching bugs.

**4. Own Short Recommendation**
*   **Layer 1 (Slug Cleanup):** Adopt ChatGPT's refined regex and truncation order. It safely handles the space-to-underscore collapse.
    ```python
    safe_title = "".join(c if c.isalnum() or c in "_ " else "" for c in str(episode_title or ""))
    safe_title = _re.sub(r"\s+", " ", safe_title).strip() # Collapse spaces BEFORE underscore conversion
    safe_title = safe_title.replace(" ", "_").lower()[:40]
    safe_title = _re.sub(r"_+", "_", safe_title).strip("_") or "untitled"
    ```
*   **Layer 2 (Ledger):** Stamp `led.data["title"] = _resolved_title` at the top level in `nodes/story_orchestrator.py` (~line 6545). Top-level is architecturally correct for a primary identity field.
*   **Layer 3 (Video Engine):** Implement the hidden read for tonight's `v2.0-alpha` sprint, but **guard it with an episode ID check** to prevent cross-run leakage. 
    ```python
    _ledger_title = ""
    try:
        from .production_ledger import get_ledger
        _led = get_ledger()
        # Only trust the ledger if it belongs to the current run (assuming you have episode_id in scope)
        if _led.data.get("episode_id") == current_episode_id: 
            _ledger_title = str(_led.data.get("title") or "").strip()
    except Exception:
        pass
    episode_title = _ledger_title or episode_title
    ```
*   **Scope Cut:** Do *not* attempt to fix the `episode_id` folder naming tonight. If the folder is created before the LLM resolves the title, fixing it requires a lifecycle refactor that threatens your VRAM/disk I/O assumptions.

**5. Uncertainties to Verify**
*   **ComfyUI Caching Behavior:** I need to know if `get_ledger()` is a purely in-memory singleton that wipes on a new ComfyUI prompt, or if it hydrates from disk. If it wipes, and `story_orchestrator` gets cached, Layer 3 will fail silently and fall back to the old behavior.
*   **`episode_id` Availability:** Does `video_engine.py:1482` have access to the current run's `episode_id`? If not, you cannot safely validate that the singleton belongs to the current run, elevating the risk of cross-run state leakage.
