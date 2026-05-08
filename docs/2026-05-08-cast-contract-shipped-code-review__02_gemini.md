# Round B -- Gemini (gemini-3.1-pro-preview-customtools) elapsed=65.4s

### 1. Core Recommendation

**PARTIALLY AGREE:** ChatGPT correctly caught a guaranteed `KeyError` trap in Element 4, but it completely missed a guaranteed `TypeError` serialization crash in Element 1 and misunderstood your explicit design goals for Element 6.

### 2. Factual Errors in ChatGPT's Review

*   **Element 6 Critique is invalid:** ChatGPT complains that `NARRATOR:` and mixed-case tags are dropped. This ignores your explicit prompt constraint: *"Goal: cheap, predictable, false-negative-prone... drop structural header words."* The extractor is functioning exactly as you designed it for Phase 0+.
*   **Element 2 Rerun logic is partially misunderstood:** ChatGPT claims normal ComfyUI reruns will fail. Because ComfyUI caches nodes with identical inputs, a pure rerun *won't* re-execute the lock node. However, ChatGPT is right that it will fail on a fresh boot or if upstream inputs change (see my recommendation below).

### 3. Important Things That Were OMITTED (The Real Weak Spots)

*   **Element 1 (Serialization Crash):** `json.dumps` will throw a `TypeError: Object of type VoiceSpec is not JSON serializable`. In Element 1, you map `"voice_spec": c.voice_spec`. Element 3 shows `VoiceSpec` is instantiated as an object (`VoiceSpec(engine=engine, preset=preset)`). Unless `VoiceSpec` inherits from `dict` or `NamedTuple`, standard `json.dumps` will hard-crash here.
*   **Element 2 (Resume Lockout):** If a downstream node (like Bark) OOMs or crashes, the user is permanently locked out of resuming that episode because `locked_path.exists()` is a blind check. If they restart ComfyUI (clearing the node cache) and hit Queue Prompt to resume, the workflow will hard-crash. 
*   **Element 4 (Undefined Helper):** You call `_coerce_voice_spec(assignments[name])`, but Element 3 defines `parse_voice_spec`. 

### 4. Short Recommendation

1.  **Fix the Element 4 `KeyError` (ChatGPT's best catch):** Do not index the original dictionary with the stripped name. Clean the dictionary first:
    ```python
    clean_assignments = {str(k).strip(): v for k, v in assignments.items() if str(k).strip()}
    sorted_names = sorted(clean_assignments.keys(), key=str.upper)
    # ... then use clean_assignments[name]
    ```
2.  **Fix the Element 1 `TypeError`:** Ensure `VoiceSpec` is a `NamedTuple`/`dataclass` and convert it to a dict in the `normalized` generator: `"voice_spec": c.voice_spec._asdict()` (or `__dict__`).
3.  **Upgrade Element 2 for ComfyUI UX:** Instead of a blind `.exists()` check, if the file exists, *read it*. If the on-disk `cast_contract_version` matches the current in-memory version, log a warning and pass through. If it differs, *then* raise the `RuntimeError`. This preserves your strict immutability rule while allowing ComfyUI crash recovery.

### 5. Uncertainties to Verify

*   **Does `CastContract` actually have a `.to_dict()` method?** Element 2 calls `contract.to_dict()`, but Element 1 manually builds a dictionary to serialize. If `.to_dict()` doesn't exist (or if it also fails to serialize `VoiceSpec`), Element 2 will crash.
*   **Is `_coerce_voice_spec` a typo?** Verify if `_coerce_voice_spec` in Element 4 is just a typo for `parse_voice_spec` from Element 3, or if it's an omitted wrapper function that handles the `"bark:"` prefix defaulting you mentioned.
