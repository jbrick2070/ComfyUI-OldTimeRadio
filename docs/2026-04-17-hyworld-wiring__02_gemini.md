# Round B -- Gemini (gemini-2.5-flash) elapsed=26.0s

I **PARTIALLY AGREE** with the core recommendation. While the overall strategy is sound and aligns with the "smallest change, largest payoff" principle, there are critical omissions regarding required inputs and robust error handling that need to be addressed.

### Factual Errors

ChatGPT's answer contains no factual errors regarding the current workflow state or the proposed node behaviors as described in the prompt. Its understanding of the ComfyUI node contract and the implications of the C7 rule is accurate.

### Important Omissions

1.  **Missing `episode_title` Wiring:** `OTR_HyworldBridge` explicitly lists `episode_title` as a **required** STRING input. ChatGPT's proposed "New links" section completely omits wiring this input. This is a critical oversight, as the workflow would fail without it. The source of this `episode_title` would need to be identified (e.g., from `OTR_Gemma4ScriptWriter` or `OTR_Gemma4Director`'s JSON outputs, or a new dedicated input node).
2.  **Dangling `OTR_HyworldPoll` Outputs:** The `OTR_HyworldPoll` node outputs `status` (STRING) and `status_detail` (STRING). While the `OTR_HyworldRenderer` implicitly handles a "FALLBACK" status from the `hyworld_assets_path` input, these `status` and `status_detail` outputs are left dangling in ChatGPT's proposed wiring. For an alpha release, these should at minimum be connected to a `PrimitiveNode` for inspection or logged to aid debugging and provide visibility into HyWorld's operation.
3.  **Node ID Allocation & JSON Structure:** The commit plan is slightly underspecified regarding the actual JSON modifications. It doesn't explicitly state that the new nodes themselves (with their default parameters) need to be added to the `nodes` array in the JSON, nor does it suggest how their ComfyUI node IDs would be allocated (e.g., sequentially after 15). While implied, explicit detail is better for a commit plan.

### My Recommendation

*   **Adopt Option A (Parallel Branches):** Proceed with wiring the HyWorld trio as a parallel branch to the existing `OTR_SignalLostVideo` (Node 12). This minimizes risk to the established audio and video pipeline, aligns with the "smallest change, largest payoff" principle for v2.0-alpha, and gracefully handles the current stub-mode nature of HyWorld. No `OTR_VideoRouter` node is necessary at this stage.
*   **Wire All Required Inputs (including `episode_title`):** Connect Node 3's `scene_manifest_json` to `HyworldBridge.scene_manifest_json`, Node 7's `output_path` to `HyworldRenderer.final_audio_path`, and `HyworldBridge.shotlist_json` to `HyworldRenderer.shotlist_json`. **Crucially, identify a source for `episode_title` (e.g., from Node 1 or 2's JSON outputs, or a new `PrimitiveNode` for a static string) and wire it to `OTR_HyworldBridge.episode_title`.**
*   **Defer Head-Start Optimization:** For v2.0-alpha, `OTR_HyworldBridge` should wait for Node 3's `scene_manifest_json`. The current CPU-only stub worker provides no benefit from an early head-start, and introducing complexity for partial inputs is unwarranted.
*   **Encapsulate VRAM Coordination:** Keep VRAM coordination logic within the `worker.py` subprocess layer, leveraging `otr_v2/hyworld/vram_coordinator.py` internally. Do not introduce a VRAM coordinator node into the ComfyUI graph for v2.0-alpha.
*   **Handle `OTR_HyworldPoll` Outputs:** Connect `OTR_HyworldPoll`'s `status` and `status_detail` outputs to `PrimitiveNode` instances. This provides essential debugging visibility during alpha testing without adding complex routing logic.
*   **Verify `output_path` Contract:** Before deployment, **verify that `OTR_EpisodeAssembler` (Node 7) fully writes and flushes the final WAV file to `output_path` *before* emitting the `output_path` string.** This is a critical check for C7 compliance and robust operation.

### Uncertainties

1.  **`OTR_EpisodeAssembler` (Node 7) `output_path` contract:** I would need to inspect the source code of `otr_v2/.../episode_assembler*.py` to confirm that the final WAV file is fully written and closed *before* the `output_path` string is returned by the node. This is paramount for `OTR_HyworldRenderer` to safely access the file and for C7 compliance.
2.  **Source of `episode_title`:** The prompt does not explicitly state where `episode_title` would originate. I'd need to review the outputs of `OTR_Gemma4ScriptWriter` (Node 1) and `OTR_Gemma4Director` (Node 2) to see if they produce a distinct `episode_title` string, or if a new `PrimitiveNode` (or similar input mechanism) is required.

---

### COMMIT PLAN

  Files to modify:   `workflows/otr_scifi_16gb_full.json`; verification-only read of `otr_v2/.../episode_assembler*.py` to confirm `output_path` contract.
  New nodes in JSON:
    - `OTR_HyworldBridge` (e.g., Node ID 16)
    - `OTR_HyworldPoll` (e.g., Node ID 17)
    - `OTR_HyworldRenderer` (e.g., Node ID 18)
    - `PrimitiveNode` for `HyworldPoll.status` (e.g., Node ID 19)
    - `PrimitiveNode` for `HyworldPoll.status_detail` (e.g., Node ID 20)
    - (Potentially) a `PrimitiveNode` for `episode_title` if not available from existing nodes (e.g., Node ID 21)
  New links:
    - `1.script_json -> 16.script_json` (HyworldBridge)
    - `2.production_plan_json -> 16.production_plan_json` (HyworldBridge)
    - `3.scene_manifest_json -> 16.scene_manifest_json` (HyworldBridge)
    - `(Source of episode_title).output -> 16.episode_title` (HyworldBridge)
    - `16.hyworld_job_id -> 17.hyworld_job_id` (HyworldPoll)
    - `17.hyworld_assets_path -> 18.hyworld_assets_path` (HyworldRenderer)
    - `7.output_path -> 18.final_audio_path` (HyworldRenderer)
    - `16.shotlist_json -> 18.shotlist_json` (HyworldRenderer)
    - `17.status -> 19.input` (PrimitiveNode for status)
    - `17.status_detail -> 20.input` (PrimitiveNode for status_detail)
  Existing links:    unchanged; no removals; Node 12 remains terminal on the existing path.
  Test after:        Validate JSON loads in ComfyUI with zero orphan/muted regressions. Run one full episode smoke test on Windows. Confirm final WAV hash matches pre-change baseline exactly. Confirm Node 12 MP4 still renders. Confirm HyWorld branch produces either a second MP4 path or clean fallback/empty-string behavior. Inspect peak VRAM during Bark to confirm no change. Verify `PrimitiveNode` outputs for HyWorld status.
  Rollback:          Revert the workflow JSON commit SHA; no data migration needed; optionally delete any `io/hyworld_in/*` and `io/hyworld_out/*` artifacts from the test run.

FIRST-MOVE: Open `workflows/otr_scifi_16gb_full.json`, identify a source for `episode_title`, and add the three HyWorld nodes plus the necessary `PrimitiveNode`s and all required links, leaving every existing link untouched.
