# Round C -- Claude Synthesis

## Ground-truth verification (done before synthesis)

Both consultants flagged the **EpisodeAssembler `output_path` contract** as an uncertainty that needed source-code verification before committing. I read both ends.

**Node 7 `OTR_EpisodeAssembler` (`nodes/scene_sequencer.py` line 1016):**
```python
output_path = "(video-only - MP4 written by OTR_SignalLostVideo)"
```
`output_path` is a **literal placeholder string**, not a real WAV path. The comment at line 1013 confirms: *"Video-only pipeline - MP4 is written by OTR_SignalLostVideo. No WAV or PNG files are saved here."*

**Node 12 `OTR_SignalLostVideo` (`nodes/video_engine.py` lines 1321, 1443):**
```python
tmp_wav = os.path.join(tempfile.gettempdir(), "otr_video_audio.wav")
# ... writes WAV, feeds ffmpeg, then:
os.remove(tmp_wav)   # deleted after encode
```
Node 12 writes a temp WAV in `%TEMP%` and deletes it after its own ffmpeg run. Not a stable path. Not exposed as an output socket.

**Conclusion: both consultants' recommended wiring `7.output_path -> HyworldRenderer.final_audio_path` is invalid.** That wire would feed the literal string `"(video-only - MP4 written by OTR_SignalLostVideo)"` into HyworldRenderer, which passes it to `ffmpeg -i` at `renderer.py` line 233. The renderer would hit the `if not audio_path.exists()` check at line 177 and return empty-string FALLBACK every single run. The HyWorld branch would never actually produce video.

This is the exact verification both consultants requested. They were right to flag it; I was right to check.

---

## Points locked in (ChatGPT and Gemini agree)

| # | Decision | Rationale |
|---|---|---|
| 1 | **Option A: parallel branches.** Node 12 stays terminal on its existing path; HyWorld trio runs in parallel and produces a second MP4 or falls back to empty-string. | Smallest change, preserves C7 audio guarantee, no router node introduced. |
| 2 | **Keep Node 12 unchanged.** | Known-good baseline. Do not demote or modify in this commit. |
| 3 | **Head-start deferred** -- Bridge waits for `scene_manifest_json` from Node 3. | CPU-only stub worker; async pre-bake buys nothing and adds partial-input complexity. |
| 4 | **VRAM coordination stays inside `worker.py` subprocess layer.** No coordinator node in the graph. | Coordinator is future-proofing for a GPU path that does not exist yet. |
| 5 | `Node 3.scene_manifest_json -> 16.scene_manifest_json` -- correct. | Bridge consumes scene timing/offsets; that is exactly what the manifest carries. |
| 6 | `16.shotlist_json -> 18.shotlist_json` -- correct. | Bridge emits shotlist, Renderer consumes it. |
| 7 | `Node 1.script_json -> 16.script_json` and `Node 2.production_plan_json -> 16.production_plan_json` -- correct. | Standard feeds. |

## Gemini's catch (adopted)

**`episode_title` is REQUIRED on `OTR_HyworldBridge`** (`otr_v2/hyworld/bridge.py` lines 112-114). ChatGPT's commit plan omitted wiring it. This would have thrown at workflow-validate time in ComfyUI.

**Resolution:** `episode_title` is a widget input on the Bridge node with no obvious source socket in the current graph (ScriptWriter/Director emit JSON, not a dedicated title string). For v2.0-alpha, set `episode_title` as a **widget literal** on the Bridge node itself -- same pattern as `episode_title` on `OTR_SignalLostVideo` (line 1203, optional widget with last-resort override). No new source node required.

If we later want the title to track the script's `{"type":"title"}` token automatically, that is a separate Bridge-internal change (mirror the resolution logic from `SignalLostVideoRenderer.render_video` lines 1239-1281). Not required for the wiring commit.

## Gemini's Poll-outputs-to-PrimitiveNode suggestion (deflected)

**Recommendation: drop for v2.0-alpha.** `HyworldRenderer.render_log` (STRING output) already carries status visibility. Adding two extra `PrimitiveNode` instances adds 2 nodes + 2 links of JSON noise for marginal debugging value. Dangling STRING outputs are legal in ComfyUI; they do not trigger orphan warnings. If we want live HyWorld telemetry, a proper diagnostic node is a better next step than primitive display.

## Blocker and resolution options (the `final_audio_path` problem)

`OTR_HyworldRenderer.final_audio_path` is declared as STRING and requires a real on-disk WAV file path. Nothing in the current v2.0-alpha graph emits one. Three minimum-change options:

### Option X1 -- Modify `OTR_HyworldRenderer` to accept AUDIO tensor type
- Change input type: `"final_audio_path": ("STRING", ...)` -> `"episode_audio": ("AUDIO", ...)`
- Inside `execute()`: write a temp WAV from the AUDIO tensor (exactly like `SignalLostVideoRenderer` does at `video_engine.py` lines 1318-1390).
- Workflow wire becomes: `Node 7.episode_audio -> 18.episode_audio` (AUDIO bus, typed, matches ComfyUI convention).
- **Cost:** ~15 lines of code change in `renderer.py`. No new node. No EpisodeAssembler modification.
- **Pro:** Cleanest graph. Typed AUDIO link. No stale tempfile concerns (renderer owns its own temp).
- **Con:** Node input contract change means any external docs/tests referencing `final_audio_path` go stale.

### Option X2 -- Modify `OTR_EpisodeAssembler` to actually save a WAV and emit the path
- Replace the placeholder string at `scene_sequencer.py` line 1016 with a real `output/old_time_radio/<title>_<timestamp>.wav` write.
- `output_path` becomes a real STRING path that both HyworldRenderer and any future consumer can use.
- **Cost:** ~20 lines of code in `scene_sequencer.py` + a test update for the EpisodeAssembler contract.
- **Pro:** Fixes a latent bug -- the current `output_path` is actively misleading. Future nodes get a clean WAV path source.
- **Con:** Touches the audio path. C7 violation risk if the WAV is not byte-identical to what SignalLostVideo currently writes and deletes. Requires byte-identical regression verification.

### Option X3 -- Add a new adapter node `OTR_AudioToWavPath`
- Minimal new node: AUDIO in, STRING out, writes WAV to a stable path and returns it.
- Workflow wire becomes: `Node 7.episode_audio -> 19.audio -> 18.final_audio_path`.
- **Cost:** ~40 lines of new node code + test + registration in `__init__.py`.
- **Pro:** Zero changes to existing nodes. Reusable for any future STRING-path-consuming node.
- **Con:** One extra node in the graph for a micro-transformation. Slight JSON bloat.

### My recommendation: **Option X1** (modify HyworldRenderer to accept AUDIO tensor)

- Smallest code surface (one file, ~15 lines).
- Cleanest workflow topology (typed AUDIO link, one wire not two).
- Keeps C7 byte-identical audio guarantee trivially because the renderer only passes the WAV to ffmpeg; it does not modify the audio. ffmpeg `-c:a copy` at renderer.py line 235 is already in place.
- Preserves Node 7 as a pure AUDIO producer (no file I/O side effects on the audio path).
- Option X2 carries real C7 risk; Option X3 is gratuitous for a one-consumer adapter.

Proceeding with Option X1 unless Jeffrey picks otherwise.

---

## FINAL COMMIT PLAN (Option X1)

**Files to modify:**
1. `otr_v2/hyworld/renderer.py` -- change `final_audio_path: STRING` input to `episode_audio: AUDIO`; add internal tempwav write; keep C7 path (audio is never re-encoded by ffmpeg; `-c:a copy` already in place at line 235).
2. `workflows/otr_scifi_16gb_full.json` -- add three HyWorld nodes + links.

**New nodes in workflow JSON** (allocate IDs sequentially after existing 15):
- `OTR_HyworldBridge` -- Node 16
  - widgets: `lane="balanced"`, `chaos_ops="[]"`, `chaos_seed=42`, `sidecar_enabled=true`, **`episode_title="<episode title>"` (widget literal)**
- `OTR_HyworldPoll` -- Node 17
- `OTR_HyworldRenderer` -- Node 18
  - widgets: `crt_postfx=true`, `output_resolution="1280x720"`, `episode_title="<same>"`

**New links** (seven, all additive):
- `1.script_json -> 16.script_json`
- `2.production_plan_json -> 16.production_plan_json`
- `3.scene_manifest_json -> 16.scene_manifest_json`
- `16.hyworld_job_id -> 17.hyworld_job_id`
- `17.hyworld_assets_path -> 18.hyworld_assets_path`
- `7.episode_audio -> 18.episode_audio`  **(AUDIO link, replaces ChatGPT's invalid `7.output_path -> 18.final_audio_path`)**
- `16.shotlist_json -> 18.shotlist_json`

**Existing links:** unchanged; no removals; Node 12 remains terminal.

**Test after:**
1. `pytest tests/test_dropdown_guardrails.py -v` -- confirm no UI surface regressions.
2. `pytest tests/v2/test_audio_byte_identical.py -v` -- confirm C7 byte-identical audio.
3. `python -m pytest "C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py" -v` -- full Bug Bible regression.
4. Validate JSON loads in ComfyUI with zero orphan/muted regressions.
5. One full episode smoke test on Windows. Confirm Node 12 MP4 still renders. Confirm HyWorld branch produces either a second MP4 path or clean empty-string FALLBACK.
6. AST parse both modified files via `py_compile`.

**Rollback:**
- Revert the two-commit sequence (renderer.py change + workflow JSON change).
- Optionally delete `io/hyworld_in/*` and `io/hyworld_out/*` test artifacts.
- No data migration.

**FIRST-MOVE:** Modify `otr_v2/hyworld/renderer.py` to accept AUDIO tensor input (Option X1), then wire the three HyWorld nodes into `workflows/otr_scifi_16gb_full.json` with the seven links above.
