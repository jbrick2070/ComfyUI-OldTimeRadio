# Cowork Live Artifact Prompt — ComfyUI Workflow QA/Debug

Drop this into Claude **inside Cowork desktop** (not web chat — `window.cowork.*`
only exists in the desktop artifact sandbox). Fill in the three paths at the
top, hit enter, and Claude will build you a pinnable live artifact that
tails your ComfyUI log in real time and auto-highlights the running node.

---

```
Build me a pinned Cowork live artifact to debug a ComfyUI workflow.

WORKFLOW JSON: <full path to your .json>
COMFYUI LOG:   <full path to comfyui_<port>.log>
GPU / VRAM:    <e.g. RTX 4090 24 GB, RTX 5080 Laptop 16 GB>

Read my workflow JSON and build a self-contained HTML artifact:

1. SVG node graph of my workflow with arrows in dependency order. Click
   a node to focus it.

2. Detail card for the focused node: 1-2 sentence purpose, inputs with
   types, outputs with types, the exact console log line to expect when
   it runs, and timing/VRAM expectations specific to my GPU.

3. Live-tail the log every 2s via
   window.cowork.callMcpTool("mcp__Desktop_Commander__read_file",
                            {path: "<my log>", offset: -200})
   Dedupe lines, buffer the last 200. Render in a dark monospace box
   color-coded: green for custom-node hits, amber for warnings, red for
   ERROR / Traceback, muted grey for the rest.

4. Pattern-match log lines to infer which node is currently running and
   highlight it with a blue pulsing border. Mark the previous node
   passed (green dot). On "Prompt executed in", mark the final node
   passed. On any ERROR / Traceback, mark the live node failed (red).

5. Buttons: Start / Stop live tail, Prev / Next (manual stepper),
   Passed / Issue (manual override), Reset state.

6. Per-node "AI: explain deeper" button that calls
   window.cowork.sample(prompt, [JSON.stringify(nodeMeta)])
   with a "in 2-3 sentences explain internals + one failure mode"
   instruction. Cache responses per node.

Technical constraints: complete HTML document (DOCTYPE through </html>),
:root { color-scheme: light }, inline all CSS/JS, no localStorage /
sessionStorage, no position:fixed, no external scripts except Chart.js
CDN if needed. Include a diagnostic panel that shows the raw
callMcpTool response if the shape differs from expectations so we can
iterate.

Build it in one shot; I'll paste the rendered result back if anything
needs adjustment.
```

---

## Notes

- **First artifact often needs a tweak pass.** The MCP tool-result
  shape varies across Desktop Commander versions. The diagnostic panel
  in step 6 is what lets you iterate — paste its contents back to
  Claude and it fixes the extractor.
- **Hit ▶ Start live tail** in the artifact header to begin polling.
  The tail only starts on demand so Cowork's Reload button doesn't
  spam tool calls.
- **Works with any workflow** — the prompt doesn't assume OTR,
  diffusion models, video pipelines, or any particular node set.
  It builds the graph from whatever's in your `.json`.

## License

Public domain / CC0. Use it, fork it, adapt it.
