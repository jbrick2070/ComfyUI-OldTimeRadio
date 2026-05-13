# Final pre-soak JSON QA — `otr_scifi_16gb_full.json`

I'm about to queue a 6-line validation soak against this ComfyUI
workflow. Five separate review passes earlier today caught 11
issues; this is the final sanity check before queue. Repo head:
`052724d` on `v2.0-alpha`. Episode pipeline: Story → Bark/Kokoro
audio → FLUX env stills → portrait render → HuMo character clips →
LTX motion clips → composite (1472×832) → upscale (1920×1080) →
procgen blend.

I do NOT want a rewrite. I want a focused yes/no on whether this
JSON will queue cleanly through ComfyUI's `validate` step and run
end-to-end without a silent mis-routing.

## What was just patched (do not flag these)

- Node 25 input now reads from Node 23 (env stills) via link 104.
  Old link 46 (which fed portrait_batch through UnloadAll) removed.
- Node 23 outputs[0].links = `[101, 104]`; Node 24 outputs[0].links
  = `[83]`; Node 25 inputs[0].link = `104`; `last_link_id` = `104`.
- Node 51 (`OTR_BatchHumoRender`) widget array extended to 17,
  with `humo_max_lines_per_process=6`, `resume_from_ledger=True`,
  `cuda_hard_reset_on_oom=True` baked in.
- Node 52 (`OTR_VideoComposite`) widget array at 15.
- Node 55 (`OTR_BatchLTXRender`) widget array trimmed to 5;
  stale `"fixed"` from a removed `seed_mode` widget dropped.

## What I want you to verify

For each numbered check, give me PASS / FAIL plus one line of
evidence (a node id, a link id, or the byte position of the
disagreement).

1. **JSON parse** — the file loads as a valid JSON object with no
   trailing junk.
2. **Link integrity** — every `links[]` entry is a 6-tuple
   `[id, src_node, src_slot, dst_node, dst_slot, type]`; every
   `dst_node`/`src_node` resolves to an existing `nodes[].id`;
   every `links[]` id is unique; `last_link_id` equals the max
   id present.
3. **Widget-count match** — for each OTR_* node listed below,
   `widgets_values` length matches the count of widget-renderable
   inputs (STRING / INT / FLOAT / BOOLEAN / enum-tuple) in the
   matching Python class's `INPUT_TYPES`. Link-only types
   (MODEL, CLIP, VAE, IMAGE, AUDIO, LATENT, CONDITIONING,
   AUDIO_ENCODER) are excluded:

   | Node id | Type | Expected widget count |
   |---|---|---|
   | 25 | OTR_SaveToEpisodeWorkspace | 2 |
   | 51 | OTR_BatchHumoRender | 17 |
   | 52 | OTR_VideoComposite | 15 |
   | 55 | OTR_BatchLTXRender | 5 |
   | 56 | OTR_RTXUpscale (or VideoUpscaleWithModel-shaped) | as-is |
   | 58 | OTR_PostUpscaleProcgenBlend | 9 |

4. **Save-to-disk routing trace** — for any node whose `widgets_values`
   include `"stills"` or `"portraits"` as a save subdir, walk back
   from its IMAGE input through any `OTR_UnloadAll` passthrough
   and confirm the IMAGE producer's identity matches the save
   intent (env stills → BatchFluxRender; portraits → PortraitRender).
   The recurring trap: `OTR_UnloadAll` is a passthrough that
   forwards whatever IMAGE entered it, so a `SaveToEpisodeWorkspace`
   downstream of UnloadAll gets the LAST image fed in, not the
   originally-intended one.

5. **HuMo → LTX sequencing** — confirm the dependency edge from
   `OTR_BatchHumoRender` to whichever node loads the LTX model is
   intact (typically Node 51 → Node 54 or Node 51 → Node 55
   `humo_clips_dir` STRING input). Without this, ComfyUI's
   topological scheduler can start LTX too early and OOM HuMo's
   16.5 GB MODEL.

6. **No cycles** — graph is a DAG.

7. **Final composite path consistency** — Node 52's `canvas_width`
   / `canvas_height` matches the upstream HuMo / LTX expectations
   (1472×832), and Node 56's upscale output (typically 1920×1080)
   matches what Node 58 expects as input.

8. **No stale widget vocabulary** — scan every `widgets_values`
   for tokens that don't appear in their target node class's
   `INPUT_TYPES`. The recurring trap: `"fixed"`, `"randomize"`,
   `"increment"` are leftover seed-mode values that ComfyUI removed
   from many node families; if they appear in a slot whose Python
   type is FLOAT/INT/STRING (not enum), it's a widget-shift bug.

## Out of scope (do not flag)

- LTX dual-LoRA stack on Nodes 60 + 61 at strengths `0.5` and `0.2`
  is intentional, not duplication.
- The `.mp4 → ledger_json` STRING edges from Node 12 are
  intentional — the receiver derives the ledger path from the
  mp4 stem.
- Cast Contract / Cast Repair / Voice Render integration is
  deferred-by-design via `story_orchestrator.py`, not via JSON.
- The `OTR_LTX_ENGINE` env var is operational (set in the launch
  environment), not a JSON field.

## Closing question

Single yes/no: does this workflow queue cleanly without a silent
mis-routing? If "no", the smallest concrete patch that gets it to
"yes".

File at: `workflows/otr_scifi_16gb_full.json`. Python class
`INPUT_TYPES` definitions live in `nodes/batch_humo_render.py`,
`nodes/batch_ltx_render.py`, `nodes/video_composite.py`,
`nodes/otr_save_to_episode_workspace.py`, etc. Repo:
https://github.com/jbrick2070/ComfyUI-OldTimeRadio (branch
`v2.0-alpha`, head `052724d`).
