# Round A -- ChatGPT (gpt-5.4) elapsed=17.9s

**Call:**

- **Q1 default:** **Union Pro 2.0 first**, but **use one control mode per render pass** and keep a **Depth-only fallback path**. Do **not** make stacked Depth+Canny the default on this machine.
- **Q2 default:** **C) Both, with A as fallback.** Accept `control_image` from shotlist, otherwise derive from Day 2 anchor.

## Why

### Q1: Union Pro 2.0 vs stacked Depth+Canny

**Pick Union Pro 2.0 as the default** because it gives you the biggest payoff for the smallest change: one adapter loaded, mode switched at inference, simpler harness, less VRAM risk than two separate ControlNets resident together.

My confidence is **moderate**, not absolute:

1. **FP8 compatibility / dequant risk**
   - In diffusers, the **base FLUX transformer** can sit in FP8 while the **ControlNet stays in bf16/fp16**. That does **not automatically imply full dequantization of the base model**.
   - The real risk is not “base explodes to fp16,” but **extra activation / adapter memory** during ControlNet forward.
   - I would **not assume** Union Pro itself runs in float8; treat it as bf16/fp16 sidecar.

2. **Stacked Depth + Canny**
   - On your card, with FLUX FP8 + CPU offload already near ~12 GB at 1024², **two ControlNets is the more likely way to blow past 13.5 GB peak**.
   - Even if weights offload, inference activations and duplicated conditioning paths are the problem.
   - So: **stacking is a “maybe later if profiling proves safe,” not the default.**

3. **Mode switch cost**
   - Union Pro’s `control_mode` is typically just a **runtime branch / embedding selection**, not a reload or recompile.
   - I’d expect it to be **cheap enough** relative to a FLUX denoise pass; **sub-500 ms is plausible**, but I would not promise it on Windows + spawn + first-call warmup.
   - Important distinction: **first call** may pay compile/kernel cache cost; **subsequent calls** should be cheap.

4. **Blackwell sm_120 gotchas**
   - No specific Union-Pro-on-sm_120 blocker I can confidently cite.
   - The likely issues are the same as Day 2/3: **new-kernel path instability, first-run warmup variance, and dtype edge cases** with FP8 + auxiliary modules.
   - So add a **hard fallback**: if Union load/infer fails, drop to **single Depth ControlNet** before giving up.

### Practical recommendation
Implement:
1. `Union Pro 2.0` with `control_mode="depth"` as default.
2. Optional `control_mode="canny"` for edge-heavy shots.
3. **No simultaneous multi-control by default.**
4. Feature-flag stacked controls later only if measured peak stays under budget.

---

## Q2: Control-image source

**Choose C: both, with A as fallback.**

Why:
- **A-only** is clean today, but it hard-codes Day 4 to Day 2 and makes future storyboard injection awkward.
- **B-only** is flexible, but it forces the shotlist/LLM layer to solve a problem you already can solve locally.
- **C** keeps the pipeline clean:
  - If `shot["control_image"]` exists, use it.
  - Else use `io/hyworld_out/<job>/shot_XXX/render.png`.
  - Then derive depth/canny locally and deterministically.

That preserves your current pipeline while opening the door for storyboard/sketch control later with **zero API break**.

## One extra thing you may be missing

For the gate “same layout across 3 prompt variations,” **Depth is the safer default than Canny**. Canny tends to over-constrain texture/edges and can become brittle when prompts diverge. So my real default is:

- **Union Pro 2.0 + depth**
- `control_image` from shotlist if present, else Day 2 anchor
- fallback to **single dedicated Depth ControlNet** if Union misbehaves on your stack

That is the lowest-risk Day 4 path.
