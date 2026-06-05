# OTR Video Engine Architecture -- PROBLEM STATEMENT

**For:** multi-AI round-robin on **how best to architect a model-agnostic, scalable, hardware-agnostic VIDEO layer** for ComfyUI-OldTimeRadio (OTR).
**Status:** architecture only -- NO build this round. Audio overhaul is already shipped (see context). We are designing the *frame* that any present or future video model plugs into.

---

## 1. One-paragraph problem
OTR turns a script into an old-time-radio-style episode: dialogue voices + announcer + music (audio) driving portraits/lip-sync/motion (video), muxed to an MP4. The AUDIO side just shipped a **model-agnostic per-role engine registry** (character voice / announcer / music -- each a swappable, profile-driven engine resolved through a single ledger, audio kept byte-identical). We now want the **symmetric thing for VIDEO**, with two added demands: (a) it must run on **any ComfyUI / any GPU / down to <=8GB VRAM and CPU-only** -- not just the developer's bleeding-edge Blackwell box -- and (b) it must **absorb new models indefinitely** (a new video model = a small adapter + a profile row, never a graph rewrite). Design the architecture that makes both true.

## 2. Why now / why this matters
- v1.7 is public; v2 must be *great*, and "great" includes **everyone gets a runnable option**, not only 16GB Blackwell owners.
- New video + lip-sync models ship monthly. A hard-wired pipeline is obsolete on arrival. The audio registry proved the pattern; we want it for video before we wire any specific model.
- The developer's stack (RTX 5080, sm_120, torch-nightly, CUDA 13) is actually the **hardest** install target; mainstream cards (3060/4060 8-12GB, older CUDA) are easier but more VRAM-constrained. The architecture must not assume any single stack.

## 3. Hard constraints (carry over from the project -- non-negotiable)
- **Audio is king.** Video is downstream and must NEVER alter the audio bytes. Audio stays byte-identical to its baseline; video determinism is allowed to be non-strict (best-effort seed-pin).
- **VRAM tiers + single-residency.** 16GB -> 12 -> 8 -> "radio" (<=8GB / CPU). One heavy model resident at a time; load -> run -> **tear down before the next** (the existing 14.5GB-ceiling discipline). On 8GB the portrait model, the lip-sync model, and any motion model cannot co-reside.
- **Hardware-agnostic + offline + local.** 100% local, open-source, no paid/API/cloud. Must degrade across GPUs and run CPU-only at the floor.
- **Per-machine dependency safety.** Each engine must pass a dep-isolation pilot (no `xformers`/`flash_attn`/torch-swap that bricks the install) -- and the *passing set differs per machine*, so the architecture must self-limit to what installs HERE.
- **Reuse the audio template.** Engine `Protocol` + `register/get_engine/engines_for_role`, a profiles YAML (per-engine params + VRAM class + model hash + license), an `assert_usable` taxonomy, and a single ledger as the source of truth.

## 4. Proposed direction (CRITIQUE this -- do not assume it's right)
The developer's current intuition, offered as a starting point for the reviewers to pressure-test:

1. **Video roles paired 1:1 to audio roles** -- a **character-video**, an **announcer-video**, and a **music-video** slot; for each role you pick any model. (Clean symmetry with the audio character/announcer/music roles.)
2. **Per-role model menu (today, extensible):** `humo | latentsync | musetalk | ltx | still+kenburns`. Designed so a future model is just another menu entry.
3. **One master "smart" workflow, parameterized by a hardware/VRAM tier** -- a single generic JSON that runs anywhere; the tier profile changes *parameters* (which model, resolution, quantization, offload, clip strategy, counts), NOT the graph topology. Explicitly NOT a pile of hand-built per-GPU JSONs.
4. **Clip-strategy sub-option per role:** either **a unique clip per audio line**, or **loop a small pool of clips** across the audio (a VRAM/time lever -- e.g. music-video loops 3 clips; character-video is unique per line).
5. **Everything hangs off the existing ledger** -- the open question is exactly *how* video choices (engine, strategy, counts) get represented and stamped so caching + reproducibility hold and the ledger stays the single source.

## 5. The technical crux -- the engines are NOT the same kind
A uniform slot interface has to normalize four different families behind one contract `{portrait, audio_clip, duration, seed} -> canonical_clip`:
- **Generative-from-audio (HuMo):** portrait + audio -> talking clip in one shot.
- **Lip-sync overlay (LatentSync, MuseTalk):** need a **base moving clip** + audio, then repaint the mouth. *Where does the base clip come from on 8GB?* (loop the portrait? a cheap motion pass?) -- this is the leakiest seam.
- **Generative-motion, audio-agnostic (LTX):** portrait -> motion, ignores the dialogue (no sync).
- **Static (still + Ken Burns):** no model; near-zero VRAM; arguably the most on-theme for "radio".

Announcer-video and music-video may not want lip-sync at all (station-ID card, visualizer, B-roll), so the contract must also cover "no face / no sync" engines.

## 6. Open architecture questions for the round-robin (the payload)
1. **Master-workflow mechanism.** ComfyUI graphs are largely static. What is the right way to get "one smart workflow, parameterized by tier"? Options to weigh: a node that branches internally on a tier input; a **build-time JSON emitter** that stamps the right graph per tier (OTR already needs a headless litegraph builder for the audio opt-in workflow -- could it emit per-tier video graphs?); a hardware-probe node that auto-selects a tier; or a hybrid. Name the ComfyUI-specific tradeoffs (caching, `IS_CHANGED`, validation, UI).
2. **Ledger integration.** How should per-role video engine + clip-strategy + counts be represented so the ledger stays the single source and the cache key is correct? A `video` section + a **"ShotLock" node** (analogue of the audio CastLock that already stamps the cast) vs a separate video ledger vs inline on each node? How does it reference the audio ledger it must stay in sync with (line ids, durations)?
3. **The uniform VideoEngine contract.** What is the minimal interface that cleanly spans all four families -- especially the base-clip dependency for the overlay engines and the "no-face" announcer/music engines -- without leaking engine specifics into the graph?
4. **Clip strategy.** How to express unique-per-line vs loop-a-pool per role, and how does each interact with (a) the cache key, (b) the audio line durations a looped clip must still cover, and (c) VRAM/time budget?
5. **VRAM tiering + graceful degradation.** How does the tier profile drive single-residency/teardown sequencing automatically, and how does the master graph degrade a role as VRAM drops (HuMo -> LatentSync -> LTX -> still) without manual rewiring?
6. **Per-machine dependency management.** How does one shipped architecture self-limit each role's engine menu to what actually installs on THIS machine (Blackwell vs 3060 vs CPU) -- a per-tier/per-host capability probe -- without forking the codebase?
7. **Scalability / add-a-model contract.** What is the exact "drop in a new model" contract so a future video model is an adapter + a profile row + (if needed) a reference-bank entry, with zero graph surgery -- mirroring how audio adds an engine?
8. **Frame normalization.** How to normalize aspect/resolution/fps across engines (portrait HuMo vs landscape LTX vs still) -- a compositor-side canonical canvas policy -- so the assembler is engine-agnostic? (Bonus: this also dissolves the parked HuMo pillarbox issue.)
9. **Determinism.** Video is non-strict. How much seed-pin/reproducibility is worth carrying, and where, given the audio is the byte-identical anchor?
10. **Role reuse.** Should announcer-video and music-video reuse the character pipeline with different engines, or be their own lighter slots (visualizer / B-roll / looped pool)? Where is the shared abstraction vs per-role specialization?

## 7. Out of scope (this round)
- No code, no graph, no model benchmarking. We are choosing an **architecture that holds any models**, not picking the models.
- Audio engine work is done and is not reopened.
- Final VRAM numbers / which engines pass on which card -- those come from later per-tier pilots; assume the menu self-limits.

## 8. What a strong answer delivers
A recommended architecture covering: the role/slot + registry shape; the VideoEngine contract that spans the four engine families; the ledger integration (and whether a ShotLock node is warranted); the master-workflow mechanism (one parameterized graph vs a per-tier emitter) with the ComfyUI tradeoffs named; the VRAM-tier + degradation model; the per-machine capability/dep-isolation strategy; and the add-a-model contract -- each justified against the constraints in section 3.
