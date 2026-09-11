# Driver anchor -- 3.5-reload: per-beat model reload as an OOM surface

Written by the driver (Claude Opus 5, Cowork, 5080) BEFORE any panel round. The panel
proposes; the driver disposes and verifies every claim against the real Windows files.

**Round shape (operator directive): ONE second opinion per round.** r1 codex, r2
cursor, r3 codex, r4 sonnet. **An arc IS coding.** A no-brainer gets no arc -- this
row is here because grounding found a genuine fork with more than one defensible
answer.

**THE OPERATOR'S BAR:** *"as long as it doesn't crash when it's not supposed to."*
Exactness is NOT the goal -- this is a fun experimental app, and aesthetic drift is
explicitly acceptable. This row is classified **DURABILITY**.

---

## 1. What is TRUE TODAY, grounded against current code

eng_ltx_video.py:load() (eng_ltx_video.py:1608-1615) resolves node CLASSES only ('the heavy weight load happens when the loader nodes execute in render_clip'). render_clip (eng_ltx_video.py:1615-1626, graph built :1655 via _build_graph_i2v) builds a FRESH graph every call -- GGUF unet loader, Gemma-3 text-encoder loader, VAE loader, projection-ckpt loader, LoRA loader all included -- executed via wrapper_bridge.run_graph (wrapper_bridge.py:468-497), a hand-rolled topo executor that instantiates each node class and calls FUNCTION fresh every time; no external_results is passed (unlike eng_ltx_8gb), so there is no cross-call weight cache. MotionEngineBase.prepare (motion_common.py:1359-1404) only takes the AS-3 GPU lease and calls load() -- it loads no weights for this engine. MotionEngineBase.teardown/_detach_patchers detaches every tracked patcher at the end of EVERY beat (V-4), and BeatSession.close() (beat_session.py:301-322) calls teardown() once per beat regardless of segment count, so even consecutive same-engine beats start cold. eng_ltx_av.py:load()/render_clip() (eng_ltx_av.py:1231-1240ff) is the identical shape, same absence of caching. The '~14 GiB' figure traces to eng_ltx_video.py:837-842's own comment: measured per-clip VRAM peak ~14804 MiB, paid again in full every beat. Contrast: eng_ltx25.py:806-930 implements begin_encoder_scope/end_encoder_scope, wired generically from render_driver.py:5216-5292 ('EPISODE-SCOPED ENGINE RESIDENCY 2026-08-20'), whose own comment records the measured baseline of this exact defect class on real hardware ('15 shot renders, 13 reads of the 8.86 GiB ... text encoder ... ~63s each') before that fix. eng_ltx_video/eng_ltx_av never received an equivalent hook, and git log on both files shows no commit that adds one.

**Key files:** `nodes/_otr_video_engines/eng_ltx_video.py:837-842`, `nodes/_otr_video_engines/eng_ltx_video.py:1572-1614`, `nodes/_otr_video_engines/eng_ltx_video.py:1615-1665`, `nodes/_otr_video_engines/eng_ltx_av.py:1202-1240`, `nodes/_otr_video_engines/eng_ltx_8gb.py:1304-1390`, `nodes/_otr_video_engines/eng_ltx25.py:806-930`, `nodes/_otr_video_engines/motion_common.py:1359-1450`, `nodes/_otr_video_engines/beat_session.py:1-40`, `nodes/_otr_video_engines/beat_session.py:301-322`, `nodes/_otr_video_engines/render_driver.py:5216-5292`, `nodes/_otr_video_engines/render_driver.py:5327-5345`, `nodes/_otr_video_engines/wrapper_bridge.py:468-497`, `docs/GO_FORWARD_PLAN.md:136`, `docs/GO_FORWARD_ARCHIVE.md:8775`

**Blast radius:** eng_ltx_video.py and eng_ltx_av.py only (the LTX-2.3 22B GGUF text-to-video and audio-in lanes). render_driver.py's episode-scope hook is already generic/duck-typed via getattr, so adding begin_encoder_scope/end_encoder_scope to these two engines changes nothing for humo, wan_ti2v, ltx25, or ltx_8gb. Primarily a 16 GB-tier (5080) concern since these are the ~14-15 GB peak lanes; eng_ltx_8gb (8 GB tier) is a separate adapter already covered by its own narrower B1b fix and is untouched either way.

## 2. Claims in the existing row description that CURRENT CODE CONTRADICTS

These are why this anchor was rebuilt from the code rather than from the backlog
text. A row description is a dated claim, not a finding.

- docs/GO_FORWARD_ARCHIVE.md:8775 prescribes 'adopt the prepare() + external_results pattern the sibling lanes use' as the fix -- that names eng_ltx_8gb's B1b mechanism (nodes/_otr_video_engines/eng_ltx_8gb.py:1304-1390), which only collapses per-SEGMENT reload into per-BEAT reload inside one multi-segment beat. It does not touch cross-beat reload at all, so it would not close this row for the common single-segment-beat case. The mechanism that actually matches the row's complaint is eng_ltx25.py's begin_encoder_scope/end_encoder_scope (eng_ltx25.py:806-930), which the archived row never cites.
- The 'OOM surface' framing (docs/GO_FORWARD_PLAN.md:136) has no docs/PROD_BUG_LOG.md entry behind it (grepped for eng_ltx_video/eng_ltx_av OOM/reload, none found). Per this repo's own admission rule only a live-verified failure earns PROD_BUG_LOG/Bible status -- current code proves the reload is real and expensive, not that it is the actual cause of an episode OOM as opposed to just wall-clock cost or feeding row 3.3's orphan-occupancy problem.

## 3. The fork -- this is what the round must pressure-test

What to cache (text encoder only, ~8.8 GB, matching the proven eng_ltx25 pattern, vs. also the ~10 GB GGUF unet for a bigger win), where to hold it (VRAM-resident for the whole episode vs. CPU-resident with a faster per-beat transfer), and how that interacts with the existing cross-engine 'inter-beat reclaim' invariant (render_driver.py:5327-5345, CS-3) which assumes a heavy engine's residency ends at its own beat's teardown so two heavy engines never co-reside on a 16 GB card. Episode-wide residency for both encoder and unet risks recreating the exact page-thrash CS-3 exists to prevent once a different engine's beat follows. The hook point to extend already exists generically (render_driver.py's begin_encoder_scope/end_encoder_scope dispatch via getattr) -- the fork is about what/where to cache and its VRAM-budget interaction, not about building new plumbing.

## 4. Questions for the reviewer -- answer these; do not restate section 1

1. **Is the mechanism in section 1 complete and correct?** Read the cited files
   yourself. Is there a step the driver missed, or something that already partially
   mitigates this?
2. **Which side of the fork survives contact with the real code?** Name the files and
   functions each choice would actually touch, and say which you would not write.
3. **What is the smallest change that resolves it?** Smallest that is CORRECT -- not
   smallest that compiles.
4. **Blast radius, measured not asserted.** This project runs on a 16 GB 5080 and an
   8 GB 4060 (CLAUDE.md section 0B). If the change touches shared code, what
   measurement proves the machine you are NOT fixing is unchanged?
5. **Is any part render-inert enough to ship before a four-machine test wave, or must
   all of it wait?** Say plainly.
6. **What would make this row WRONG to do at all?** Argue the other side once.

## 5. Hard constraints -- a proposal violating any of these is rejected on sight

- **No new content gates**, story-rejection gates, word/duration/cast-size limits,
  standalone checkers, chunkers, or recursive retry loops.
- **No prompt rewrites.** Prompts are hand-crafted per model and CHARACTER-BUDGETED
  (`motion_registers` 240 chars enforced at load, BUG-LOCAL-112; `_fit_motion_slot`
  truncates to 60). Adding conditional nuance spends budget that does not exist.
- **Story/prose QUALITY work is DONE** by operator directive. Correctness defects are
  still open; better prose is not.
- **Do not make an OOM silent.** An OOM is the only acceptable killer and must stay
  truthful. The goal is to stop CAUSING avoidable ones, never to hide them.
- **Do not reduce how many episodes reach `otr/obs/`.** A fix that refuses more than
  it saves is a worse fix.
- **One canonical graph.** No second workflow JSON, no new output-path owner, no
  reviving deleted code.
- **Never blanket-kill Python processes** -- that kills the agent's own tooling.
- Style: no curse words, never the name "dummy".
