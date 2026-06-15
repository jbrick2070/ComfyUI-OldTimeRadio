# Headless Wan-Lane Soak -- 2026-06-15 (post flux-eviction OOM fix)

Branch v2.0-alpha. M1 no-runtime-fallback gate ON (`--strict-fallback`); a shot
that silently falls off its requested engine FAILS the leg. Fresh OS-entropy
cast/style/story per leg (no OTR_C7 / CAST_SEED / STYLE_SEED) = free audio +
content permutation coverage. Audio spine frozen; byte-identical asserted per leg.

---

## The Wan-lane OOM fix (the headline result)

The 14B `wan_i2v` OOM'd at the ksampler before this work. Root cause + fix
(committed, suite green throughout):

| Commit | What |
|--------|------|
| `3da56ba` | first attempt: measured ComfyUI-only phase-boundary free (revealed the real residue) |
| `0e1dc22` | **fix**: pre-render boundary calls the canonical `free_otr_pipeline_residue()` |
| `3bffd81` | removed the now-redundant ComfyUI-only helper + test |
| `c95ab96` | env-overridable `POLL_TIMEOUT_S` (the now-real Wan episodes run longer) |

**Root cause:** under ComfyUI's dynamic-VRAM "Staged" model the detach-only
reclaim freed 0 (measured live: torch-free 6939MB -> 6939MB). The ~7-8GB resident
residue at the stills->video boundary is the OUT-OF-BAND transformers caches (the
writer LLM Mistral-Nemo + Bark) loaded through OTR's own loaders -- invisible to
ComfyUI's `model_management`, so `free_memory`/`empty_cache` cannot touch them.
That residue + the 14B UNET (13.6GB) busts 16GB -> OOM.

**Fix:** the pre-render boundary now calls `free_otr_pipeline_residue()` (the
existing Lever-1 freer): unload_llm + _unload_bark FIRST, then ComfyUI FLUX
(detach + unload_all_models), then the allocator flush. **PROVEN LIVE
2026-06-15:** allocated **7991 -> 65 MB**, **free 14739 MB** at the boundary;
`wan_i2v` then rendered the 14B with **no OOM** (ksampler 20/20 in 28s, 1.47s/it).

**Known follow-on (NOT this fix):** a full Wan episode's wall-clock at the 16GB
edge exceeds the old 90-min poll ceiling -- the per-sampler speed is fine, but
heavy engines accumulate across beats (wan stays resident as the next beat
loads). That is the CS-3 inter-beat reclaim, tracked separately. The overnight
soak runs with OTR_SOAK_POLL_TIMEOUT_S=10800 so the real renders complete.

---

## Overnight soak run (ACTIVE)

`python scripts\otr_coverage_sweep.py --strict-fallback --exclude latentsync --exclude triposg`
(server: full in-process enable-set -- HuMo + wan_i2v(14B) + wan_ti2v(5B GGUF) +
ltx_video + ltx_orbit + still_parallax + mesh_stage; OTR_SOAK_POLL_TIMEOUT_S=10800;
no C7). Started 05:35.

Excluded: latentsync + triposg (sidecar venvs), hunyuan3d/trellis (cu128 toolchain
-- auto-skip).

| # | Leg | Verdict | Elapsed | Notes |
|---|-----|---------|---------|-------|
| 1 | sweep_announcer_visual_ltx_video | (incomplete) | 30+ min | thrashing -- see below |

### OUTCOME: soak HALTED -- surfaced CS-3 inter-beat thrash (the real blocker)

Two things happened:

1. **Launch bug (mine):** the background sweep was started with a 10-min
   `run_in_background` timeout, which killed the sweep WRAPPER (the poller) at
   t=600s. The ComfyUI server kept grinding the first prompt with no client. Fix
   for next time: long/no timeout on the background launch.

2. **CS-3 inter-beat thrash (the real finding):** the FIRST leg
   (announcer=ltx_video, music=ltx_video, other_beats=humo_1.7B) pinned ~16 GB
   and a video ksampler crawled at **29 s/it at 1% GPU util** -- catastrophic
   weight PAGING, not compute. Contrast the clean wan verify earlier the same
   night: the FIRST video beat after the residue free ran **1.47 s/it** (fast).
   So the residue/OOM fix is sound; the slowness is LATER beats: heavy engines
   ACCUMULATE across beats with NO inter-beat reclaim (ltx 12.5 GB + humo 7 GB
   co-resident -> 19.5 GB on a 16 GB card -> page-thrash). The pre-render residue
   free fires ONCE (before beat 1); it does not drain the prior heavy engine
   before a DIFFERENT engine loads on the next beat. **That is CS-3** (section 5,
   reframed): "the inter-beat reclaim drains the prior heavy engine before the
   next beat loads." My fix EXPOSED it (engines now actually render + accumulate
   instead of OOMing/falling back in seconds).

**Net:** the assigned OOM fix is DONE + proven (no OOM, first beat fast). A
PRACTICAL multi-heavy-engine soak is BLOCKED on CS-3 inter-beat reclaim --
without it, every leg that crosses two heavy engines thrashes. Per the
"don't burn 10 hours on the same root cause" stop condition, the soak was
halted rather than left grinding.

**Recommended next fix (CS-3):** in `render_driver.run_episode`'s beat loop,
reclaim the prior heavy engine before a beat that loads a DIFFERENT heavy engine
(reuse when the next beat is the same engine; honor the retained Wan unet
patcher). This makes the multi-beat Wan/HuMo episode fit + run fast, and unblocks
the soak. Needs a GPU verify cycle.
