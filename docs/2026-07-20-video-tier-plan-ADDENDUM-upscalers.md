# Addendum — Upscalers

**To:** `2026-07-20-8gb-16gb-video-tier-plan.md`
**Date:** 2026-07-20
**Purpose:** Refine the "upscaling = deferred" line (§5). No change to the 8GB build or the 16GB append-only work.

---

## 1. The internal recipe upscaler STAYS ACTIVE (it was never the deferred item)

The **x2 spatial upscaler inside the 16GB two-stage HQ recipe** — `LTXVLatentUpsampler` + `LatentUpscaleModelLoader`, using assets already on disk (`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`, 0.93GB) — runs on **already-installed ComfyUI-LTXVideo nodes**. **No additional nodes.** It runs today and stays active, exactly as-is.

The **8GB engines stay single-pass / no upscale** to fit the 8GB budget.

## 2. What is actually deferred: a model-agnostic multi-bank UPSCALER REGISTRY

The separate upscaling conversation is **not** ad-hoc nodes. The intended shape is a pluggable **upscaler "bank"** — a **fourth parallel namespace** alongside the existing video / image / audio engine registries and the LLM banks, on the **same `engine_registry_base` pattern**:

- pluggable upscaler adapters, each self-registering,
- per-adapter `CAPABILITIES` rows (device backends, model requirements),
- "registry IS the menu" — every registered upscaler is selectable, **no gating**,
- the current internal x2 upscaler becomes the **default entry**; the bank makes upscalers swappable and extensible.

Scoped and built in that separate conversation — **out of scope for the current 8GB build.**

---

## Net effect on the reviewed plan

- **8GB build:** unchanged (single-pass, no upscale).
- **16GB:** unchanged — append `16gb` labels only; internal x2 upscaler stays active.
- **§5 "upscaling deferred":** refined — *internal recipe upscaler active now; additional upscalers arrive later as a registry/bank*, not ad-hoc nodes.
