# SA-104 watch-list addition (Sprint D SPRINT.md v2.1+ watch-list)

**Origin:** triage pass over `docs/retrospectives/2026-05-15-sprint-c-triage-findings.md` §6 Adjudication.
**Captured:** 2026-05-16 on triage branch `triage-sprint-c-retrospective-2026-05-15`.
**For:** Sprint D planning, OR whichever sprint owns `SPRINT.md` when this lands. Paste the bullet below verbatim into the existing "v2.1+ watch-list (deferred decisions, parked)" section of `SPRINT.md`.

The canonical SPRINT.md state at Sprint C close (the `sprint-c-story-brief-v2` branch) already contains a v2.1+ watch-list section with two entries: `artokun/comfyui-mcp` evaluation and LTX 2.3 LipDub IC-LoRA. SA-104 below is a THIRD entry to append to that same list.

---

## Bullet to paste

```markdown
- **B3SUM tier-2 perceptual audio hash supplement (SA-104, deferred from Sprint C triage 2026-05-16).** Chromaprint via `fpcalc` subprocess (or `librosa.feature.chroma_cqt` + cosine similarity as fallback) wired in as a tier-4 disambiguator AFTER SA-102's hardware snapshot tier-2 strict-check. Tier ordering: (1) b3sum byte-identical = PASS; (2) hardware_snapshot.json `--check` ADVISORY = halt for operator review; (3) tier-2 advisory-passes but tier-1 failed = STRICT FAIL real regression; (4) tier-2 strict-fails = run perceptual hash, >=95% similarity = VERSION-DRIFT-TOLERANT PASS with diagnostic dump, below threshold = STRICT FAIL. **Deferred because:** SA-102 already covers env-drift detection for a solo-developer single-fixed-machine setup; a four-tier fallback ladder is dragon-chasing until SA-102 produces ambiguous results in practice. **Revisit trigger:** SA-102 strict-fails twice or more in Sprint A with audio that operator subjectively judges to be perceptually identical, indicating env drift is happening without semantic regression and the b3sum-only gate is over-strict.
```

---

## Why this is a v2.1+ watch-list item, not a Sprint A row

Per §6 Adjudication: SA-102 (hardware snapshot) is enough for the floating-point-determinism failure mode that actually applies to the operator's single-RTX-5080 single-machine setup. Cross-hardware risk is theoretical; same-machine time-axis drift is the real risk, and the hardware snapshot's strict-check on `torch.version` / `gpu.compute_capability` / `cudnn_version` / `backends_flags` / `transformers.version` / `bitsandbytes.version` catches it deterministically without a perceptual-similarity threshold to argue about.

Building Chromaprint + a four-tier ladder before observing whether SA-102 alone is sufficient is the dragon-chasing pattern Jeffrey called out in the `feedback_no_vram_dragons.md` memory. Park, observe, revisit if needed.

---

## Source references

- `docs/retrospectives/2026-05-15-sprint-c-triage-findings.md` §3 (b3sum hardware determinism spec), §6 Adjudication
- `SPRINT.md` on `sprint-c-story-brief-v2`, "v2.1+ watch-list (deferred decisions, parked)" section (existing entries: `artokun/comfyui-mcp` evaluation, LTX 2.3 LipDub IC-LoRA)
