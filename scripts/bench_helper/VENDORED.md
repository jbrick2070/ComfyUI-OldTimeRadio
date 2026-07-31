# Vendored bench-measurement nodes

`otr_bakeoff_helper/` is a **tracked copy** of the diagnostic ComfyUI custom-node
package that provides the bench's per-stage torch probes.

**Why it lives here.** The working copy at
`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\otr_bakeoff_helper` is in **no git
repository** (`git rev-parse` -> "fatal: not a git repository"; `git ls-files`
from this repo -> "outside repository"). A pushed harness cannot depend on
unshipped local code, so this copy is the source of truth and the sibling is an
install target (bench spec, gate G3).

**Install contract.** `run_video_arm_bakeoff.install_bench_helper()` copies this
file into `custom_nodes/otr_bakeoff_helper/` when the digest differs, then
re-hashes the installed file and **raises** on mismatch. A bench whose probes are
a different build than the one in the commit is measuring an unknown. Build
order: vendor -> install -> server restart -> `/object_info` contract check ->
submission.

**What it registers.**

| node | edge | purpose |
|---|---|---|
| `OTR_BakeoffVramReset` | LATENT | `torch.cuda.reset_peak_memory_stats()` immediately before the sampler -- opens the order-safe segment |
| `OTR_BakeoffVramProbe` | IMAGE | logs `max_memory_allocated` / `max_memory_reserved` immediately after decode -- closes it |
| `OTR_BakeoffReclaim` | LATENT | encoder-only eviction (from the 2026-06-27 HuMo bakeoff); **not wired into any bench graph here** |

Both probes are always-dirty (`IS_CHANGED` returns a fresh uuid) so the executor
can never cache-skip them, and each prints a unique marker. `parse_stage_probe`
fails the cell closed unless **exactly one** reset and **exactly one** probe
appear in that cell's log slice.

**Scope.** DIAGNOSTIC ONLY. These nodes are not part of the production graph, are
not registered by the OTR pack, and measure only the sample+decode segment. The
text-encode / image-encode boundaries need forced node ordering, which changes
the memory schedule under test -- that is operator decision O7, deferred to a
separate diagnostic campaign whose data may never drive a fit verdict.

Do not edit the installed sibling. Edit this copy and re-install.
