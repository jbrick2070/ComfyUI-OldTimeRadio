# Final problem statement -- OTR v2.0-alpha 2026-05-08 sprint

## Context for the reviewer

Repo: https://github.com/jbrick2070/ComfyUI-OldTimeRadio (branch
`v2.0-alpha`, head `513a461`).

Across roughly 12 hours of autonomous work (predecessor + Jeffrey
overnight + Jeffrey morning), this sprint shipped:

- Cast Contract Phase 0+ §1+§2+§3+§4+§5 skeletons + helpers
- Voice Backend Abstraction (skeleton, unregistered)
- Old-timey LLM period prompts module
- Soak watcher PowerShell script
- BUG-LOCAL-121 (padded keys KeyError)
- BUG-LOCAL-122 (lock_to_episode read-and-compare-version)
- BUG-LOCAL-123 (plateau loop crash on DISCARD/LEAK/NEW)
- BUG-LOCAL-124 (lock_to_episode atomic write)
- BUG-LOCAL-125 (voice backends lazy registry init)
- BUG-LOCAL-126 (HuMo soak fatal abort -- alarm plumbing + cap)
- BUG-LOCAL-127 (save_ledger_safe atomic write)
- Ledger schema l3-2026-05-08 (BUG-126 telemetry + Cast Contract
  pre-wiring)

Three round-robin code reviews (ChatGPT gpt-5.5 + Gemini
gemini-3.1-pro-preview-customtools) were run during the sprint and
each caught real bugs that were fixed in the same loop.

Test floor at HEAD: 27/27 ledger l3 + 9/9 BUG-126 + 33/33 LTX
regression + 113/113 Phase 0+ stack. AST clean.

**This document lists the fixes I'm LEAST confident about** -- not
the green-and-shipped-clean ones. Every numbered item below is
something an outside reviewer should weigh in on before the next
unattended overnight soak.

Stack: Windows, Python 3.12, RTX 5080 Laptop 16 GB Blackwell,
torch 2.10/CUDA 13, SDPA + SageAttention. No cloud. C7 byte-identity
of audio is sacred.

---

## What I want from you

For each numbered item below, give me:

- **Confidence level** that the fix is correct as shipped (high /
  medium / low)
- **Single concrete failure mode** you'd watch for in the next FULL
  acceptance soak
- **Whether you would ship a v2 fix now** or **defer until live data
  arrives**
- One line: **what missing data would let you decide with high
  confidence?**

NOT what I want: rewriting whole subsystems, suggestions to chase
weight-streamer / FA2/3 / quantization paths (the OTR project
explicitly rejects that class of work), or deep speculation on
ComfyUI internals you can't verify.

---

## Item 1 -- BUG-LOCAL-126 cleanup chain efficacy

The fix shipped a cleanup chain that runs after every CAUGHT HuMo OOM:

```python
def _hard_reset_cuda_context() -> None:
    # NOTE: name says "hard reset", actual behavior is soft cleanup.
    # Running OUTSIDE the except block (Element 3 round-robin fix)
    # so the failed frame's traceback doesn't pin tensors.
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache(force=True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # no-op on Windows but kept for portability
```

**Both prior reviewers (ChatGPT + Gemini) agreed this is NOT a true
CUDA context reset.** It's a model-patcher unload + allocator cache
drain. If the underlying allocator drift is caused by retained
references that live beyond the model patcher (custom node globals,
weakly-rooted torch modules, extension-level state), the chain will
log "OK" and reclaim nothing.

The overnight soak observed PyTorch's allocator pool drifting from
~14 GiB in-use to **24.50 GiB allocated on a 15.92 GiB device** after
2 caught OOMs. Stack trace indicates the third sample tripped a
C-level abort.

**Verification status:** UNIT tests pass (the chain runs without
raising even when torch is missing). LIVE behavior on the
allocator-drift case is UNKNOWN.

**Question:** Is this fix worth shipping at all, or is the per-process
cap (Item 2) the only real defense?

---

## Item 2 -- HuMo soak cap of 6 lines per process

```python
"humo_max_lines_per_process": ("INT", {
    "default": 0,  # disabled by default
    # widget tooltip: 6-8 is a safe per-process budget
}),
```

The number 6 is empirical from a single overnight run that survived 9
HuMo lines before fatal-aborting. It's `n=1`. Pessimistic to be
conservative, but the actual ceiling on this stack might be 8-10
on a typical 56-line episode, OR could degrade earlier with a
heavier prompt or longer character lines.

**Verification status:** Loop wiring + structured exit are unit
tested. The number 6 is a guess from a single failure point.

**Question:** What soak-design discipline would let me empirically
narrow this number in 2-3 short runs without burning a full day of
GPU time on each? Is there a less-arbitrary heuristic
(e.g. "track allocator high-water-mark per line and stop when
it crosses 13.5 GiB")?

---

## Item 3 -- Plateau-bounded repair loop `decided_residuals` set

After the round-robin Element 3 catch, the loop now tracks orphan tags
the classifier explicitly placed in DISCARD / NARRATIVE_LEAK /
GENUINELY_NEW so they don't trigger a false plateau:

```python
decided_residuals: set[str] = set()
for iteration in range(1, max_iterations + 1):
    live_residual = _residual_orphans(script, contract) - decided_residuals
    if not live_residual:
        return RepairOutcome(...)  # success
    if prev_residual is not None and live_residual == prev_residual:
        raise CastContractUnreparable(...)
    prev_residual = set(live_residual)

    results = [classifier(tag, contract, script) for tag in live_residual]
    _, applied = apply_classifications(contract, results)
    for r in results:
        if r.bucket in (OrphanClass.DISCARD, OrphanClass.NARRATIVE_LEAK, OrphanClass.GENUINELY_NEW):
            decided_residuals.add(r.orphan_tag)
```

The set never shrinks. If the classifier classifies a tag as
GENUINELY_NEW on iteration 1, it stays decided for the rest of the
loop. That's correct for the stub classifier (deterministic).
With a real LLM classifier:
- Could the LLM legitimately revise a verdict on iteration 2 with
  more context?
- If yes, should the set have a TTL or be re-evaluable?

**Verification status:** 22/22 unit tests pass against the stub
classifier. Real-LLM behavior is not yet wired (orchestrator hooks
deferred).

**Question:** When the real LLM classifier ships, should
`decided_residuals` be permanent-per-loop or reconsidered on each
iteration?

---

## Item 4 -- `detect_aliases` first-match-wins prefix heuristic

```python
ALIAS_PREFIX_LEN = 4

def detect_aliases(script, contract) -> dict[str, str]:
    aliases_found = {}
    for tag in _extract_dialogue_tags(script):
        if contract.lookup(tag) is not None:
            continue
        for character in contract.characters:
            cn = character.canonical_name
            if len(tag) >= 4 and len(cn) >= 4 and tag[:4] == cn[:4]:
                aliases_found[tag] = character.character_id
                break  # first match wins
```

If two canonicals share a 4-char prefix (MARLA / MARLON), an
unfamiliar tag like MARLENE arbitrarily picks the first canonical.
**§4 adversarial classification is supposed to disambiguate, but
that's the LLM-call path that hasn't yet shipped wired.** Without §4,
this heuristic can silently route an entire episode's MARLENE lines
to MARLA's voice preset without any indication in the ledger or
log.

The `decided_residuals` set in §5 means once detect_aliases makes a
wrong call, §4 never re-evaluates that orphan because it's in the
"already aliased" bucket.

**Verification status:** Unit tests cover the deterministic
first-match behavior. Real-script behavior on a name-collision episode
is untested.

**Question:** Should `detect_aliases` (a) refuse to apply on prefix
collision and escalate to §4, (b) require a longer prefix when
collisions exist, or (c) ship as-is and accept silent mis-routing
until §4 is wired?

---

## Item 5 -- sha-8 cast contract version collision risk

```python
def stamp_version(self) -> str:
    blob = json.dumps(normalized, separators=(",", ":"), sort_keys=True)
    sha = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]
    self.version = f"sha:{sha}"
    return self.version
```

8 hex chars = 32 bits = ~4 billion possibilities. `production_ledger`
will reject merges in O(1) by string comparison of these. After
~10K episodes the birthday-paradox collision probability is
non-negligible.

The audio_gates[] sha256 uses 32 hex chars (128 bits) which is safe.

**Question:** Should the cast contract version be 12 chars (48 bits,
~280 trillion) or 16 chars (64 bits, secure)? Is the 32-bit form
load-bearing somewhere I haven't seen?

---

## Item 6 -- OOM detection wrap case

```python
def _is_oom_exception(exc: BaseException) -> bool:
    try:
        import torch
        oom_cls = getattr(torch, "OutOfMemoryError", None) or ...
    except Exception:
        oom_cls = None
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "out of memory" in msg or "outofmemoryerror" in msg:
            return True
    return False
```

ChatGPT noted in the BUG-126 round-robin: ComfyUI / PyTorch could
wrap an OOM in a higher-level exception whose top-level
`str(exc)` lacks "out of memory". My check would then miss and the
cleanup chain wouldn't fire.

**Verification status:** Both forms confirmed present in the overnight
log. Wrapped case: theoretical, no observed instance.

**Question:** Worth defensively walking `exc.__cause__` /
`exc.__context__` for OOM signatures, or YAGNI?

---

## Item 7 -- Voice backend registry race condition

```python
_DEFAULTS_REGISTERED = False

def _ensure_defaults_registered() -> None:
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return
    _DEFAULTS_REGISTERED = True
    try:
        _register_default_drivers()
    except Exception:
        pass
```

ComfyUI runs node `execute()` calls in worker threads. Two threads
hitting `get_factory("bark")` on a cold process could both see
`_DEFAULTS_REGISTERED == False`, both flip it, both run the import
chain. The bark/kokoro module-level `register(...)` calls are
idempotent on the registry dict (last write wins, same value), so
this is benign in practice -- but the lazy-init pattern technically
isn't thread-safe.

**Question:** Wrap the lazy init in a `threading.Lock`, or document
"benign race, registers idempotently" and move on?

---

## Item 8 -- `CastContractMismatch` inheritance

```python
class CastContractMismatch(RuntimeError):
    """Raised by lock_to_episode when an existing locked contract has
    a different cast_contract_version than the contract being locked."""
```

It inherits from `RuntimeError`. Some downstream callers might catch
`RuntimeError` broadly (`except Exception` is even broader), which
swallows the mismatch signal. The structured intent is "this is
NOT a generic runtime error -- it's a specific drift signal that
the orchestrator should react to differently than a transient
fault."

**Question:** Is `RuntimeError` the right base, or should it inherit
from something more specific (or even `Exception` directly with no
`RuntimeError` parent) so a `except RuntimeError` clause doesn't
accidentally catch it?

Same question applies to `HumoSoakCapReached`.

---

## Item 9 -- `HumoSoakCapReached` UX in ComfyUI

When the soak cap fires:

```python
raise HumoSoakCapReached(
    lines_completed=rendered,
    cap=humo_max_lines_per_process,
)
```

ComfyUI's prompt executor catches every exception thrown by a node
and surfaces it to the UI as a red "Workflow Failed" with the
exception message. From a USER perspective, the cap firing looks
identical to a real fault.

The intent is "this is a clean stop, please rerun with
resume_from_ledger=True". But the user sees "FAILED: HuMo soak cap
reached: rendered 6 of cap 6; queue a follow-up run with
resume_from_ledger=True to continue."

**Verification status:** Behavior in ComfyUI's UI is not yet observed.

**Question:** Is there a ComfyUI idiom for "soft stop, user should
take action" that doesn't look like a fault? Custom result type with
a status field, custom node-level early-return, sentinel return
value?

---

## Item 10 -- Atomic save tempfile cleanup race

```python
tmp_fd, tmp_name = tempfile.mkstemp(
    prefix=".ledger.save.",
    suffix=".tmp.json",
    dir=tmp_dir,
)
try:
    with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_name, target)
except Exception:
    try:
        os.unlink(tmp_name)
    except OSError:
        pass
    raise
```

If the process is `kill -9`'d between `tempfile.mkstemp` and the
`os.replace`, the temp file is left on disk. Cleanup never runs.
Over many crashes the directory accumulates `.ledger.save.*.tmp.json`
debris.

Not catastrophic (each is small, episode dirs are ephemeral), but
worth knowing.

**Question:** Add a startup-time `glob('.ledger.save.*.tmp.json')`
sweep on the episode dir, or accept the debris?

---

## Closing question

Across these 10 items, which one would you triage as highest priority
to nail down BEFORE the next 6-line validation soak runs? Which can
wait for live data?

(I am specifically NOT asking you to rewrite anything. The shipped
code is at HEAD `513a461` and tests pass. This is post-fix
peace-of-mind.)
