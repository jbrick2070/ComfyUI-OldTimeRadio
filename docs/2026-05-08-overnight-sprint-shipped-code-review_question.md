# Code review: OTR overnight sprint shipped to v2.0-alpha

## Background

Last night a previous round-robin pass reviewed the §1+§2+§3 cast contract
skeletons + the §1 helpers (8 elements; two real bugs caught and fixed in the
same loop). Overnight I shipped four more chunks of new code and want a
similar pre-soak peer review before the next session wires the orchestrator
hooks. The orchestrator integration is the only thing that can break the
audio pipeline -- everything below is helper / abstraction / scaffolding
code that the orchestrator hooks will call. So the question is: are these
helpers right, or do they have traps that the orchestrator session will
trip on?

Stack: Windows, Python 3.12, no torch coupling in any of these modules
(stdlib only). 106-test cast-contract+voice+prompts suite green; LTX
regression 33/33 unchanged; AST clean.

## What I want from you

For each of the SEVEN numbered code elements below, give me:
- **Per-element fix-needed probability % (0-100)**.
- **One-line reasoning**.
- **Verdict badge**: GREEN (<15%), AMBER (15-30%), RED (>30%).
- One concrete failure mode you'd watch for in the next FULL acceptance
  soak.

Then a short closing section:
- **Where would you push back?** (one or two strongest disagreements)
- **What's the load-bearing weak spot?** (single element most likely to
  fail in production)

NOT what I want: prevention plans, "things to consider", or suggestions
to add §6 / Voice Backend production code this round. We are intentionally
holding the migration steps for the next session.

## Constraints + non-goals

- All elements are stdlib-only on purpose. No torch / no LLM HTTP / no
  ComfyUI imports at module-load time.
- The bark/kokoro drivers are stubs by design -- production code stays in
  `batch_bark_generator.py` / `kokoro_announcer.py` until the
  orchestrator session migrates them.
- Period-prompt module is pluggable but NOT yet wired into the LLM call
  site -- that's the orchestrator session's job.

## The seven elements under review

### Element 1 -- `lock_to_episode` read-and-compare upgrade (BUG-LOCAL-122 fix)

```python
class CastContractMismatch(RuntimeError):
    """Raised by lock_to_episode when an existing locked contract has
    a different cast_contract_version than the contract being locked."""

def lock_to_episode(contract: CastContract, episode_dir: Path) -> Path:
    episode_dir = Path(episode_dir)
    if not episode_dir.is_dir():
        raise FileNotFoundError(...)
    if not contract.version:
        contract.stamp_version()
    locked_path = episode_dir / LOCKED_FILENAME
    if locked_path.exists():
        try:
            existing = json.loads(locked_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"existing locked contract at {locked_path} is unreadable...") from exc
        existing_version = (existing or {}).get("version", "")
        if existing_version == contract.version:
            return locked_path  # idempotent on rerun
        raise CastContractMismatch(...)
    locked_path.write_text(json.dumps(contract.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return locked_path
```

Both reviewers from last night's pass agreed on this design. Want a sanity
check that the implementation matches the design intent, especially around
TOCTOU (file may appear / disappear between exists() and read_text()).

### Element 2 -- `OrphanClass` 5-bucket Enum + `apply_classifications`

```python
class OrphanClass(str, Enum):
    TYPO_OF_EXISTING = "TYPO_OF_EXISTING"
    ALIAS_OF_EXISTING = "ALIAS_OF_EXISTING"
    GENUINELY_NEW = "GENUINELY_NEW"
    NARRATIVE_LEAK = "NARRATIVE_LEAK"
    DISCARD = "DISCARD"

@dataclass(frozen=True)
class ClassificationResult:
    orphan_tag: str
    bucket: OrphanClass
    target_character_id: str = ""
    confidence: float = 1.0
    rationale: str = ""

def apply_classifications(contract, results):
    applied = []
    for r in results:
        if r.bucket not in (OrphanClass.TYPO_OF_EXISTING, OrphanClass.ALIAS_OF_EXISTING):
            continue
        if not r.target_character_id:
            continue
        target = next((c for c in contract.characters if c.character_id == r.target_character_id), None)
        if target is None:
            continue
        existing_upper = {a.upper() for a in target.aliases}
        if r.orphan_tag.upper() in existing_upper:
            continue
        if r.orphan_tag.upper() == target.canonical_name.upper():
            continue
        target.aliases.append(r.orphan_tag)
        applied.append(r)
    if applied:
        contract.stamp_version()
    return contract, applied
```

Mutates the contract in place for TYPO/ALIAS buckets; other buckets are
caller's responsibility. Re-stamps version IFF any alias was actually
added.

### Element 3 -- Plateau-bounded `repair_orphans` loop (§5)

```python
def repair_orphans(script, contract, classifier=classify_orphans_stub, max_iterations=3):
    if not isinstance(contract, CastContract):
        raise TypeError("contract must be a CastContract")

    # Cheap path first: heuristic alias detect, applied as ALIAS_OF_EXISTING
    cheap_aliases = detect_aliases(script, contract)
    cheap_results = [
        ClassificationResult(orphan_tag=tag, bucket=OrphanClass.ALIAS_OF_EXISTING,
                             target_character_id=cid, confidence=0.5,
                             rationale="4-char prefix match")
        for tag, cid in cheap_aliases.items()
    ]
    _, cheap_applied = apply_classifications(contract, cheap_results)
    aliases_added_total = len(cheap_applied)
    classifications_seen_total = len(cheap_results)

    prev_residual = None
    for iteration in range(1, max_iterations + 1):
        residual_orphans = _residual_orphans(script, contract)
        if not residual_orphans:
            return RepairOutcome(iterations=iteration - 1, ...)
        if prev_residual is not None and residual_orphans == prev_residual:
            raise CastContractUnreparable(orphans=list(residual_orphans), iterations=iteration - 1)
        prev_residual = set(residual_orphans)
        results = [classifier(tag, contract, script) for tag in residual_orphans]
        classifications_seen_total += len(results)
        _, applied = apply_classifications(contract, results)
        aliases_added_total += len(applied)

    final = _residual_orphans(script, contract)
    if not final:
        return RepairOutcome(iterations=max_iterations, ...)
    raise CastContractUnreparable(orphans=list(final), iterations=max_iterations)
```

Plateau detection uses set equality on orphan tags (not classification
results) -- the same residuals back means the classifier learned nothing
this round.

### Element 4 -- `_voice_backends` registry + lazy driver self-registration

```python
KNOWN_ENGINES: set[str] = {"bark", "kokoro", "cosyvoice", "xtts", "piper"}
_REGISTRY: dict[str, Callable[[], VoiceBackend]] = {}

def register(engine, factory):
    if not isinstance(engine, str) or not engine.strip():
        raise ValueError("engine must be a non-empty string")
    _REGISTRY[engine.strip().lower()] = factory

def get_factory(engine):
    e = (engine or "").strip().lower()
    if e not in _REGISTRY:
        registered = sorted(_REGISTRY.keys())
        raise KeyError(f"voice backend {engine!r} not registered (currently registered: {registered})")
    return _REGISTRY[e]

def _register_default_drivers():
    from nodes._voice_backends import bark as _bark
    from nodes._voice_backends import kokoro as _kokoro
```

Drivers self-register at module import (each calls `register(...)` at
module scope). `_register_default_drivers()` exists so callers that just
want `available_engines()` to confirm registry shape don't pay heavy
import cost.

### Element 5 -- `VoiceBackend` runtime-checkable Protocol

```python
@runtime_checkable
class VoiceBackend(Protocol):
    engine_name: str

    def load(self, preset: str) -> None: ...
    def generate(self, text: str, **kwargs: Any) -> bytes: ...
    def unload(self) -> None: ...
```

Structural typing -- implementers don't need to subclass, can pass
`isinstance(obj, VoiceBackend)` for shape conformance.

### Element 6 -- `OTR_VoiceRender.render` dispatch with try/finally cleanup

```python
def render(self, text, voice_model, voice_preset, temperature=0.7):
    spec_string = f"{voice_model}:{voice_preset}"
    spec = parse_voice_spec(spec_string)  # raises ValueError on malformed
    try:
        factory = get_factory(spec.engine)
    except KeyError:
        registered = available_engines()
        raise RuntimeError(f"voice_model={voice_model!r} not registered (available: {registered}). ...")
    backend = factory()
    backend.load(spec.preset)
    try:
        audio = backend.generate(text, temperature=temperature)
    finally:
        backend.unload()
    return (audio,)
```

Pattern: parse spec -> resolve factory -> instantiate backend -> load ->
try/finally generate+unload. Fresh backend per call (no caching).

### Element 7 -- `render_prompt` few-shot assembly + period system prompt

```python
def render_prompt(user_instruction, include_few_shot=True, max_exemplars=2,
                  system_prompt=OTR_PERIOD_SYSTEM_PROMPT):
    if include_few_shot:
        few_shot = render_few_shot_block(max_exemplars=max_exemplars)
        user_prompt = f"{few_shot}\n\n--- Now write the requested episode ---\n\n{user_instruction.strip()}"
    else:
        user_prompt = user_instruction.strip()
    return system_prompt, user_prompt
```

Three exemplars in PERIOD_EXEMPLARS (Lighthouse Keeper / The Wireless /
Last Train Out). System prompt forbids "okay" / "guys" / "cool" / "hey"
and pins the 1940s broadcast convention (NARRATOR open + close, [SOUND:
...] stage directions, CHARACTER:dialogue tags).

## Non-goals

We are NOT looking for: real LLM call wired into repair_orphans (stub is
on purpose); Bark/Kokoro driver migration (waits for next session);
OTR_VoiceRender registration in __init__.py (waits for next session);
period-prompt integration into story_orchestrator.py LLM call site (waits
for next session). All four are explicitly deferred until the
orchestrator hooks land.

Repo: https://github.com/jbrick2070/ComfyUI-OldTimeRadio (branch
`v2.0-alpha`, head `8009dd0`). Last night's review at
`docs/2026-05-08-cast-contract-shipped-code-review__*.md` -- this round
is the follow-up on what shipped after that review.
