# Question -- 2026-05-07

# Code review: OTR Cast Contract Phase 0+ §1+§2 (shipped to v2.0-alpha)

## Background

OTR (ComfyUI-OldTimeRadio) had a recurring character drift bug: the LLM-written
script would tag dialogue with names like `MONTGOMERY` while the Director plan
had assigned voice presets only to `MONTY`. BatchBark would silently pool the
unknown name to the wrong voice slot. We're closing this gap with a
"Cast Contract" — a versioned, episode-locked roster of characters, each with a
canonical name, alias list, and `engine:preset` voice spec. Five pieces in the
design (per `ROADMAP.md` "Phase 0+ candidates"): §1 versioning, §2 episode lock,
§3 character canon, §4 adversarial classification of orphan tags, §5 plateau-bounded
repair loop. This consult is about the §1 + §2 + §3-skeleton code that just
shipped to `v2.0-alpha` — three commits, three modules + tests + helpers, NOT
yet wired into `story_orchestrator.py`.

Stack: Windows, Python 3.12, no torch/VRAM coupling in any of these modules
(stdlib only). Test suite is 48/48 green; LTX regression 33/33 unchanged.

## What I want from you

For each of the SIX numbered code elements below, give me:
- **Per-element fix-needed probability % (0–100)** — your best estimate of how
  likely this element will need a follow-up code change once it's wired into
  `story_orchestrator.py` in the next session.
- **One-line reasoning** for that estimate.
- **Verdict badge**: GREEN (<15%), AMBER (15–30%), RED (>30%).
- One specific concrete failure mode you'd watch for in the next FULL
  acceptance soak.

Then a short closing section:
- **Where would you push back?** (one or two strongest disagreements with the
  design choices below)
- **What's the load-bearing weak spot?** (the single element most likely to
  fail in production)

NOT what I want: a prevention plan, a list of "things to consider", or
suggestions to add §4 / §5 work this session. We are intentionally only
landing §1 + §2 + §3-skeleton this round.

## The six elements under review

### Element 1 — `CastContract.stamp_version` (sha-8 content-addressed versioning)

```python
def stamp_version(self) -> str:
    normalized = sorted(
        (
            {
                "character_id": c.character_id,
                "canonical_name": c.canonical_name,
                "aliases": sorted(c.aliases),
                "voice_spec": c.voice_spec,
            }
            for c in self.characters
        ),
        key=lambda d: d["character_id"],
    )
    blob = json.dumps(normalized, separators=(",", ":"), sort_keys=True)
    sha = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]
    self.version = f"sha:{sha}"
    return self.version
```

Goal: every dialogue line in `production_ledger.py` will carry
`cast_contract_version: "sha:HEX..."`. Production ledger merge code will reject
mismatches as a hard fail.

### Element 2 — `lock_to_episode` (immutable per-episode lock)

```python
LOCKED_FILENAME = "cast_contract.locked.json"

def lock_to_episode(contract: CastContract, episode_dir: Path) -> Path:
    episode_dir = Path(episode_dir)
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"episode dir does not exist: {episode_dir}")
    locked_path = episode_dir / LOCKED_FILENAME
    if locked_path.exists():
        raise RuntimeError(
            f"cast contract already locked at {locked_path}; "
            "refusing to overwrite (immutable per §2)"
        )
    if not contract.version:
        contract.stamp_version()
    locked_path.write_text(
        json.dumps(contract.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return locked_path
```

Goal: episode workspace gets a single canonical `cast_contract.locked.json` at
the moment Bark health check passes. Any later call to lock the same episode
hard-fails. Downstream code reads from this file, not from the in-memory
Director plan.

### Element 3 — `parse_voice_spec` with forward-compat unknown-engine pass-through

```python
KNOWN_ENGINES: set[str] = {"bark", "kokoro", "cosyvoice", "xtts", "piper"}

def parse_voice_spec(s: str) -> VoiceSpec:
    if not s or ":" not in s:
        raise ValueError(...)
    engine, _, preset = s.partition(":")
    engine = engine.strip().lower()
    preset = preset.strip()
    if not engine: raise ValueError(...)
    if not preset: raise ValueError(...)
    # Forward-compat: unknown engines pass through unchanged.
    return VoiceSpec(engine=engine, preset=preset)
```

Goal: cast contract entries can carry `bark:v2/en_speaker_5` today and
`cosyvoice:robotic_calm` later without this module needing changes. Backend
dispatch lives in the future Voice Backend Abstraction work.

### Element 4 — `build_contract_from_director_plan` (helper)

```python
def build_contract_from_director_plan(director_plan: dict) -> CastContract:
    plan = director_plan or {}
    assignments = plan.get("voice_assignments") or {}
    if not isinstance(assignments, dict):
        raise TypeError(...)
    sorted_names = sorted(
        (str(name).strip() for name in assignments.keys() if str(name).strip()),
        key=str.upper,
    )
    characters = []
    for idx, name in enumerate(sorted_names, start=1):
        cid = f"c{idx:02d}"
        voice_spec = _coerce_voice_spec(assignments[name])  # bark: prefix default
        characters.append(CharacterEntry(
            character_id=cid,
            canonical_name=name.upper(),
            aliases=[],
            voice_spec=voice_spec,
        ))
    contract = CastContract(characters=characters)
    contract.stamp_version()
    return contract
```

Sort by canonical name (alphabetical) then assign `c01`, `c02`, ... in that
order. Default engine prefix is `bark:` if the value is a bare preset like
`v2/en_speaker_3`. Accepts both string and dict voice-assignment values.

### Element 5 — `detect_aliases` (pure heuristic, no LLM)

```python
ALIAS_PREFIX_LEN = 4  # min shared prefix to consider name an alias of canonical

def detect_aliases(script: str, contract: CastContract) -> dict[str, str]:
    if not isinstance(contract, CastContract):
        raise TypeError("contract must be a CastContract")
    aliases_found: dict[str, str] = {}
    if not script:
        return aliases_found
    for tag in _extract_dialogue_tags(script):
        if contract.lookup(tag) is not None:
            continue  # already canonical or registered alias
        tag_u = tag.upper()
        for character in contract.characters:
            cn = character.canonical_name.upper()
            if not cn:
                continue
            n = ALIAS_PREFIX_LEN
            if len(tag_u) >= n and len(cn) >= n and (
                tag_u[:n] == cn[:n]
            ):
                aliases_found[tag] = character.character_id
                break
    return aliases_found
```

Heuristic: 4-character shared prefix in either direction (truncation
MONTGOMERY -> MONTY *and* expansion MONT -> MONTY). First-match-wins on prefix
collisions; §4 adversarial classification is the canonical disambiguator.
First-seen iteration order preserved so merge logs are deterministic. NO
mutation of contract; the caller decides whether to apply each alias and
re-stamp the version, or escalate to §4 for ambiguity.

### Element 6 — `_extract_dialogue_tags` (regex-free dialogue tag extractor)

```python
def _extract_dialogue_tags(script: str) -> list[str]:
    if not script:
        return []
    seen: dict[str, None] = {}
    for raw_line in script.splitlines():
        line = raw_line.lstrip()
        if not line:
            continue
        colon = line.find(":")
        if colon <= 0 or colon > 60:
            continue
        candidate = line[:colon].strip()
        if not candidate:
            continue
        letters = [c for c in candidate if c.isalpha()]
        if not letters:
            continue
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio < 0.8:
            continue
        first_token = candidate.split()[0].upper() if candidate.split() else ""
        if first_token in {"SCENE", "ACT", "FADE", "INT", "EXT", "NARRATOR"}:
            continue
        seen.setdefault(candidate.upper(), None)
    return list(seen.keys())
```

Non-regex: scan each line, take prefix before the first colon (capped at
60 chars), require >=80% uppercase letters, drop a small list of structural
header words. Goal: cheap, predictable, false-negative-prone (better than
false-positive given §4 fallback exists).

## Constraints + non-goals (so you don't redirect)

- All six elements are stdlib-only on purpose. No torch / no VRAM / no
  ComfyUI imports. Module is callable from any node, including subprocess
  workers.
- §1 + §2 + §3-skeleton ONLY this round. §4 + §5 are explicitly out of scope
  this session.
- We are NOT going to switch the alias detector to an LLM call now. The
  whole point of `detect_aliases` is to be the cheap fast-path so the §4
  LLM classifier only sees the residual ambiguous cases.
- We are NOT going to add YAML/Pydantic/Protobuf/JSON-Schema. The contract
  is a small dataclass + json.dump for a reason: zero dependencies + diff-able
  on disk.
- We are NOT going to add per-character voice fingerprinting. Out of scope
  per ROADMAP.md "Rejected from autonovel".

Refer to ROADMAP.md for the full design context. Repo:
https://github.com/jbrick2070/ComfyUI-OldTimeRadio (branch `v2.0-alpha`).
