# Audit S11.4 — `_visual_plan_from_script_json` dict shape

**Date:** 2026-05-12
**Branch:** `v2.0-alpha`
**Predecessor:** S11.3 commit `6ed0fd8` (function rename only)
**Verdict:** SHAPE STILL MIRRORS DELETED DIRECTOR — flatten sprint scheduled as S11.6.

## Method

Read the live source at `nodes/otr_video_plan.py:276-307`. Compare the
returned dict's key set against the deleted `LLMDirector` output
contract (`nodes/story_orchestrator.py` lines 4319-4669, deleted in
S7.1 commit `b6fb314`). Compare against the access patterns of the
three downstream consumers (`build_pass1_char_prompts`,
`build_pass2_scene_prompts`, `build_shot_plan`).

## Current return shape

```python
def _visual_plan_from_script_json(script_json: str) -> dict:
    ...
    return {
        "visual_plan":       visual_plan,                              # nested dict
        "voice_assignments": _OTRLC.voice_assignments_from_cast(led),  # nested dict
        "style":             meta.get("style") or "",                  # str
        "genre":             visual_plan.get("genre") or "",           # str
    }
```

## Deleted Director contract (per `_DIRECTOR_SCHEMA`, pre-S7.1)

```
{
  "voice_assignments": dict,
  "sfx_plan":          list,
  "music_plan":        list,
  "visual_plan":       dict,    # also nested with characters / scenes
  "style":             str,
  "genre":             str,
  ...
}
```

## Mirror analysis

| Key                  | In live shape | In deleted Director | Matches mental model? |
|----------------------|---------------|---------------------|-----------------------|
| `voice_assignments`  | yes           | yes                 | YES (same name, same nested structure) |
| `visual_plan`        | yes           | yes                 | YES (same name, same nested structure) |
| `style`              | yes           | yes                 | YES                   |
| `genre`              | yes           | yes                 | YES                   |
| `sfx_plan`           | no            | yes                 | n/a (gone in S6/S10)  |
| `music_plan`         | no            | yes                 | n/a (gone in S6/S10)  |

**4 of 4 surviving Director keys are mirrored.** The audit verdict is
unambiguous: the projection shape is the deleted Director's output
shape minus the audio-plan keys (which the audio path no longer
reads from the director). A reader who only sees this dict would
reasonably believe a director object still flows through the system.

## Downstream access patterns

```
build_pass1_char_prompts:
  director.get("visual_plan", {}).get("characters", {})
  director.get("voice_assignments", {})    # for portrait_prompt fallback chain

build_pass2_scene_prompts:
  director.get("visual_plan", {}).get("scenes", [])

build_shot_plan:
  director.get("visual_plan", {}).get("characters", {})
  director.get("visual_plan", {}).get("scenes", [])
  director.get("voice_assignments", {})
  director.get("style", "")
```

Each consumer dives into `visual_plan` to reach `characters` and
`scenes`. None of them needs the indirection -- it exists only
because the seam returns a Director-shaped envelope.

## Proposed flatten sprint (S11.6)

Skip the wrapper. Have `_visual_plan_from_script_json` return:

```python
return {
    "characters":        visual_plan.get("characters") or {},
    "scenes":            visual_plan.get("scenes")     or [],
    "voice_assignments": _OTRLC.voice_assignments_from_cast(led),
    "style":             meta.get("style") or "",
    "genre":             visual_plan.get("genre") or "",
}
```

Helper-by-helper migration:

| Helper | Pre-flatten | Post-flatten |
|--------|-------------|--------------|
| `build_pass1_char_prompts` | `director.get("visual_plan", {}).get("characters", {})` | `derived.get("characters", {})` |
| `build_pass2_scene_prompts` | `director.get("visual_plan", {}).get("scenes", [])` | `derived.get("scenes", [])` |
| `build_shot_plan` | same `visual_plan` digs | same hoist; `derived.get("scenes", [])` and `derived.get("characters", {})` |
| All three | `director.get("voice_assignments", {})`, `director.get("style", "")` | unchanged at the access site (still top-level) |

**Local variable rename in lockstep:** `director = _visual_plan_from_script_json(...)`
becomes `derived = _visual_plan_from_script_json(...)` (or `proj`,
or any neutral name -- whatever the reviewer prefers). This closes
the standing-directive #11 obligation that S11.3 deferred.

**Test surface:** 21 fixtures in `tests/test_otr_video_plan.py` build
the director-shape dict via `_sample_director()` + `_ledger_wrap()`
helpers. The flatten changes only `_visual_plan_from_script_json`'s
return; the L3 ledger fixture format is unchanged, so the helpers'
test inputs need no update. The helpers' assertions read from the
PROJECTED dict and would need updating in lockstep.

## Risk analysis

- **No behavior change.** Same data flows through the same code
  paths; the projection shape is the only difference.
- **No fixture update.** `_ledger_wrap()` still wraps a Director-
  shape dict because that's the L3 ledger's `meta.visual_plan` shape;
  the wrapper-vs-flatten distinction lives one layer up.
- **One commit per helper feasible** but bundle is cleaner. Single
  flatten commit touches the function + 3 helpers + their tests
  atomically. Pattern matches the S6.5 OTRVideoPlan adapter rip
  (commit `b0810df`) -- proven shipping shape.

## Decision

**Schedule S11.6 — `_visual_plan_from_script_json` shape flatten.**

S11.5 (QA doc diagram + typo fixes) is already in the plan's
sequence after S11.4. S11.6 fits cleanly between S11 and S12 with
no cross-sprint dependency: S12 (cache hardening) doesn't touch
the video projection.

The plan's sprint sequencing diagram:

```
S11 — symbolic + doc housekeeping (5 items)
  S11.1 module docstring
  S11.2 comment housekeeping
  S11.3 _director_from_script_json -> _visual_plan_from_script_json
  S11.4 dict-shape mirror audit          <-- THIS DOC closes S11.4
  S11.5 QA doc diagram + typo fixes
```

extends to:

```
  S11.4 dict-shape mirror audit -- SCHEDULES S11.6
  S11.5 QA doc diagram + typo fixes
  S11.6 (NEW) projection-shape flatten + local-var rename
```

S12 starts after S11.6, not S11.5. One additional commit on the
critical path; no other sprint affected.

## Acceptance for this commit (S11.4)

- This audit doc lives at `docs/audit-S11.4.md`.
- Verdict + migration plan + test impact + risk analysis all
  present.
- No code change in this item per the plan ("Audit is read-only --
  no code change in this item").
- S11.6 scheduled in the audit doc with concrete helper-by-helper
  migration table.
