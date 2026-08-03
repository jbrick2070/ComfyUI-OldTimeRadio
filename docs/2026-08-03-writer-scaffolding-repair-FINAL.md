# Writer scaffolding repair -- the consensus method (r1 final)

**Panel-hardened build spec. 2026-08-03. Supersedes
`docs/2026-08-03-writer-scaffolding-repair-PLAN.md`, whose central premise was
wrong.** Judgment trail: `kibitz-runs/2026-08-03-writer-scaffolding/r1/`
(driver anchor, codex gpt-5.6-sol, Opus 5 seat, judgment). Consensus: Codex and
Opus independently converged on the decisive fact; the one divergence is ruled
in the judgment.

## The fact that reframed everything

**There is no repair rung at HEAD.** `_run_markup_ladder` rebuilds messages
every attempt from `system` + `base_user` + defect strings
(`_otr_scifi_fable2.py:1672-1692`); the rejected script is parsed and
discarded, never re-supplied. The model is asked to "keep the same wording" of
a script it cannot see, with defect line numbers into a document absent from
its context. Four identical failures were the system working exactly as built.
Any fix that only re-words the repair prompt is dead on arrival.

## The method (ruled: quarantine-and-adjudicate core, bounded fallback)

**Increment 0 -- deterministic, ships first, covers BOTH dead legs' shared
defect:**
- Extend `_canonicalize_transport_line` (`_otr_fable2_markup.py:52-109`) with
  the fourth balanced shape -- whole-line wrapper -- RESTRICTED to lines whose
  unwrapped form matches the transport grammar (TITLE / MUSIC / SCENE / CODA /
  END). Branch runs LAST in the colon path; interior marker count must be
  balanced; anything else stays loud. Fixture pins that a stripped non-transport
  line still dies on the roster (no scaffolding-to-dialogue promotion).
- Prevention one-liner ("plain text only -- no markdown emphasis, no headings
  beyond the transport lines shown") lands in `_script_user_prompt`'s format
  reminder (`:1553-1558`) AND the `fable2_script_system` seam (`:1751`). The
  `format_example` parameter route is dead in production -- do not use it for
  this.

**Increment 1 -- retain the draft.** Keep the post-strip `raw` per attempt.
Needed by both the core and the fallback; enables row extraction with line
numbers that match `rendered` defects.

**Increment 2 -- the core: bounded row adjudication.** When a parse fails with
row-scoped defects (UNKNOWN_SPEAKER, BAD_LINE_SHAPE, and kin), do NOT
regenerate. One tiny LLM call receives ONLY the defect-bearing rows (stable row
IDs, minimal neighbouring structure, the exact roster) and returns one verdict
per row:

    DIALOGUE_BY <exact roster name> | DROP | UNRESOLVED

Verdicts merge deterministically; **every non-quarantined row is preserved
byte-for-byte by construction** -- that is the safety property, and it is
structural, not tuned. Then the FULL parser re-runs on the merged script.
`DIRECTION` is explicitly NOT a first-build verdict (`ParsedScript` has no
destination for it). No announcer defaulting ever: an ANNOUNCER line mid-scene
closes the scene (`:388-400`) and the base prompt forbids it (`:1557`);
orphan dialogue goes to a cast member already speaking in that scene, else
UNRESOLVED.

**Increment 3 -- the fallback, now a true repair.** UNRESOLVED rows (or
non-row-scoped defects) fall through to the existing full-regeneration rung --
which now re-supplies the stripped draft as an assistant turn using the
message shape already present as dead code (`:1681-1692`), BOUNDED by
`ProviderCapacityMessages` headroom (re-supply only while output room remains;
else defects-only; log which mode ran -- `prompt_no_room` is permanently
non-rerollable).

**Increment 4 -- guard as telemetry first.** Log per-character
character-word ratios (rejected-attempt floor computed deterministically via
the parser's own counters; accepted side from `parsed.character_word_count`)
on every adjudicated acceptance. ENFORCE NOTHING until measured across sweeps.
`CAST_MEMBER_SILENT` already makes total deletion loud. When enforcement turns
on later, it needs the full trace work in one change: third
`PassAttemptTrace` outcome + `__post_init__` set + `selected` invariant +
sequence validator + **seal-hash re-baseline** (the trace is hashed into
`artifact_hash`) + one trace row per consumed call + a refusal reason carried
into the exhaustion message so a guard-refused clean parse is never reported
as "last defects" from an earlier attempt.

## Proof obligations

Unit (no GPU, seconds; template `tests\test_45word_failure_regressions.py:90-115`;
ALWAYS supply four scripted responses -- the ladder has four rungs and a short
iterator masks assertions with StopIteration):
1. Adjudication path: defect rows fixed per verdicts, non-quarantined rows
   byte-identical, full parse clean, every cast member still speaking.
2. Scaffolding-only removal accepted (the false-positive path -- most
   important: a guard that refuses correct repairs is a new outage).
3. Announcer-laundering measured by the telemetry (character-words catches what
   raw mass cannot).
4. Increment-0 fixtures: fourth-shape strip; `**BO NI:** Hello **world**`
   preserved; stripped non-transport still loud.

Live (after unit-green): re-run `ltx_audio_in` and `viz_mxc_cpu` through
`workflows/otr_canonical.json`; require RESULT SUCCESS + `obs_publish OK` +
canonical assets. Liveness proof only -- the unit tests are the safety proof.

Both live failures enter `PROD_BUG_LOG.md` (live headless evidence; fixtures
cite the logs).

## Out of scope, unchanged

`OTR_CastLock` freeze cascade (`wan_ti2v`); the OUTPUT_TRUNCATED 16384-token
slot (NOTE: increment 3 couples to it -- bounded re-supply is the interim
answer); missing `wan2.2-i2v.safetensors` (download, no code).
