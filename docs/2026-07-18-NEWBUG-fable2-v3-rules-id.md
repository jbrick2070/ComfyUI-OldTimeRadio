# NEWBUG -- scifi_fable2_v3 cannot run: fable2 revision_contract hardcodes rules_id == 'scifi_fable2'

**Discovered:** 2026-07-18, Sonnet-4.5 cross-bank bake-off (render window), baseline HEAD `60c73618`.
**Class:** hard lane-contract defect (model-independent). NOT a Sonnet content fail, NOT a deterministic
SFW/quality gate. Repeatable live production failure -> candidate PBUG (coder/campaign window owns admission).

## Symptom
`scifi_fable2_v3` story-only leg fails almost immediately (t=22s, before any real generation);
`RESULT FAIL canonical_runner_exit=1`. Reproduced with creative=`anthropic/claude-sonnet-4.5`; the failure
is upstream of any writer output, so it is model-independent (would fail with Mistral, gemma, etc.).

## Error (verbatim, server log)
```
!!! Exception during processing !!! [scifi_fable2] pass 'revision_contract' failed:
story_rules.rules_id must be 'scifi_fable2', got 'scifi_fable2_v3' (no fallback to legacy_many_pass)
nodes._otr_scifi_fable2.Fable2ScriptError: [scifi_fable2] pass 'revision_contract' failed: ...
```

## Root cause (code-grounded)
`nodes/_otr_scifi_fable2.py:2307`:
```python
if getattr(story_rules, "rules_id", None) != "scifi_fable2":
    raise Fable2ScriptError(
        "revision_contract",
        "story_rules.rules_id must be 'scifi_fable2', got "
        f"{getattr(story_rules, 'rules_id', None)!r}",
    )
```
The fable2 lane hardcodes the expected `rules_id` to the literal `"scifi_fable2"`. The 2026-07-17 roster
trim (`499386aa`) made every lane own its pack + `story_rules` by EXACT id, so the `scifi_fable2_v3` bank's
story_rules carries `rules_id = "scifi_fable2_v3"` and its pipeline `fable2_multipass_v3` routes to this same
`_otr_scifi_fable2` code -- which then rejects the v3 id. Net: `scifi_fable2_v3` is a runnable=True bank in
`banks.json` that can never complete a leg.

## Blast radius
- `scifi_fable2_v3` ONLY (pipeline `fable2_multipass_v3` -> `_otr_scifi_fable2`).
- `scifi_fable2` base is unaffected (rules_id == 'scifi_fable2') -- it succeeded in this bake-off.
- Other `_v3` banks are unaffected: `media_archive_v3` / `public_domain_story_v3` / `shakespeare_v3`
  (legacy_many_pass_v3) and `scifi_codex_v3` (scifi_codex_circuit_v3) and `scifi_sonnet_v3` all succeeded --
  their lane code does not hardcode a single rules_id.

## Fix direction (defer to a coder window; NOT landed mid-sweep per the no-code-mid-sweep rule)
Root fix, no shim: the revision_contract check must accept the lane's DECLARED rules_id rather than a single
literal. Options (coder chooses): (a) pass the expected rules_id in from the bank/pipeline config and compare
against that; (b) accept a family match (id == base OR id startswith "scifi_fable2"); (c) validate against the
set of fable2-family rules_ids from the registry. Add executable coverage: a `scifi_fable2_v3` contract test
that asserts the revision_contract pass accepts `rules_id == "scifi_fable2_v3"`. If promoted, follow the
Three-File Contract (Bug Bible YAML + README + regression in one commit).

## Verify condition (automatable)
A `scifi_fable2_v3` story-only leg (any model) passes the `revision_contract` pass (no `Fable2ScriptError`)
and reaches generation. Bake-off status for this bank is recorded FAIL (this bug) at both 420w and 720w.
