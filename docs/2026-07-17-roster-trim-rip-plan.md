# Roster Trim -- Clean Rip Plan (2026-07-17)

**Baseline:** HEAD `f265c044` on `v2.0-alpha`. Suite 7984 / Bible 17 green.
**Precedent:** `git show 3312aec7` (scifi_gemini + original_codex56sol rip).
**Operator decisions folded:** (1) rip the 13 scoreboard-trim lanes; (2) ALSO rip
`science_news_v3` -> the **whole science_news family goes**; (3) new system default
bank = **scifi_fable2** ("in the dropdown scifi_fable2 v1").

This is a DESTRUCTIVE canonical change. Rip FIRST, prove self-containment BEFORE
any base deletion, everything in ONE commit + re-validate + full suite + Bible +
push + verify HEAD==origin. `/kibitz` this plan before executing.

## Final roster

**KEEP 10 lanes + custom (11 registry rows), registry order preserved:**
`media_archive`, `original_radio`, `scifi_fable2`, `scifi_codex`,
`media_archive_v3`, `public_domain_story_v3`, `shakespeare_v3`,
`scifi_fable2_v3`, `scifi_codex_v3`, `scifi_sonnet_v3`, `custom_source_bank`.

**RIP 14 lanes:**
`science_news` (base), `science_news_v2`, `science_news_v3`, `media_archive_v2`,
`public_domain_story` (base), `public_domain_story_v2`, `shakespeare` (base),
`shakespeare_v2`, `original_radio_v2`, `original_radio_v3`, `scifi_fable2_v2`,
`scifi_codex_v2`, `scifi_sonnet` (base), `scifi_sonnet_v2`.

= all 8 `_v2` + base-v1 of {public_domain_story, science_news, scifi_sonnet,
shakespeare} + `original_radio_v3` + `science_news_v3`.

## Self-containment proof (the "yikes" -- kept lanes must NOT read a deleted parent)

Grounded against HEAD:
- **Registry rows are self-describing.** The writer resolves the SELECTED id via
  `_otr_story_routing.get_bank(id)` / `require_runnable_bank(id)`
  (`OTR_LedgerScriptWriter.py:1507/3736`); v3 rows carry their own `defaults`,
  `required_seams`, `default_story_model`, `default_story_pipeline`. No base-row
  read for a v3 lane.
- **Packs are full copies, not merges.** `resolve_story_pack(id)` ->
  `get_bank(id)` -> `_pack_path(id, bank.default_story_model)`
  (`_otr_story_routing.py:516-529`); each kept v3 has its own pack file
  (e.g. `public_domain_story_v3/faithful_radio_adaptation_v3.json`). No
  `extends`/`clone_of`/`base_pack` keys in the v3 packs.
- **`base_source_bank_id()` only keys FAMILY-named code resources** that ALL
  survive because every kept family retains >=1 lane:
  - `_otr_story_rules.resolve_story_rules` -> `rules[base_source_bank_id(id)]`
    (`:305`); rules dict loaded from `story_rules/<family>.json`. KEEP all 7
    surviving families' files.
  - `_otr_style_catalog` style pool by family name (`:811`).
  - `_otr_source_payload` strict_v4 set membership `{scifi_codex, scifi_sonnet}`
    (`:285-292`) -- string test, no row.
  - `OTR_LedgerScriptWriter.py:4288` adaptation-cast classification
    `in ("shakespeare","public_domain_story")` -- string test.
  - `_otr_source_snapshot` manifest key -- C7 replay only, not production.
- **Story-rules validator stays satisfied:** `_otr_story_rules` requires
  `{base_source_bank_id(b) for b in _runnable_bank_ids()} subset of set(rules)`
  (`:278`). Post-rip runnable families = {media_archive, public_domain_story,
  shakespeare, original_radio, scifi_fable2, scifi_codex, scifi_sonnet} (7) --
  all keep their `story_rules/<family>.json`. Only `science_news.json` is removed
  (its family is gone) and `DEFAULT_RULES_ID` repointed.

**Conclusion:** the kept v3 lanes are already independent. The rip must simply
NOT delete any surviving family's `story_rules` file or style pool -- and it
doesn't (every kept family keeps a lane). No promotion/merge of base data
needed. (Optional future refactor: rename the v3-only survivors to their base
id to drop the nominal `_v3` suffix -- NOT in scope; operator's KEEP list pins
the v3 ids and the scoreboard/Sonnet step reference them.)

## science_news removal -- the default/fallback repoint (new default = scifi_fable2)

`science_news` is the hardcoded system default; repoint every site to
`scifi_fable2` (a kept lane, same `science_rss` fetcher so the empty-selection
fetch guard still pulls a science wire):

- `OTR_LedgerScriptWriter.py`: `source_bank: str = "science_news"` (:1407);
  `get_bank(source_bank or "science_news")` (:1507, :1618); `or "science_news"`
  (:1518, :1532, :1696); INPUT_TYPES `"default": "science_news"` (:3251) + tooltip
  rewrite; `source_bank="science_news"` test/helper default (:3690); `if base ==
  "science_news"` (:1963) -- audit this branch's intent (dead vs needs remap).
- `_otr_story_rules.py:51` `DEFAULT_RULES_ID = "science_news"` -> `scifi_fable2`.
- `_otr_reroll.py:661`, `_otr_story_spine.py:174/222/233` `or "science_news"`.
- `_otr_creative_prompt_router.py:136` `_SCIENCE_BANK_ID = "science_news"` --
  audit: this routes science-specific creative prompt; with no science_news lane
  the branch is dead-but-harmless. Decide keep-as-const vs remove.
- `_otr_line_composer.py` / `_otr_outline.py` param defaults + `== "science_news"`
  creative_repo_id branches (:2050/2105/2441/3257/3285/3562/3598/3647/3732,
  :1782) -- these are function-param defaults overridden at runtime; repoint the
  literal defaults to `scifi_fable2` so standalone/test calls resolve a live bank;
  the `== "science_news"` branches become dead (harmless) -- audit each.
- **Canonical** `workflows/otr_canonical.json`: writer node id 1 widget slot
  **[23]** = `'science_news'` -> `'scifi_fable2'` (value-only change at a FIXED
  positional slot; B5-safe -- no insert/shift). Structure stays 23 nodes / 57
  links / last_node_id 95 / last_link_id 283.

`news_interpreter` code + tests are RETAINED infra (removing them is separate
scope); sci-fi lanes use the `science_rss` fetcher directly (empty interpreter),
so the fetcher family is untouched.

## Surfaces to touch (ONE commit)

1. **`nodes/story_packs/banks.json`** -- delete the 14 rip rows; keep 10 + custom.
2. **`nodes/story_packs/<lane>/`** -- delete all 14 rip lane dirs.
3. **`nodes/story_packs/pipelines.json`** -- remove the 2 now-dead pipelines:
   `sonnet_archive_multipass` (base; only scifi_sonnet v1+v2 used it) and
   `original_multi_pass_v3` (only original_radio_v3 used it). KEEP
   `sonnet_archive_multipass_v3`, `original_multi_pass`, `legacy_many_pass(_v3)`,
   `fable2_multipass(_v3)`, `scifi_codex_circuit(_v3)`, `simple_4_prompt_experimental`.
4. **`nodes/OTR_LedgerScriptWriter.py`** dispatch maps: drop
   `"sonnet_archive_multipass"` key from `_RUNNER_BY_PIPELINE` (:1994) and
   `"original_multi_pass_v3"` from `_INLINE_V3_PIPELINES` (:2006). **KEEP the
   `_run_scifi_sonnet_lane` function** -- the `_make_v3_runner` wrapper for
   `sonnet_archive_multipass_v3` uses it (VERIFY at build). No runner FILE deleted.
5. **`nodes/story_rules/science_news.json`** -- delete (family gone).
6. **Default repoint** -- section above.
7. **Canonical widget [23]** -- section above; re-run validator + audits.
8. **Tests** -- rewrite roster pins + drop science_news-specific tests:
   - `test_fable2_registry.py`: the pinned 25-item order tuple (:273-284) -> new
     11-row order; `ids[-4:]` pin (:54); the "science_news untouched" test
     (:241-258) -> remove/rewrite.
   - `test_bank_variants.py`, `test_source_bank_widget_2c.py`,
     `test_scifi_lane_schema_parity.py`, `test_story_routing_stage2.py`,
     any `TestChunk2V2Rows`/`TestChunk4V3Rows`, `test_scifi_sonnet_lane.py`,
     and any "24 runnable / 25 visible" count assertions -- enumerate every
     roster/count/pin at build via grep and update.

## Validation (all before push)

- `OTR_WorkflowValidator` OK; JSON round-trip; strict link/input; live
  widget-vector drift = 0; generated-variant audit.
- Registry dry-load: 10 runnable + 1 non-runnable (custom) = 11 rows; every kept
  lane's pipeline present + pack file exists (routing validator).
- Full Windows venv suite (baseline 7984, expected to DROP as roster tests are
  removed) + Bug Bible 17.
- AST parse touched .py; JSON parse touched .json; no BOM; no 0-byte.
- Commit AND push to `v2.0-alpha`; verify `HEAD == origin/v2.0-alpha`.

## Kibitz r3 fold (codex, grounded 2026-07-17) -- MUST-FIX

1. **Legacy-seam default is NOT scifi_fable2 (build-breaker).** `scifi_fable2`'s
   pack declares only `fable2_*` stages (`required_seams: []`); the legacy-seam
   helper defaults in `_otr_line_composer.py` (:2050/:2441/:3257/:3562/:3647) and
   `_otr_outline.py:1782` resolve legacy seams (`outline_macro_system`,
   `line_composer_system`, `coda_system`, ...) and `_otr_story_pack.py:237-244`
   RAISES on a missing seam. Split the repoint: LANE-selecting defaults (writer
   `source_bank` param, INPUT_TYPES default, widget) -> `scifi_fable2`; legacy-seam
   helper defaults -> a legacy-seam bank (**media_archive**) or require explicit id.
2. **Deletions are ONE atomic edit group.** `_sweep_and_crossref`
   (`_otr_story_routing.py:351-365/386-390`) + rules-stem check
   (`_otr_story_rules.py:252-255`) hard-fail any intermediate registered/dir/rules
   mismatch. Make ALL registry/pack/rules/pipeline/map edits, THEN load/validate
   once -- never import or run a test between edits.
3. **Snapshot preflight.** Ensure no stale `OTR_SOURCE_SNAPSHOT_MANIFEST` keyed to
   science_news is set for the suite/validation run (else `load_snapshot_for_bank`
   `:170-183` hard-fails on the new base). Normal runs have no manifest -> None.
4. **`otr_story_only.json:539` ALSO stores `science_news`** (slot [23]) -- the
   Sonnet-check harness. Repoint it to a kept bank in the same commit. Update
   `README.md:92-96` default-bank docs too.
5. **Test step = grep-driven checklist.** Grep `tests/` for `science_news`,
   `science_news_v2`, `science_news_v3`, `sonnet_archive_multipass`,
   `original_multi_pass_v3` AND the ripped base ids; classify each hit
   updated/deleted/retained. Codex named extra pins: `test_workflow_json_guardrails.py`,
   `test_canonical_headless_api.py`, `test_source_ref_widget.py`,
   `test_visual_style_widget_3c.py`, `test_story_pack_stage1.py`, `test_story_rules_4a.py`.
6. Dead-pipeline cleanup (sonnet_archive_multipass base + original_multi_pass_v3)
   is NOT a runtime break if left; codex CUT it from the primary commit if risky
   -- may ship as a second green commit.

## TRUE-INDEPENDENCE workstream (operator 2026-07-17: "independent banks, real future proof")

Operator wants each kept lane standalone -- NO `base_source_bank_id` family
dependency -- while KEEPING the `_v3` ids. Grounded: rules resolve via
`base_source_bank_id` into a family stem (`_otr_story_rules.py:278/305`; stems
must be registered bank ids, `:252`). To sever the family tie for kept lanes:

- Add a `story_rules/<exact_id>.json` for each kept `_v3` lane (media_archive_v3,
  public_domain_story_v3, shakespeare_v3, scifi_fable2_v3, scifi_codex_v3,
  scifi_sonnet_v3) = copies of their family rules; change `resolve_story_rules`
  (:305) + the runnable-coverage check (:278) to use the EXACT id, not base-map.
- Update the small exact-id membership/classification sets: `strict_v4_banks`
  (`_otr_source_payload.py:285`) -> add scifi_codex_v3/scifi_sonnet_v3;
  adaptation-cast class (`OTR_LedgerScriptWriter.py:4288-4290`) -> add
  shakespeare_v3/public_domain_story_v3.
- Style pool (`_otr_style_catalog.py:811`) is a COSMETIC family pool (visual
  style, not source content) -- either exact-id it too or leave family-keyed
  (defensible: not a "source clone"). Operator call.
- Net effect: after the rip+independence, NO runnable lane relies on family
  base-mapping; `base_source_bank_id` becomes vestigial (advisory + C7 snapshot
  only) and the bake-off variant-family mechanism is effectively retired -- the
  future-proof end-state.

This is materially MORE than a deletion -- it touches the rules contract + adds 6
rules files. Scope + sequencing to confirm with operator before build.

## Risks / kibitz focus

- Dead-pipeline removal (#3/#4) is the highest-risk optional cleanup -- confirm
  NO surviving code/test path references `sonnet_archive_multipass` (base) or
  `original_multi_pass_v3` beyond the entries being removed, and that the sonnet
  v3 wrapper does not resolve the base pipeline id.
- Every `== "science_news"` behavioral branch: prove dead-and-harmless or remap.
- Exhaustive test enumeration -- a missed roster pin fails the suite (that's the
  guard working); a missed COUNT in a fixture/golden is the subtle one.
- Confirm no golden fixture / ledger fixture pins a ripped bank id.
