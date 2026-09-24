# Handoff: execute the widget tier

Paste the block at the bottom into a fresh window. Everything above it is the
grounding it refers to.

Written 2026-09-13 by the window that cleared TASK ZERO and got the tier its
verified QA verdict. **No widget code has been written.**

---

## The state you are inheriting

`docs/2026-09-13-widget-cleanup-QA-VERDICT.md` is the **plan of record**. It is
Cursor's verified QA on the brief, with its own contrarian folded in. Where it
and `docs/2026-09-13-widget-cleanup-QA-BRIEF.md` disagree, the verdict wins.

Already done, do not redo:

* **TASK ZERO is cleared.** All 14 suite deltas classified and fixed; the four
  "newly passing" proved to be worktree artifacts, not fixes.
* **`test_workflow_link_target_indexes` is fixed** -- it was collecting ONE item
  of 17. Now `rglob`, collects 17, passes 17.
* **`test_text_metric_ownership` is cleared** (`5184e33b`). The QA report lists
  it as a red blocker on step 1; that report was written before the fix.
* **`scripts/otr_widget_surgery.py` exists, and its contract changed on
  2026-09-13 after a QA pass found a blocker in it.** Identity-based `dst_slot`
  repair, descriptor-vs-value index handling, and a `verify()` that re-asserts
  the backstop property. Use it; do not re-derive the procedure by hand.

  **`reorder_widgets` now returns `(touched, repairs)` and repairs the link
  table ITSELF.** An earlier version said a reorder "needs zero link repairs",
  which was true of the canonical and false in general: a WIDGET descriptor can
  carry a live link, and three do today -- `OTR_SceneSequencer.script_json`
  (link 277), `OTR_SignalLostVideo.script_json` (16) and `news_used` (110).
  Moving one left its row pointing at the slot it vacated. The tool no longer
  trusts the caller to remember part 3, and it REFUSES a node whose
  `widgets_values` is shorter than its descriptor count rather than backfilling
  `None`.

  **`remove_widget` got the SAME three properties**, and this document did not
  say so until now -- which mattered, because Task C below tells you to use it.
  It returns `(touched, repairs)`, repairs the link table itself, and refuses a
  `widgets_values` that is short OR absent.

  **The trap that return type sets, which has already been sprung once:**

  ```python
  touched = ws.remove_widget(wf, "OTR_LedgerScriptWriter", "some_widget")
  assert touched          # ALWAYS TRUE -- a 2-tuple is truthy
  ```

  Bind both names. A removal that matched nothing returns `([], [])`, and the
  one-name form hides it behind a truthy tuple. That exact line was written into
  the tool's own test and survived a green run before it was caught.

  `tests/test_widget_surgery_tool.py` proves all of it against the real
  canonical -- eleven tests. Run it before you trust a change you make to the
  tool.

## The one blocker, and it is not code

**There is no exact 36-widget Writer order.** The brief gives a leading block
and broad groups; every model, provider, runtime, source, scaffold, validation
and replay field is unplaced. The QA verdict stops coding at step 8 for exactly
this reason.

The settled leading block (13 of 36):

```
episode_title, source_bank, custom_premise, num_characters, act_count,
include_act_breaks, visual_style, creativity, lemmy_cameo,
story_characters, story_plot, story_setting, story_author
```

The rest remain unplaced:

```
creative_writing_model, technical_model, min_p, repetition_penalty,
max_new_tokens_cap, use_exchange, enable_production_stage3_validators,
news_briefs_required, openrouter_slot_a_model, openrouter_slot_b_model,
comfy_slot_a_model, comfy_slot_b_model, story_scaffold,
google_api_slot_a_model, google_api_slot_b_model, source_ref, llm_device,
llm_attn_impl, llm_quant_policy, llm_vram_ceiling_gb, replay_from
```

UPDATED 2026-09-24: this list used to name `gguf_n_ctx` and `gguf_quant` too,
and used to count the widgets. Both widgets were removed with the writer backend
behind them, and `episode_language` has been appended since. DO NOT resume this
handoff against the counts in it -- read the live `INPUT_TYPES()` and rebuild
both lists before proposing an order, because this plan was never executed and
the schema has moved twice underneath it.

Propose one explicit ordering of the live set, get the operator's yes, and make
that list an executable test before touching `INPUT_TYPES`.

## The traps, measured

**`migrateWidgetsValues` fires at exactly one removal.** The Writer's mask is
`(37 - k) + 1` against 37 saved values, so `k == 1` is the dangerous case:

| removed | migration | widgets holding a wrong value |
|---|---|---|
| `perfect_run_spacesaver` alone | FIRES | 23 |
| it + `min_p` | no | 27 |
| trailing `story_author` alone | **FIRES** | 4 |
| last two trailing together | no | **0** |

"A trailing widget is nearly free" is FALSE. This is old-workflow behaviour;
correctly regenerated shipped graphs are fine.

**`gate_in` moves.** It is descriptor 32 today and carries link 279. Removing
descriptor 8 moves it to 31. Repair by identity, never by subtracting one, and
never hardcode 32 in a later reorder.

**Renames are answered without renaming.** Frontend 1.51.10 separates the UI
label from the internal key and `node_info` forwards the option, so
`display_name` gets the wording with none of the blast radius. Do not rename
internal keys. Do not add aliases.

**Two no-goes, on evidence.** Voice-engine consolidation (five blockers, the
sharpest being that `char_voice_engine` is legitimately stamped literal `"auto"`
when CastLock resolves nothing). And `custom_source_bank` cannot be fixed as
proposed at all -- the dropdown is fed scalar ids by `list_bank_ids()`, so
`banks.json`'s `label` is not a per-option display label.

## Two guards the verdict requires BEFORE any reorder

1. **Live schema-order comparison.** Every graph's ordered widget descriptor
   names against the live `INPUT_TYPES` order. The existing tests prove the 17
   graphs agree with EACH OTHER -- they could all agree on the same wrong order.
2. **Name-paired migration with SYMBOLIC values.** Real defaults are `""`,
   `False` and `0`, so two swapped widgets can hold equal values and hide the
   corruption. Unique marker per widget, asserted to travel with its name.

---

## PASTE THIS INTO THE NEW WINDOW

```
Execute the widget tier on ComfyUI-OldTimeRadio. Read
docs/2026-09-13-widget-cleanup-QA-VERDICT.md FIRST -- it is the plan of record,
it is an independent verified QA of my brief, and where it disagrees with
docs/2026-09-13-widget-cleanup-QA-BRIEF.md the verdict wins. Then CLAUDE.md
sections 0 and 0B. Then docs/HANDOFF_WIDGET_TIER_EXECUTE.md for what is already
done.

Start current: git fetch origin main && git pull --rebase origin main
Branch is main. v2.0-alpha is retired -- never push to it.

STEP 1 IS NOT CODE. There is no exact 36-widget Writer order and the verdict
stops coding until there is one. Propose ONE explicit ordering naming every
remaining widget exactly once -- the 13-widget leading block is settled, the
other 23 are listed in the handoff and are yours to place and justify. Show me
the list and wait for my yes. Then make it an executable test BEFORE you touch
INPUT_TYPES.

While you wait for that yes, do the work that does not depend on it, in this
order:

  A. Add the two guards the verdict requires before any reorder: a live
     schema-order comparison (every graph's descriptor names against live
     INPUT_TYPES -- the existing tests only prove the 17 graphs agree with each
     other, so they could all agree on the same wrong order), and a name-paired
     migration test using SYMBOLIC values (real defaults are "", False and 0, so
     two swapped widgets can hold equal values and hide the corruption).

  B. Apply the UI labels via display_name. Frontend 1.51.10 separates the label
     from the stable internal key and node_info forwards the option. Do NOT
     rename internal keys, do NOT add aliases, do NOT touch descriptor name or
     widget.name. Update localized_name in the canonical for consistency and
     regenerate variants.

  C. Remove perfect_run_spacesaver. Removal surface and the link consequence are
     in the verdict; gate_in moves from descriptor 32 to 31 and link 279 must be
     repaired BY IDENTITY. Replace its existing "must remain" test with an
     absence test -- do not just delete it.

  D. Remove all FIVE deprecated ffmpeg widgets (CaptionBurn, MasterAudioMux,
     SilentComposite, PostUpscaleProcgenBlend, SceneAwareScopes). The last two
     are off the canonical but still registered and still ship. Keep the
     trusted internal resolver arguments; only the public widget goes. Rewrite
     tests/test_widget_cannot_name_the_binary.py to assert no node exposes the
     widget while keeping the resolver-security coverage.

  E. Remove seed_mode and request_seed from OTR_VideoDirector ONLY. Do not touch
     OTR_ImageDirector's -- those feed the live image dispatcher.

Then, separately and tested on its own, the title consolidation (the Assembler
already has a ledger source: canonical link 289 carries the full v2_ledger_json
into replay_descriptor). Leave voice ownership and custom_source_bank alone --
both are no-goes on evidence, reasons in the verdict.

USE scripts/otr_widget_surgery.py, and note the contract changed on 2026-09-13.
BOTH remove_widget and reorder_widgets now return (touched, repairs) and repair
the link table themselves -- a WIDGET descriptor can carry a live link and three
do today -- and both refuse a widgets_values that is short or absent rather than
backfilling None.

UNPACK BOTH NAMES. `touched = ws.remove_widget(...)` followed by
`assert touched` is ALWAYS TRUE, because a 2-tuple is truthy even when nothing
matched; that line was written into the tool's own test and passed a green run
before it was caught. A no-match returns ([], []).

Identity-based dst_slot repair is idempotent -- a second pass returns empty --
and reports a stale dst_node as well as a stale slot. Run
tests/test_widget_surgery_tool.py before trusting any change you make to it. Do
not re-derive the three-part removal by hand.

THE TRAP THAT INVERTS THE OBVIOUS: migrateWidgetsValues fires when EXACTLY ONE
widget is removed ((37-k)+1 against 37 saved values), so removing a single
trailing widget corrupts replay_from and the three My Story fields while
removing the last two together corrupts nothing. "Trailing is free" is false.

GATES, every stage, before any push:
  python scripts/build_variants.py --all && python scripts/build_variants.py --check
  python -m pytest -q -p no:cacheprovider tests/test_widget_value_alignment.py \
    tests/test_canonical_widget_input_parity.py \
    tests/test_workflow_link_target_indexes.py \
    tests/test_workflow_live_passes_validator.py \
    tests/test_no_shipped_graph_carries_a_premise.py \
    tests/test_input_types_signature_parity.py \
    tests/test_workflow_apply.py tests/test_capability_profiles.py
  plus the full suite and the Bug Bible regression.

SUITE DISCIPLINE: never read a raw failure count -- diff failure SETS against a
worktree at the same HEAD, and NAME THAT WORKTREE WITH A HYPHEN (a hyphen-less
path makes test_package_loads_by_path_despite_the_hyphenated_directory fail by
design, and three more fail for unrelated worktree reasons -- that is four fake
"fixes" in your diff). Re-verify the FIXED side of any diff at HEAD in a
worktree before believing it. The known-fail guard can eat tracebacks, so run
pytest through a plugin printing report.longreprtext from
pytest_runtest_logreport.

CONSTRAINTS: never git add -A; commit messages via -F file; do NOT touch
pyproject.toml (it auto-fires a registry publish); push to main.

One contrarian reviewer on the finished diff before any push, briefed to REFUTE
and told explicitly not to execute the pipeline -- no node run(), no model load,
no bare python -c importing nodes.* render paths. An agent did that today and
held 17.9 GB of GPU for 40 minutes.

Runtime proof before you call the tier done: inspect live /object_info for
stable internal names and the expected display labels, convert the canonical to
an API prompt and verify every realized input, then run the real canonical
workflow to RESULT SUCCESS + obs_publish OK with the asset under otr/obs/.
```
