VERDICT: yes-with-fixes. The cleanup arc is directionally coherent, but caption ownership and “hide/remove” semantics are not build-ready.

MUST-FIX BEFORE BUILD:
1. [Cleanup plan #3] Caption ownership is unresolved. The plan says node 93 owns live caption burn and punts node 86 to “operator call,” but `nodes/otr_caption_burn.py:1-13` says caption burn moved into `OTR_CaptionBurn`, while `nodes/otr_post_upscale_procgen_blend.py:901` / `:988-995` still burns captions and the canonical workflow routes node 84 -> 86 -> 93 with node 86 off and node 93 on (`workflows/otr_scifi_16gb_full.json`, nodes 84/86/93, links 247/266/250). Concrete fix: choose exactly one caption owner before build. Either remove/re-route node 86 and make node 93 the documented owner, or turn off/remove node 93 caption widgets and make node 86 the documented owner. Update code comments, workflow JSON, env behavior for `OTR_BURN_CAPTIONS`, and tests in the same change.

2. [Cleanup plan #1 / Top 5 confusion offenders] “Remove/hide stereo_policy + delivery_profile” conflates UI clutter with behavior removal. `stereo_policy` is actively consumed for mono conversion in `nodes/_otr_voice_node_common.py:235`, `:274`, `:362` and `nodes/stable_audio_theme.py:119`, `:138`, `:195`; `delivery_profile` is validated and stamped into metadata/cache identity in `nodes/cast_lock.py:100`, `:129`, `:142`, `:175` and `nodes/_otr_delivery_profiles.py:1-8`, `:49-55`. Concrete fix: specify “remove from the widget surface only; preserve internal constants/defaults and output semantics.” Do not describe this as deleting the feature.

3. [Cleanup plan #2] “Tooltip/label batch (no positional risk)” is only true for tooltip text. Renaming actual INPUT_TYPES keys such as `engine`, `oom_index`, `story_scaffold`, slot model widgets, or video model widgets is a public graph/API change: `_otr_workflow_apply.py:139-144`, `:493-503` and `workflows/otr_scifi_16gb_full.json` address those widgets by name. Concrete fix: split tooltip-only work from any real key rename. If names change, treat it as a workflow/schema migration with validator and profile-applier updates.

SHOULD-FIX:
1. [Cleanup plan #1] The migration recipe is too vague for node 80. `delivery_profile` is mid-list in `nodes/cast_lock.py:88-106`, and the workflow has `widgets_values` `default | auto_registry | neutral | true` for node 80. Concrete fix: state the exact before/after widget-value mapping by live INPUT_TYPES order, not by serialized `inputs[]`, then validate node 80 separately.

2. [Cleanup plan #4] The video alias dedupe item appears to rest on a false premise. `OTR_VideoDirector` accepts aliases only in `_engine_id_from_pick` (`nodes/otr_video_director.py:110-127`), while the dropdown is built from `registry.all_engine_names()` (`nodes/otr_video_director.py:137-149`; `nodes/_otr_shared/engine_registry_base.py:160-166`). Concrete fix: cut this unless a UI capture proves duplicate displayed choices.

OPTIONAL / NICE-TO-HAVE:
- Add a small before/after table for each affected node: visible widgets removed, constants preserved, workflow `widgets_values` count before/after, and expected user-facing behavior.
- Record that the reviewed nodes are registered: `__init__.py:259`, `:301`, `:312` and `_otr_class_registry.py:50-75`.

CUT THESE (scope / over-engineering):
1. [Cleanup plan #4] Cut registry-level video alias dedupe for this pass. It does not serve the widget-clutter goal unless duplicates are proven visible; legacy alias compatibility is useful and low-cost.
2. [Cleanup plan #2] Cut actual widget key renames from the first build. Tooltips or display-only labels address confusion without graph/API migration risk.
