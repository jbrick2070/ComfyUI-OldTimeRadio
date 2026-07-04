# AUDIT: rip ALL hard-baked VRAM-TIER code, keep the OOM seatbelt

Operator directive 2026-07-03: remove the hard-baked VRAM *tier classification* entirely.
KEEP the runtime OOM ceiling guard (the seatbelt). This is a big surface in a big repo --
the audit's job is to find EVERY touchpoint, bucket it RIP vs KEEP, flag entanglement +
breakage, and give a safe rip order.

## RIP (the hard-baked tier classification -- all of it)
- `nodes/_otr_video_engines/registry.py` CAPABILITIES table: per-engine `vram_class`
  (cpu/light/medium/heavy) + `vram_estimate_mb` (the "DRAFT estimates pending operator
  probe" numbers that never got probed) + `vram_tier_label()` and the `(~6.8GB)` dropdown
  label suffix it feeds (used in otr_video_director `_label_for`).
- The capability-profile VRAM tier-FILTERING: any place a profile/enable-set is gated,
  cross-validated, or an engine excluded BY vram_class / vram_estimate_mb / a "tier"
  (nodes/_otr_shared/capability_profiles.py + the 3 profiles + widget_mapping).
- The `vram_ceiling_gb` WIDGET on the graph node (OTR_LedgerFreezeCascade, the `14.0` in
  node "1b") -- the hard-baked graph number. After removal the ceiling comes from the
  env/default only.
- Any other hard-baked per-engine VRAM tier tables / assumptions
  (_otr_model_catalog.py VRAM fields, eng_humo safe_render_frames caps if they are tier
  guesses vs measured, _vram_log tier buckets, etc.) -- classify each.

## KEEP (the seatbelt + real plumbing -- do NOT remove)
- The RUNTIME OOM ceiling guard: `motion_common.dynamic_vram_ceiling_mb()` (env
  OTR_VRAM_CEILING_MB, default 14500) + `assert_vram_within_ceiling()` -- the fail-LOUD
  guard that prevents an OOM crash. This is NOT a tier; it stays.
- Actual VRAM RECLAIM/free-between-renders (wrapper_bridge.reclaim_idle_models,
  free_after_use, the BUG-291 detach) -- how the single-heavy budget is met at runtime.
- Live VRAM MEASUREMENT (nvml/vram_used_mb) that the seatbelt reads.

## Questions for the panel (ground every claim, cite file:line)
1. Enumerate EVERY touchpoint of the RIP list above across nodes/scripts/tests/config --
   the CAPABILITIES fields, vram_tier_label + label suffix, profile vram filtering, the
   vram_ceiling_gb widget + its node + its downstream consumers.
2. For each: is it purely tier-classification (RIP), or does something load-bearing read
   it (a render decision, the enable-set, a fail-closed gate)? If load-bearing, what
   replaces it after the rip (e.g., the runtime ceiling already guards OOM, so a static
   vram_estimate gate is redundant)?
3. The `vram_ceiling_gb` widget: which node owns it, who reads the emitted value, and does
   removing the widget cleanly fall back to dynamic_vram_ceiling_mb()? Any positional-widget
   (BUG-LOCAL-097) impact on that node + its JSON?
4. Capability profiles: after removing vram_class filtering, do the 3 profiles + the
   applier + widget_mapping still work as plain per-role engine pick maps (no tier logic)?
   What tests break?
5. Confirm the SEATBELT (dynamic_vram_ceiling_mb + assert + reclaim) is fully independent
   of the tier classification and survives untouched.
6. A safe RIP ORDER + the full test-breakage inventory. Flag anything that would silently
   break (esp. the profiles we just consolidated, the enable-set, the dropdowns).

Invariants: KEEP the OOM seatbelt; node/JSON same-change (BUG-LOCAL-097); NO fallbacks; NO
back-compat shim; audio spine untouched; suite + Bug Bible + B7 green; push per green chunk.
