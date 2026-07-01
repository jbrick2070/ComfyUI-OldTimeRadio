# Claude anchor -- r3 (wiring)

Grounded against the live maps (line-checked 2026-06-30).

## CONFIRMED WIRING SET (mirror viz_mxc_cpu exactly)
- `nodes/_otr_video_engines/__init__.py` -- viz_mxc_cpu is imported at L139 (`from . import eng_viz_rainbow`)
  inside the guarded block. ADD `from . import eng_viz_mandala as _eng_viz_mandala` in the same block.
  CONFIRMED the __init__ pattern.
- `render_driver.py:64` ENGINE_FAMILY -- viz_mxc_cpu at L75. ADD `"viz_mxc_mandala": "abstract"`. CONFIRMED.
- `render_driver.py:748` `_uses_ambient_master_audio` -- the whitelist at L760 is literally
  `str(engine_id) in ("visualizer", "viz_mxc_cpu")`. ADD "viz_mxc_mandala" to that tuple. CONFIRMED.
- `content_oracle.py:42` _FAMILY_FALLBACK -- viz_mxc_cpu at L57. ADD `"viz_mxc_mandala": "abstract"`.
  CONFIRMED (this is what makes it motion-exempt so the content oracle never demands a luma/motion floor).
- `registry.CAPABILITIES` -- ADD a viz_mxc_mandala row (cpu_ok True, required_toolchain None,
  model_requirements []). Enforced by test_capability_profiles.py:215 (registry<->CAPABILITIES bijection).

## FLAG (verify, do not blind-copy)
- `scripts/otr_video_soak.py:56` has its OWN ENGINE_FAMILY copy, but **viz_mxc_cpu is NOT in it** (grep
  found only the map header, no viz_mxc_cpu row). So either (a) the soak map is a curated subset that
  doesn't need every abstract engine, or (b) viz_mxc_cpu shipped WITHOUT this entry (latent gap). ACTION:
  before adding viz_mxc_mandala there, confirm what the soak actually does with ENGINE_FAMILY (L217 uses
  `.get(nxt, shot["family"])` -- it FALLS BACK to the shot family, so a missing row is non-fatal). If the
  soak never selects viz_mxc_*, the entry is optional; add it for symmetry only if the soak can drive it.

## DROPDOWN / JSON
- The video-engine dropdown is registry-driven -> registering the engine makes it selectable; no enum
  edit needed. NO saved-widget default in otr_scifi_16gb_full.json (opt-in selectable, like viz_mxc_cpu).
  Still run OTR_WorkflowValidator + JSON round-trip + link/widget audit after the code lands (CLAUDE.md 0)
  even though no widget value changes -- the audit proves the new engine name resolves in the dropdown.

## TESTS WIRING
- `pytest.importorskip("cairo")` at top of test_video_viz_mandala.py: the registration/capability/wiring
  tests do NOT need cairo (they don't paint), so importorskip at MODULE top would over-skip. Split:
  keep registration/contract/map tests cairo-free (engine import must be cairo-free anyway -- V-12);
  gate ONLY the real-paint determinism/visual-smoke tests behind importorskip. This preserves wiring
  coverage on cairo-less CI.

## VERDICT: wiring is a known quantity (mirrors the shipped viz_mxc_cpu); the only genuine decision is the
otr_video_soak.py entry (verify vs blind-add) and the importorskip granularity (per-test, not per-module).
