# r2 Claude anchor (coding plan / implementability)

VERDICT: yes-with-fixes. FORMAT=ENGINE is the right shape; the
implementability gaps are the family taxonomy, engine-composes-engine
dispatch, and default-resolution precedence.

MUST-FIX:
1. [1 FORMAT=ENGINE] Family taxonomy: fmt rows need a `family` value.
   Existing families (video registry protocol) do not include a
   composite/format concept; `static_motion` misdescribes a board pan
   with embedded lipsync crops. Fix: add ONE new family value
   `format_composite` + CAPABILITIES rows (cpu_ok=True, 0 VRAM) +
   role_compat/profile-derivation awareness, OR justify reusing
   static_motion explicitly. Decide here, not at build.
2. [1/3 F1-c, 4 F2-d] ENGINE COMPOSES ENGINE: format render_clip
   invokes the kling lipsync ROW. It must route through the SAME
   resolver + session/budget/cache machinery as any dispatch (request-
   level composition), NEVER a direct adapter/class import -- else
   budget/ledger/fallback are bypassed inside formats. Define the
   composition API (a scoped sub-request carrying the parent
   request_id; ledger links parent/child).
3. [1 visual_format] Default-resolution PRECEDENCE must be stated:
   explicit per-role pick > visual_format widget/env > cloud-profile
   DEFAULT-OVERRIDE map (pass04 sec 6) > registry default. Without
   this, S4's profile override and the format switch fight silently.
4. [3 F1-b] The camera desk consumes ledger line timings; the render
   request template must be shown to carry them (or the format engine
   declares a ledger_json requirement). verify: request shape at
   VideoRenderBatch includes per-line timing for the beat.
5. [4 F2-c] Blender render determinism: the local plate cache key
   (mesh, camera, duration) must include the CAMERA PATH HASH + fixed
   render settings (seed/samples), else cache hits return wrong
   plates. Also name where stage templates live (repo `assets/blender/`
   checked in; scenes are code-adjacent, not episode assets).

SHOULD-FIX:
1. still->silent-clip helper: shared home `nodes/_otr_shared/`
   (consumed by F1-c and F2-d), not duplicated per engine.
2. Post-paste face-similarity: reuse the EXISTING portrait-hash
   machinery (the pipeline already computes portrait hashes); verify
   which module owns it and its threshold semantics at build.
3. Tests to name in the plan: registry registration + CAPABILITIES
   consistency for fmt rows; universal-slot resolve incl. fmt rows;
   golden-30s scripts per format; precedence-order unit tests.
4. Blender subprocess: timeout + LOUD failure classification
   (corrupt_output vs timeout) consistent with the S0 error taxonomy.
