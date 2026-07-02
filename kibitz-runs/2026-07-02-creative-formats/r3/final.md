# r3 synthesis (anchor + codex + claude)

Grounding: both agents cite exact lines (FAMILIES closed tuple +
import guard schemas.py:31-75; extra=forbid :79; director :275-343;
render lifecycle :1752-55; ENGINE_FAMILY :70-87; lipsync base drift
:1836-39; has_audio=False invariant :217-223). No material misreads.

ADOPTED:
1. family: KEEP `format_composite` -- added to FAMILIES +
   FAMILY_REQUIRED_INPUTS in the SAME change as the engines
   (alternative of reusing static_motion/image_to_video considered and
   rejected: ledger/soak classification clarity wins; the schema edit
   is small and guarded). ENGINE_FAMILY gains both fmt entries.
2. format_ctx EXACT shape: `format_ctx: Optional[FormatContext] =
   None` on VideoRequest; FormatContext is its own _Forbid sub-model;
   NO VideoRequest validator reads it (engines read it). Wiring:
   manifest path stamped into the patched ledger BEFORE the video
   batch node; build_request_from_shot copies it into format-engine
   requests.
3. fmt-row usability: rows REGISTER always (menu invariant); adapter
   assert_usable fails CLOSED with a named error when format_ctx is
   absent (stamping did not run). Manual per-role pick of a fmt row is
   honored WHEN the format context exists; else the error names the
   visual_format switch. (Universal-slot rule preserved with an
   honest precondition.)
4. explicit-vs-default detection: direct() computes the registry
   default per role at resolution time; widget value == default =>
   inheritable (visual_format may override), != default => explicit.
   Documented edge: explicitly picking the default value is
   indistinguishable and treated as inheritable.
5. visual_format widget appended AFTER gate_in (forceInput slot
   undisturbed); node's widgets_values grows by EXACTLY 1 at the end;
   saved-workflow round-trip test.
6. Bridge media-type seam: S0/S3 invoke bridge converts canonical
   mp4/wav PATHS -> Comfy VIDEO/AUDIO objects; normalize the existing
   base_clip_ref {"path": ...} drift in S3; test the exact
   cloud_kling_lipsync payload.
7. Format render_clip obeys the STANDARD post-render contract: silent
   video (has_audio=False; Kling output stripped with -an), target
   frame count matched; canonicalize runs after as usual. Per-line
   lipsync failures are handled INSIDE render_clip (stay-still line +
   LOUD stamp) -- driver-level engine fallback is NOT the inner retry
   path.
8. V1 kill-switch DECOUPLED from the 3D pin: run it with a CHECKED-IN
   fixture GLB (tiny known-good mesh) as soon as the S3 kling adapter
   exists -- before any 3D row pinning or F2 cache machinery.
9. Headless env preflight in every format acceptance: the four envs
   (flag, credentials, budget, Kling concurrency) checked before
   smokes; launcher/soak wiring included.
10. Board manifest coordinates: top-left origin, integer pixels on
    the 4K canvas, rounding + paste-scaling rules declared;
    layout_seed consumed via random.Random(layout_seed) permutation.
11. Estimate report includes per-line lipsync rows (cached vs billed
    visible). Mesh key simplification: tin_toy profile changes flow
    through the CONCEPT-SHEET content hash (already in the key);
    keep mesher_version for mesher changes; no extra dimension.

CUTS: whole-episode tin-toy wiring (again, gone); bespoke F1 cache
gate (S0 cache tests own it; one ledger assertion in the smoke).
NTFS hardlink alternative noted once more; operator copy directive
stands.
