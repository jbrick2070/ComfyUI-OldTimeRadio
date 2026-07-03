# Claude anchor review -- r3 (wiring / integration / sequencing)

VERDICT: yes-with-fixes. The r2-hardened plan names the right seams; three
wiring-order risks remain.

## MUST-FIX

1. **[CONFIRMED] Sprint A wiring order: JSON audit must land in the SAME commit
   as the Policy/widget default flip.** The saved otr_scifi_16gb_full.json
   carries a positional widgets_values entry for allow_auto_fallback on the
   OTR_VideoDirector node. Sequence inside A3: flip code defaults -> re-audit
   the JSON widget-by-NAME (never index) -> OTR_WorkflowValidator + round-trip
   + link/widget audit -> suite. If the JSON is touched in a separate commit,
   the intermediate state ships a True widget against a False-forcing runtime
   (harmless by design (c), but the validator must PASS at every commit).
2. **[CONFIRMED] Sprint B dropdown wiring depends on how the image dropdowns are
   BUILT.** Video engine dropdowns build from the video registry; the stills
   dropdowns (announcer/music/other/character image models) surface through the
   image dispatcher's INPUT_TYPES. Verify at build whether the option list is
   registry-derived (new engines auto-appear -- then the JSON change is only the
   SAVED VALUE audit) or a static list (then the list + JSON change together).
   B7 must name which one it is before the coder touches the JSON.
3. **[CONFIRMED] Sequencing hazard: A4 (test triage) must complete BEFORE A1/A2
   land, not in parallel.** Several consumer tests pin valid non-fallback
   behavior (e.g. test_cs3_inter_beat_reclaim exercises inter-beat reclaim via
   run_episode(fallback_of=...) -- the reclaim assertions must survive with a
   no-fallback runner signature). If run_episode's fallback_of parameter is
   removed, its signature change ripples to every caller (tests + batch nodes +
   scripts) -- grep run_episode( callers as part of A4, not after the rip.

## SHOULD-FIX

4. Sprint D wiring: config/audio_engine_profiles.yaml rows + _LEGACY_FIRST_ENGINES
   + adapter import must land together or the dropdown shows a row whose engine
   import fails at pick time. Fail-LOUD at pick is acceptable (directive) but an
   ImportError traceback is not a NAMED error -- wrap registration like
   eng_cloud_video does.
5. Sprint E: the AudioMotionProfile schema field lands with at least ONE consumer
   assertion (a contract test that a request built by build_request_from_shot
   carries the field) or the field ships dead and drifts.
6. GPU-gate interleave: soak2 QA + proof9d (render windows) do not block Sprint A
   (CPU), but Sprint A's soak-contract rewrite (A1) CHANGES what the next soak
   asserts -- run proof9d BEFORE landing A1, or re-pin the soak harness in the
   same window (avoid a window where neither old nor new soak contract is green).

## UNVERIFIABLE (verify at build)

- Image dropdown construction mechanism (registry-derived vs static) -- must-fix 2.
- run_episode signature ripple extent -- enumerated by the A4 grep.
