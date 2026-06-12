# PASS 07 REVIEW FOCUS: PRE-MORTEM / RED TEAM

You are one panelist. THIS pass is the PRE-MORTEM: assume it is six weeks
from now and the LTX-AV lane SHIPPED AND THEN FAILED PAINFULLY in
production. Work backwards: what killed it? Pass01-06 are LOCKED -- your
job is not to redesign but to find the failure the plan does not yet
survive, and the cheapest mitigation that fits the locked design.

Rank the kill-list (most likely x most damaging first). For each: the
failure story in one sentence, the earliest SIGNAL (log line / gate /
test that would have caught it), and the cheapest MITIGATION consistent
with pass01-06. Cover AT LEAST these candidates plus any you invent:

1. FALLBACK STORM: operator sets OTR_ENABLE_LTX_AV=1 but weights are
   absent/wrong-path on the box -> EVERY talk/music beat walks the
   chain every episode -> humo-heavy episodes, double render cost,
   nobody notices for a week because episodes still complete (the
   "never aborts" property hides it). What is the storm DETECTOR (e.g.
   N degrades of the same origin engine in one episode -> one screaming
   summary line / tracker count)?
2. RELOAD THRASH: consecutive same-engine clips re-load the 13-16 GiB
   transformer + 13.2 GB encoder PER CLIP (free_after_use semantics?)
   -> 3-clip episode pays 3x model load. Does the batch order clips by
   engine? Is keep-resident-across-consecutive-clips safe under the
   AS-3 lease, and who decides (wrapper_bridge policy vs adapter)?
   Find the existing behavior in the grounding and name the cheapest
   v1 answer (even if it is "accept the reload, record the cost in
   M0").
3. CANCEL MID-SAMPLE: operator cancels in Comfy Desktop during the
   transformer phase -> executor-thread state, lease held, VRAM
   resident, next render starts on a poisoned GPU. What does the
   existing machinery do (teardown finally-blocks? lease timeout?) and
   what is the v1 discipline (e.g. always-restart-after-cancel rule in
   the operator docs vs code)?
4. PARTIAL/CORRUPT DOWNLOADS + broken symlinks (HF resume, the
   cache+symlink pattern): earliest signal and cheapest gate (file
   size/hash check in assert_usable weights probe? M0 inventory only?).
5. MODULE-CACHE STALENESS: new adapter code on disk, Comfy Desktop
   still running old module -> dropdown shows engines but render uses
   stale code / engines missing entirely. Restart discipline per build
   (Desktop needs RESTART; headless boots fresh) -- where is it
   WRITTEN so the operator hits it (adapter docstring? M0 checklist?
   error message?)?
6. GPU CONTENTION with the ACTIVE acceptance-test window: M0 runs
   while the 30w acceptance render is live -> both fail mysteriously.
   The plan says "after the acceptance window" -- is a schedule note
   enough, or does M0's launcher check :8000 liveness first (the soak
   launcher pattern)?
7. NVML UNAVAILABLE: nvml_available() False on some driver state ->
   the 14.5 ceiling silently unenforced. Fail-open or fail-closed for
   THIS lane (the heaviest engine yet)?
8. PAD-TAIL ABUSE: a systematic timing bug upstream makes every beat
   exceed the cap -> every clip is 19.9s render + frozen tail; the
   per-clip LOUD line exists, but what aggregates it (same storm
   detector?)?
9. CAPTIONS/CREDITS/TIMELINE: pad-tail and trimmed clips join the
   compositor timeline + node-93 caption ledger + credits-tail cap
   (MASTER-WAV duration). Any interaction where a padded clip shifts
   captions or the credits gate? Name the M4 grep that proves none.
10. GOLDEN-FIXTURE ROT: the dark-lane goldens break on every unrelated
    driver change -> developers update them mechanically -> the guard
    is dead. Mitigation (scope the golden to the fields that matter?
    regenerate-with-review policy?).
11. SLICE-CACHE STALENESS: cache key is (start,dur,path) -- a re-run
    after a master re-render reuses stale slices (path unchanged).
    Cheapest key fix (mtime+size) and where.
12. DESKTOP NODE LAG (#13194/#13308): Desktop build lacks
    LTXVReferenceAudio while headless has it (or vice versa) ->
    operator look-QA renders differ from production renders. M0
    records both; what is the RUNTIME guard (assert_usable node gate
    runs per-process, so each build self-gates -- confirm that is
    sufficient)?

Rules: cite grounding or VERIFY-AT-BUILD; mitigations must be additive
and consistent with the locked design; prefer signals that land in
EXISTING grep surfaces (swap-log, ledger, tracker). Output: RANKED
numbered list (failure -> signal -> mitigation), then SHOULD-CONSIDER,
then OPEN-QUESTIONS. Terse.
