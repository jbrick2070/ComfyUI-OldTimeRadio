# pass07 pre-mortem -- Claude panelist review (before reading the panel)

RANKED KILL-LIST (likelihood x damage)

1. FALLBACK STORM (flag on, weights absent / path wrong). Story: every
   episode silently becomes humo+kenburns at double wall cost; "never
   aborts" hides it for days. Signal TODAY: one swap-log line per beat
   (easy to scroll past). MITIGATION: episode-end STORM SUMMARY -- if
   N>=2 degrades share the same origin engine in one episode, emit ONE
   screaming line ("[ltx_av] STORM: 3/3 beats degraded ltx_av_talk ->
   humo: <reason>") + a manifest/tracker count field. Cheap: counts
   exist in runtime_fallback_decisions; aggregation is a few lines in
   run_episode (additive) + an M4 grep.
2. RELOAD THRASH (per-clip model reload). Story: 3-clip episode pays 3x
   (load 13-16 GiB transformer + 13.2 GB encoder), wall triples vs the
   M0 single-clip numbers; lane passes M0 and fails the FIRST real
   episode. Signal: per-clip wall in the ledger vs M0 sheet. MITIGATION
   v1: (a) M0 adds a TWO-CONSECUTIVE-CLIP row (the marginal cost IS the
   gate number); (b) batch order already groups by engine? VERIFY in
   video_engine/render batch; if not, accept reload in v1 and record
   honestly -- keep-resident-across-clips is a wrapper_bridge policy
   question deferred unless M0's 2-clip row fails the episode budget.
3. CANCEL MID-SAMPLE / poisoned GPU. Story: operator cancels in
   Desktop; lease held or VRAM resident; next render OOMs confusingly.
   VERIFY: teardown finally-blocks + lease release on exception paths
   in MotionEngineBase; gpu_residency staleness/timeout semantics.
   MITIGATION: v1 operator rule in the M0 checklist + adapter docstring
   ("after a mid-render cancel, restart the server before the next
   render"); plus assert_usable's existing below-ceiling wait fails
   LOUD if VRAM is still resident (names the engine).
4. WEIGHTS PRESENT BUT CORRUPT/PARTIAL (resume artifacts, broken
   symlink). Signal: load explodes mid-phase after minutes. MITIGATION:
   assert_usable weight probe checks existence AND size >= a per
   -artifact floor (file sizes are now judge-verified constants);
   hash checks stay M0-only (too slow per-render).
5. MODULE-CACHE STALENESS. Story: Desktop shows the dropdown but runs
   old code (or misses the engines), operator concludes "lane broken".
   MITIGATION: RESTART discipline written in THREE places (M0
   checklist, adapter docstring, the ship-notes section of the plan);
   headless boots fresh by design.
6. GPU CONTENTION with the live acceptance render. MITIGATION: M0
   checklist step 0 = the soak-launcher pattern (check :8000 listener /
   agree with the operator the box is idle); never run M0 during the
   acceptance window -- schedule note + checklist gate, no code.
7. NVML UNAVAILABLE. Story: driver hiccup -> nvml_available False ->
   ceiling unenforced for the heaviest engine yet. MITIGATION: for
   ltx_av_* ONLY, treat NVML-unavailable as FAIL-CLOSED in
   assert_usable (named error) -- the lane is opt-in and can afford
   strictness; existing engines unchanged. (VERIFY gpu_residency's
   current fail-open/closed stance first.)
8. PAD-TAIL SYSTEMIC ABUSE (upstream timing bug -> every beat capped).
   Signal: per-clip pad-tail markers. MITIGATION: fold into the storm
   summary (count pad-tail events per episode; >=2 -> one screaming
   line). No new machinery.
9. CAPTIONS/CREDITS INTERPLAY. Clips are trimmed/padded to EXACTLY
   target_frame_count, so the compositor timeline math is unchanged by
   construction; captions key on the audio ledger, not clip length;
   credits-tail cap keys on MASTER WAV duration. Risk is therefore
   LOW; M4 grep: duration_check OK + captions events line + credits
   post-roll restored on the forced-lane smoke (same greps as the
   acceptance harness -- no new surface).
10. GOLDEN-FIXTURE ROT. MITIGATION: goldens compare a PROJECTION
    (engine_id, canvas, text_prompt, audio_ref presence, asset_refs,
    timing, seed_bundle) -- not the full dict -- so unrelated request
    fields don't churn them; regeneration requires a one-line judgment
    note in the commit ("golden regen: <why>").
11. SLICE-CACHE STALENESS. Key gains (mtime, size) of the master --
    one-line change at _slice_master_audio key build, additive, also
    fixes the same latent issue for HuMo slices.
12. DESKTOP NODE LAG. Per-process assert_usable self-gates each build
    (sufficient at runtime); M0 records both builds; ship-notes tell
    the operator Desktop may lag core for #13111 nodes and the
    headless build is authoritative for renders.

SHOULD-CONSIDER

13. M0 checklist as a literal checkbox file in the campaign dir (the
    operator runs it top to bottom; steps 0-N with expected outputs).
14. Tracker row gains a "degrade count last smoke" cell -- makes storms
    visible outside logs.

OPEN-QUESTIONS

15. Does wrapper_bridge keep models resident across consecutive clips
    of the same engine today (free_after_use semantics per engine)?
    Decides kill-item 2's branch. (Panel: find it in the grounding.)
16. gpu_residency: lease timeout/stale-holder behavior after a crashed
    process? (Decides kill-item 3's depth.)
