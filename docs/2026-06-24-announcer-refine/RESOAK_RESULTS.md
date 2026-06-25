# KILL 2 / KILL 4 -- LIVE RE-SOAK RESULTS (2026-06-24)

Build under test: C1-C4 on `v2.0-alpha` @ `b7bf7fc3` (C1 `14704f98` -> C2 `69125683`
-> C3 `e58fba40` -> C4 `b7bf7fc3`), all behind the `story_scaffold` widget, no
workflow-JSON change. Per-chunk: full suite green vs the 5 pre-existing `267a53e`
workflow-pin fails; Bug Bible 16/7/3; +65 unit tests across the four chunks.

Soak harness: `scripts/_tmp_kill2_resoak.py` (throwaway) against a live headless
server (canonical `workflows/otr_scifi_16gb_full.json`, LTX lane,
`OTR_BYPASS_FREEZE_HALT=1`). The toggle is flipped per-prompt via the
`story_scaffold` widget (no reboot). Fixed `custom_premise` (an interstellar-comet
news seed) so the run is deterministic and the coda's real-news payload is
verifiable. `news_briefs_required=False`.

## LEG 1 -- mistral-nemo, story_scaffold=ON -- FULL END-TO-END PASS

Ledger `signal_lost_comets_trail_20260624_232917`; `Prompt executed in 00:25:25`;
VRAM peaked 14.5 GB (host, under the 14.5 GB ceiling); zero render tracebacks.

| Chunk | Live evidence | Verdict |
|---|---|---|
| **C1 StoryContract (KILL 2)** | `meta.story_contract = {slug: retirement_home_ghost_story, label: retirement-home ghost story, ending_tag: bittersweet_parting}`; server log `story-grammar ON: style=retirement_home_ghost_story ending_tag=bittersweet_parting climax_beat=b008 final_beat_crisis=0`. Style selected pre-outline from the brief/seed; climax shape is the non-doomsday `bittersweet_parting`, NOT the console-standoff default. | PASS |
| **C2 announcer OPEN (input starvation)** | `meta.story_quality.open_safe_fallback = False` (the LLM safe-open pass succeeded, not the fallback); server log `[OTR_AnnouncerPass] safe-open pass ok (125 chars)`. Open line: *"Evening at Sunny Meadows, where Alice Corben's unexpected revelation hangs in the air like the comet she claims to have seen."* -- orients era/place/cast + raises intrigue, **states no outcome or twist**. | PASS |
| **C3 news CODA** | `meta.story_quality.news_coda_emitted = True`; close line: *"From tonight's headlines: In a historic first, astronomers confirm the detection of an interstellar comet passing through our solar system."* -- the **real news** (the seed fact) delivered, NOT the fictional ending. mistral's dynamic bridge failed the bridge validator both attempts -> `news_coda_fallback = True` -> the sha256(cast_seed) rotating-pool floor picked "From tonight's headlines:" and appended the deterministic fact. Exactly the designed fail-safe. | PASS (via floor) |
| **C4 / grounding** | `meta.story_quality.ungrounded_crisis = {matches: 0, total: 149}` -- **zero** ungrounded crisis nouns across 149 shipped body tokens; `body_gate_reroll=2` fired. The body is a character-driven retirement-home ghost story (Alice Corben, a former comet scientist, + Drake) framing the comet news -- not a console standoff. | PASS |

Pipeline integrity: freeze landed `frozen_with_doctor_edits` (reviewer=improved);
master WAV 78.0 s, `emit audio_done`; **`audio_byte_identical OK (0e97b0c78f21)`**
(the mux-LAST archival-PCM invariant holds); `obs_publish OK ->
otr/obs/signal_lost_comets_trail_20260624_232917_silent_procgen_blended_final.mp4`
(51.8 MB). The new code did not perturb the frozen audio spine or the render path.

## LEG 2 -- gemma-4-12b (Ollama), story_scaffold=ON -- PASS (cross-writer)

First attempt failed in 16.5 s on `OllamaCallFailedError` -- the gemma lane's
`hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M` tag was not pulled (`ollama list`
empty); an ENVIRONMENTAL block, not a C1-C4 regression (the in-process mistral
lane ran clean). Pulled the model (7.3 GB) and re-ran. Ledger
`pending_20260624_235459`; freeze `frozen_with_doctor_edits`.

| Chunk | Live evidence (gemma) | Verdict |
|---|---|---|
| **C1 StoryContract** | `meta.story_contract = {slug: final_message_before_silence, ending_tag: quiet_acceptance}`; `story-grammar ON: style=final_message_before_silence ending_tag=quiet_acceptance climax_beat=b008`. A DIFFERENT non-doomsday style than mistral (fresh per-episode cast_seed -- production rolls OS entropy). | PASS |
| **C2 safe OPEN** | `open_safe_fallback = False`; `safe-open pass ok (169 chars)`. Open: *"Deep within the Vera Rubin Observatory, Quasimodo Bouvier races against the ticking clock while Yuki Voss watches the shadows. Will their data survive the coming strike?"* -- place + cast + intrigue, no outcome. | PASS |
| **C3 news CODA -- DYNAMIC bridge** | `news_coda_emitted = True`, `news_coda_fallback = False` -> the dynamic LLM bridge VALIDATED (compose flag `news_coda_bridge_reroll`, 2nd attempt). Close: *"While we ponder the fate of our stars and final transmissions: Researchers have confirmed the first detection of an interstellar comet passing through our solar system..."* -- gemma's own premise-specific segue + the real news appended. This is the FULL intended C3 behavior the weaker mistral could not reach. | PASS (dynamic) |
| **C4 / grounding** | `ungrounded_crisis = {matches: 0, total: 181}` -- zero ungrounded crisis nouns; `body_gate_reroll=1`. | PASS |

## VERDICT

C1-C4 are **proven live end-to-end on BOTH local writers**. All four behaviors
fired on each; the body grounding is clean (mistral 0/149, gemma 0/181); both
froze `frozen_with_doctor_edits`. mistral published a clean OBS final with
`audio_byte_identical OK` (the writer-agnostic render + mux-LAST spine is
unbroken); gemma additionally proved the **dynamic coda bridge** path. The toggle
flips live (`story_scaffold=on -> OTR_ENABLE_STYLE_GRAMMAR=1 (widget override)`).
The OFF path is byte-identical (unit-proven across +65 tests + the prior bake-off's
live OFF episodes).

Minor content notes (PRE-EXISTING / upstream of C1-C4, not regressions):
1. **News-brief artifact** -- gemma's coda ended with a leaked `"Central object,
   if useful."` clause. This text is in the `news_close_brief` itself (from the
   central-object injection `inject_central_object_into_brief`, writer ~:4003),
   which the coda appends deterministically and verbatim. The fix belongs in the
   brief builder, not the coda. Worth a follow-up scrub.
2. **gemma prose register** -- despite a clean grounding metric (0/181) and a
   quiet_acceptance climax SHAPE, gemma still writes imperative techno-tension
   dialogue ("force the calibration logs", "step back from that console"). The
   model ceiling the plan flagged: the scaffold fixes climax shape + grounding +
   cross-episode sameness, not raw prose tone.
3. **Coda bridge on weak writers** -- mistral's dynamic bridge was rejected both
   attempts and fell to the (correct) sha256 floor; gemma's validated. If a
   dynamic segue is wanted from the weakest writers too, the bridge prompt /
   validator bands are the dial (deferred -- the floor always lands the news).

prod/main + tags remain GATED.
