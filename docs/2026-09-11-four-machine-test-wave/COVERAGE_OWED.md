# Qualification coverage owed

**THIS IS NOT LIVE WORK YET.** Testing does not start until the go-forward plan's
ARC and CODE sections are empty (operator, 2026-09-12: *"let's just do the coding
and arcs first, let's not even talk testing yet"*). This file exists so the
coverage is not LOST while it waits -- it was lifted out of
[GO_FORWARD_PLAN](../GO_FORWARD_PLAN.md) so that plan holds only what is being
built now.

When the plan's first two sections are empty, this page and the two lane
documents beside it are the test wave.

**The head is NOT frozen.** It freezes only when sections 1 and 2 are empty; the
hash then goes into the `WAVE HEAD:` line of
[PROMPTS.md](2026-09-11-four-machine-test-wave/PROMPTS.md) and the four lanes
start. Do not cut a head early.

Copy-paste prompts, one block per machine:
[PROMPTS.md](2026-09-11-four-machine-test-wave/PROMPTS.md) (5080, RunPod) and
[NATIVE_PLAN_4060_AND_MAC.md](2026-09-11-four-machine-test-wave/NATIVE_PLAN_4060_AND_MAC.md)
(4060, Mac). Lane assignments, per-lane legs and the rules every lane follows
live THERE and are not repeated here -- including the 4060's `kernel_source`
phone-home, which is where the Ghost Half B ranked-tier number comes from.

Every qualifying run loads `workflows/otr_canonical.json` through
`scripts/otr_canonical_api_run.py` -- no `--workflow` override, no replay
substitute, no `partial_execution_targets`. Assets go to canonical episode
paths; final publication must exist in `otr/obs`. These are the coverages the
wave does not already carry.

| Coverage | Required evidence |
|---|---|
| Six-act repeatability | Three fresh full canonical runs. Record observed loads/reuse; claim clean boot and resident reuse only where the runtime actually supports it. |
| Source and cast variety | One-act monologue, three-act ensemble, detailed six-act. Requested vs actual cast, supplied/neutral byline, breaks on and off. After the wave, check what its legs already covered and run only the gap. |
| Listening | Opening/middle/ending on at least two publications including a six-act. Operator's ear only -- no agent in the wave can ingest audio, and reading TTS text or checking a waveform is not listening. |
| 2.2 Ghost CUDA | Live five-act forced-Ghost CUDA publication with stored prompt/admission/reuse inspection. Needs a CUDA host. **Also the first live measurement of Half B on a forced-Ghost leg, which the wave does not carry:** count `kernel_source` across the episode's shots -- `authored_subject` (the model ranked it), `key_object_in_beat` (the dialogue named it), `key_object` (odometer). The deterministic tier's ceiling was 26.3%; how far the ranked tier lifts it is the number nobody has. Report the `[OTR_ShotLock] Ghost Half B: eligible=N submitted=N admitted=N candidates=N` line too: admitted/submitted is the admission rate, submitted/eligible the abstention rate -- together they say whether the model is choosing from the list or inventing. |
| 5.7 chunked music | **Settle whether this leg can exist before booking it.** The chunk branch is `scene_sequencer.py:2151-2186` and its threshold is `_MUSIC_MAX_CHUNK_DUR_S = 22.0` (`:2094`), but the composer's own cue lengths are `CUE_DURATIONS = {opening 12, closing 8, interstitial 4}` (`nodes/_otr_music_prompt.py:51`), so a composed cue can never reach it and `_chunk_count` is always 1. The only way in is an AUTHORED ledger music row carrying `target_duration_s > 22`, which `stable_audio_theme.py:372` honours over the default -- and `scifi_news_pro` is the only lane that authors music rows. So: either the leg supplies such a row (full canonical on `otr_8gb_fastwan` -- that is the PROFILE id; `fastwan_8gb` is the ENGINE id inside it and is not a profile -- status `draft`), or the row closes as unreachable code that no episode exercises. |
| Inherited regression debt | **Load-bearing, not a footnote, and right now the guard cannot do its job.** `tests/conftest.py:184` sets `EXPECTED_FAILED_NODEIDS = frozenset()` -- an EMPTY baseline -- so the known-fail guard reports every failure as a new regression and cannot tell one apart from the inherited set. Measured 2026-09-12 on the 5080 in a normal checkout: **49 failed of 15,383 collected**; the count moves with where it runs (some fail only inside a worktree, and a two-file subset still fails the four named in the plan's CODE section). Re-ground the OTR and Bible failure sets against baseline IDs **and normalized payloads**, then load the baseline so a real regression stands out. The Bible strict metadata validator separately has 149 unchanged issues. Never quarantine silently; never call the suite all-green. |

**The six-act fixture** is specified in full in PROMPTS.md, which restates it on
purpose so it can be pasted standalone. For every attempt record source fields,
code/canonical/graph hashes, actual model/quantization/profile, prompt ID,
elapsed time, memory and loads, all repair attempts, requested vs actual acts and
cast, ledger seals and final paths. Preserve terminal failure evidence before
asserting anything. Keep the full denominator.

**After the wave, the next day begins in `otr/obs/`, not in the editor.** Count
what landed against the legs promised, read the four phone-homes, and triage any
crash-class failure FIRST.
