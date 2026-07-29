VERDICT: no. The document combines two different policies—native beat coverage and presentation-layer still holds—without defining a measurable motion contract, resolving the WAN exception, or covering explicit still engines.

MUST-FIX BEFORE BUILD:

1. [The ruling, claims 1-3 / WHERE IT BITES #1] The stated scope does not include its lead regression. Claims 1 and 3 govern beats rendered by a video model, but credits are produced by `OTRCreditsRoll`, not a `VideoEngine` beat (`nodes/_otr_video_engines/registry.py:53-107`; `nodes/otr_credits_roll.py:1361-1384`). Concrete fix: state two separate invariants: (a) video-model beats receive exact native-render coverage; (b) no presentation component—including the credits backdrop—may hold one source frame longer than four seconds.

2. [The ruling, claim 2] “Somewhere around 2-4 s” is not enforceable, and “real moving video” does not distinguish native forward render, still pan, loop, reverse/ping-pong, or scrolling overlay. The registry includes `still_motion`, `still_pan`, deliberately dead-flat `still_flat`, and dead-flat `still_word` as video-engine choices (`nodes/_otr_video_engines/cheap_families.py:274-377`). Concrete fix: pin four seconds as `round(4 * final_fps)` and publish a route matrix defining which engine families owe native motion, which explicit still routes are legal, and which repetition modes count.

3. [WHERE IT BITES #2, case 1] The claimed new behavior already exists. For any contract permitting tail trim, `partition_beat` renders the smallest legal length at or above the target and trims the excess (`nodes/_otr_video_engines/coverage_plan.py:287-297`). Discrete menus are already required to permit trimming (`nodes/_otr_video_engines/frame_contract.py:156-167`). Concrete fix: remove “target below min_frames” as a new implementation chunk; instead audit that every affected adapter declares the correct contract and that execution preserves the stamped render/trim plan.

4. [WHERE IT BITES #2 / WIRE-W3b] The universal motion claim and the WAN 8-GB survival contract contradict each other. `coverage_plan` forbids ping-pong as coverage (`nodes/_otr_video_engines/coverage_plan.py:1-6`), while `wan_ti2v` deliberately mirror-extends a VRAM-limited native render (`nodes/_otr_video_engines/eng_wan_ti2v.py:695-735`). `PBUG-20260723-02` explicitly preserves that behavior and excludes WAN from planning-cap splitting (`docs/PROD_BUG_LOG.md:2679-2719`). Concrete fix: choose one policy before WIRE-W3b: either native multi-render coverage supersedes the 8-GB exception, or WAN ping-pong remains a named `repeated_motion` compatibility mode that does not claim strict native coverage. Do not let one receipt claim both.

5. [WHERE IT BITES #2 / What is already built] The plan leaves unavoidable still ownership undefined. A still emitted for a beat frozen to a video engine either lies about the delivered `engine_id` or conflicts with WIRE-W5’s planned per-shot frozen-route comparison (`kibitz-runs/2026-07-28-local-engine-obs-wiring/r4/final.md`, A6). Concrete fix: for video-model routes, an uncoverable beat must refuse rather than silently become a still; explicit still engines need their own declared cadence policy and receipt.

6. [WHERE IT BITES #1] The credits proposal remains a menu of unresolved creative choices, not a design. A short arbitrary loop can repeat excessively, include a static fade-out, or expose a hard seam. Concrete fix: use the longest available closing body slice: `N = min(body_duration, credits_duration)`. Play it forward once; loop only when the body is shorter than the roll, with a short crossfade at the seam. Keep extraction, looping, and encoding inside the existing presentation-only boundary at `nodes/otr_credits_roll.py:1421-1447`.

7. [What the panel is being asked for / WIRE-W5] Route and frame-count receipts cannot prove the backdrop moved. The current credits test intentionally proves scrolling text over a constant backdrop (`tests/test_credits_roll_spec.py:446-470`), so a whole-frame motion check would pass the exact frozen-background defect. Concrete fix: grade source components before overlays: native/rendered frame provenance per beat, extension mode, longest repeated-frame run, and credits-backdrop motion separately. OBS publication and engine-ID matching remain independent checks.

8. [Title / What is already built] “And what it costs” contains no cost or latency envelope. The difficult contracts are provider minimums—Veo 100/150/200 frames (`nodes/_otr_video_engines/eng_google_veo_video.py:503-520`) and Pixverse 125/200 (`nodes/_otr_video_engines/eng_cloud_video.py:949-960`)—but the proposed live proof covers only 18 local engines, while cloud engines are explicitly parked pending spend approval (`docs/GO_FORWARD_PLAN.md:970-971`). Concrete fix: add deterministic no-provider contract tests now, quantify discarded rendered seconds per beat, and keep live cloud qualification separately gated. [ASSUMPTION] Verify provider billing before assigning monetary totals.

SHOULD-FIX:

1. [WHERE IT BITES #1, failure-boundary question] This is already answered: body/ledger truth is established above the presentation try, while extraction, rendering, and append failures return the body with a zero tail (`nodes/otr_credits_roll.py:1396-1447`). Remove it as an open design question; require new backdrop work to remain inside that boundary.

2. [WHERE IT BITES #2, audio-in question] Do not reopen mouth-still ownership here. `accepts_still` governs init-image minting and is explicitly distinct from continuity (`nodes/_otr_video_engines/frame_contract.py:13-18`). State that WIRE-W7 is unchanged unless a concrete execution path substitutes the init still for rendered frames.

3. [ComfyUI contract] `OTRCreditsRoll` reads the body file and global production ledger but defines no `IS_CHANGED` (`nodes/otr_credits_roll.py:1361-1414`). [ASSUMPTION] If episode runs can reuse the same path and manifest string, ComfyUI can cache stale credits. Add invalidation keyed to body-file identity and ledger revision, or prove inputs are unique per run.

4. [Canonical workflow] The existing node is registered and wired as node 95 with no widgets (`__init__.py:319-322`; `workflows/otr_canonical.json`, node 95). Derive N internally. If a new control is unavoidable, append it as the final optional widget and update the canonical workflow in the same chunk.

OPTIONAL / NICE-TO-HAVE:

Record credits source offsets, loop count, crossfade duration, longest static run, and backdrop classification in the credits report so the published artifact explains exactly how it passed.

CUT THESE (scope / over-engineering):

1. [What the panel is being asked for] Cut exact file ordering from R1. First freeze the policy matrix and acceptance semantics; file/chunk sequencing belongs in R2/R3.

2. [WHERE IT BITES #1] Cut synthesized “slow drift” after the tail. It converts a still into cosmetic movement, adds another visual authority, and weakens the native-motion rule.

3. [WHERE IT BITES #2] Cut “several short stills” as an automatic fallback. Repeating or renaming stills merely games the four-second bound; explicit still-engine cadence must be a separate declared route.

4. [What is already built] Cut WIRE-W4 audio-slicer and WIRE-W7 mouth-ownership redesign from this campaign. Preserve their existing specifications; this ruling directly changes credits, coverage semantics, WAN repetition, and WIRE-W5 acceptance.
