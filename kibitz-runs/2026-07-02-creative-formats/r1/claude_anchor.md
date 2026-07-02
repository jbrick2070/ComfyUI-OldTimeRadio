# r1 Claude anchor (arc / creative coherence) -- CREATIVE_FORMATS_PLAN pass00

VERDICT: yes-with-fixes. The two formats are coherent, era-true, and
ride proven machinery; the arc-level gaps are an unresolved switch
mechanism with a known drift hazard, an undefined cache lifetime word,
and a board-layer contradiction.

MUST-FIX:
1. [2 F1-d / 3 F2-e] The format-switch mechanism is punted ("widget OR
   episode toggle -- decide at wiring round") but it is ARC-load-
   bearing: a NEW WIDGET on OTR_VideoDirector changes widgets_values
   POSITIONALLY (BUG-LOCAL-097 class) -- legal ONLY as an APPEND-at-end
   optional widget in the same change as the JSON (standing rule). The
   plan must state that constraint NOW so r3 designs within it, and
   must prefer one mechanism as default (lean: append-at-end widget on
   the policy node; episode-level env override for headless).
2. [2 F1-c] kling_lipsync consumes a BASE CLIP (base_clip_ref +
   audio_ref -- pinned schema), not a still. The polaroid crop must
   become a short still-video (trivial local ffmpeg loop at role fps)
   BEFORE the lipsync call. One sentence, but without it the chain as
   written does not run.
3. [3 F2-b / 6] "Per season" is not a concept the repo has (episodes
   exist; seasons do not). The cache lifetime is ALREADY correct via
   keying (portrait hash + tin_toy profile version = re-mint exactly
   when the character design changes); DELETE the word "season" and
   state the key-driven lifetime plainly.
4. [2 F1-a] Board cache contradiction: "re-billed only when the cast
   changes" vs per-episode CLUES pinned to the board. Resolve as two
   layers: (a) cast-polaroid layer cached per portrait-hash set;
   (b) episode dressing layer (clues/notes) minted per episode via
   cheap Flux fill on top of the cached base. State the split and
   which layer each credit touches.
5. [5] Local-compositor shots (board pans, Blender toy plates) BYPASS
   the video engine registry -- no engine row renders them. The plan
   must state their citizenship explicitly: either the formats register
   as (local, zero-cost) engine rows so ShotLock/ledger/reactivity
   classification still see every shot, or the bypass is sanctioned
   and stamped in the ledger as FORMAT_LOCAL_SHOT. Silent bypass of
   the policy layer is the one thing r1 cannot leave open.

SHOULD-FIX:
1. [4 V2] Extend: verify Meshy rig/animate EXPORT FORMAT is Blender-
   importable (GLB w/ baked animation vs proprietary) -- the F2 chain
   dies quietly here if not.
2. [2 F1-b] Note output canvas: crops composite at the role canvas
   (1472x832), so the 8K board only ever streams as crops -- no
   full-8K frame renders; keeps the camera desk trivially cheap.
3. [7] Name the F1 acceptance episode: reuse an existing regression
   script (e.g. a 30w smoke) in board format so acceptance is
   comparable to prior lanes.

CONFIRMED against repo/pin: kling_lipsync inputs (pinned yaml:
base_clip_ref-shaped input + audio; verify exact input names at
adapter build), MeshyRig/Animate/MultiImage nodes present (live dump),
Blender 4.5.10 shipped (0-E Phase A), widgets_values positional rule
(BUG-LOCAL-097), portrait-hash chain live. UNVERIFIABLE here: LTX
outpaint 8K stitch quality (V3 stands), Kling-on-tin-face texture
behavior (V1 stands, correctly ordered FIRST).
