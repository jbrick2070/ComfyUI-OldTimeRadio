# r3 Claude anchor (wiring / integration / sequencing)

VERDICT: yes-with-fixes. Contracts are now concrete; the wiring
residue is where the stamping phase runs in the graph, who mints the
board BEFORE video legs, and the format_ctx schema-version handshake.

MUST-FIX:
1. [1b FORMAT CONTEXT] WHERE does board mint run in the GRAPH? The
   stamping phase is named (ShotLock/ImageGenDispatcher) but the board
   mint consumes STILLS-lane outputs (cast polaroids) and must finish
   BEFORE any video leg renders. Wiring: board mint belongs to the
   IMAGE phase (ImageGenDispatcher emits the board manifest as part of
   image_done), so the existing image_done -> video gating carries the
   dependency with ZERO new gates. State it.
2. [1b visual_format] The widget lands on OTR_VideoDirector (append-
   only) -- but headless `OTR_VISUAL_FORMAT` must be read at the SAME
   resolution point (direct()) so widget and env produce identical
   policy JSON; profile applier must NOT also implement it (one
   resolution point, no double-apply).
3. [1b format_ctx] Schema versioning handshake: format_ctx carries
   `format_ctx_version`; eng_evidence_board.assert_usable rejects a
   mismatched version LOUDLY (unsupported_schema) -- prevents silent
   drift between the stamping phase and the engine across commits.
4. [3 F1-c] The lipsync sub-request needs the SESSION: render_clip
   runs inside VideoRenderBatch's execution; the S0 session is keyed
   by prompt_id via hidden inputs on the BATCH node -- the format
   engine reaches the session through the invoke bridge's internal
   resolution (pass04 sec 3), NOT by receiving auth itself. Confirm
   VideoRenderBatch is in the hidden-input list for S3 (it is, per
   pass04 sec 2) and note the format engine needs NOTHING extra.
5. [7] Sequencing correction: F1 needs S1 (stills) + the kling row --
   which is an S3 deliverable. State that F1 may begin when S3's
   kling_lipsync adapter EXISTS, not when all of S3's matrix
   acceptance is done (narrower gate, earlier start, zero risk since
   F1 exercises exactly one row).

SHOULD-FIX:
1. Golden 30s sample scripts live with the format engines' tests
   (tests/goldens/formats/) and run under OTR_RUN_CLOUD_SMOKE=1 (they
   spend Kling credits).
2. The visual_format widget-vector test must cover the SAVED workflow
   round-trip (load full.json, append widget, save, reload -- values
   stable) per the S4 same-change rule.
3. board_manifest.json is an EPISODE asset (placement directive) AND
   an input to render -- write-once at image phase, read-only after
   (stamp its sha into the ledger for audit).
