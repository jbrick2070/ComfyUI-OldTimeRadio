VERDICT: build-ready as-is? no. Critical audio-pipeline and freeze-cascade bypasses are omitted, ShotLock re-planning violates determinism, and the proposed ledger write path bypasses the process singleton.

MUST-FIX BEFORE BUILD:
1. [D3 & §4] Missing bypasses for Node 62, Node 83, Node 4, and Node 7, and misattributed master audio ownership.
   - Defect: In workflows/otr_canonical.json, audio execution flows Node 1 -> Node 62 (OTR_LedgerFreezeCascade) -> Node 80 (OTR_CastLock) -> Nodes 81/82/83 -> Node 3 (OTR_SceneSequencer) -> Node 4 (OTR_AudioEnhance) -> Node 7 (OTR_EpisodeAssembler). D3 claims "the voice nodes and the sequencer see meta.replay_from and copy the frozen line WAVs and master mix instead of rendering". SceneSequencer (nodes/scene_sequencer.py:122) does not own or write the master mix; EpisodeAssembler (nodes/scene_sequencer.py:1451-1477) owns and writes output_path (_master_wav). Furthermore, Node 62 (nodes/OTR_LedgerFreezeCascade.py:260-297) is unmentioned in §4 and will load the technical LLM into VRAM and re-mint meta.freeze_timestamp (nodes/_otr_ledger_freeze.py:1097), causing immediate hard rejection downstream in nodes/otr_image_gen_dispatcher.py:594 and nodes/otr_shot_lock.py:184. Node 83 (nodes/stable_audio_theme.py:397) will load MusicGen/Stable Audio onto GPU, and Node 4 (nodes/audio_enhance.py:460) will re-apply DSP filtering.
   - Fix: Centralize frozen audio pass-through at Node 7 (OTR_EpisodeAssembler). When meta.replay_from is set, nodes 62, 80, 81, 82, 83, 3, and 4 must immediately return pass-through stubs without loading models or mutating meta.freeze_timestamp. Node 7 copies the bundle master WAV directly to the new episode directory, assigns output_path, loads the waveform tensor into episode_audio, and emits audio_done.

2. [D3 & §2.2] ShotLock re-planning invokes non-deterministic LLM generation on replay.
   - Defect: D3 assumes "ShotLock re-plans deterministically (same brief, same cast -> same hashes)". In nodes/otr_shot_lock.py:1312-1420, ShotLock executes llm_fn (resolving the writer LLM from meta) to generate creative directives (expression, motion, camera, text_prompt). Re-running ShotLock on replay invokes the LLM non-deterministically unless bypassed, mutating the prompt text and breaking the A/A null requirement in §2.2.
   - Fix: In nodes/otr_shot_lock.py, check meta.get("replay_from"). When present, skip llm_fn execution and creative derivation entirely, reusing the planned video section preserved from the frozen bundle ledger and re-verifying route consistency against the live video policy.

3. [D1 & §1 Fact 3] Disk-only ledger writes bypass the process singleton and risk clobbering.
   - Defect: D1 prescribes that ShotLock write its planned video section to disk via load_ledger_safe / save_ledger_safe. However, nodes/production_ledger.py maintains an in-memory singleton (_CURRENT, accessed via peek_ledger() at line 508). Later nodes in the canonical graph (e.g. OTR_SignalLostVideo at nodes/video_engine.py:2066) invoke Ledger.save(), which merges in-memory data with disk via _merge_with_disk (line 1506). If ShotLock writes only to disk without updating the singleton, any subsequent save risks state desynchronization. Furthermore, save_ledger_safe is fail-soft (returns False), whereas the established contract for required credit/shot receipts is loud failure.
   - Fix: In nodes/otr_shot_lock.py, persist the planned video section using stamp_durable(sections={"video": video_section}, source="OTR_ShotLock") (nodes/production_ledger.py:527). Add "video" to TOP_PRESERVE in nodes/production_ledger.py:1592 so any intermediate Ledger.save() preserves it.

4. [D3 & §1 Fact 5] Missing singleton lifecycle and workspace directory initialization in OTR_LedgerScriptWriter.
   - Defect: When replay_from is provided to Node 1, minting <slug>_replay_<stamp> requires creating a new on-disk workspace under output/otr/episodes/<new_id>/audio/ and updating the process singleton. If Node 1 merely emits JSON strings on the ComfyUI wire without calling production_ledger.new_ledger(episode_id=new_id), downstream functions calling in_flight_ledger_path() (nodes/_otr_ledger.py:515) fail to resolve the active workspace or bind to a stale directory.
   - Fix: In nodes/OTR_LedgerScriptWriter.py, branch on replay_from before randomizer rolls or premise resolution: instantiate the new episode via production_ledger.new_ledger(episode_id=replay_episode_id), populate led.data from the bundle ledger, record meta.replay_from and meta.replay_of_episode, preserve original bank/style rolls and freeze_timestamp, and save the skeleton to disk before returning wire outputs.

SHOULD-FIX:
1. [D1 & §2.1 & §5 Question 2] Redundant schema expansion: top-level render_trace[] vs meta.render_engines.per_clip.
   - Defect: D1 proposes adding render_trace[] to the top-level ledger and TOP_PRESERVE. Concurrently, §2.1 requires adding prompt_sha8 and request_seed to meta.render_engines.per_clip. VideoRenderBatch already builds and stamps meta.render_engines via stamp_durable (nodes/otr_video_render_batch.py:304). Creating a separate top-level list duplicates per-clip receipts, requires schema bumps, and complicates merge logic.
   - Fix: Keep the trace inside meta (e.g. meta["render_trace"] or enriched within meta["render_engines"]["per_clip"]). meta is already merged and persisted across saves via stamp_durable, avoiding modifications to TOP_PRESERVE.

2. [D3 & §4] CastLock auto-registry policy re-rolls cast on replay.
   - Defect: In workflows/otr_canonical.json (node 80), cast_voice_policy is set to "auto_registry". D3 states "CastLock runs in preserve_ledger", but does not specify how this widget is overridden. If left as-is, CastLock will re-evaluate cast pools and potentially re-assign voices.
   - Fix: In nodes/cast_lock.py, check if meta.get("replay_from") is present. If so, force policy to preserve_ledger internally regardless of the widget value, logging the enforcement.

3. [D5 & §2.3] Premature blinding and scorecard harness in Campaign Item 0.
   - Defect: §2.3 and D5 introduce neutral-title blinding, _blind_key.json, and a 3-arm comparison scorecard into scripts/otr_canonical_api_run.py. The primary goal of Item 0 is proving receipt persistence and achieving an A/A null replay. Coupling multi-arm blinding infrastructure into this PR inflates test surface and delays receipt verification.
   - Fix: Scope Item 0 strictly to --replay-from and A/A null equality verification. Defer automated key withholding and subjective scorecards to the subsequent evaluation campaign.

OPTIONAL / NICE-TO-HAVE:
1. [D4] Asset pre-validation in scripts/otr_freeze_replay_bundle.py.
   - Ensure the freeze script explicitly verifies that all referenced still images, character portraits, and audio files exist and have non-zero size before generating manifest.json.

CUT THESE (scope / over-engineering):
1. [D3] Per-line WAV file copying across voice nodes.
   - Why safe to cut: VideoRenderBatch already supports slicing beat audio directly from master_audio_path (nodes/otr_video_render_batch.py:420-426, nodes/_otr_video_engines/render_driver.py:4930-4935). Copying dozens of individual line WAV files between workspaces during replay is unnecessary I/O; passing the verified master WAV via Node 7 satisfies all downstream audio motion and muxing requirements.
2. [D1 & §5 Question 2] Top-level render_trace in TOP_PRESERVE.
   - Why safe to cut: Placing render_trace under meta leverages existing recursive meta preservation without modifying TOP_PRESERVE or altering top-level schema validation.
