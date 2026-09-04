VERDICT: yes-with-fixes. The migration breaks real standalone provisioning loaders; those import regressions block acceptance of the semantics-neutral claim.

MUST-FIX BEFORE BUILD:
1. [Hunt 1, 3] CONFIRMED — The new environment imports break standalone provisioning and portable voice-bank generation. nodes/_otr_model_catalog.py:26 and nodes/_otr_voice_bank.py:36 fall back to importing _otr_shared, but their callers load them under nonpackage names without exposing nodes/ on sys.path: scripts/otr_provision.py:638, scripts/otr_provision.py:963, and scripts/otr_make_portable_voice_bank.py:106. All three real helpers fail in fresh Windows interpreters with “No module named '_otr_shared'”; both pre-migration authority modules load successfully. Fix the standalone loaders to establish the correct package context before importing these authorities. Add fresh-interpreter coverage for all three helpers. Existing tests exercise them inside the shared pytest process, masking the missing import context (tests/test_otr_pod_runtime.py:583; tests/test_make_portable_voice_bank.py:17).

SHOULD-FIX:
1. [Hunt 1, 3] CONFIRMED — The blend node has the same import-order regression. nodes/otr_post_upscale_procgen_blend.py:49 imports proc before its own sys.path bootstrap at :60. An isolated flat load succeeds before the migration and fails afterward. Move the proc import below that bootstrap and test the documented flat-load mode. Normal package loading is unaffected; tests/test_post_upscale_procgen_blend.py:36 uses that successful path.

2. [Hunt 4] CONFIRMED — The executable allowlist checks the nominal command, while executable= can replace the actual binary. Both wrappers in nodes/_otr_shared/proc.py:147 and :154 accept (['ffmpeg'], executable='cmd.exe') and forward it to subprocess. Verified with mocked spawn functions; nothing was launched. No current migrated caller uses this override, so this is a boundary-contract hole rather than an established render failure. Smallest fix: refuse non-None executable= overrides on both entry points and test refusal before spawning. tests/test_shared_env_and_proc_owners.py:220 tests keyword forwarding but omits this case.

3. [Hunt 5] CONFIRMED — The ratchets miss the bypass forms explicitly requested for review. tests/test_env_single_owner.py:127 and tests/test_process_single_owner.py:112 return no findings for rebound module aliases, literal getattr access, star imports, and importlib.import_module calls. The process guard also misses “run = subprocess.run; run(...)” and calling the unguarded identity export otr_proc.Popen, defined at nodes/_otr_shared/proc.py:58. Both baseline scans pass because their finder fixtures omit these forms. Add bounded detection or explicit rejection for these spellings, including calls through the Popen export while preserving its identity for annotations/type checks.

4. [What was done; Hunt 1] CONFIRMED — Document the intentional behavioral exception to neutrality: nodes/_otr_shared/gpu_residency.py:94 now treats an unknown Windows PID as alive when psutil fails, disabling stale-lease reclamation. This is explicitly implemented and tested in tests/test_gpu_residency_pid_liveness.py; the review brief should acknowledge it.

OPTIONAL / NICE-TO-HAVE:
None.

CUT THESE:
1. [Hunt 5] Cut any requirement for a general Python dataflow analyzer. Explicitly cover the demonstrated bypass spellings and state the ratchet’s limits; arbitrary runtime reflection detection is unnecessary for this migration.

VERIFY-AT-BUILD checklist:
- [R4 input] UNVERIFIABLE — Earlier-round UNVERIFIABLE flags are not included in kibitz-runs/2026-09-04-prepublish-collapse/r4/input.md. The judge must reconcile its carry-forward list; completeness cannot be certified from this input.
- [Hunt 1, 3] Verify all three standalone provisioning helpers and the blend flat import in fresh interpreters without inherited nodes/ paths or cached _otr_shared modules.
- [Hunt 2] Verify deployment-resolved ffmpeg, ffprobe, sidecar Python, and Blender paths through the real process boundary. No second concrete rejected binary was established; the static receipt in docs/2026-09-04-registry-findings-collapse/argv0_receipt.txt does not prove runtime resolution.
- [Hunt 4, 5] Verify the added refusal and finder cases fail before their fixes and pass afterward; rerun the existing owner/ratchet tests and required regression gates.
- [What was done] Verify the packaged install cold-boots and completes workflows/otr_canonical.json through an existing final artifact in otr/obs/, as required by docs/PRODUCTION_SPRINT_LESSONS.md §7. No live-render result was established by this review.
