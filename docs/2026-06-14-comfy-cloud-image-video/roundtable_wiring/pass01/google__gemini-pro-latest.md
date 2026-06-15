<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The adapter pattern and role compatibility are correct, but the lease-skip logic and episode-level cost guard have critical gaps in the dispatcher.

MUST-FIX BEFORE BUILD:
1. [Section 4.1] Unconditional NVML probe stall. `dispatch_images` unconditionally calls `_lease.wait_until_below_mb(15000)` after the `finally` block. If you skip the lease for network engines, you MUST also conditionally skip this probe (e.g., `if not is_network:`), otherwise the dispatcher will stall polling the local GPU after every cloud render.
2. [Section 4.2] Cross-phase budget tracking gap. You propose a "per-episode credit ceiling" enforced in the dispatcher. `dispatch_images` runs before the video dispatcher and only knows about images. To enforce an episode-level ceiling, `dispatch_images` MUST read and write a running `credits_spent` total to the `ledger` (e.g., `ledger["billing"]`), so the downstream video dispatcher knows how much budget remains.
3. [Section 4.2] Image dispatcher lacks duration. The plan claims the dispatcher has "w/h/duration/engine there" for the cost estimate. `dispatch_images` does NOT have `duration`, `fps`, or `target_frame_count` in its request object (it only processes stills). The image cost estimate must be strictly per-run/per-image, not per-second.

SHOULD-FIX:
1. [Section 2a] Unnecessary disk-write constraint. The plan forbids returning a ComfyUI IMAGE tensor and forces writing a PNG to disk. However, `flux_gen1.py` successfully returns an in-memory numpy array by calling `_wb.images_to_uint8(images)[0]`. If the Comfy API node yields an IMAGE tensor, the adapter should just convert it to a numpy array and return it, avoiding the disk I/O and `wait_for_file_ready` timeout risks entirely.
2. [Section 3] Phantom image CAPABILITIES. The plan says to add a `CAPABILITIES` row in `_otr_image_engines/registry