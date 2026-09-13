# Apple Silicon

It works, and the short version is that you do not have to do anything special.
Install as in [INSTALL.md](INSTALL.md), open a Mac graph, press Queue. Four
ship, and each has published an episode on a Mac mini M4 / 16 GB:

```
workflows/variants/otr_mac16_low.json          procedural lanes, no weights
workflows/variants/otr_mac16_still.json        sd15 stills with motion
workflows/variants/otr_mac16_video.json        ltx098 video diffusion
workflows/variants/otr_mac16_animatediff.json  SD 1.5 motion from the prompt
```

Start with `otr_mac16_low` if you want the fastest proof that it works at all.
Each is the canonical with Mac-appropriate dropdowns already saved. The
canonical itself also runs — it names no vendor and resolves your device at run
time — but these pick the lanes that have receipts here.

(`otr_mac_mps` is the PROFILE those are cut from, not a file you open.)

What runs on a Mac and what does not is the **Mac 16 GB** column in
[MACHINES.md](MACHINES.md). That table is generated from the same data the code
uses, so it is the one to trust; this page is only the things that are different
*because* it is a Mac.

---

## The one warning that matters

**An out-of-memory on Apple Silicon can take the machine down, not just the
render.** Unified memory means the GPU allocation and the system's are the same
pool, so an overcommit does not fail politely the way a discrete card's does —
it can reboot you.

This is why the Mac column is conservative, and why lanes that merely *fit* on a
16 GB NVIDIA card read as **not offered** here. They are not omissions.

## The writer is the memory hog, not the video

Counter-intuitive but consistently true: the lane most likely to push a 16 GB Mac
over is the language model writing the script, not anything that draws pixels.
The Mac graph enforces a **10 GB ceiling** on it for that reason, and ships
`Qwen/Qwen3.5-4B` unquantized.

**Close other applications before a run.** A browser with many tabs is a real
factor on a 16 GB machine, and it is the cheapest thing you can change.

## Images

The Mac graph ships **SD 1.5**, which downloads itself on first use — one 2 GB
checkpoint, ungated, no account.

Z-Image-Turbo is what the NVIDIA graphs use and is a ~19 GB download; leaving it
selected on a Mac with a still-consuming video lane starts that download. If you
did not mean to, that is the usual cause.

## Video

The Mac graph ships **three procedural lanes** — they draw their own frames and
download nothing. All four `still_*` lanes are proven here too, plus the
AnimateDiff Lightning lane.

**LTX 0.9.8 is a swap, not a default.** It is real video diffusion and it is
proven on Apple Silicon, but selecting it starts a ~16 GiB download. It is
image-to-video, so it consumes the SD 1.5 still — which the Mac graph already
selects, so that half costs you nothing extra.

## Python and voices

3.12 or earlier runs Kokoro on torch; 3.13 runs it through `kokoro-onnx` on the
CPU automatically; 3.14 has no Kokoro build and is refused rather than half
working. Details in [INSTALL.md](INSTALL.md) section 5.

Kokoro is the shipped voice on every platform. The cloning engines need a
Windows-only installer and are not available here.

## Fonts

Captions and credits need a real font. macOS has plenty, so this is normally a
non-issue — unlike Linux, where a headless image often has none.

---

## Going deeper

`docs/MAC_PORTABILITY_GUIDE.md` in the GitHub tree is the full record — every
measurement, every dead end, and the reasoning behind each row above. It is a lab
notebook rather than a guide, which is why the useful half is here instead. It
does not ship in a Manager install; read it on GitHub.

## Shipping set, 2026-09-13

Tested `8fd43146e776e17b84abe75acacc90ffc65bae85` on the 16 GB M4 Mac,
using Python 3.13.12, torch 2.12.1, MPS, and ComfyUI 0.35.1. The four
shipping graphs ran sequentially, one act each, with a 10,800-second timeout
per leg and no title override. Run `shipping_set_20260913_051313` finished
with `DONE 07:15:27` PDT. Minutes below are the harness's integer values.

Verified local runtime paths:

```text
REPO=/Users/rentamac/Documents/otr-mac/repo
COMFY=/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI
PY=/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/bin/python
OBS=/Users/rentamac/ComfyUI-Shared/output/otr/obs
URL=http://127.0.0.1:8188
```

OBS follows this server's `--output-directory` override. AnimateDiff-Evolved
was already installed; the Lightning fetcher verified the checkpoint, motion
module, and VAE before boot.

**The canonical Mac launch line, found the hard way on 2026-09-13** (a bare
restart dropped both of these and cost two wasted diagnostic passes -- see
the animatediff entries below): a plain `python main.py --listen 127.0.0.1
--port 8188` boots with an EMPTY `models/checkpoints/` and `models/vae/`
(only placeholder files) and publishes to `$COMFY/output/otr/obs` instead of
the shared one. Both flags are required together:

```
python main.py --listen 127.0.0.1 --port 8188 \
  --extra-model-paths-config "$HOME/Library/Application Support/Comfy Desktop/instance-model-paths/inst-<id>.yaml" \
  --output-directory /Users/rentamac/ComfyUI-Shared/output
```

The `inst-<id>.yaml` name is per-install (find it with `ls "$HOME/Library/
Application Support/Comfy Desktop/instance-model-paths/"`); it already maps
`checkpoints`, `vae`, `animatediff_models`, and `animatediff_motion_lora` to
`/Users/rentamac/ComfyUI-Shared/models`, so `config/otr_mac_extra_model_paths.yaml`
does not need to be filled in or passed alongside it for those categories.
Verify before spending a leg on it, per the API rather than eyeballing
folders: `curl -s http://127.0.0.1:8188/object_info/CheckpointLoaderSimple`
should list the checkpoint under `ckpt_name`.

| Graph | RESULT | Minutes | OBS filename |
| --- | --- | ---: | --- |
| otr_mac16_low | SUCCESS | 15 | `the_unbroken_wrist_20260913_052629__vart__vcam__none__koko__orig__q354b__mgen_final.mp4` |
| otr_mac16_still | SUCCESS | 26 | `the_festival_in_washington_20260913_054210__vstb__stmo__sd15__koko__marc__q354b__mgen_final.mp4` |
| otr_mac16_video | SUCCESS | 56 | `the_lady_in_the_nightdress_who_guards_th_20260913_060712__arch__lx8g__sd15__koko__pubd__q354b__mgen_final.mp4` |
| otr_mac16_animatediff | FAIL | 23 | None — no pass |

The three successful files were created during their respective legs and each
contains H.264 video and AAC audio (86.96, 100.52, and 75.80 seconds). Both
the leg log and ComfyUI log were read at each leg's five-minute gate when
no OBS MP4 had landed, with subsequent checks recorded in the run directory.

`otr_mac16_animatediff` failed at node 92 (`OTR_VideoRenderBatch`), shot
`shot_music_opening_001`, engine `animatediff15_lightning_video`. All eight
sampling steps completed in 5:11 with 88 latents and a 16-frame context;
the subsequent VAE decode ran out of MPS memory. The failed allocation was
2.67 GiB, with 14.29 GiB allocated by MPS, 3.65 GiB in other allocations,
and a 20.13 GiB limit. The wrapper reported `FailureKind.INVALID_DAG`; the
underlying exception is the MPS allocation failure shown below. No OBS MP4
was published for this leg. ComfyUI remained alive and the queue was empty
after the harness finished. No source fixes or memory-limit changes were
applied during the set.

Logs: `otr/legs/shipping_set_20260913_051313/otr_mac16_animatediff.log` and
`otr/legs/mac_comfyui_20260913.log` in the checkout above. The complete
chained traceback is included for the 5080 git handoff.

## otr_mac16_animatediff single-leg re-test, 2026-09-13 (afternoon)

Handoff from a Codex session that ran out of credits mid three-act rung. That
rung (`shipping_set_20260913_111821`) finished `otr_mac16_low` (SUCCESS, 30
min) and `otr_mac16_still` (SUCCESS, 54 min), then `otr_mac16_video` got stuck:
ComfyUI's server process had exited (no crash log found, nothing listening on
`:8188`) while `scripts/otr_canonical_api_run.py` kept polling and logging
`status=pending` every 5s regardless, because it does not distinguish a
connection failure from a still-pending prompt. That worker and its harness
shell were killed manually; no source was touched.

`git pull --rebase origin v2.0-alpha` then fast-forwarded `d1a81d16` ->
`bda17f48487576146d6cf1fe6515203ca37e494233` (30 files). ComfyUI was fully
restarted (old process confirmed gone, port confirmed free, then relaunched)
so the new modules were actually loaded before the next leg, per the
handoff's explicit warning that a plain `git pull` does not reach a server
that already imported the old code at boot.

One leg, one act, ran clean against the fresh server:

```
OTR_ACT_COUNT=1 OTR_LEG_TIMEOUT=9000 scripts/otr_shipping_set_legs.sh \
  http://127.0.0.1:8188 /Users/rentamac/ComfyUI-Shared/output/otr/obs \
  .../.venv/bin/python otr_mac16_animatediff
```

| Graph | RESULT | Minutes | OBS filename |
| --- | --- | ---: | --- |
| otr_mac16_animatediff | FAIL | 13 | None — no pass |

**This did not retest the MPS chunked-decode fix.** The leg never reached VAE
decode: it failed at `assert_usable` for shot `shot_music_opening_001`,
engine `animatediff15_lightning_video`, `FailureKind.DEPENDENCY_MISSING` --
`checkpoint=v1-5-pruned-emaonly-fp16.safetensors` (folder_paths category
`checkpoints`) and `decoder=vae-ft-mse-840000-ema-pruned.safetensors`
(folder_paths category `vae`) not found, no fallback, nothing downloaded at
render time. The `[ghost-signal] decoded N frame(s) in M chunk(s)` line the
handoff asked us to watch for is **absent** -- the engine never got past its
own usability check to attempt a decode.

Checked by hand (report only, no fix applied): the checkpoint file exists on
disk, but in the HuggingFace hub cache
(`models/huggingface/hub/models--Comfy-Org--stable-diffusion-v1-5-archive/snapshots/.../v1-5-pruned-emaonly-fp16.safetensors`),
not under a `folder_paths` `checkpoints` directory the loader will see. No
`vae-ft-mse-840000-ema-pruned.safetensors` was found anywhere under
`models/` in this search. The 2026-09-13 05:13 run of this same graph noted
"the Lightning fetcher verified the checkpoint, motion module, and VAE before
boot" and got past this point to the MPS OOM this fix targets -- so the
model-path resolution for this engine now fails where it previously passed,
on the same machine, same day. Peak MPS allocation: not applicable this run
(no decode attempted); ComfyUI stayed alive after the failure, queue empty.

<details>
<summary>otr_mac16_animatediff traceback (2026-09-13 afternoon, missing-model failure)</summary>

```text
[ERROR] [OTR video] render FAILED (no fallback) shot shot_music_opening_001 engine animatediff15_lightning_video: EngineUnusable: video engine 'animatediff15_lightning_video' is not usable for role 'text_to_video': missing_model -- animatediff15_lightning_video artifact(s) not found: checkpoint=v1-5-pruned-emaonly-fp16.safetensors (folder_paths category 'checkpoints'), decoder=vae-ft-mse-840000-ema-pruned.safetensors (folder_paths category 'vae') -- drop them in the matching folder or register it in extra_model_paths.yaml. There is no fallback list and nothing is downloaded at render time.
Traceback (most recent call last):
  File ".../nodes/_otr_video_engines/render_driver.py", line 4726, in render_shot
    clip = _render_one(eng, request, force_oom=force, host_caps=host_caps, profile=profile, segment=segment)
  File ".../nodes/_otr_video_engines/render_driver.py", line 4461, in _render_one
    eng.assert_usable(host_caps=_caps, profile=_prof, request_template=request)
  File ".../nodes/_otr_video_engines/eng_ghost_signal_lightning.py", line 352, in assert_usable
    super().assert_usable(host_caps, profile, request_template)
  File ".../nodes/_otr_video_engines/eng_ghost_signal.py", line 1017, in assert_usable
    raise EngineUnusable(...)
comfyui-old-time-radio.nodes._otr_shared.engine_registry_base.EngineUnusable: video engine 'animatediff15_lightning_video' is not usable for role 'text_to_video': missing_model -- checkpoint=v1-5-pruned-emaonly-fp16.safetensors (folder_paths category 'checkpoints'), decoder=vae-ft-mse-840000-ema-pruned.safetensors (folder_paths category 'vae')
```

</details>

Logs: `otr/legs/shipping_set_20260913_131946/otr_mac16_animatediff.log` in the
checkout above. No source fixes or memory-limit changes were applied.

## otr_mac16_animatediff second re-test: chunked MPS decode fix CONFIRMED, 2026-09-13

Root cause of the DEPENDENCY_MISSING failure above, found jointly with the
5080 window: my restart in the previous entry launched bare `main.py
--listen 127.0.0.1 --port 8188` with no `--extra-model-paths-config` flag,
so `models/checkpoints/` and `models/vae/` under `ComfyUI-Installs/ComfyUI/
ComfyUI/` were empty (only `put_checkpoints_here` / `put_vae_here`
placeholders) and folder_paths had nothing to resolve against. Not a code
regression -- `nodes/_otr_video_engines/eng_ghost_signal.py`'s
`_resolve_model_file_by_token` is a flat `folder_paths.get_full_path` call,
untouched by any commit that day. The real checkpoint and VAE files exist,
just outside any registered folder: `/Users/rentamac/ComfyUI-Shared/models/
checkpoints/v1-5-pruned-emaonly-fp16.safetensors` and `.../models/vae/
vae-ft-mse-840000-ema-pruned.safetensors`. No repo config file needed
editing: the Comfy Desktop generated file at `~/Library/Application
Support/Comfy Desktop/instance-model-paths/inst-1788817218536.yaml` already
maps `checkpoints`, `vae`, `animatediff_models`, and `animatediff_motion_lora`
to that shared folder as `base_path`.

Relaunched:

```
python main.py --listen 127.0.0.1 --port 8188 \
  --extra-model-paths-config "$HOME/Library/Application Support/Comfy Desktop/instance-model-paths/inst-1788817218536.yaml"
```

Verified before spending a leg on it, per the API rather than eyeballing
folders:

```
curl -s http://127.0.0.1:8188/object_info/CheckpointLoaderSimple | python3 -c "..."   # -> True
curl -s http://127.0.0.1:8188/object_info/VAELoader | python3 -c "..."                # -> True
```

Re-ran the same leg:

| Graph | RESULT | Minutes | OBS filename |
| --- | --- | ---: | --- |
| otr_mac16_animatediff | SUCCESS | 65 | `the_sealed_compact_of_lost_lands_20260913_135656__anim__adlt__none__koko__sspr__q354b__mgen_final.mp4` |

**This confirms the chunked MPS decode fix.** All 8 beats (6 story shots
`shot_b001`-`shot_b006` plus the opening and closing music-visual beats)
went through AnimateDiff Lightning sampling and VAE decode with zero MPS
OOM, each logging the target line:

```
[ghost-signal] decoded 88 frame(s) in 22 chunk(s) of 4 at 36x64 latent -- Apple silicon decode budget    (shot_music_opening_001)
[ghost-signal] decoded 76 frame(s) in 19 chunk(s) of 4 at 36x64 latent -- Apple silicon decode budget    (shot_b001)
[ghost-signal] decoded 124 frame(s) in 31 chunk(s) of 4 at 36x64 latent -- Apple silicon decode budget   (shot_b002)
[ghost-signal] decoded 136 frame(s) in 34 chunk(s) of 4 at 36x64 latent -- Apple silicon decode budget   (shot_b003)
[ghost-signal] decoded 136 frame(s) in 34 chunk(s) of 4 at 36x64 latent -- Apple silicon decode budget   (shot_b004)
[ghost-signal] decoded 88 frame(s) in 22 chunk(s) of 4 at 36x64 latent -- Apple silicon decode budget    (shot_b005)
[ghost-signal] decoded 64 frame(s) in 16 chunk(s) of 4 at 36x64 latent -- Apple silicon decode budget    (shot_b006)
[ghost-signal] decoded 76 frame(s) in 19 chunk(s) of 4 at 36x64 latent -- Apple silicon decode budget    (shot_music_closing_001)
```

**Discrepancy from the handoff worth flagging:** the handoff described this
profile as 832x480 (60x104 latent) expecting one frame per decode call.
What actually ran was 512x288 (36x64 latent) in chunks of 4 frames each.
Both are consistent with a chunked-decode-by-budget design, just not the
exact numbers named in the handoff -- possibly a different profile/graph
variant than assumed, not a defect in what was observed.

The pipeline completed normally after decode: composite (2799 frames @
25fps 1920x1080), captions burned, credits appended (24.0s), master audio
muxed (duration check `v=135.960s a=111.954s tail_budget=24.0s OK`,
`audio_byte_identical OK`), and `obs_publish OK`.

**One real miss, mine, not the fix's:** the publish landed in
`/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/output/otr/obs/`, not
`/Users/rentamac/ComfyUI-Shared/output/otr/obs/` -- my restart also omitted
an `--output-directory` override pointing at the shared output tree that
the original (pre-crash) process evidently had, so `OTR_OUTPUT_DIR` pinned
to the node-relative default instead. The harness's own `SUMMARY.txt`
therefore reads `obs=0` even though the episode genuinely finished and
published. The file is real and on disk at the path above (111,333,128
bytes, `the_sealed_compact_of_lost_lands_20260913_135656__anim__adlt__none__koko__sspr__q354b__mgen_final.mp4`,
dated 14:55) -- it was not moved or copied per this session's "report only,
no fixes" instruction.

Peak MPS could not be read directly from a decode-phase log line since no
OOM ever fired to print one; the periodic `MEMORY_SNAPSHOT` instrumentation
in this log only covers the LLM-writer phase and its high-water mark this
run was `mps_current_bytes` 8.41 GB / `mps_driver_bytes` 18.54 GB (writer
generation, not decode). The qualitative result stands regardless: eight
decode calls, zero MPS OOM.

Logs: `otr/legs/shipping_set_20260913_134932/otr_mac16_animatediff.log` in
the checkout above; ComfyUI's own log is not committed (`/tmp/
comfyui_restart3_20260913.log` on this machine, not preserved past this
session). No source fixes or memory-limit changes were applied -- config
and launch-flag corrections only.

<details>
<summary>otr_mac16_animatediff traceback</summary>

```text
Traceback (most recent call last):
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/custom_nodes/comfyui-old-time-radio/nodes/_otr_video_engines/wrapper_bridge.py", line 596, in run_graph
    out = normalize_node_output(fn(**kwargs))
                                ~~^^^^^^^^^^
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/nodes.py", line 338, in decode
    images = vae.decode(latent)
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/comfy/sd.py", line 1257, in decode
    model_management.raise_non_oom(e)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/comfy/model_management.py", line 399, in raise_non_oom
    raise e
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/comfy/sd.py", line 1249, in decode
    out = self.first_stage_model.decode(samples, **vae_options).to(device=self.output_device, dtype=self.vae_output_dtype(), copy=True)
          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/comfy/ldm/models/autoencoder.py", line 254, in decode
    dec = self.decoder(dec, **decoder_kwargs)
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/comfy/ldm/modules/diffusionmodules/model.py", line 810, in forward
    h1 = self.up[i_level].block[i_block](h1, temb, conv_carry_in, conv_carry_out, **kwargs)
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/comfy/ldm/modules/diffusionmodules/model.py", line 216, in forward
    h = self.norm1(h)
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/comfy/ops.py", line 646, in forward
    return super().forward(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/lib/python3.13/site-packages/torch/nn/modules/normalization.py", line 334, in forward
    return F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/lib/python3.13/site-packages/torch/nn/functional.py", line 2994, in group_norm
    return torch.group_norm(
           ~~~~~~~~~~~~~~~~^
        input, num_groups, weight, bias, eps, torch.backends.cudnn.enabled
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/lib/python3.13/site-packages/torch/_refs/__init__.py", line 3415, in native_group_norm
    out = out + unsqueeze_bias
          ~~~~^~~~~~~~~~~~~~~~
RuntimeError: MPS backend out of memory (MPS allocated: 14.29 GiB, other allocations: 3.65 GiB, max allowed: 20.13 GiB). Tried to allocate 2.67 GiB on shared pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/custom_nodes/comfyui-old-time-radio/nodes/_otr_video_engines/render_driver.py", line 4696, in render_shot
    clip = _render_one(eng, request, force_oom=force,
                       host_caps=host_caps, profile=profile,
                       segment=segment)
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/custom_nodes/comfyui-old-time-radio/nodes/_otr_video_engines/render_driver.py", line 4464, in _render_one
    raw = eng.render_clip(request, sess.prepared)
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/custom_nodes/comfyui-old-time-radio/nodes/_otr_video_engines/eng_ghost_signal.py", line 1389, in render_clip
    images = _wb.run_graph(
             ~~~~~~~~~~~~~^
        decode_graph,
        ^^^^^^^^^^^^^
        external_results={"sampled_latent": sampled_latent,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                          "vae": owners["vae"]},
                          ^^^^^^^^^^^^^^^^^^^^^^
        terminal=NODE_DECODE)[0]
        ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/custom_nodes/comfyui-old-time-radio/nodes/_otr_video_engines/wrapper_bridge.py", line 598, in run_graph
    raise GraphExecutionError(
        "node %r (%s) raised %s: %s"
        % (nid, fn_name, type(exc).__name__, exc))
/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/custom_nodes/comfyui-old-time-radio.nodes._otr_video_engines.wrapper_bridge.GraphExecutionError: node 'decode' (decode) raised RuntimeError: MPS backend out of memory (MPS allocated: 14.29 GiB, other allocations: 3.65 GiB, max allowed: 20.13 GiB). Tried to allocate 2.67 GiB on shared pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/execution.py", line 306, in process_inputs
    result = f(**inputs)
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/custom_nodes/comfyui-old-time-radio/nodes/otr_video_render_batch.py", line 548, in render
    report, manifest_payload, name = self._render_episode(
                                     ~~~~~~~~~~~~~~~~~~~~^
        _rd, patched_ledger_json,
        ^^^^^^^^^^^^^^^^^^^^^^^^^
        master_audio_path=str(master_audio_path or ""))
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/custom_nodes/comfyui-old-time-radio/nodes/otr_video_render_batch.py", line 661, in _render_episode
    ep = _rd.run_real_episode(ledger,
                              master_audio_path=str(master_audio_path or ""))
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/custom_nodes/comfyui-old-time-radio/nodes/_otr_video_engines/render_driver.py", line 5667, in run_real_episode
    return run_episode(ledger, request_builder=rb, canvas=canvas)
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/custom_nodes/comfyui-old-time-radio/nodes/_otr_video_engines/render_driver.py", line 5467, in run_episode
    clip, out_shot, attempts, used = render_beat_coverage(
                                     ~~~~~~~~~~~~~~~~~~~~^
        shot, ledger, request=request, request_builder=request_builder,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        canvas=canvas, oom_engines=oom_engines, oom_shot_id=oom_shot_id,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        host_caps=_episode_host_caps, profile=_episode_profile)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/custom_nodes/comfyui-old-time-radio/nodes/_otr_video_engines/render_driver.py", line 4795, in render_beat_coverage
    return render_shot(shot, single, oom_engines=oom_engines,
                       oom_shot_id=oom_shot_id, host_caps=host_caps,
                       profile=profile)
  File "/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/custom_nodes/comfyui-old-time-radio/nodes/_otr_video_engines/render_driver.py", line 4705, in render_shot
    raise RenderError(
        "shot %s engine %r failed to render; fallbacks are disabled (%s) -- "
        "fix the engine or its inputs: %s" % (sid, eng, kind, exc)) from exc
/Users/rentamac/ComfyUI-Installs/ComfyUI/ComfyUI/custom_nodes/comfyui-old-time-radio.nodes._otr_video_engines.render_errors.RenderError: shot shot_music_opening_001 engine 'animatediff15_lightning_video' failed to render; fallbacks are disabled (FailureKind.INVALID_DAG) -- fix the engine or its inputs: node 'decode' (decode) raised RuntimeError: MPS backend out of memory (MPS allocated: 14.29 GiB, other allocations: 3.65 GiB, max allowed: 20.13 GiB). Tried to allocate 2.67 GiB on shared pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).
```

</details>
