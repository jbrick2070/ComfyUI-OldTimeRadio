# Apple Silicon

It works, and the short version is that you do not have to do anything special.
Install as in [INSTALL.md](INSTALL.md), open the Mac graph, press Queue.

```
workflows/variants/otr_mac_mps.json
```

That graph is the canonical with Mac-appropriate dropdowns already saved. The
canonical itself also runs — it names no vendor and resolves your device at run
time — but the Mac variant picks the lanes that have receipts here.

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
