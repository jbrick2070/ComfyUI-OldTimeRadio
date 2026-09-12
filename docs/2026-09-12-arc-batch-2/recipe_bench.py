"""WHICH SA3 RECIPE SHIPS -- the bench arm for the plan's section 1 row.

Same graph, prompt families, seeds and measures as the music-model bench
(docs/2026-09-12-music-model-bench/music_model_bench.py), ONE checkpoint -- the
fetch default, small base -- at the engine's base guidance of 4.0. The only
variable is the sampler recipe:

  dpmpp100  dpmpp_3m_sde_gpu / exponential, 100 steps   (shipped today; the
            Stable Audio 1.0 template's pair)
  lcm50     lcm / simple, 50 steps                      (Comfy-Org's own
            SA3 base template recipe, at our cfg)

Bursts, loopiness, repeat, pulse, tempo, peak, wall seconds and VRAM per
render. FLACs under output/organ_bench/mrb/, MP3s for the ear under
output/otr/obs/music_recipe_bench/. A stock-node bench; it qualifies nothing.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\docs\2026-09-12-music-model-bench")
import music_model_bench as mmb  # noqa: E402

CKPT = "stable_audio_3_small_music_base.safetensors"
RECIPES = {
    "dpmpp100": dict(cfg=4.0, steps=100, sampler="dpmpp_3m_sde_gpu", scheduler="exponential"),
    "lcm50": dict(cfg=4.0, steps=50, sampler="lcm", scheduler="simple"),
}
mmb.OBS_SUBDIR = "otr/obs/music_recipe_bench"
mmb.FLAC_SUBDIR = "organ_bench/mrb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(RECIPES))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--base-seed", type=int, default=90120001)
    args = ap.parse_args()

    fams = [f for f in mmb.families() if not args.families or f["key"] in args.families]
    rows = []
    if os.path.isfile(args.json_out):
        with open(args.json_out, encoding="utf-8") as fh:
            rows = json.load(fh)
    done = {(r["arm"], r["family"], r["seed"]) for r in rows}
    for arm in args.arms:
        mmb.SA3_RECIPE.clear()
        mmb.SA3_RECIPE.update(RECIPES[arm])
        print("== ARM %s %s" % (arm, json.dumps(RECIPES[arm])), flush=True)
        for i, fam in enumerate(fams):
            for k in range(args.seeds):
                seed = args.base_seed + i * 17 + k * 1000
                if (arm, fam["key"], seed) in done:
                    continue
                label = "%s__%s__seed%d" % (arm, fam["key"], k)
                graph = mmb.sa3_graph(CKPT, fam, seed, label)
                try:
                    flac, mp3, took, vram = mmb.render(graph)
                except Exception as exc:  # noqa: BLE001 -- a bench never dies mid-arm
                    print("  %-34s FAILED: %s" % (label, str(exc)[:300]), flush=True)
                    rows.append({"arm": arm, "family": fam["key"], "seed": seed,
                                 "error": str(exc)[:300]})
                    continue
                if not flac:
                    print("  %-34s no flac came back" % label, flush=True)
                    continue
                m = mmb.organ.measure(flac)
                m.update(mmb.rhythm(flac))
                m.update({"arm": arm, "family": fam["key"], "lane": fam["lane"], "seed": seed,
                          "seed_index": k, "flac": flac, "mp3": mp3,
                          "wall_s": round(took, 1), "vram_peak_mib": vram,
                          "cue_seconds": fam["seconds"], "ckpt": CKPT,
                          "recipe": RECIPES[arm]})
                rows.append(m)
                print("  %-34s %5.1fs vram=%5d bursts=%-2d loop=%.3f@%.2fs onsets/min=%5.1f "
                      "tempo=%5.1f pulse=%.2f peak=%6.2f rms=%6.2f"
                      % (label, took, vram, m["bursts"], m["loopiness"], m["loop_lag_s"],
                         m["onsets_per_min"], m["tempo_bpm"], m["pulse"], m["peak_dbfs"],
                         m["rms_dbfs"]), flush=True)
                with open(args.json_out, "w", encoding="utf-8") as fh:
                    json.dump(rows, fh, indent=1)
    print("BENCH DONE rows=%d" % len(rows), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
