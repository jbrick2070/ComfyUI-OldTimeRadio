"""Fresh CPU check of the audio segment in the real canonical workflow.

Loads live registrations, links and widget values on every invocation. Only the
incoming voice/music assets and ledger are supplied by this check. Sequencing,
DSP, assembly, ledger persistence and WAV writing are the production functions.
This is a bounded segment check, not an end-to-end episode/render qualification.
Run in a fresh process; no pytest fixtures or old harness modules are imported.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "workflows" / "otr_canonical.json"
ROUTE = ("OTR_SceneSequencer", "OTR_AudioEnhance", "OTR_EpisodeAssembler")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_package():
    # ComfyUI's real folder_paths is used, not the suite's collection-time stub.
    sys.path.insert(0, str(ROOT.parents[1]))
    name = ROOT.name.replace("-", "_")
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return package


class CanonicalAudioRoute:
    def __init__(self, package):
        raw = CANONICAL.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.graph = json.loads(raw)
        self.nodes = {n["id"]: n for n in self.graph["nodes"]}
        self.links = {link[0]: link for link in self.graph["links"]}
        self.selected = {}
        self.classes = {}
        self.values = {}
        self.boundaries = []
        self.calls = []
        for kind in ROUTE:
            matches = [n for n in self.nodes.values() if n["type"] == kind]
            require(len(matches) == 1, f"Expected one canonical {kind}")
            node = matches[0]
            require(node.get("mode", 0) == 0, f"{kind} is muted/bypassed")
            self.selected[kind] = node
            self.classes[kind] = package.NODE_CLASS_MAPPINGS[kind]
            schema = self.classes[kind].INPUT_TYPES()
            declared = {**schema.get("required", {}), **schema.get("optional", {})}
            require(all(i["name"] in declared for i in node["inputs"]),
                    f"{kind} canonical input names differ from live INPUT_TYPES")
            widgets = [i for i in node["inputs"] if i.get("widget")]
            require(len(widgets) == len(node["widgets_values"]),
                    f"{kind} positional widget count differs")
            # Confirm serialized widget order against the live input schema.
            live_widgets = [key for key, spec in declared.items()
                            if (isinstance(spec[0], list)
                                or spec[0] in ("STRING", "INT", "FLOAT", "BOOLEAN"))
                            and not (len(spec) > 1 and spec[1].get("forceInput"))]
            require([i["name"] for i in widgets] == live_widgets,
                    f"{kind} widget order differs from live INPUT_TYPES")
            for index, item in enumerate(node["inputs"]):
                if item.get("link") is not None:
                    link = self.links[item["link"]]
                    require(link[3:5] == [node["id"], index],
                            f"{kind}.{item['name']} link target differs")
                    source = self.nodes[link[1]]["outputs"][link[2]]
                    require(link[0] in (source.get("links") or []),
                            f"{kind}.{item['name']} missing source fan-out")
                    require(source["type"] == link[5] == item["type"],
                            f"{kind}.{item['name']} link type differs")
        for source, target, input_name in (
            (ROUTE[0], ROUTE[1], "audio"), (ROUTE[1], ROUTE[2], "scene_audio")
        ):
            item = self.input(target, input_name)
            link = self.links[item["link"]]
            require(link[1] == self.selected[source]["id"] and link[2] == 0,
                    f"Canonical route no longer connects {source} to {target}")

    def input(self, kind, name):
        return next(i for i in self.selected[kind]["inputs"] if i["name"] == name)

    def supply(self, kind, name, value, source_type):
        item = self.input(kind, name)
        require(item.get("link") is not None, f"Boundary {kind}.{name} is unwired")
        link = self.links[item["link"]]
        require(self.nodes[link[1]]["type"] == source_type,
                f"{kind}.{name} now comes from a different producer; requalify the boundary")
        require(link[1] not in {n["id"] for n in self.selected.values()},
                "Cannot replace an internal production edge with test data")
        self.values[(link[1], link[2])] = value
        description = ({"sample_rate": value["sample_rate"],
                        "shape": list(value["waveform"].shape)}
                       if isinstance(value, dict) and "waveform" in value
                       else {"value": value})
        self.boundaries.append({"input": f"{kind}.{name}", "link": link,
                                "source_type": self.nodes[link[1]]["type"],
                                "supplied": description})

    def run(self, kind):
        node = self.selected[kind]
        widgets = iter(node["widgets_values"])
        kwargs = {}
        for item in node["inputs"]:
            if item.get("widget"):
                kwargs[item["name"]] = next(widgets)
            if item.get("link") is not None:
                link = self.links[item["link"]]
                kwargs[item["name"]] = self.values[(link[1], link[2])]
        cls = self.classes[kind]
        result = getattr(cls(), cls.FUNCTION)(**kwargs)
        require(len(result) == len(cls.RETURN_TYPES), f"{kind} returned wrong arity")
        for slot, value in enumerate(result):
            self.values[(node["id"], slot)] = value
        self.calls.append({"type": kind, "node_id": node["id"],
                           "function": cls.FUNCTION,
                           "source_sha256": hashlib.sha256(
                               Path(inspect.getfile(cls)).read_bytes()).hexdigest(),
                           "widgets_values": node["widgets_values"]})
        return result


def check_case(package, output_root, opening):
    import numpy as np
    import torch
    from scipy.signal import correlate

    namespace = package.__name__ + ".nodes."
    pl = importlib.import_module(namespace + "production_ledger")
    cm = importlib.import_module(namespace + "_otr_cue_manifest")
    route = CanonicalAudioRoute(package)
    # Roomtone and tape hiss are real DSP randomness. Control the fixture RNG
    # so same-environment before/after WAV comparisons can detect a code change.
    rng_seed = 20260910
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    episode = "canonical_audio_opening" if opening else "canonical_audio_no_opening"
    audio_dir = output_root / "otr" / "episodes" / episode / "audio"
    led = pl.new_ledger(episode, str(audio_dir))
    led.data["meta"] = {"title": "Canonical audio check"}
    led.data["cast"] = [{"char_id": "c01", "name": "Mira", "voice_preset": "af_heart"}]
    led.data["lines"] = [
        {"line_id": "l001", "speaker_role": "announcer", "text": "Opening announcement."},
        {"line_id": "l002", "speaker_role": "character", "char_id": "c01", "text": "A reply."},
    ]
    led.save()
    require(Path(led.path).is_file(), "Production ledger did not save")
    rate = 48000

    def audio(seconds, frequency):
        t = torch.arange(round(seconds * rate), dtype=torch.float32) / rate
        signal = 0.1 * torch.sin(2 * torch.pi * (frequency * t + 180 * t * t))
        return {"waveform": signal.reshape(1, 1, -1), "sample_rate": rate}

    route.supply(ROUTE[0], "script_json", json.dumps(led.data), "OTR_CastLock")
    route.supply(ROUTE[0], "tts_audio_clips", audio(1, 430), "OTR_BatchCharacterVoices")
    route.supply(ROUTE[0], "announcer_audio_clips", audio(1, 710), "OTR_AnnouncerVoice")
    route.run(ROUTE[0])
    before = json.loads(Path(led.path).read_text(encoding="utf-8"))
    require(all(row.get("start_s_space") == "scene_audio" for row in before["lines"]),
            "Sequencer did not persist line positions")
    positions = {row["line_id"]: row["start_s"] for row in before["lines"]}
    enhanced = route.run(ROUTE[1])[0]
    require(torch.isfinite(enhanced["waveform"]).all().item(), "DSP produced nonfinite audio")
    rate = int(enhanced["sample_rate"])

    # Durable prior state: no mock of ledger loading, patching, or saving.
    current = json.loads(Path(led.path).read_text(encoding="utf-8"))
    current["clips"] = [
        {"line_id": "l001", "start_s": 0.25},
        {"line_id": "l002", "start_s": 7.0, "start_s_space": "master_mix"},
        {"line_id": "missing_time"},
    ]
    led.data = current
    led.save()
    cue_audio = audio(2, 230) if opening else audio(1, 230)
    placement = "opening" if opening else "closing"
    duration = cue_audio["waveform"].shape[-1] / rate
    rows = [{"cue_id": placement, "batch_index": 0,
             "sample_count": cue_audio["waveform"].shape[-1], "sample_rate": rate,
             "prompt": "Instrumental tone.", "prompt_sha256": cm.prompt_sha256("Instrumental tone."),
             "cue_spec_sha256": None, "placement": placement, "anchor_line_id": None,
             "seed": 1, "requested_duration_s": duration, "actual_duration_s": duration,
             "output_path": ""}]
    route.supply(ROUTE[2], "music_cue_audio", cue_audio, "OTR_StableAudioTheme")
    route.supply(ROUTE[2], "music_cue_manifest_json", cm.dumps(cm.build_manifest(rate, rows)),
                 "OTR_StableAudioTheme")
    route.supply(ROUTE[2], "video_policy_json", "", "OTR_VideoDirector")
    route.supply(ROUTE[2], "replay_descriptor", "", "OTR_LedgerFreezeCascade")
    result = route.run(ROUTE[2])
    master = Path(result[1]).resolve()
    require(master.parent == audio_dir.resolve() and master.is_file(), "Master WAV missing/misplaced")
    with wave.open(str(master), "rb") as wav:
        saved_rate = wav.getframerate()
        decoded = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        decoded = decoded.reshape(-1, wav.getnchannels()).mean(axis=1)
    # Locate the actual enhanced signal inside the written master. This measures
    # placement independently: it does not reproduce the production shift loop.
    template = enhanced["waveform"][0].mean(dim=0).numpy()
    start, stop = round(0.65 * rate), round(1.25 * rate)
    lag = int(np.argmax(correlate(decoded, template[start:stop], mode="valid", method="fft"))) - start
    require(saved_rate == rate, "Master rate differs from the measured signal")
    measured_offset = lag / saved_rate
    require(lag > 0 if opening else lag == 0, "Written audio placement is wrong")
    saved = json.loads(Path(led.path).read_text(encoding="utf-8"))
    by_id = {row["line_id"]: row for row in saved["lines"]}
    for line_id, start_s in positions.items():
        row = by_id[line_id]
        require(abs(row["start_s"] - start_s - measured_offset) <= 1 / rate,
                f"{line_id}: persisted timing disagrees with written audio")
        require(row["start_s_space"] == ("master_mix" if opening else "scene_audio"),
                f"{line_id}: wrong coordinate-space marker")
    require(abs(saved["clips"][0]["start_s"] - 0.25 - measured_offset) <= 1 / rate,
            "Pre-existing clip did not follow the written audio")
    require(saved["clips"][1]["start_s"] == 7.0, "Master-space clip shifted again")
    require("start_s" not in saved["clips"][2], "Missing clip time was invented")
    timing = [(row.get("line_id"), row.get("start_s"), row.get("start_s_space"))
              for row in saved["lines"] + saved["clips"]]
    route.run(ROUTE[2])
    repeated = json.loads(Path(led.path).read_text(encoding="utf-8"))
    require(timing == [(row.get("line_id"), row.get("start_s"), row.get("start_s_space"))
                       for row in repeated["lines"] + repeated["clips"]],
            "Real assembler repeat changed persisted timing")
    require(hashlib.sha256(CANONICAL.read_bytes()).hexdigest() == route.sha256,
            "Canonical changed during check; evidence is stale")
    for call in route.calls:
        current_hash = hashlib.sha256(Path(inspect.getfile(route.classes[call["type"]])).read_bytes()).hexdigest()
        require(current_hash == call["source_sha256"], "Production code changed during check")
    return {"episode": episode, "canonical_sha256": route.sha256,
            "rng_seed": rng_seed,
            "calls": route.calls, "supplied_boundaries": route.boundaries,
            "measured_scene_offset_s": measured_offset, "ledger": str(led.path),
            "master": str(master), "master_sha256": hashlib.sha256(master.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    require(not output.exists(), "Use a fresh output root; existing evidence is never overwritten")
    os.environ.update(CUDA_VISIBLE_DEVICES="", OTR_TEST_MODE="1",
                      OTR_OUTPUT_DIR=str(output), OTR_EXTRA_OUTPUT_ROOTS=str(output))
    os.environ.pop("OTR_OBS_DIR", None)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    package = load_package()
    import torch
    torch.set_num_threads(1)
    cases = [check_case(package, output, opening) for opening in (True, False)]
    require(head == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "HEAD changed during check")
    receipt = {"scope": "canonical non-foley CPU audio segment; synthetic upstream assets; no model/server/publish",
               "head": head,
               "canonical": str(CANONICAL), "cases": cases}
    path = output / "canonical_audio_check.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"CANONICAL AUDIO CHECK PASSED: {path}")


if __name__ == "__main__":
    main()
