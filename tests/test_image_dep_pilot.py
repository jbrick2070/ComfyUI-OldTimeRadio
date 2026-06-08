"""CPU tests for the image dependency-pilot (scripts/otr_image_dep_pilot.py).

The third-namespace twin of the audio/video dep pilots. It enumerates the
pluggable image engines, classifies each as sidecar-isolated (Z-Image) vs in-stack
(the GGUF/native peers) vs the gen-1 default (Flux), and carries a headless-safe
BEFORE/AFTER import probe for the sidecar engines. All CPU; the live sidecar import
is an operator GPU-box step.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PILOT_SRC = REPO_ROOT / "scripts" / "otr_image_dep_pilot.py"


def _load_pilot():
    spec = importlib.util.spec_from_file_location("otr_image_dep_pilot", PILOT_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pilot = _load_pilot()


def test_enumerate_covers_all_registered_engines():
    from nodes._otr_image_engines import registry as ireg
    engines = pilot.enumerate_engines()
    names = {e["engine"] for e in engines}
    assert names == set(ireg.all_engine_names())
    # the full peer set is present (Flux gen-1 + C2/C3 + C4-C8)
    assert {"flux_gen1", "z_image_turbo", "qwen_image", "hidream_i1",
            "chroma_hd", "lumina_image", "sd35_large", "flux2_klein"} <= names


def test_isolation_classification():
    by_name = {e["engine"]: e for e in pilot.enumerate_engines()}
    # Flux gen-1 = default (no flag, no sidecar, no model env gate)
    assert by_name["flux_gen1"]["isolation"] == "default"
    assert by_name["flux_gen1"]["enable_flag"] is None
    # Z-Image (C2) = sidecar (exposes SIDECAR_ENV)
    assert by_name["z_image_turbo"]["isolation"] == "sidecar"
    assert by_name["z_image_turbo"]["gate_env"] == "OTR_ZIMAGE_SIDECAR"
    # the GGUF/native peers = in-stack (expose MODEL_ENV)
    for n in ("qwen_image", "hidream_i1", "chroma_hd", "lumina_image",
              "sd35_large", "flux2_klein"):
        assert by_name[n]["isolation"] == "in_stack", n
        assert by_name[n]["gate_env"], n


def test_commercial_clean_reported_honestly():
    by_name = {e["engine"]: e for e in pilot.enumerate_engines()}
    assert by_name["qwen_image"]["commercial_clean"] is True       # Apache
    assert by_name["hidream_i1"]["commercial_clean"] is True       # MIT
    assert by_name["sd35_large"]["commercial_clean"] is False      # Community (conditional)
    assert by_name["flux2_klein"]["commercial_clean"] is False     # FLUX-family (verify)


def test_snapshot_violations_pure_logic():
    # torch swap is a violation
    assert pilot.snapshot_violations({"torch": "2.10"}, {"torch": "2.7"})
    # a banned dep pulled in is a violation
    before = {"torch": "2.10", "xformers": False, "flash_attn": False, "sageattention": False}
    after = {"torch": "2.10", "xformers": True, "flash_attn": False, "sageattention": False}
    viol = pilot.snapshot_violations(before, after)
    assert any("xformers" in v for v in viol)
    # clean -> no violations
    assert pilot.snapshot_violations(before, before) == []


def test_run_pilot_counts():
    report = pilot.run_pilot()
    assert report["engine_count"] == len(pilot.enumerate_engines())
    assert report["sidecar_count"] >= 1          # at least Z-Image
    assert report["in_stack_count"] >= 6         # the GGUF/native peers
    assert "sidecar_probe" not in report         # no probe unless --lib given


def test_sidecar_probe_lib_absent_headless():
    """A missing sidecar library is a clean lib_absent verdict, never a crash."""
    v = pilot.probe_sidecar_import("a_module_that_does_not_exist_otr")
    assert v["status"] == "lib_absent"
    assert v["import_clean_ready"] is False


def test_pilot_runs_as_script_json():
    r = subprocess.run(
        [sys.executable, "scripts/otr_image_dep_pilot.py", "--json"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert '"engines"' in r.stdout and '"in_stack"' in r.stdout


def test_pilot_cold_import_clean():
    """Importing the pilot pulls NO heavy lib (it enumerates the cold-import-clean
    registry, never a model framework)."""
    code = (
        "import importlib.util, sys;"
        f"spec=importlib.util.spec_from_file_location('p', r'{PILOT_SRC}');"
        "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        "m.enumerate_engines();"
        "heavy=[x for x in ('torch','transformers','diffusers') if x in sys.modules];"
        "print('HEAVY', heavy); sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs at import:\n{r.stdout}\n{r.stderr}"
