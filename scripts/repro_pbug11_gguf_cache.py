"""MINIMAL REPRODUCTION of PBUG-20260829-11 -- the GGUF lane can never cache.

No ComfyUI, no graph, no episode, no competing VRAM consumer. OTR's own
request_slot, called TWICE for the same GGUF model.

  healthy lane : call 2 is a cache HIT, well under a second, no disk read
  PBUG-11 live : call 2 cold-loads 7 GB again and logs "abandoned
                 (cache epoch advanced)"

Runtime ~2 min. Deterministic. Gives the third fix attempt something exact to
be verified against, with no OOM confound (n_gpu_layers 35 is known-good here).
"""
import importlib.util
import logging
import os
import sys
import time
import types
from pathlib import Path

# BOX-AGNOSTIC RESOLUTION (5080 addition). The 4060 wrote this against its own
# absolute paths; both boxes need to run it, and a harness that reproduces only
# on the machine that found the bug is half a harness. Env overrides still win,
# so the original invocation on the 4060 is unchanged.
REPO = Path(os.environ.get("OTR_REPO") or Path(__file__).resolve().parents[1])


def _repro_models_root() -> Path:
    """OTR's own authority for where weights live -- never a hardcoded guess."""
    for var in ("OTR_COMFYUI_MODELS_ROOT", "COMFYUI_MODELS_ROOT"):
        v = os.environ.get(var)
        if v:
            return Path(v)
    for cand in (Path("C:/ComfyUI-Models"), REPO.parents[1] / "models"):
        if cand.exists():
            return cand
    return Path("C:/ComfyUI-Models")


MODELS = _repro_models_root()
MODEL = Path(os.environ.get("OTR_REPRO_GGUF") or (
    MODELS / "LLM" / "converted" / "gemma-4-12b-it" / "gemma-4-12b-it-Q4_K_M.gguf"))
INSTALL = REPO.parents[1]

os.environ.setdefault("HF_HOME", str(MODELS / "huggingface"))
os.environ.setdefault("OTR_COMFYUI_MODELS_ROOT", str(MODELS))

if not MODEL.is_file():
    raise SystemExit("repro: GGUF not found at %s -- set OTR_REPRO_GGUF" % MODEL)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

pkg = types.ModuleType("otrpack"); pkg.__path__ = [str(REPO)]; sys.modules["otrpack"] = pkg
nodes = types.ModuleType("otrpack.nodes"); nodes.__path__ = [str(REPO / "nodes")]
sys.modules["otrpack.nodes"] = nodes
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(INSTALL))


def load(mod, rel):
    s = importlib.util.spec_from_file_location(mod, REPO / rel)
    m = importlib.util.module_from_spec(s); sys.modules[mod] = m; s.loader.exec_module(m)
    return m

ml = load("otrpack.nodes._otr_model_loader", Path("nodes/_otr_model_loader.py"))
gb = load("otrpack.nodes._otr_gguf_backend", Path("nodes/_otr_gguf_backend.py"))
pol_mod = load("otrpack.nodes._otr_shared.llm_policy", Path("nodes/_otr_shared/llm_policy.py"))

ROW = "unsloth/gemma-4-12b-it-GGUF"


def epoch():
    f = getattr(ml, "_current_cache_epoch", None)
    return f() if callable(f) else getattr(ml, "_CACHE_EPOCH", "?")


def cached():
    c = getattr(ml, "LLM_CACHE", None)
    return (c or {}).get("model_id") if isinstance(c, dict) else "?"


policy = pol_mod.LLMRuntimePolicy(
    device="cuda", attn_impl="sdpa", quant_policy="none",
    vram_ceiling_gb=12.5, gguf_n_ctx=4096, gguf_quant="Q4_K_M")

cfg = gb.GGUFLoadConfig(
    repo_id=ROW, model_path=str(MODEL), quant="Q4_K_M", n_ctx=4096,
    n_batch=512, n_gpu_layers=35, kv_gb_per_1k=0.70, seed=42,
    stop_tokens=tuple(), think_policy="none")

print(f"\n[repro] start   epoch={epoch()}  cached={cached()!r}")
print(f"[repro] model   {MODEL.name}  n_gpu_layers=35 (known-good, no OOM confound)\n")

for i in (1, 2):
    print(f"===== CALL {i} =====")
    e0, t0 = epoch(), time.time()
    try:
        res = ml.request_slot("creative", ROW, policy, cfg)
        dt = time.time() - t0
        print(f"[repro] call {i}: OK in {dt:6.2f}s   epoch {e0} -> {epoch()}   "
              f"cached now={cached()!r}")
        print(f"[repro] call {i}: returned keys={sorted(res)[:8] if isinstance(res, dict) else type(res)}")
    except Exception as ex:
        dt = time.time() - t0
        print(f"[repro] call {i}: FAILED in {dt:6.2f}s  {type(ex).__name__}: {str(ex)[:200]}")
        print(f"[repro] call {i}: epoch {e0} -> {epoch()}  cached={cached()!r}")

print("\n===== VERDICT =====")
print(f"final epoch  : {epoch()}   (0 would mean no invalidation ever fired)")
print(f"final cached : {cached()!r}  (None => the lane NEVER populated LLM_CACHE)")
print("PBUG-11 CONFIRMED if call 2 took roughly as long as call 1 and cached is None.")
print("PBUG-11 FIXED    if call 2 was ~instant and cached names the row.")
