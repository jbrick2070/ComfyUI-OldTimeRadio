"""LTX 2.5 refuses a GGUF pack missing any required semantic patch fact."""
from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

from nodes._otr_video_engines import eng_ltx25


_RAW_NAMES = (
    "audio_embeddings_connector.learnable_registers",
    "keyframes_abs_pos_embedding",
    "video_embeddings_connector.learnable_registers",
)


def _source(*, gemma4=True, raw_names=_RAW_NAMES, condition=None, dtype="float32"):
    arches = "{'t5', 'gemma4'}" if gemma4 else "{'t5'}"
    names = "{" + ", ".join(repr(item) for item in raw_names) + "}"
    condition = condition or (
        "tensor.tensor_type == gguf.GGMLQuantizationType.BF16 and "
        "(len(shape) <= 1 or (arch_str == 'ltxv' and "
        "tensor_name in LTXV_BF16_PARAMETERS))")
    return (
        "TXT_ARCH_LIST = %s\n"
        "LTXV_BF16_PARAMETERS = %s\n"
        "def gguf_sd_loader():\n"
        "    if %s:\n"
        "        state_dict[sd_key] = dequantize_tensor("
        "state_dict[sd_key], dtype=torch.%s)\n"
        % (arches, names, condition, dtype)
    )


def _registered_class(tmp_path: Path, monkeypatch, source=None, *, module_file=True):
    pack = tmp_path / ("pack_" + str(len(list(tmp_path.iterdir()))))
    pack.mkdir()
    if source is not None:
        (pack / "loader.py").write_text(source, encoding="utf-8")
    module_name = "_otr_gguf_semantic_%s" % pack.name
    module = types.ModuleType(module_name)
    if module_file:
        module.__file__ = str(pack / "nodes.py")
    monkeypatch.setitem(sys.modules, module_name, module)
    return type("CLIPLoaderGGUF", (), {"__module__": module_name})


def test_complete_semantic_patch_passes(tmp_path, monkeypatch):
    loader_cls = _registered_class(tmp_path, monkeypatch, _source())
    path, gaps = eng_ltx25._inspect_ltx25_gguf_patch(loader_cls)

    assert path.endswith("loader.py")
    assert gaps == ()


def test_clean_and_gemma_only_sources_fail_all_unmet_facts(tmp_path, monkeypatch):
    clean = (
        "TXT_ARCH_LIST = {'t5'}\n"
        "def gguf_sd_loader():\n"
        "    if len(shape) <= 1 and "
        "tensor.tensor_type == gguf.GGMLQuantizationType.BF16:\n"
        "        state_dict[sd_key] = dequantize_tensor("
        "state_dict[sd_key], dtype=torch.float32)\n"
    )
    _path, gaps = eng_ltx25._inspect_ltx25_gguf_patch(
        _registered_class(tmp_path, monkeypatch, clean))
    assert len(gaps) == 3
    assert any("gemma4" in item for item in gaps)
    assert any("LTXV_BF16_PARAMETERS" in item for item in gaps)
    assert any("materialization branch" in item for item in gaps)

    gemma_only = clean.replace("{'t5'}", "{'t5', 'gemma4'}")
    _path, gaps = eng_ltx25._inspect_ltx25_gguf_patch(
        _registered_class(tmp_path, monkeypatch, gemma_only))
    assert len(gaps) == 2
    assert not any("TXT_ARCH_LIST" in item for item in gaps)


@pytest.mark.parametrize("missing_name", _RAW_NAMES)
def test_each_raw_parameter_name_is_required(tmp_path, monkeypatch, missing_name):
    names = tuple(item for item in _RAW_NAMES if item != missing_name)
    _path, gaps = eng_ltx25._inspect_ltx25_gguf_patch(
        _registered_class(tmp_path, monkeypatch, _source(raw_names=names)))
    assert any(missing_name in item for item in gaps)


@pytest.mark.parametrize("condition,dtype", [
    ("len(shape) <= 1 and tensor.tensor_type == "
     "gguf.GGMLQuantizationType.BF16", "float32"),
    ("tensor.tensor_type == gguf.GGMLQuantizationType.BF16 or "
     "(len(shape) <= 1 or (arch_str == 'ltxv' and "
     "tensor_name in LTXV_BF16_PARAMETERS))", "float32"),
    (None, "float16"),
])
def test_materialization_branch_must_match_condition_and_dtype(
        tmp_path, monkeypatch, condition, dtype):
    _path, gaps = eng_ltx25._inspect_ltx25_gguf_patch(
        _registered_class(
            tmp_path, monkeypatch, _source(condition=condition, dtype=dtype)))
    assert any("materialization branch" in item for item in gaps)


def test_missing_provenance_sibling_and_syntax_fail_closed(tmp_path, monkeypatch):
    no_provenance = _registered_class(
        tmp_path, monkeypatch, _source(), module_file=False)
    path, gaps = eng_ltx25._inspect_ltx25_gguf_patch(no_provenance)
    assert path == "" and "no readable __file__" in gaps[0]

    no_loader = _registered_class(tmp_path, monkeypatch, None)
    path, gaps = eng_ltx25._inspect_ltx25_gguf_patch(no_loader)
    assert path.endswith("loader.py") and "cannot parse" in gaps[0]

    syntax_error = _registered_class(tmp_path, monkeypatch, "if:\n")
    _path, gaps = eng_ltx25._inspect_ltx25_gguf_patch(syntax_error)
    assert "SyntaxError" in gaps[0]


def _mapping_for(engine, loader_cls):
    fallback = type("InstalledNode", (), {})
    mapping = {
        candidate
        for candidates in engine._node_candidates().values()
        for candidate in candidates
    }
    result = {name: fallback for name in mapping}
    result["CLIPLoaderGGUF"] = loader_cls
    return result


def test_assert_usable_checks_patch_before_weight_paths(tmp_path, monkeypatch):
    from nodes._otr_video_engines import wrapper_bridge

    loader_cls = _registered_class(
        tmp_path, monkeypatch, _source(gemma4=True, raw_names=()))
    engine = eng_ltx25.Ltx25VideoEngine()
    monkeypatch.setattr(eng_ltx25._MC, "assert_sage_not_patched", lambda *_args: None)
    monkeypatch.setattr(
        wrapper_bridge, "node_class_mappings",
        lambda _mapping=None: _mapping_for(engine, loader_cls))
    monkeypatch.setattr(
        engine, "_weight_paths",
        lambda: (_ for _ in ()).throw(AssertionError("weights resolved first")))

    with pytest.raises(eng_ltx25.EngineUnusable, match="not LTX 2.5 compatible"):
        engine.assert_usable({}, {})


def test_assert_usable_proceeds_after_complete_patch(tmp_path, monkeypatch):
    from nodes._otr_video_engines import wrapper_bridge

    loader_cls = _registered_class(tmp_path, monkeypatch, _source())
    engine = eng_ltx25.Ltx25VideoEngine()
    monkeypatch.setattr(eng_ltx25._MC, "assert_sage_not_patched", lambda *_args: None)
    monkeypatch.setattr(
        wrapper_bridge, "node_class_mappings",
        lambda _mapping=None: _mapping_for(engine, loader_cls))
    monkeypatch.setattr(engine, "_weight_paths", lambda: [])

    assert engine.assert_usable({}, {}) == engine.name
