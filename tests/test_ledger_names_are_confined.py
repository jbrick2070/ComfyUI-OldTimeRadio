"""A ledger-chosen name reaches no filesystem sink unconfined.

Three claims from the 2026-10-30 registry-sinks review (Codex Astra, grounded
by executed probes): a patched ledger row's `content_hash` became a filename
tail and walked a copy out of the stills directory; a shot's `char_id` in
`conditioning_refs` was treated as a path, stat'ed and hashed wherever it
pointed; the writer download's `tokenizer*` prefix admitted a `.py`. Each test
reproduces the escape the review named and asserts the refusal.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def test_a_hash_with_separators_cannot_leave_the_stills_dir(tmp_path, monkeypatch):
    from nodes._otr_video_engines import render_driver as RD
    stills = tmp_path / "otr" / "episodes" / "ep1" / "stills"
    stills.mkdir(parents=True)
    src = tmp_path / "otr" / "episodes" / "_shared" / "cache" / "src.png"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 80)
    monkeypatch.setattr(RD, "_trusted_still_source",
                        lambda candidate, root: candidate == str(src))
    row = {"object_id": "x", "path": str(src), "content_hash": "\\..\\..\\..\\zz"}
    out = RD._still_spine_materialize_row(row, str(stills))
    assert out and Path(out).parent == stills                 # inside, never beside
    assert Path(out).name.startswith("x_") and Path(out).suffix == ".png"
    assert len(Path(out).stem) == len("x_") + 12               # a real digest's tail
    assert not (tmp_path / "otr" / "episodes" / "zz.png").exists()
    assert not list(tmp_path.glob("**/zz*"))
    # a hex digest keeps its own first twelve characters
    row2 = {"object_id": "y", "path": str(src), "content_hash": "ab" * 32}
    assert Path(RD._still_spine_materialize_row(row2, str(stills))).name == "y_" + "ab" * 6 + ".png"


def test_a_char_id_outside_the_read_roots_is_never_hashed(tmp_path):
    from nodes import _otr_producer_clips as PC
    outside = Path(sys.executable)                 # exists, readable, in no read root
    proj = PC.request_projection({"conditioning_refs": {"char_id": str(outside)}})
    assert proj["conditioning_refs"]["char_id"] == str(outside)
    unc = r"\\10.0.0.9\share\secret.png"
    proj = PC.request_projection({"conditioning_refs": {"char_id": unc}})
    assert proj["conditioning_refs"]["char_id"] == unc
    proj = PC.request_projection({"conditioning_refs": {"char_id": "announcer"}})
    assert proj["conditioning_refs"]["char_id"] == "announcer"
    # a reference inside the roots (pytest's tmp tree is a declared root) still
    # enters the key as its digest, which is what the key is for
    inside = tmp_path / "ref.png"
    inside.write_bytes(b"\x89PNG ref")
    proj = PC.request_projection({"conditioning_refs": {"init_image": str(inside)}})
    assert proj["conditioning_refs"]["init_image"] == {
        "sha256": hashlib.sha256(b"\x89PNG ref").hexdigest()}


def test_the_writer_download_never_fetches_code():
    from nodes import _otr_model_catalog as MC
    assert "consolidated*.safetensors" in MC.IGNORE_PATTERNS
    got = MC._fetched_paths([
        "tokenizer_payload.py", "tokenizer.pyd", "tokenizer.exe", "tokenizer/x.py",
        "tokenizer.sh", "tokenizer.json", "tokenizer.model", "tokenizer.model.v3",
        "model.safetensors", "config.json", "README.md",
    ])
    assert got == {"tokenizer.json", "tokenizer.model", "tokenizer.model.v3",
                   "model.safetensors", "config.json", "README.md"}
