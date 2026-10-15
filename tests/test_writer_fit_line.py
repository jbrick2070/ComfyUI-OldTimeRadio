"""What the writer widgets' tooltip says about each writer (2026-10-14).

Until 2026-10-14 this was the dropdown's size badge (vram_badge_for); the
dropdown now offers bare names and the same facts moved to the tooltip, built
by writer_fit_line / writer_tooltip_table. The promises carry over:
PBUG-20260829-17 -- no line carries a context term (every row is a download
whose size does not move with the context asked for) -- and a row nothing can
estimate never breaks INPUT_TYPES.
"""
from __future__ import annotations

from nodes import _otr_model_catalog as catalog
from nodes._otr_model_catalog import writer_fit_line, writer_tooltip_table


def test_lines_state_the_measured_download_and_no_context_term():
    for repo in ("google/gemma-4-E2B-it", "google/gemma-2-2b-it", "Qwen/Qwen3.5-4B"):
        line = writer_fit_line(repo)
        assert line.startswith("about "), (repo, line)
        assert "ctx" not in line and "context" not in line, (repo, line)
    assert writer_fit_line("google/gemma-4-E2B-it").startswith("about 10.2 GB")


def test_the_fit_words_come_from_the_gate_s_own_projection():
    for repo in ("Qwen/Qwen3.5-4B", "unsloth/Llama-3.2-3B-Instruct", "Qwen/Qwen3.8-27B"):
        tags = catalog.fit_tags_for(repo)
        assert tags, repo
        assert ("fits " + " ".join(tags)) in writer_fit_line(repo), repo


def test_a_gated_row_says_so_after_its_size_and_native_rows_claim_no_fit():
    gated = writer_fit_line("google/gemma-2-2b-it")
    assert gated.split("; ")[1].startswith("gated"), gated
    native = writer_fit_line("comfy_native:gemma4-e2b-it-int8-convrot")
    assert native.startswith("about 5.2 GB") and "fit not rated" in native
    assert "fits " not in native


def test_a_hosted_row_claims_no_download_and_no_fit():
    hosted = [m.repo_id for m in catalog._active_curated_models()
              if getattr(m, "provider", "local") not in ("local", "comfy_native")]
    assert hosted
    for repo in hosted:
        line = writer_fit_line(repo)
        assert line.startswith("hosted"), (repo, line)
        assert "fits " not in line, (repo, line)


def test_a_projection_that_throws_reads_as_unknown_never_as_fits_none(monkeypatch):
    def broken(_repo):
        raise RuntimeError("projection unavailable")
    monkeypatch.setattr(catalog, "fit_tags_for", broken)
    line = writer_fit_line("Qwen/Qwen3.5-4B")
    assert line.endswith("fit unknown (could not be estimated)"), line
    assert "fits none" not in line


def test_a_row_without_a_size_basis_reads_as_unknown_never_as_fits_none(monkeypatch):
    """fit_tags_for answers () for a row it cannot rate (no size basis); the
    line must not turn that into 'fits none of the rated machines'."""
    import dataclasses
    repo = "Qwen/Qwen3.5-4B"
    row = dataclasses.replace(catalog._by_repo_id()[repo], approx_safetensors_gb=0.0)
    monkeypatch.setattr(catalog, "_by_repo_id", lambda: {repo: row})
    assert catalog.fit_tags_for(repo) == ()
    line = writer_fit_line(repo)
    assert line.endswith("fit unknown (no size basis)"), line
    assert "fits none" not in line


def test_a_line_never_raises_on_an_unknown_row_and_claims_nothing_about_it():
    """A tooltip must render even for a row nothing can estimate -- and say it
    is unknown, never that it fits nothing."""
    assert writer_fit_line("definitely/not-a-real-model-xyz") == \
        "not in the writer catalog; size and fit unknown"


def test_the_table_lists_every_local_and_native_writer_then_the_legend():
    table = writer_tooltip_table()
    for m in catalog._active_curated_models():
        if getattr(m, "provider", "local") not in ("local", "comfy_native"):
            continue
        if m.repo_id == catalog.DEFAULT_LLM_NF4:
            continue
        assert "- %s: " % m.repo_id in table, m.repo_id
    assert "openrouter:" not in table
    assert table.endswith(catalog.WRITER_FIT_LEGEND)
    assert "GB download" not in table
