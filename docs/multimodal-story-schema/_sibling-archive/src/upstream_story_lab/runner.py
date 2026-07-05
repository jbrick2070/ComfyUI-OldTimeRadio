"""Minimal lab runner for lab-executable pipelines (simple_4 only, v1 scope).

Python owns pass sequencing, pass status, and failure reporting. The runner
refuses descriptive pipelines, refuses to skip a failed pass, and never falls
back to another pipeline (kibitz r1: without this runner the experimental
pipeline is fiction; with more than this it is scope creep).
"""

from __future__ import annotations

from typing import Callable

from .contracts import PipelineSpec, StoryPack


class PipelineRunError(RuntimeError):
    """A pass failed. Message names the exact pass. No fallback exists."""


#: FakeLLM signature: (slot_role, pass_id, prompt) -> output text.
LlmCallable = Callable[[str, str, str], str]


def _context_block(pack: StoryPack, source_material_text: str,
                   ledger_schema_text: str) -> str:
    """The run context every pass can see (roundtable pass01, all three
    panelists CONFIRMED the vacuum: pass prompts referenced guardrails and
    schema the runner never supplied). Tone guardrails are POSITIVE
    constraints and are injected; forbidden patterns stay metadata (the
    locked negation-copy rule) and are enforced by the post-generation scan."""

    parts = []
    if source_material_text.strip():
        parts.append("SOURCE MATERIAL:\n" + source_material_text.strip())
    if pack.tone_guardrails:
        parts.append("TONE GUARDRAILS:\n- " + "\n- ".join(pack.tone_guardrails))
    if ledger_schema_text.strip():
        parts.append("LEDGER SCHEMA:\n" + ledger_schema_text.strip())
    return "\n\n".join(parts)


def run_pipeline(pack: StoryPack, pipeline: PipelineSpec,
                 llm: LlmCallable, *,
                 source_material_text: str = "",
                 ledger_schema_text: str = "") -> dict[str, str]:
    if not pipeline.executable_in_lab:
        raise PipelineRunError(
            f"pipeline {pipeline.story_pipeline_id!r} is descriptive metadata "
            "(production-native); the lab runner refuses to execute it"
        )
    if pack.story_pipeline_id != pipeline.story_pipeline_id:
        raise PipelineRunError(
            f"pack {pack.story_model_id!r} declares pipeline "
            f"{pack.story_pipeline_id!r}, not {pipeline.story_pipeline_id!r}"
        )
    context = _context_block(pack, source_material_text, ledger_schema_text)
    outputs: dict[str, str] = {}
    previous: str = ""
    for decl in pipeline.passes:
        if not decl.seam_refs:
            raise PipelineRunError(
                f"pass {decl.pass_id!r} declares no seam prompt to execute"
            )
        seam = decl.seam_refs[0]
        prompt = (pack.prompt_stages.get(seam) or "").strip()
        if not prompt:
            raise PipelineRunError(
                f"pass {decl.pass_id!r}: pack {pack.story_model_id!r} has no "
                f"prompt for seam {seam!r} (JSON owns content; nothing is invented)"
            )
        # Output CHAINING: each pass sees the previous pass's output - without
        # this the 4-pass experiment is fiction (roundtable pass01).
        pieces = [prompt]
        if context:
            pieces.append("RUN CONTEXT:\n" + context)
        if previous:
            pieces.append("PREVIOUS PASS OUTPUT:\n" + previous)
        full_prompt = "\n\n".join(pieces)
        try:
            result = llm(decl.slot, decl.pass_id, full_prompt)
        except Exception as exc:
            raise PipelineRunError(
                f"pass {decl.pass_id!r} raised: {exc} - failing loudly; "
                "no fallback pipeline exists"
            ) from exc
        if not (result or "").strip():
            raise PipelineRunError(
                f"pass {decl.pass_id!r} returned empty output - failing loudly; "
                "no fallback pipeline exists"
            )
        outputs[decl.pass_id] = result
        previous = result
    return outputs
