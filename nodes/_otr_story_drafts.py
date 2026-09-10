"""My Story drafts -- the listener's submitted input, saved before it is spent.

THE ONE STORAGE OWNER. The writer and the workflow validator both reach this
module and nothing else writes a draft, because two writers of one file is how
a validator and a writer end up disagreeing about what was submitted.

WHY A DRAFT EXISTS AT ALL. A person types an idea and presses Run; several
minutes of model work follow, and any of it can fail. Without this file their
words are gone with the run. The draft is written at ADMISSION -- after the
input is judged admissible and before any generation starts -- so a failure, a
cancellation or a crash leaves the submission on disk to retry from.

WHAT IT IS NOT. It is not a replay system and not an episode archive: nothing
here restores a rendered episode, imports an old bundle, or reconstructs
metadata. Reusing a draft means generating a fresh episode from the same words.

IDENTITY IS THE CONTENT, NOT THE CLOCK. The digest covers the submitted fields
and the pre-roll controls only, so the validator and the writer compute the
same identity for one run, and an unchanged resubmission verifies the existing
draft instead of writing a second copy of it. Timestamps and the submitting
node ride ALONGSIDE the content, never inside its identity.

Stdlib only, lazy ComfyUI import. UTF-8, no BOM, ASCII source.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    from . import _otr_paths as _OTRP
except ImportError:  # pragma: no cover -- flat / standalone test import
    import _otr_paths as _OTRP  # type: ignore

try:
    from . import _otr_story_input as _SI
except ImportError:  # pragma: no cover -- flat / standalone test import
    import _otr_story_input as _SI  # type: ignore

log = logging.getLogger("OTR")

#: Directory under the per-machine state tier. The output contract admits only
#: `episodes` and `obs` at the top level, so drafts live inside the shared
#: state directory rather than adding a third top-level entry.
DRAFTS_DIRNAME = "story_drafts"

#: The one filename inside a draft directory.
DRAFT_FILENAME = "input.json"

STATUS_CREATED = "created"
STATUS_VERIFIED = "verified"


class StoryDraftError(RuntimeError):
    """The draft could not be written or could not be trusted.

    RAISED, never swallowed. "We saved your input" is a promise; a storage
    failure that let generation proceed anyway would make it a false one, and
    the listener would only discover it after the run they were relying on.
    """


@dataclass(frozen=True)
class DraftReceipt:
    """Where the submission landed and whether this call wrote it."""

    path: str
    digest: str
    status: str

    def to_meta(self) -> dict:
        return {
            "schema_version": _SI.STORY_INPUT_SCHEMA,
            "digest": self.digest,
            "path": self.path,
            "status": self.status,
        }


@dataclass(frozen=True)
class SubmissionContext:
    """Who submitted this run, for the record beside the content.

    Never part of the digest: the same words submitted from a different node
    are the same submission.
    """

    prompt_id: str = ""
    node_id: str = ""
    caller: str = ""

    def as_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "node_id": self.node_id,
            "caller": self.caller,
        }

    @property
    def is_empty(self) -> bool:
        return not (self.prompt_id or self.node_id or self.caller)


def drafts_root() -> Path:
    """``<output>/otr/episodes/_shared/state/story_drafts``."""
    return _OTRP.otr_state_dir() / DRAFTS_DIRNAME


def draft_dir(digest: str) -> Path:
    return drafts_root() / str(digest)


def draft_path(digest: str) -> Path:
    return draft_dir(digest) / DRAFT_FILENAME


def _executing_context() -> Optional[SubmissionContext]:
    """ComfyUI's own answer to "which prompt and node is running?".

    Imported INSIDE the function so this module can be imported by a test with
    no ComfyUI on sys.path. The installed context is a NamedTuple with
    ``prompt_id`` / ``node_id`` / ``list_index``; a dict-shaped guess would
    silently read nothing.
    """
    try:
        from comfy_execution.utils import get_executing_context  # type: ignore
    except Exception:  # noqa: BLE001 -- standalone / older core
        return None
    try:
        ctx = get_executing_context()
    except Exception:  # noqa: BLE001 -- never let context lookup kill a run
        return None
    if ctx is None:
        return None
    prompt_id = str(getattr(ctx, "prompt_id", "") or "")
    node_id = str(getattr(ctx, "node_id", "") or "")
    if not (prompt_id or node_id):
        return None
    return SubmissionContext(prompt_id=prompt_id, node_id=node_id)


def resolve_context(
    context: "SubmissionContext | None" = None,
    caller: str = "",
) -> SubmissionContext:
    """Explicit context wins, then ComfyUI's, then an explicit caller.

    A call with none of the three RAISES rather than inventing a shared
    identity. A made-up default (a fixed node id, say) would make two
    independent writers look like the same submitter, which is exactly the
    collision this refusal prevents.
    """
    if context is not None and not context.is_empty:
        return context
    live = _executing_context()
    if live is not None:
        return live
    named = str(caller or "").strip()
    if named:
        return SubmissionContext(caller=named)
    raise StoryDraftError(
        "my_story: cannot record who submitted this input. Inside ComfyUI the "
        "execution context supplies it; a direct call must pass an explicit "
        "context= or caller= instead."
    )


def _read_existing(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StoryDraftError(
            "my_story: an existing draft at %s could not be read (%s: %s). "
            "Move or delete that file and run again."
            % (path, type(exc).__name__, exc)
        ) from exc


def _verify_existing(path: Path, bundle: Any) -> DraftReceipt:
    """An existing draft must still hash to the identity it is filed under.

    Recomputed from the stored fields rather than trusting the stored digest:
    a file whose contents were edited by hand would otherwise keep answering
    for a submission it no longer holds.
    """
    stored = _read_existing(path)
    if not isinstance(stored, dict):
        raise StoryDraftError("my_story: draft at %s is not an input object" % path)
    fields = stored.get("fields")
    request = stored.get("request")
    if not isinstance(fields, dict) or not isinstance(request, dict):
        raise StoryDraftError(
            "my_story: the draft at %s is missing its fields or request "
            "block. Move or delete that file and run again." % path
        )
    try:
        recomputed = _SI.compute_digest(
            _SI.RawStoryFields(**{k: str(v or "") for k, v in fields.items()}),
            _SI.StoryRequest(**{
                "num_characters": int(request.get("num_characters") or 0),
                "act_count": str(request.get("act_count") or ""),
                "include_act_breaks": bool(request.get("include_act_breaks")),
                "source_bank_requested": str(
                    request.get("source_bank_requested") or ""),
                "visual_style_requested": str(
                    request.get("visual_style_requested") or ""),
            }),
        )
    except (TypeError, ValueError) as exc:
        raise StoryDraftError(
            "my_story: the draft at %s does not match the current input "
            "schema (%s). Move or delete that file and run again."
            % (path, exc)
        ) from exc
    if (recomputed != bundle.digest
            or stored.get("digest") != bundle.digest
            or stored.get("schema_version") != _SI.STORY_INPUT_SCHEMA
            or stored.get("fields") != bundle.fields.as_dict()
            or stored.get("request") != bundle.request.as_dict()
            or stored.get("normalized") != bundle.normalized.as_dict()):
        raise StoryDraftError(
            "my_story: the draft directory %s holds input that hashes to %s, "
            "not %s. Two different submissions cannot share one identity; "
            "move or delete that directory and run again."
            % (path.parent, recomputed[:12], bundle.digest[:12])
        )
    return DraftReceipt(path=str(path), digest=bundle.digest,
                        status=STATUS_VERIFIED)


def _atomic_write(path: Path, payload: dict) -> None:
    """Write beside the target, then replace it in one step.

    Same directory on purpose: ``os.replace`` is atomic only within a
    filesystem, and a temp file elsewhere can land on a different volume. A
    half-written draft that looked complete is worse than no draft.
    """
    tmp = path.parent / ("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(
            json.dumps(payload, sort_keys=True, ensure_ascii=True, indent=1),
            encoding="utf-8",
        )
        os.replace(str(tmp), str(path))
    except OSError as exc:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:  # pragma: no cover -- cleanup is best-effort
            pass
        raise StoryDraftError(
            "my_story: could not save your story input to %s (%s: %s). "
            "Nothing was generated; fix the storage problem and run again."
            % (path, type(exc).__name__, exc)
        ) from exc


def ensure_draft(
    bundle: Any,
    *,
    context: "SubmissionContext | None" = None,
    caller: str = "",
    now: str = "",
) -> DraftReceipt:
    """Persist the submission, or verify the one already filed under it.

    Idempotent by content: the validator admits and writes, the writer admits
    the same bundle minutes later and verifies the same file. Returns the
    receipt the ledger stamps.
    """
    submitter = resolve_context(context, caller)
    path = draft_path(bundle.digest)
    if path.exists():
        return _verify_existing(path, bundle)
    payload = dict(bundle.as_dict())
    # Alongside the content, never inside its identity.
    from datetime import datetime, timezone
    payload["created_at"] = str(now or datetime.now(timezone.utc).isoformat())
    payload["submitted_by"] = submitter.as_dict()
    _atomic_write(path, payload)
    log.info("[my_story] saved story input %s -> %s", bundle.digest[:12], path)
    return DraftReceipt(path=str(path), digest=bundle.digest,
                        status=STATUS_CREATED)


__all__ = [
    "DRAFTS_DIRNAME",
    "DRAFT_FILENAME",
    "DraftReceipt",
    "STATUS_CREATED",
    "STATUS_VERIFIED",
    "StoryDraftError",
    "SubmissionContext",
    "draft_dir",
    "draft_path",
    "drafts_root",
    "ensure_draft",
    "resolve_context",
]
