"""Shared provider-side fan-out -- video, stills, TTS, music.

Partner HTTP waits, not VRAM. Jobs may finish in any order; callers
commit in the original sequence. Local GPU work never belongs here.
"""
from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field

try:
    from . import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore


class CloudFanoutError(RuntimeError):
    """A ready-queue that never drained, or an item with no id."""


@dataclass
class FanoutOutcome:
    results: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    stuck_ids: list = field(default_factory=list)
    halted_ids: list = field(default_factory=list)


def cloud_fanout_workers() -> int:
    """How many provider-side jobs may be in flight.

    ``OTR_CLOUD_FANOUT`` is the one knob. ``OTR_CLOUD_VIDEO_FANOUT`` is
    the older name and still wins if the new one is unset, so existing
    launch.env pins keep working. Unset defaults to 4 (operator
    2026-09-16: 8-wide Luma 429s and stacked LTX reserves). ``1`` (or
    less) is serial -- callers skip this helper.
    """
    raw = str(otr_env.get("OTR_CLOUD_FANOUT", "") or "").strip()
    if not raw:
        raw = str(otr_env.get("OTR_CLOUD_VIDEO_FANOUT", "") or "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            return 4
    return 4


def snapshot_prompt_id():
    """Prompt id from THIS Comfy execution thread.

    Partner invoke reads comfy_execution.utils.get_executing_context,
    which is executor-thread local. ThreadPool workers do not see it,
    so callers snapshot here and this helper re-attaches it inside
    each worker.
    """
    try:
        from .cloud_media_invoke import current_prompt_id
        pid = str(current_prompt_id() or "").strip()
    except Exception:  # noqa: BLE001 -- unit tests have no Comfy context
        return None
    return pid or None


def adapter_is_cloud_side(obj) -> bool:
    """True when this adapter is a partner/API job, not a local GPU load."""
    if obj is None:
        return False
    if getattr(obj, "provider_side", False):
        return True
    if getattr(obj, "native", True) is False:
        return True
    name = str(getattr(obj, "name", "") or getattr(obj, "engine_id", "") or "")
    return name.startswith("cloud_")


def adapter_draws_partner_heartbeat(obj) -> bool:
    """True when this adapter's calls go through a Comfy partner node.

    The ``cloud_*`` engines call ``cloud_media_invoke.invoke_partner_node``,
    whose wait loop redraws the node's progress bar every 20 s. Every other
    engine -- local ones, and the direct Google API lanes (``google_tts``,
    ``google_image``, ``google_lyria``) that are cloud-side without being
    partner nodes -- draws nothing while it works, so a caller may give it
    an item bar (``node_progress.NodeProgress``) without two bars fighting.
    """
    if obj is None:
        return False
    name = str(getattr(obj, "name", "") or getattr(obj, "engine_id", "") or "")
    return name.startswith("cloud_")


def run_cloud_fanout(
    items,
    *,
    item_id,
    execute,
    predecessors=None,
    workers=None,
    prompt_id=None,
    on_item_done=None,
):
    """Request many provider jobs; collect by id.

    ``execute(item)`` runs in a worker. ``predecessors(item)`` returns
    ids that must SUCCEED first (default empty -- every cheap Vidu /
    Luma still / ElevenLabs line is ready at t=0).

    Results and errors are keyed by id. The caller walks the original
    ``items`` order to commit or to raise the first error, so a clip
    that landed first never becomes beat one.

    ``on_item_done(item)``, when given, is called on the CALLING thread
    each time a job finishes, success or error -- the executing node's
    own thread, which is where a ComfyUI progress bar must be updated.
    """
    items = list(items or ())
    if not items:
        return FanoutOutcome()
    n_workers = cloud_fanout_workers() if workers is None else int(workers)
    n_workers = max(1, min(n_workers, len(items)))
    pred_fn = predecessors or (lambda _item: ())

    def _pid(item):
        sid = str(item_id(item) or "").strip()
        if not sid:
            raise CloudFanoutError("cloud fan-out item has an empty id")
        return sid

    pending = list(items)
    submitted = set()
    finished_ok = set()
    results = {}
    errors = {}
    futs = {}
    halt_submit = False

    def _ready():
        ready = []
        for item in pending:
            deps = tuple(
                str(p) for p in (pred_fn(item) or ()) if p not in (None, ""))
            if all(d in finished_ok for d in deps):
                ready.append(item)
        return ready

    def _submit(pool):
        from .cloud_media_invoke import bind_prompt_id

        if halt_submit:
            return
        for item in _ready():
            # Keep only n_workers in flight. Submitting the whole ready
            # set at t=0 queues every remaining beat, so a spend-cap
            # refusal cannot stop the rest (live 2026-09-16: hundreds of
            # LTX reserves after $40).
            if len(futs) >= n_workers:
                break
            sid = _pid(item)
            if sid in submitted:
                continue
            submitted.add(sid)

            def _run(it=item):
                if prompt_id:
                    with bind_prompt_id(str(prompt_id)):
                        return execute(it)
                return execute(it)

            futs[pool.submit(_run)] = item

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        _submit(pool)
        while futs:
            done, _not = concurrent.futures.wait(
                tuple(futs),
                return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done:
                item = futs.pop(fut)
                sid = _pid(item)
                pending = [s for s in pending if _pid(s) != sid]
                try:
                    results[sid] = fut.result()
                    finished_ok.add(sid)
                except Exception as exc:  # noqa: BLE001 -- raise in caller order
                    errors[sid] = exc
                    from .cloud_media_backend import is_cloud_budget_error
                    if is_cloud_budget_error(exc):
                        halt_submit = True
                if on_item_done is not None:
                    on_item_done(item)
            _submit(pool)

    leftover = [_pid(s) for s in pending]
    if halt_submit:
        return FanoutOutcome(
            results=results, errors=errors, stuck_ids=[],
            halted_ids=leftover)
    return FanoutOutcome(
        results=results, errors=errors, stuck_ids=leftover)
