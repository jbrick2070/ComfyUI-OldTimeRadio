"""``episode_title`` -- the ONE rule for what an episode is called on screen.

Two nodes need the episode's title and neither of them authors it: the video
renderer paints it on the title card, and the Episode Assembler names it in the
log line and in its ``episode_info`` output. The title itself is written by
``OTR_LedgerScriptWriter`` -- its J.5 post-composition pass stamps
``meta.episode_title`` on every run -- so both consumers are reading the same
ledger field, and the chain that reads it belongs in one place rather than
inline in the consumer that happened to need it first.

THE CHAIN, in order, and the order is the whole contract::

    1. led["meta"]["episode_title"]   the writer's stamp; the normal answer
    2. led["meta"]["title"]           forward-compat slot
    3. led["title"]                   top-level; pre-LPL ledgers read off disk
    4. "Signal Lost <timestamp>"      last resort, so a render always has a card

``news_used[0].headline`` and ``meta.news_seed.headline`` are deliberately NOT
rungs. Both surface news/outline text as a title and neither is what belongs on
screen (Path B, confirmed 2026-05-09).

THERE IS NO WIDGET RUNG (2026-09-14). A typed override used to sit between
``led.title`` and the timestamp, declared on the renderer and on the Assembler.
It could only ever win on a run whose ledger carried no title at all, and having
three nodes declare the same field made it ambiguous which one the operator was
supposed to type into. The writer's widget is the single workflow-facing owner;
everything downstream reads the ledger.

THE LAST RESORT IS LOAD-BEARING, not a nicety. The published episode's title
card comes from this chain, and a run that reaches the end of it must still
produce a title rather than fail -- the daily stream of episodes reaching
``otr/obs/`` is never interrupted for a cosmetic title problem. A caller that
sees ``timestamp_lastresort`` should say so loudly in its own log; it must not
refuse to render.

Pure stdlib, nothing heavy at import time -- both consumers pull this in without
dragging anything behind it.
"""
from __future__ import annotations

import time

#: Rungs in the order they are consulted: (source label, container, key). The
#: source label is what the caller records and what the ledger's forensics read
#: back, so these strings are part of the contract, not decoration.
TITLE_RUNGS = (
    ("led.meta.episode_title", "meta", "episode_title"),
    ("led.meta.title", "meta", "title"),
    ("led.title (legacy stamp)", "led", "title"),
)

#: The source recorded when every rung above came back empty.
LAST_RESORT_SOURCE = "timestamp_lastresort"

#: Prefix of the generated last-resort title.
LAST_RESORT_PREFIX = "Signal Lost"


def last_resort_title() -> str:
    """A unique, always-available title so a run can finish.

    Shaped ``Signal Lost YYYYMMDD HHMMSS`` -- readable on a title card and
    distinct per run, which matters because it is also how an operator scanning
    ``otr/obs/`` spots that the writer produced no title.
    """
    return "%s %s" % (LAST_RESORT_PREFIX, time.strftime("%Y%m%d %H%M%S"))


def title_candidates(ledger) -> dict:
    """Every rung's stripped value, in chain order, keyed by its source label.

    Returned whole rather than as a bare winner because the losing rungs are
    the useful half of a title complaint: "which slots were empty" is the
    question asked when a card comes up wrong, and a caller that re-derives them
    is a second copy of this chain.
    """
    led = ledger if isinstance(ledger, dict) else {}
    meta = led.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    found = {}
    for source, container, key in TITLE_RUNGS:
        holder = meta if container == "meta" else led
        # str() before strip(): a ledger loaded off disk can carry a number or
        # None here, and a title chain whose job is to always produce a title
        # must not be the thing that raises on one.
        found[source] = str(holder.get(key) or "").strip()
    return found


def resolve_episode_title(ledger) -> tuple:
    """Resolve ``ledger`` to ``(title, source)``.

    ``source`` names the rung that won -- one of the labels in
    :data:`TITLE_RUNGS` or :data:`LAST_RESORT_SOURCE` -- and is worth recording:
    it is the difference between "the writer titled this episode" and "nothing
    did, and you are looking at a timestamp".
    """
    found = title_candidates(ledger)
    for source, _container, _key in TITLE_RUNGS:
        value = found.get(source) or ""
        if value:
            return value, source
    return last_resort_title(), LAST_RESORT_SOURCE
