"""otr_ledger_view.py -- read any OTR ledger fast, and grade it honestly.

    python scripts/otr_ledger_view.py                  # newest episode
    python scripts/otr_ledger_view.py <episode_id>     # by id, or a fragment
    python scripts/otr_ledger_view.py <path_to.json>   # an explicit ledger
    python scripts/otr_ledger_view.py --list 20        # the 20 newest
    python scripts/otr_ledger_view.py --html out.html  # a page to eyeball
    python scripts/otr_ledger_view.py --json           # machine-readable
    python scripts/otr_ledger_view.py a b --html x.html   # compare two runs
    python scripts/otr_ledger_view.py --watch          # LIVE, while it writes

THE LIVE WINDOW
---------------
``--watch`` is for the question "is it working, or is it stuck?", which from
the outside look identical: a healthy decode and a decode cycling the same
paragraph both keep the GPU busy and both keep the token counter climbing.
The window shows the three things that tell them apart -- which pass is
running, the token rate, and the TAIL OF THE STREAM sampled over time. When
successive samples say the same thing while the counter keeps rising, the
model is repeating itself, and the window says so in as many words.

It only READS: the ledger on disk, the server log, and nvidia-smi. It never
touches the run.

WHAT IT GRADES, AND WHAT IT REFUSES TO GRADE
--------------------------------------------
Operator ruling 2026-08-14, and it is only two things:

    F1  ACTION IN THE LEDGER -- a spoken row carrying stage business,
        delivery notes or narration. Every sealed line becomes TTS audio,
        so a stage direction inside a line gets read aloud on air.
    F2  SPEAKER / LINE MISMATCH -- a row carrying another character's
        words, or carrying no speaker at all.

NOTHING ELSE IS A FAILURE. Prose quality is not a defect (the 2026-08-04
"story quality is done" directive); character consistency explicitly is. So
this tool GRADES F1 and F2, and REPORTS everything else -- length, shape,
pass receipts -- as an observation with no verdict attached.

It DETECTS and explains. It never edits a ledger, and it never writes one:
rewriting spoken prose is a model's job, never Python's.

The F1 detectors are deliberately noisy in one direction. A false positive
costs a human five seconds of reading; a false negative ships a stage
direction to the microphone. Every finding names the exact phrase it tripped
on so the reader can overrule it at a glance.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Where rendered episodes live. Overridable so a bench tree or a copied
#: episode can be inspected without moving files around.
EPISODES_ROOT = Path(
    os.environ.get("OTR_OUTPUT_DIR")
    or r"C:\Users\jeffr\Documents\ComfyUI\output"
) / "otr" / "episodes"

VOICED_ROLES = ("character", "announcer")


# ---------------------------------------------------------------------------
# F1 -- action inside a row that is supposed to be pure speech
# ---------------------------------------------------------------------------
# Each pattern is one WAY narration leaks in, kept separate so a finding can
# say which one fired instead of "looks like prose".

_F1_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "stage direction in brackets",
        re.compile(r"\([^)]{2,}\)|\[[^\]]{2,}\]|\*[^*]{2,}\*"),
    ),
    (
        "attribution verb",
        re.compile(
            r"\b(?:said|says|saying|murmurs?|murmured|whispers?|whispered"
            r"|shouts?|shouted|replies|replied|asks|asked|answers?|answered"
            r"|adds?|added|continues?|continued|mutters?|muttered|snaps?"
            r"|snapped|breathes?|breathed|calls? out|called out)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "third-person action",
        re.compile(
            r"\b(?:turns?|turned|leans?|leaned|reaches?|reached|walks?|walked"
            r"|steps?|stepped|nods?|nodded|smiles?|smiled|frowns?|frowned"
            r"|glances?|glanced|stares?|stared|drums?|drummed|paces?|paced"
            r"|grabs?|grabbed|enters?|entered|leaves?|exits?|exited|stands?"
            r"|stood|sits?|sat|rises?|rose|sighs?|sighed|shrugs?|shrugged"
            r"|laughs?|laughed|pauses?|paused)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "delivery note",
        re.compile(
            r"\b(?:her|his|their|its) (?:voice|eyes|hands?|face|gaze|tone"
            r"|breath|shoulders?|fingers?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "quoted speech inside prose",
        re.compile(r"[\"\u201c\u2018\u2019\u201d].{8,}[\"\u201c\u2018\u2019\u201d]"),
    ),
    (
        "speaker label",
        re.compile(r"^\s*[A-Z][A-Z .'\u2019-]{1,30}:", re.MULTILINE),
    ),
)


@dataclass
class RowFinding:
    line_id: str
    kind: str
    detail: str


@dataclass
class LedgerReport:
    path: Path
    episode_id: str
    source_bank: str
    owner_bank: str
    rows: list[dict[str, Any]]
    cast: dict[str, str]
    beats: dict[str, dict[str, Any]]
    f1: list[RowFinding] = field(default_factory=list)
    f2: list[RowFinding] = field(default_factory=list)
    coda: dict[str, Any] = field(default_factory=dict)
    receipts: dict[str, Any] = field(default_factory=dict)

    @property
    def voiced(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r.get("speaker_role") in VOICED_ROLES]

    @property
    def spoken_words(self) -> int:
        return sum(int(r.get("word_count") or 0) for r in self.voiced)

    @property
    def clean(self) -> bool:
        """The operator's acceptance test, and only that."""
        return not self.f1 and not self.f2 and bool(self.voiced)


def _first_match(text: str, pattern: re.Pattern[str]) -> str:
    found = pattern.search(text)
    if not found:
        return ""
    snippet = " ".join(found.group(0).split())
    return snippet[:60]


def source_anchors(data: dict) -> list[str]:
    """Verbatim strings a coda could say to name its real source.

    Same rule the lane's own verifier uses: entity names of three characters
    or more (a two-letter name matches inside ordinary words and proves
    nothing) plus source-spanned figures.
    """
    lane = (data.get("meta") or {}).get("scifi_codex") or {}
    index = lane.get("fact_index") or {}
    anchors: list[str] = []
    for entity in index.get("entities") or []:
        name = str((entity or {}).get("name") or "").strip()
        if len(name) >= 3:
            anchors.append(name)
    for number in index.get("numbers") or []:
        token = str((number or {}).get("verbatim") or "").strip()
        if token:
            anchors.append(token)
    seen: set[str] = set()
    ordered: list[str] = []
    for anchor in anchors:
        key = anchor.casefold()
        if key not in seen:
            seen.add(key)
            ordered.append(anchor)
    return ordered


def names_anchor(text: str, anchors: Iterable[str]) -> str:
    for anchor in anchors:
        pattern = (
            r"(?<![A-Za-z0-9])" + re.escape(anchor) + r"(?![A-Za-z0-9])"
        )
        if re.search(pattern, text, re.IGNORECASE):
            return anchor
    return ""


def grade(path: Path) -> LedgerReport:
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("meta") or {}
    lane = meta.get("scifi_codex") or {}
    rows = [r for r in (data.get("lines") or []) if isinstance(r, dict)]
    cast = {
        str(r.get("char_id")): str(r.get("name") or "")
        for r in (data.get("cast") or []) if isinstance(r, dict)
    }
    beats = {
        str(r.get("beat_id")): r
        for r in (data.get("beats") or []) if isinstance(r, dict)
    }
    report = LedgerReport(
        path=path,
        episode_id=str(data.get("episode_id") or path.stem),
        source_bank=str(meta.get("source_bank") or ""),
        owner_bank=str(meta.get("owner_bank") or ""),
        rows=rows,
        cast=cast,
        beats=beats,
    )

    for row in rows:
        if row.get("speaker_role") not in VOICED_ROLES:
            continue
        line_id = str(row.get("line_id") or "")
        text = str(row.get("text") or "")

        for kind, pattern in _F1_PATTERNS:
            hit = _first_match(text, pattern)
            if hit:
                report.f1.append(RowFinding(line_id, kind, hit))

        speaker = str(row.get("speaker") or "").strip()
        beat = beats.get(str(row.get("beat_id") or "")) or {}
        owner = str(beat.get("speaker") or "").strip()
        if not speaker:
            report.f2.append(RowFinding(
                line_id, "no speaker",
                "the row does not say who is talking"))
        elif owner and speaker != owner:
            report.f2.append(RowFinding(
                line_id, "row disagrees with its beat",
                f"row says {speaker!r}, beat says {owner!r}"))
        elif row.get("speaker_role") == "character":
            name = cast.get(str(row.get("char_id") or ""))
            if name and speaker != name:
                report.f2.append(RowFinding(
                    line_id, "row disagrees with the cast",
                    f"row says {speaker!r}, cast says {name!r}"))

    anchors = source_anchors(data)
    voiced = report.voiced
    last = voiced[-1] if voiced else None
    flagged = [
        r for r in rows if "news_coda" in (r.get("compose_flags") or [])
    ]
    report.coda = {
        "anchors": anchors,
        "row_count": len(flagged),
        "is_last": bool(last is not None and flagged and last is flagged[-1]),
        "announcer": bool(
            flagged and flagged[-1].get("speaker_role") == "announcer"),
        "text": str(flagged[-1].get("text") or "") if flagged else "",
        "names": names_anchor(
            str(flagged[-1].get("text") or "") if flagged else "", anchors),
        "lane_status": ((lane.get("news_coda") or {}).get("status") or ""),
    }

    schedule = (lane.get("call_journal") or {}).get("script_schedule") or {}
    report.receipts = {
        "act_count": (lane.get("call_journal") or {})
        .get("script_transport", {}).get("act_count"),
        "schedule": schedule.get("shape") or "",
        "beat_jobs": schedule.get("beat_dialogue_jobs"),
        "review_jobs": schedule.get("scene_review_jobs"),
        "creative_model": meta.get("creative_writing_model")
        or meta.get("creative_model") or "",
        "freeze_verdict": meta.get("freeze_verdict") or "",
        "news_coda_status": report.coda["lane_status"],
    }
    return report


# ---------------------------------------------------------------------------
# Finding a ledger
# ---------------------------------------------------------------------------

def episode_ledgers(root: Path | None = None) -> list[Path]:
    """Every episode ledger under the output tree, newest first."""
    base = root or EPISODES_ROOT
    if not base.is_dir():
        return []
    found: list[Path] = []
    for episode in base.iterdir():
        if not episode.is_dir() or episode.name.startswith("_"):
            continue
        found.extend((episode / "audio").glob("*_ledger.json"))
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found


def resolve(target: str | None, root: Path | None = None) -> Path:
    """A path, an episode id, a fragment of one, or the newest run."""
    if target:
        direct = Path(target)
        if direct.is_file():
            return direct
        if direct.is_dir():
            found = sorted((direct / "audio").glob("*_ledger.json"))
            if found:
                return found[0]
    candidates = episode_ledgers(root)
    if not candidates:
        raise SystemExit(
            f"no episode ledger under {root or EPISODES_ROOT}")
    if not target or target == "latest":
        return candidates[0]
    needle = target.casefold()
    for path in candidates:
        if needle in path.parent.parent.name.casefold():
            return path
    raise SystemExit(
        f"no episode matches {target!r}; try --list to see what is there")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _verdict_line(report: LedgerReport) -> str:
    if not report.voiced:
        return "EMPTY -- the ledger has no spoken rows at all"
    if report.clean:
        return "CLEAN -- no action in a spoken row, every row names its speaker"
    parts = []
    if report.f1:
        parts.append(f"{len({f.line_id for f in report.f1})} row(s) with action")
    if report.f2:
        parts.append(f"{len({f.line_id for f in report.f2})} row(s) misattributed")
    return "DIRTY -- " + ", ".join(parts)


def render_text(report: LedgerReport) -> str:
    out: list[str] = []
    add = out.append
    add(f"LEDGER   {report.path}")
    add(f"episode  {report.episode_id}")
    add(f"bank     {report.source_bank or '?'}"
        + (f" (owner {report.owner_bank})" if report.owner_bank else ""))
    receipts = report.receipts
    if receipts.get("schedule"):
        add(f"schedule {receipts['schedule']} "
            f"beats={receipts.get('beat_jobs')} "
            f"reviews={receipts.get('review_jobs')}")
    if receipts.get("act_count") is not None:
        add(f"acts     {receipts['act_count']}")
    if receipts.get("freeze_verdict"):
        add(f"freeze   {receipts['freeze_verdict']}")
    add(f"rows     {len(report.rows)} total, {len(report.voiced)} spoken, "
        f"{report.spoken_words} spoken words")
    if report.cast:
        add("cast     " + ", ".join(
            f"{cid}={name}" for cid, name in report.cast.items()))
    add("")
    add(f"VERDICT  {_verdict_line(report)}")
    add("         (F1 action and F2 speaker are the ONLY failures; prose "
        "quality is not graded)")
    add("")

    f1_by_row: dict[str, list[RowFinding]] = {}
    for finding in report.f1:
        f1_by_row.setdefault(finding.line_id, []).append(finding)
    f2_by_row: dict[str, list[RowFinding]] = {}
    for finding in report.f2:
        f2_by_row.setdefault(finding.line_id, []).append(finding)

    add("--- ROWS " + "-" * 62)
    for row in report.rows:
        line_id = str(row.get("line_id") or "")
        role = str(row.get("speaker_role") or "")
        speaker = row.get("speaker")
        flags = ",".join(row.get("compose_flags") or [])
        mark = ""
        if line_id in f1_by_row or line_id in f2_by_row:
            mark = "  <-- see findings"
        add("")
        add(f"{line_id:<18} {role:<10} speaker={speaker!r}"
            + (f"  [{flags}]" if flags else "") + mark)
        text = str(row.get("text") or "")
        if text:
            for chunk in _wrap(text, 74):
                add(f"    {chunk}")
        for finding in f1_by_row.get(line_id, []):
            add(f"    !! F1 {finding.kind}: {finding.detail}")
        for finding in f2_by_row.get(line_id, []):
            add(f"    !! F2 {finding.kind}: {finding.detail}")

    add("")
    add("--- CODA " + "-" * 62)
    coda = report.coda
    if not coda["row_count"]:
        add("  NO CODA ROW. The episode never names the story it came from.")
    else:
        add(f"  rows={coda['row_count']}  last_spoken={coda['is_last']}  "
            f"announcer={coda['announcer']}  "
            f"lane_status={coda['lane_status'] or '?'}")
        for chunk in _wrap(coda["text"], 74):
            add(f"    {chunk}")
        add(f"  names the source: {coda['names'] or 'NO'}")
    add(f"  indexed anchors: {', '.join(coda['anchors']) or '(none)'}")
    return "\n".join(out)


def _wrap(text: str, width: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]


def _mark_html(text: str) -> str:
    """Escape, then highlight every phrase an F1 detector would trip on."""
    escaped = html.escape(text)
    for _kind, pattern in _F1_PATTERNS:
        escaped = pattern.sub(
            lambda m: f"<mark>{m.group(0)}</mark>", escaped)
    return escaped


def render_html(reports: list[LedgerReport]) -> str:
    parts: list[str] = []
    add = parts.append
    title = " vs ".join(r.episode_id for r in reports)
    add(f"<title>Ledger: {html.escape(title)}</title>")
    add(_HTML_STYLE)
    add("<main>")
    add("<h1>OTR ledger</h1>")
    add("<p class='sub'>Two things are failure: <b>action in a spoken row</b> "
        "and <b>a row carrying the wrong speaker</b>. Everything else here is "
        "an observation, not a grade &mdash; prose quality is not a defect.</p>")
    for report in reports:
        add("<section class='ep'>")
        add(f"<h2>{html.escape(report.episode_id)}</h2>")
        state = "clean" if report.clean else "dirty"
        add(f"<p class='verdict {state}'>{html.escape(_verdict_line(report))}</p>")
        add("<dl class='meta'>")
        for label, value in (
            ("bank", report.source_bank or "?"),
            ("acts", report.receipts.get("act_count")),
            ("schedule", report.receipts.get("schedule") or "-"),
            ("beat jobs", report.receipts.get("beat_jobs")),
            ("review jobs", report.receipts.get("review_jobs")),
            ("spoken rows", len(report.voiced)),
            ("spoken words", report.spoken_words),
            ("coda", report.receipts.get("news_coda_status") or "-"),
            ("freeze", report.receipts.get("freeze_verdict") or "-"),
        ):
            if value in (None, ""):
                continue
            add(f"<dt>{html.escape(str(label))}</dt>"
                f"<dd>{html.escape(str(value))}</dd>")
        add("</dl>")

        f1_rows = {f.line_id for f in report.f1}
        f2_rows = {f.line_id for f in report.f2}
        add("<div class='rows'>")
        for row in report.rows:
            line_id = str(row.get("line_id") or "")
            role = str(row.get("speaker_role") or "")
            if role not in VOICED_ROLES:
                add(f"<div class='row music'><span class='id'>"
                    f"{html.escape(line_id)}</span>"
                    f"<span class='role'>{html.escape(role)}</span></div>")
                continue
            bad = "bad" if line_id in f1_rows or line_id in f2_rows else "ok"
            speaker = str(row.get("speaker") or "")
            add(f"<div class='row {bad}'>")
            add(f"<div class='head'><span class='id'>{html.escape(line_id)}"
                f"</span><span class='who'>"
                f"{html.escape(speaker or 'NO SPEAKER')}</span>"
                f"<span class='role'>{html.escape(role)}</span></div>")
            add(f"<p class='text'>{_mark_html(str(row.get('text') or ''))}</p>")
            for finding in report.f1:
                if finding.line_id == line_id:
                    add(f"<p class='find f1'>F1 {html.escape(finding.kind)}: "
                        f"<code>{html.escape(finding.detail)}</code></p>")
            for finding in report.f2:
                if finding.line_id == line_id:
                    add(f"<p class='find f2'>F2 {html.escape(finding.kind)}: "
                        f"{html.escape(finding.detail)}</p>")
            add("</div>")
        add("</div>")

        coda = report.coda
        add("<div class='coda'>")
        add("<h3>Coda</h3>")
        if not coda["row_count"]:
            add("<p class='find f1'>No coda row. The episode never names the "
                "story it came from.</p>")
        else:
            named = coda["names"]
            add(f"<p class='text'>{html.escape(coda['text'])}</p>")
            add(f"<p class='{'okline' if named else 'find f1'}'>"
                f"names the source: {html.escape(named or 'NO')}</p>")
        add("<p class='anchors'>indexed anchors: "
            f"{html.escape(', '.join(coda['anchors']) or '(none)')}</p>")
        add("</div>")
        add("</section>")
    add("</main>")
    return "\n".join(parts)


_HTML_STYLE = """<style>
:root{--bg:#faf8f4;--fg:#1d1b18;--dim:#6b6558;--line:#e0dad0;--card:#fff;
--bad:#a4243b;--badbg:#fbeaed;--good:#2f6b3f;--goodbg:#e9f3ec;--mark:#ffe9a8;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#16161a;--fg:#eceae6;--dim:#9a948a;--line:#2e2e35;--card:#1e1e24;
--bad:#ff8fa0;--badbg:#33202a;--good:#8fd6a4;--goodbg:#1d2c22;--mark:#5a4a1e;}}
:root[data-theme="dark"]{--bg:#16161a;--fg:#eceae6;--dim:#9a948a;--line:#2e2e35;
--card:#1e1e24;--bad:#ff8fa0;--badbg:#33202a;--good:#8fd6a4;--goodbg:#1d2c22;
--mark:#5a4a1e;}
body{background:var(--bg);color:var(--fg);margin:0;
font:16px/1.55 ui-serif,Georgia,'Times New Roman',serif;}
main{max-width:900px;margin:0 auto;padding:2.5rem 1.25rem 5rem;}
h1{font-size:1.9rem;margin:0 0 .4rem;letter-spacing:-.01em;}
h2{font-size:1.25rem;margin:2.5rem 0 .5rem;font-family:ui-sans-serif,system-ui;}
h3{font-size:1rem;margin:1.5rem 0 .4rem;font-family:ui-sans-serif,system-ui;}
.sub{color:var(--dim);margin:0 0 1.5rem;font-size:.95rem;}
.verdict{font-family:ui-sans-serif,system-ui;font-weight:600;padding:.55rem .8rem;
border-radius:.4rem;margin:.3rem 0 1rem;}
.verdict.clean{background:var(--goodbg);color:var(--good);}
.verdict.dirty{background:var(--badbg);color:var(--bad);}
dl.meta{display:grid;grid-template-columns:auto 1fr;gap:.15rem .8rem;
font-family:ui-sans-serif,system-ui;font-size:.85rem;margin:0 0 1.4rem;}
dl.meta dt{color:var(--dim);}
dl.meta dd{margin:0;}
.row{background:var(--card);border:1px solid var(--line);border-left-width:4px;
border-radius:.4rem;padding:.7rem .9rem;margin:.55rem 0;}
.row.ok{border-left-color:var(--good);}
.row.bad{border-left-color:var(--bad);}
.row.music{border-left-color:var(--line);color:var(--dim);font-size:.8rem;
font-family:ui-sans-serif,system-ui;padding:.4rem .9rem;}
.head{display:flex;gap:.7rem;align-items:baseline;font-family:ui-sans-serif,
system-ui;font-size:.78rem;margin-bottom:.35rem;}
.id{color:var(--dim);}
.who{font-weight:700;letter-spacing:.02em;text-transform:uppercase;}
.role{color:var(--dim);margin-left:auto;}
.text{margin:0;}
mark{background:var(--mark);color:inherit;padding:0 .1em;border-radius:.15em;}
.find{font-family:ui-sans-serif,system-ui;font-size:.78rem;margin:.4rem 0 0;
color:var(--bad);}
.okline{font-family:ui-sans-serif,system-ui;font-size:.8rem;color:var(--good);}
.anchors{font-family:ui-sans-serif,system-ui;font-size:.78rem;color:var(--dim);}
code{font:.85em ui-monospace,Menlo,Consolas,monospace;}
section.ep{border-top:1px solid var(--line);padding-top:.5rem;}
</style>"""


# ---------------------------------------------------------------------------
# The live window
# ---------------------------------------------------------------------------
#: Where the headless launcher writes its server log. A run started another
#: way can point --server-log wherever it actually landed.
DEFAULT_SERVER_LOG = REPO_ROOT / "tmp" / "otr_story_server.log"

_HEARTBEAT = re.compile(
    r"heartbeat:\s*(\d+)\s*tok\s*\|\s*([\d.]+)\s*tok/s\s*\|\s*([\d.]+)s\s*\|"
    r"\s*(?:\.\.\.)?(.*)$"
)
_PASS_LINE = re.compile(
    r"'([a-z0-9_]+:[A-Z0-9]+)'\s+attempt\s+(\d+)/(\d+)")
_HALT_LINE = re.compile(r"DECODE HALTED \(([a-z_]+)\)")


def _tail_bytes(path: Path, limit: int = 200_000) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > limit:
                handle.seek(size - limit)
            raw = handle.read()
    except OSError:
        return ""
    return raw.decode("utf-8", errors="replace").replace("\r", "\n")


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@dataclass
class LiveState:
    pass_id: str = ""
    attempt: str = ""
    tokens: int = 0
    rate: float = 0.0
    elapsed: float = 0.0
    tails: list[str] = field(default_factory=list)
    halts: list[str] = field(default_factory=list)
    repeating: str = ""


def read_live(server_log: Path) -> LiveState:
    """Everything the window knows, read from the server log alone."""
    state = LiveState()
    text = _strip_ansi(_tail_bytes(server_log))
    if not text:
        return state
    lines = text.splitlines()
    for line in lines:
        found = _PASS_LINE.search(line)
        if found:
            state.pass_id = found.group(1)
            state.attempt = f"{found.group(2)}/{found.group(3)}"
        found = _HALT_LINE.search(line)
        if found:
            state.halts.append(found.group(1))
    beats = [_HEARTBEAT.search(line) for line in lines]
    beats = [b for b in beats if b]
    if beats:
        last = beats[-1]
        state.tokens = int(last.group(1))
        state.rate = float(last.group(2))
        state.elapsed = float(last.group(3))
        # Keep a long run of samples: the window stitches them back into one
        # stream, and more samples means more of the sentence the model is
        # actually typing. Turn the pulse rate up with
        # OTR_WRITER_HEARTBEAT_EVERY (default 64 tokens) for a finer stream.
        state.tails = [
            " ".join(b.group(4).split()) for b in beats[-24:]
        ]
    state.repeating = _repetition_note(state.tails)
    return state


def _repetition_note(tails: list[str]) -> str:
    """Say plainly when the stream is saying the same thing again.

    Two samples of the tail of a growing stream should not look alike. When
    they do -- while the token counter keeps climbing -- the decode is
    cycling. This is the SAME evidence the code-side guard acts on, surfaced
    for a human watching; it never stops anything itself.
    """
    # LOOK AT WHAT THE HUMAN IS LOOKING AT. An earlier cut examined the last
    # four samples while the window rendered twenty-four, so a loop plainly
    # readable on screen went unflagged. A detector with a narrower view than
    # the display is worse than none: it reads as an all-clear.
    recent = [t for t in tails[-14:] if t]
    if len(recent) < 2:
        return ""
    if len(set(recent)) == 1:
        return "the last samples are IDENTICAL -- the decode is cycling"

    # RECONSTRUCT THE STREAM FIRST. Each heartbeat prints the tail of a
    # GROWING string, so consecutive samples legitimately overlap: the end of
    # one is the start of the next. Searching the raw concatenation for a
    # repeated phrase therefore fires on every healthy run -- measured, on the
    # first live leg this window ever watched. Stitching the samples back into
    # one stream removes that whole class of false alarm, and it is also what
    # makes the real tell visible: a cycle survives the stitch, because its
    # repeats are not at the seam.
    stream = _merge_tails(recent)
    words = stream.split()
    for size in range(min(12, len(words) // 2), 3, -1):
        seen: dict[str, int] = {}
        for start in range(len(words) - size + 1):
            phrase = " ".join(words[start:start + size])
            seen[phrase] = seen.get(phrase, 0) + 1
            if seen[phrase] >= 2 and len(phrase) >= 24:
                return (f"a {size}-word phrase came back: "
                        f"“{phrase[:64]}”")
    return ""


def _merge_tails(tails: list[str]) -> str:
    """Stitch overlapping heartbeat samples back into one stream.

    Samples that do not overlap are joined with an ellipsis rather than
    silently butted together: time passed between them, and a phrase that
    comes back ACROSS that gap is still the model returning to itself.
    """
    merged = ""
    for tail in tails:
        if not merged:
            merged = tail
            continue
        overlap = 0
        for size in range(min(len(merged), len(tail)), 5, -1):
            if merged.endswith(tail[:size]):
                overlap = size
                break
        merged += tail[overlap:] if overlap else f" … {tail}"
    return merged


def read_gpu() -> str:
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8, check=False,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return "gpu: unavailable"
    if not out:
        return "gpu: unavailable"
    used, total, util = [p.strip() for p in out[0].split(",")]
    try:
        frac = int(used) / max(int(total), 1)
    except ValueError:
        frac = 0.0
    bar = "#" * int(frac * 28)
    return (f"gpu  {used:>6} / {total} MiB  {util:>3}% util  "
            f"[{bar:<28}]")


def render_live(report: LedgerReport | None, state: LiveState,
                gpu: str) -> str:
    out: list[str] = []
    add = out.append
    add("=" * 78)
    add("OTR LIVE  --  ledger as it is written")
    add("=" * 78)
    add(gpu)
    if state.pass_id:
        add(f"pass {state.pass_id}  attempt {state.attempt}   "
            f"{state.tokens} tok @ {state.rate:.1f} tok/s over "
            f"{state.elapsed:.0f}s")
    else:
        add("pass  (no pass has logged yet)")
    if state.halts:
        add(f"HALTS so far: {', '.join(state.halts)}  "
            "(the guard stopped a runaway; the ladder rerolls it)")
    if state.repeating:
        add("")
        add("  !!  REPEATING  --  " + state.repeating)
    add("")
    add("writing now:")
    if state.tails:
        # The samples stitched back into the sentence the model is typing.
        # Reading disjoint slices is what made a runaway hard to see; reading
        # the prose is what makes it obvious.
        stream = _merge_tails(state.tails)[-900:]
        for chunk in _wrap(stream, 72):
            add(f"   {chunk}")
    else:
        add("   (nothing streaming)")
    add("")
    if report is None:
        add("ledger: none on disk yet")
        return "\n".join(out)
    add(f"ledger {report.path.name}")
    composed = [r for r in report.voiced if str(r.get('text') or '').strip()]
    add(f"  {len(composed)} of {len(report.voiced)} spoken rows composed, "
        f"{report.spoken_words} words")
    for row in composed[-6:]:
        speaker = str(row.get("speaker") or "NO SPEAKER")
        text = " ".join(str(row.get("text") or "").split())
        add(f"   {speaker:<14} {text[:56]}")
    if report.f1 or report.f2:
        add(f"  findings so far: F1={len({f.line_id for f in report.f1})} "
            f"F2={len({f.line_id for f in report.f2})}")
    return "\n".join(out)


def render_live_html(frame: str, interval: float) -> str:
    """The same frame as a page that refreshes itself.

    A terminal window means opening a terminal. This is the same information
    as a thing you can leave open on a second monitor and glance at.
    """
    alarm = "REPEATING" in frame
    return (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta http-equiv='refresh' content='{max(int(interval), 1)}'>"
        f"<title>OTR live</title>{_LIVE_STYLE}</head><body"
        f"{' class=alarm' if alarm else ''}>"
        f"<pre>{html.escape(frame)}</pre></body></html>"
    )


_LIVE_STYLE = """<style>
:root{color-scheme:dark;}
body{background:#101014;color:#d8d4cc;margin:0;padding:1.2rem 1.4rem;
font:14px/1.45 ui-monospace,Consolas,Menlo,monospace;}
body.alarm{background:#2a1116;color:#ffd9df;}
pre{margin:0;white-space:pre-wrap;word-break:break-word;}
</style>"""


def watch(interval: float, server_log: Path, root: Path | None,
          html_out: Path | None = None) -> int:
    import time
    try:
        while True:
            try:
                report = grade(resolve(None, root))
            except (SystemExit, OSError, ValueError, json.JSONDecodeError):
                report = None
            frame = render_live(report, read_live(server_log), read_gpu())
            if html_out is not None:
                html_out.parent.mkdir(parents=True, exist_ok=True)
                tmp = html_out.with_suffix(html_out.suffix + ".tmp")
                tmp.write_text(
                    render_live_html(frame, interval), encoding="utf-8")
                os.replace(tmp, html_out)
            else:
                sys.stdout.write("\x1b[2J\x1b[H" + frame + "\n")
                sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0


def report_json(report: LedgerReport) -> dict[str, Any]:
    return {
        "path": str(report.path),
        "episode_id": report.episode_id,
        "source_bank": report.source_bank,
        "clean": report.clean,
        "verdict": _verdict_line(report),
        "spoken_rows": len(report.voiced),
        "spoken_words": report.spoken_words,
        "f1": [f.__dict__ for f in report.f1],
        "f2": [f.__dict__ for f in report.f2],
        "coda": report.coda,
        "receipts": report.receipts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read an OTR ledger and grade it on F1 (action in a "
                    "spoken row) and F2 (speaker mismatch). Nothing else "
                    "is a failure.")
    parser.add_argument(
        "target", nargs="*",
        help="ledger path, episode id or id fragment; omit for the newest")
    parser.add_argument("--list", type=int, nargs="?", const=15, default=None,
                        metavar="N", help="list the N newest episode ledgers")
    parser.add_argument("--html", default=None, metavar="PATH",
                        help="write a readable page instead of text")
    parser.add_argument("--json", action="store_true",
                        help="emit the machine-readable verdict")
    parser.add_argument("--episodes-root", default=None,
                        help="override the episode tree to search")
    parser.add_argument(
        "--watch", type=float, nargs="?", const=4.0, default=None,
        metavar="SECONDS",
        help="live window: redraw every SECONDS (default 4) while a run "
             "writes. Ctrl-C to stop. Read-only.")
    parser.add_argument("--server-log", default=str(DEFAULT_SERVER_LOG),
                        help="server log the live window reads")
    args = parser.parse_args(argv)

    root = Path(args.episodes_root) if args.episodes_root else None

    if args.watch is not None:
        return watch(
            max(args.watch, 0.5), Path(args.server_log), root,
            Path(args.html) if args.html else None,
        )

    if args.list is not None:
        found = episode_ledgers(root)[: args.list]
        if not found:
            print(f"no episode ledger under {root or EPISODES_ROOT}")
            return 1
        for path in found:
            print(f"{path.parent.parent.name}")
        return 0

    targets = args.target or [None]
    reports = [grade(resolve(t, root)) for t in targets]

    if args.json:
        print(json.dumps([report_json(r) for r in reports], indent=2))
    elif args.html:
        out = Path(args.html)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_html(reports), encoding="utf-8")
        print(f"wrote {out}")
    else:
        for report in reports:
            print(render_text(report))
            print()

    return 0 if all(r.clean for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
