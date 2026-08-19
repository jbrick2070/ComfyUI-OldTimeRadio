"""Build a self-contained listening page for everything published to otr/obs/.

WHY IT LIVES IN obs/ AND USES RELATIVE PATHS. The operator listens from the 4060
laptop, which reaches `otr/obs` over a direct Ethernet link as a mapped drive --
not over the apartment wifi. A page sitting beside the episodes with `src="<file>"`
therefore works unchanged from the render box, from the mapped drive, and from a
copied folder. An absolute path would break every one of those but the first.

FULLY OFFLINE. No CDN, no web fonts, no external anything -- this project is
offline-first and the listening box may have no internet at all. Everything is
inlined, so the page is one file plus the videos already in the folder.

IT NEVER MOVES, RENAMES OR DELETES ANYTHING IN obs/. It only writes its own
index file. `otr/obs/` is the operator's success signal and its contents are not
tidied, sorted or relocated -- the page reads the folder and adds one artifact.

Re-run it any time; it rebuilds from whatever is in the folder now:
    python scripts/otr_build_obs_listen_page.py
"""
from __future__ import annotations

import glob
import html
import json
import os
import re
from pathlib import Path

OBS = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\obs")
EPISODES = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes")
OUT_NAME = "LISTEN.html"

BANK_LABEL = {
    "shakespeare": "Shakespeare",
    "public_domain": "Public Domain",
    "scifi_news": "Sci-Fi News",
    "scifi_news_pro": "Sci-Fi News Pro",
    "media_archive": "Media Archive",
    "original": "Original",
}


def collect() -> list[dict]:
    rows: list[dict] = []
    for name in os.listdir(OBS):
        if not name.lower().endswith(".mp4"):
            continue
        stem = name.split("_silent")[0]
        stamp_match = re.search(r"(20\d{6})_(\d{6})$", stem)
        when = sort_key = ""
        title = stem
        if stamp_match:
            d, t = stamp_match.group(1), stamp_match.group(2)
            when = f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}"
            sort_key = d + t
            title = stem[: stamp_match.start()].rstrip("_")
        title = title.replace("signal_lost_", "").replace("_", " ").strip()
        title = title.title() if title else stem

        bank, duration, voices = "", 0.0, []
        found = glob.glob(str(EPISODES / stem / "audio" / "*_ledger.json"))
        if found:
            try:
                data = json.loads(Path(found[0]).read_text(encoding="utf-8"))
                meta = data.get("meta") or {}
                bank = str(meta.get("source_bank") or "")
                duration = float(data.get("total_episode_dur_s") or 0)
                for row in data.get("cast") or []:
                    ref = row.get("voice_ref_id")
                    who = str(row.get("name") or "")
                    if ref and not who.upper().startswith("ANNOUNCER"):
                        voices.append({"who": who, "ref": str(ref)})
            except Exception:  # noqa: BLE001 -- a bad ledger must not lose the episode
                pass

        rows.append({
            "file": name,
            "title": title,
            "when": when,
            "bank": bank,
            "bankLabel": BANK_LABEL.get(bank, bank or "unknown"),
            "mb": round(os.path.getsize(OBS / name) / 1048576, 1),
            "dur": duration,
            "voices": voices,
            "sort": sort_key or "0",
        })
    rows.sort(key=lambda r: r["sort"], reverse=True)
    return rows


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Signal Lost - listening room</title>
<style>
  :root {
    --bg:#12100e; --panel:#1c1916; --edge:#2e2823; --ink:#f0e6d8;
    --dim:#a2968a; --amber:#e0a33e; --amber-soft:#7a5a1e;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
    font:16px/1.55 "Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif; }
  header { padding:28px 20px 18px; border-bottom:1px solid var(--edge);
    background:linear-gradient(180deg,#1a1613,#12100e); }
  .wrap { max-width:1080px; margin:0 auto; padding:0 20px; }
  h1 { margin:0 0 4px; font-size:30px; letter-spacing:.02em; }
  h1 .dot { color:var(--amber); }
  .sub { color:var(--dim); font-size:14px; }
  .controls { display:flex; flex-wrap:wrap; gap:8px; align-items:center;
    margin:18px 0 6px; }
  input[type=search] { flex:1 1 220px; min-width:180px; background:var(--panel);
    border:1px solid var(--edge); color:var(--ink); padding:9px 12px;
    border-radius:7px; font:inherit; font-size:14px; }
  .chip { background:var(--panel); border:1px solid var(--edge); color:var(--dim);
    padding:7px 13px; border-radius:999px; cursor:pointer; font-size:13px;
    font-family:system-ui,sans-serif; }
  .chip[aria-pressed=true] { background:var(--amber-soft); border-color:var(--amber);
    color:#fff5e2; }
  main { padding:22px 0 64px; }
  .ep { background:var(--panel); border:1px solid var(--edge); border-radius:10px;
    margin:0 0 14px; overflow:hidden; }
  .ep > summary { list-style:none; cursor:pointer; padding:14px 16px;
    display:flex; gap:12px; align-items:baseline; flex-wrap:wrap; }
  .ep > summary::-webkit-details-marker { display:none; }
  .ep[open] { border-color:var(--amber-soft); }
  .t { font-size:19px; flex:1 1 260px; }
  .badge { font-family:system-ui,sans-serif; font-size:11px; letter-spacing:.06em;
    text-transform:uppercase; color:var(--amber); border:1px solid var(--amber-soft);
    padding:2px 8px; border-radius:999px; white-space:nowrap; }
  .meta { color:var(--dim); font-size:13px; font-family:system-ui,sans-serif;
    white-space:nowrap; }
  .body { padding:0 16px 16px; }
  video { width:100%; border-radius:8px; background:#000; display:block; }
  .cast { margin:12px 0 0; font-size:13px; font-family:system-ui,sans-serif;
    color:var(--dim); }
  .cast b { color:var(--ink); font-weight:600; }
  .cast code { color:var(--amber); font-size:12px; }
  .empty { color:var(--dim); text-align:center; padding:48px 0;
    font-family:system-ui,sans-serif; }
  footer { color:var(--dim); font-size:12px; font-family:system-ui,sans-serif;
    border-top:1px solid var(--edge); padding:16px 0 40px; }
</style></head><body>
<header><div class="wrap">
  <h1>Signal Lost<span class="dot">.</span></h1>
  <div class="sub">__COUNT__ episodes published to <code>otr/obs</code> &middot; newest first &middot; built __BUILT__</div>
  <div class="controls">
    <input type="search" id="q" placeholder="Search titles, casts, voices...">
    <span id="chips"></span>
  </div>
</div></header>
<main class="wrap"><div id="list"></div><div class="empty" id="empty" hidden>Nothing matches.</div></main>
<footer class="wrap">Plays straight off the folder &mdash; relative paths, so this works from the
render box or the mapped drive. Nothing here is moved or renamed; the page only reads.</footer>
<script>
const EPS = __DATA__;
const list = document.getElementById('list');
const empty = document.getElementById('empty');
const chips = document.getElementById('chips');
let bank = '', q = '';

const banks = [...new Set(EPS.map(e => e.bank).filter(Boolean))].sort();
chips.innerHTML = ['<button class="chip" data-b="" aria-pressed="true">All</button>']
  .concat(banks.map(b => {
    const label = (EPS.find(e => e.bank === b) || {}).bankLabel || b;
    const n = EPS.filter(e => e.bank === b).length;
    return `<button class="chip" data-b="${b}" aria-pressed="false">${label} ${n}</button>`;
  })).join(' ');

chips.addEventListener('click', ev => {
  const btn = ev.target.closest('.chip'); if (!btn) return;
  bank = btn.dataset.b;
  [...chips.querySelectorAll('.chip')].forEach(c =>
    c.setAttribute('aria-pressed', String(c === btn)));
  render();
});
document.getElementById('q').addEventListener('input', e => {
  q = e.target.value.toLowerCase().trim(); render();
});

function mmss(s) {
  if (!s) return '';
  const m = Math.floor(s / 60), r = Math.round(s % 60);
  return m + ':' + String(r).padStart(2, '0');
}

function render() {
  const hits = EPS.filter(e => {
    if (bank && e.bank !== bank) return false;
    if (!q) return true;
    const hay = (e.title + ' ' + e.bankLabel + ' ' +
      e.voices.map(v => v.who + ' ' + v.ref).join(' ')).toLowerCase();
    return hay.includes(q);
  });
  empty.hidden = hits.length > 0;
  list.innerHTML = hits.map(e => `
    <details class="ep">
      <summary>
        <span class="t">${e.title}</span>
        ${e.bank ? `<span class="badge">${e.bankLabel}</span>` : ''}
        <span class="meta">${e.when}${e.dur ? ' &middot; ' + mmss(e.dur) : ''} &middot; ${e.mb} MB</span>
      </summary>
      <div class="body">
        <video controls preload="none" src="${e.file}"></video>
        ${e.voices.length ? `<div class="cast">${e.voices.map(v =>
          `<b>${v.who}</b> <code>${v.ref}</code>`).join(' &nbsp;&middot;&nbsp; ')}</div>` : ''}
      </div>
    </details>`).join('');
  // only one plays at a time
  list.querySelectorAll('video').forEach(v => v.addEventListener('play', () => {
    list.querySelectorAll('video').forEach(o => { if (o !== v) o.pause(); });
  }));
}
render();
</script></body></html>
"""


def main() -> int:
    rows = collect()
    if not rows:
        print("no episodes found in obs")
        return 1
    import datetime
    page = (PAGE
            .replace("__DATA__", json.dumps(rows, ensure_ascii=False))
            .replace("__COUNT__", str(len(rows)))
            .replace("__BUILT__", datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    out = OBS / OUT_NAME
    out.write_text(page, encoding="utf-8", newline="\n")
    with_cast = sum(1 for r in rows if r["voices"])
    print(f"wrote {out}")
    print(f"  {len(rows)} episodes, {with_cast} with cast metadata")
    print(f"  newest: {rows[0]['title']} ({rows[0]['when']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
