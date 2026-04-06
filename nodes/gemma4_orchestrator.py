r"""
Gemma 4 Orchestrator — Script Writer + Director for "SIGNAL LOST"
===================================================================

Two nodes:
  1. Gemma4ScriptWriter — Fetches real daily science news via RSS, feeds it to
     Gemma 4 to generate a full audio drama script. Contemporary sci-fi anthology
     format (Black Mirror / NPR Invisibilia / Arrival). News-as-spine: real
     headlines become the inciting incident, extrapolated to dramatic extremes.
     Includes a hard-science epilogue citing real sources (ArXiv, Nature, etc.).

  2. Gemma4Director — Takes a finished script and generates a production plan:
     TTS voice assignments, SFX cue list, music cues, timing, and spatial audio
     settings. Outputs structured JSON that drives all downstream nodes.

Gemma 4 runs via transformers (local GPU). Content safety filter catches
profanity/NSFW that slips past the prompt policy.

v1.0  2026-04-04  Jeffrey Brick
"""

import json
import logging
import os
import random
import re
import socket
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Lazy heavy imports (Section 8) — torch, numpy, transformers inside methods/classes only

log = logging.getLogger("OTR")

# BaseStreamer for custom heartbeat logic.
# Graceful stub allows importing this module in test environments without
# a GPU or transformers installed — ScriptParser and pure-logic tests work fine;
# actual Gemma4 generation will raise ImportError at call time as expected.
try:
    from transformers.generation.streamers import BaseStreamer, TextStreamer
except ImportError:
    class BaseStreamer:  # type: ignore[no-redef]
        """Stub — transformers not installed in this environment."""
        def put(self, value): pass
        def end(self): pass
    class TextStreamer(BaseStreamer):  # type: ignore[no-redef]
        pass

def _runtime_log(msg):
    """Write a persistent heartbeat to otr_runtime.log for monitoring."""
    try:
        ts = datetime.now().strftime("%H:%M:%S")
        log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "otr_runtime.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except: pass

# ─────────────────────────────────────────────────────────────────────────────
# LOG CLEANUP — compliant fixes handle most warnings at the source.
# These catch residual library noise from urllib3/httpx cache checks.
# ─────────────────────────────────────────────────────────────────────────────
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# CONTENT SAFETY FILTER — catches profanity/NSFW that slips past the prompt
# ─────────────────────────────────────────────────────────────────────────────

# Word list: common profanity, slurs, and explicit terms.
# Kept as a set for O(1) lookup. Checked against whole-word boundaries only
# so words like "assembly" or "hell" (in sci-fi context) aren't false-flagged.
_BLOCKED_WORDS = {
    # profanity
    "fuck", "fucking", "fucked", "fucker", "motherfucker", "motherfucking",
    "shit", "shitting", "shitty", "bullshit",
    "damn", "damned", "dammit", "goddamn", "goddammit",
    "ass", "asshole", "arse", "arsehole",
    "bitch", "bitches", "bastard", "crap", "crappy",
    "piss", "pissed", "pissing",
    "dick", "cock", "pussy", "tits", "boobs",
    "whore", "slut", "skank",
    # slurs (abbreviated to avoid reproducing full slurs in source)
    "nigger", "nigga", "faggot", "fag", "retard", "retarded",
    "spic", "chink", "kike", "wetback", "coon",
    # violence-adjacent shock terms
    "disembowel", "dismember", "decapitate", "eviscerate",
    "rape", "raped", "raping", "molest",
}

# Replacement word — period-appropriate euphemism
_REPLACEMENT = "[BLEEP]"


def _content_filter(text: str) -> tuple:
    """Scrub blocked words from generated script text.

    Returns (cleaned_text, list_of_replacements_made).
    Uses whole-word regex matching to avoid false positives.
    """
    replacements = []
    def _replace(match):
        word = match.group(0)
        replacements.append(word.lower())
        return _REPLACEMENT

    # Build regex: whole-word match, case-insensitive
    if not _BLOCKED_WORDS:
        return text, []
    pattern = r'\b(?:' + '|'.join(re.escape(w) for w in sorted(_BLOCKED_WORDS, key=len, reverse=True)) + r')\b'
    cleaned = re.sub(pattern, _replace, text, flags=re.IGNORECASE)

    if replacements:
        log.warning("[ContentFilter] Replaced %d blocked word(s): %s",
                    len(replacements), ", ".join(set(replacements)))

    return cleaned, replacements


# ─────────────────────────────────────────────────────────────────────────────
# NEWS FETCHER — pulls real science headlines to seed the story
# ─────────────────────────────────────────────────────────────────────────────

SCIENCE_NEWS_FEEDS = [
    # ── Open-access: full article text fetchable, no paywall ──
    "https://www.sciencedaily.com/rss/all.xml",           # Best: full articles, open
    "https://www.eurekalert.org/rss/technology_engineering.xml",  # Press releases, open
    "https://www.eurekalert.org/rss/space.xml",           # Press releases, open
    "https://www.eurekalert.org/rss/biology.xml",         # Press releases, open
    "https://www.eurekalert.org/rss/chemistry_physics.xml", # Press releases, open
    "https://www.eurekalert.org/rss/earth_environment.xml", # Press releases, open
    # ── Government / institutional (fully open) ──
    "https://www.nasa.gov/rss/dyn/breaking_news.rss",     # NASA, open
    "https://www.nih.gov/news-events/news-releases.xml",  # NIH, open
    "https://www.nsf.gov/rss/rss_www_news.xml",           # NSF, open
    # ── UCLA Newsroom (open-access institutional research) ──
    "https://newsroom.ucla.edu/cats/health_+_behavior.xml",      # Best: full content:encoded in RSS
    "https://newsroom.ucla.edu/cats/science_+_technology.xml",   # Open-access, URL scrape works
    "https://newsroom.ucla.edu/cats/environment_+_climate.xml",  # Open-access, URL scrape works
    # ── Open journalism (full text accessible) ──
    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",  # BBC, open
    "https://feeds.arstechnica.com/arstechnica/science",  # Ars, open
    "https://theconversation.com/us/science/rss",         # The Conversation, open
    "https://cosmosmagazine.com/feed/",                   # Cosmos, open
]


def _fetch_full_article(url, timeout=20):
    """Fetch the full text of a science article from its URL.

    Uses requests + BeautifulSoup to strip HTML boilerplate and extract
    the article body. Returns the raw text (up to 12000 chars) so Gemma
    gets real science content — methodology, findings, implications —
    not just the RSS teaser. Falls back to empty string on any failure
    (paywalls, bot blocks, timeouts) so the caller can degrade gracefully.

    The scraper tries a cascade of CSS selectors before falling back to
    the full document, so it handles sites that don't use semantic
    <article>/<main> tags (e.g. UCLA Newsroom, institutional press pages).
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return ""

    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; OTR-ScriptBot/1.0)"}
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Strip boilerplate — nav, ads, footer, sidebar, scripts
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "form", "noscript", "iframe"]):
            tag.decompose()

        # Cascade of content selectors — most specific to least.
        # Covers: semantic HTML5, WordPress/CMS class conventions,
        # institutional press release pages (UCLA, NIH, NSF, EurekaAlert).
        _SELECTORS = [
            "article",
            "main",
            '[class*="article-body"]',
            '[class*="article__body"]',
            '[class*="story-body"]',
            '[class*="entry-content"]',
            '[class*="post-content"]',
            '[class*="content-body"]',
            '[class*="wysiwyg"]',
            '[class*="rich-text"]',
            '[class*="body-copy"]',
            '[class*="release-body"]',      # EurekaAlert press releases
            '[class*="article-content"]',
            '[id*="article-body"]',
            '[id*="main-content"]',
            "div.content",
            "div.body",
        ]

        body = None
        for selector in _SELECTORS:
            body = soup.select_one(selector)
            if body:
                break
        if body is None:
            body = soup  # last resort — full stripped document

        # Extract paragraphs AND headings — h2/h3 carry section context
        # (methodology, implications, researcher quotes) that's often the
        # richest science content buried past the lede.
        content_tags = body.find_all(["p", "h2", "h3"])
        text = " ".join(tag.get_text(" ", strip=True) for tag in content_tags)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:12000]
    except Exception:
        return ""


def _fetch_science_news(max_feeds=10):
    """Fetch science stories from multiple RSS feeds in parallel.

    Uses ThreadPoolExecutor to hit all feeds simultaneously, dramatically
    reducing the wait time when feeds are slow or unresponsive. Each feed
    has its own timeout.
    """
    try:
        import feedparser
    except ImportError:
        msg = (
            "╔══════════════════════════════════════════════════════════════════╗\n"
            "║  CRITICAL: feedparser is missing.                              ║\n"
            "║  Run `pip install feedparser` to enable live science news.     ║\n"
            "║  The OTR ScriptWriter REQUIRES real headlines — no fallback.   ║\n"
            "╚══════════════════════════════════════════════════════════════════╝"
        )
        log.error(msg)
        raise ImportError(msg)

    def _fetch_single_feed(feed_url):
        data = []
        FEED_TIMEOUT = 7
        try:
            # Set socket timeout locally for this thread
            _prev_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(FEED_TIMEOUT)
            try:
                feed = feedparser.parse(feed_url)
            finally:
                socket.setdefaulttimeout(_prev_timeout)

            for entry in feed.entries[:6]:
                title = entry.get("title", "").strip()
                if not title:
                    continue

                # ── Headline pre-filter: reject non-article content ──────────
                # Teaser/media headlines have no science payload for Gemma to
                # work with. A podcast slug or video title gives Gemma 90 chars
                # about content it can't access. Drop these at the source.
                _SKIP_PREFIXES = (
                    "podcast:", "watch:", "video:", "listen:", "opinion:",
                    "q&a:", "quiz:", "gallery:", "photos:", "slideshow:",
                    "live:", "webinar:", "event:", "in photos:",
                )
                _SKIP_PHRASES = (
                    "behind-the-scenes", "tour of", "in conversation with",
                    "ask the expert", "meet the", "alumni spotlight",
                    "faculty spotlight", "student spotlight", "donate",
                    "how to apply", "registration open",
                )
                title_lower = title.lower()
                if any(title_lower.startswith(p) for p in _SKIP_PREFIXES):
                    log.debug("[NewsFetcher] Skipping non-article (prefix): %s", title[:60])
                    continue
                if any(p in title_lower for p in _SKIP_PHRASES):
                    log.debug("[NewsFetcher] Skipping non-article (phrase): %s", title[:60])
                    continue
                # ─────────────────────────────────────────────────────────────

                content_candidates = entry.get("content", [])
                rss_full = ""
                if content_candidates:
                    rss_full = content_candidates[0].get("value", "")
                    rss_full = re.sub(r'<[^>]+>', '', rss_full).strip()
                summary = entry.get("summary", "")[:2000].strip()
                summary = re.sub(r'<[^>]+>', '', summary).strip()
                data.append({
                    "headline": title,
                    "summary": summary,
                    "rss_full": rss_full,
                    "source": feed.feed.get("title", feed_url.split("/")[2]),
                    "date": entry.get("published", str(datetime.now().date())),
                    "link": entry.get("link", ""),
                })
            return data
        except Exception as e:
            log.debug("[NewsFetcher] Feed failed %s: %s", feed_url, e)
            return []

    pool = []
    feeds_hit = 0
    shuffled_feeds = SCIENCE_NEWS_FEEDS[:]
    random.shuffle(shuffled_feeds)

    log.info("[NewsFetcher] Starting parallel fetch from %d sources...", len(shuffled_feeds))
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=len(shuffled_feeds)) as executor:
        futures = {executor.submit(_fetch_single_feed, url): url for url in shuffled_feeds}
        for future in as_completed(futures):
            results = future.result()
            if results:
                pool.extend(results)
                feeds_hit += 1

    fetch_time = time.time() - start_time
    log.info("[NewsFetcher] Parallel fetch complete in %.2fs. Pool: %d headlines from %d feeds.",
             fetch_time, len(pool), feeds_hit)

    if not pool:
        log.error("[NewsFetcher] ALL feeds failed — check network connectivity")
        raise RuntimeError(
            "No science headlines could be fetched. Check your internet connection. "
            "The OTR ScriptWriter requires live RSS feeds to generate scripts."
        )

    # Shuffle pool before choosing
    random.shuffle(pool)

    # Content quality floor: try up to 5 candidates until we get a rich article.
    # Thin content (<400 chars) gives Gemma too little to extrapolate from —
    # the story ends up generic rather than grounded in real science.
    CONTENT_FLOOR = 400
    MAX_ATTEMPTS = 5
    chosen = None

    for candidate in pool[:MAX_ATTEMPTS]:
        result = candidate

        # Resolve full article text — 3-tier: rss_full → URL scrape → summary
        if result.get("rss_full") and len(result["rss_full"]) > 300:
            result["full_text"] = result["rss_full"][:12000]
            log.info("[NewsFetcher] Full text from RSS content field: %d chars", len(result["full_text"]))
        elif result.get("link"):
            log.info("[NewsFetcher] Attempting to fetch full article: %s", result["link"])
            fetched = _fetch_full_article(result["link"], timeout=5)
            if fetched and len(fetched) > 300:
                result["full_text"] = fetched
                log.info("[NewsFetcher] Full article fetched: %d chars", len(result["full_text"]))
            else:
                result["full_text"] = result["summary"]
                log.info("[NewsFetcher] Article fetch failed or blocked — falling back to RSS summary (%d chars)", len(result["full_text"]))
        else:
            result["full_text"] = result["summary"]
            log.info("[NewsFetcher] No URL — using RSS summary (%d chars)", len(result["full_text"]))

        if len(result.get("full_text", "")) >= CONTENT_FLOOR:
            chosen = result
            break
        else:
            log.warning("[NewsFetcher] Article too thin (%d chars) — trying next candidate: %s",
                        len(result.get("full_text", "")), result["headline"][:60])

    if chosen is None:
        # All candidates were thin — take the richest one we found
        chosen = max(pool[:MAX_ATTEMPTS], key=lambda x: len(x.get("full_text", x.get("summary", ""))))
        chosen.setdefault("full_text", chosen.get("summary", ""))
        log.warning("[NewsFetcher] All %d candidates were thin — using richest available (%d chars): %s",
                    MAX_ATTEMPTS, len(chosen["full_text"]), chosen["headline"][:60])

    return [chosen]


# ─────────────────────────────────────────────────────────────────────────────
# GEMMA 4 INFERENCE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def _load_gemma4(model_id="google/gemma-4-E4B-it", device="cuda"):
    """Load Gemma 4 via transformers. Caches globally with device tracking.

    BEST PRACTICES applied (per survival guide):
      - Section 3:  Lazy loading — never load at import time
      - Section 5:  Device/dtype alignment
      - Section 34: Cache invalidation on device change
      - Section 40: Manual VRAM management since we're outside ComfyUI model registry
      - Section 47: No device_map="auto" (conflicts with ComfyUI's torch.set_default_device)
      - Section 49: No trust_remote_code=True (Gemma 4 is natively supported)
    """
    global _GEMMA4_CACHE

    # Check for device change — invalidate if needed
    if (_GEMMA4_CACHE["model"] is not None and
            str(_GEMMA4_CACHE["device"]) != str(device)):
        log.info("Gemma 4 device changed %s → %s, reloading", _GEMMA4_CACHE["device"], device)
        _unload_gemma4()

    if _GEMMA4_CACHE["model"] is None:
        log.info(f"Loading Gemma 4 model: {model_id}")
        try:
            # Lazy import — only pay the cost when actually generating
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM

            # Enable TF32 for faster matmuls on Ampere/Ada/Blackwell GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            import gc
            gc.collect()
            torch.cuda.empty_cache()

            # gemma-4-E4B-it uses AutoProcessor (handles text+vision+audio)
            # local_files_only=True stops HF Hub ETag HEAD requests — no
            # "phoning home" on every render once models are cached locally.
            try:
                tokenizer = AutoProcessor.from_pretrained(model_id, local_files_only=True)
            except OSError:
                tokenizer = AutoProcessor.from_pretrained(model_id)

            # Using bfloat16 + sdpa for maximum speed on RTX 5000-series (Ada/Blackwell) GPUs.
            load_dtype = torch.bfloat16

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    dtype=load_dtype,
                    low_cpu_mem_usage=True,
                    attn_implementation="sdpa",  # Scaled Dot Product Attention — turbo speed
                    local_files_only=True,
                ).to(device).eval()
            except OSError:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    dtype=load_dtype,
                    low_cpu_mem_usage=True,
                    attn_implementation="sdpa",
                ).to(device).eval()

            _GEMMA4_CACHE["model"] = model
            _GEMMA4_CACHE["tokenizer"] = tokenizer
            _GEMMA4_CACHE["device"] = device
            log.info("Gemma 4 loaded: %s on %s", type(model).__name__, device)
        except Exception as e:
            log.exception("Failed to load Gemma 4: %s", e)  # Section 49: log.exception for full traceback
            raise
    return _GEMMA4_CACHE["model"], _GEMMA4_CACHE["tokenizer"]


# Bounded model cache with device tracking (Section 34)
_GEMMA4_CACHE = {"model": None, "tokenizer": None, "device": None}


def _unload_gemma4():
    """Explicitly unload Gemma 4 to free VRAM (Section 3, Section 40).

    gc.collect() forces Python to destroy the model object before
    torch.cuda.empty_cache() runs. Without it, Python's lazy GC may
    leave dead tensors in VRAM when Bark tries to load — OOM on
    long 25+ minute renders.
    """
    global _GEMMA4_CACHE
    import gc
    import torch
    if _GEMMA4_CACHE["model"] is not None:
        del _GEMMA4_CACHE["model"]
        del _GEMMA4_CACHE["tokenizer"]
        _GEMMA4_CACHE = {"model": None, "tokenizer": None, "device": None}
        gc.collect()              # force Python to destroy the object NOW
        torch.cuda.empty_cache()  # THEN release the VRAM
        log.info("Gemma 4 unloaded, VRAM freed (gc.collect + empty_cache)")


class GemmaHeartbeatStreamer(BaseStreamer):
    """Custom streamer that pulses heartbeats to _runtime_log for Canonical Tokens.

    Hooks into model.generate() at the token level. Every time Gemma completes
    a line that contains a recognizable script tag (=== SCENE, [VOICE:], [SFX:],
    [ENV:], (beat)), it writes a timestamped entry to otr_runtime.log immediately.

    Also tracks:
      - Scene count (how many === SCENE === tags so far)
      - Dialogue line count (how many [VOICE:] tags)
      - Unique character names seen
      - Token generation speed (tokens/sec, reported every 100 tokens)

    The OTR Monitor (otr_monitor.py) tails otr_runtime.log and folds these
    heartbeats into the live dashboard — so you can watch the script being
    written in real time without touching ComfyUI.
    """

    def __init__(self, tokenizer, skip_prompt=False, **decode_kwargs):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs
        self.token_cache = []
        self.on_prompt_end = True
        self.print_streamer = TextStreamer(tokenizer, skip_prompt=skip_prompt, **decode_kwargs)

        # Counters for the dashboard
        self.scene_count = 0
        self.dialogue_count = 0
        self.sfx_count = 0
        self.characters_seen = set()

        # Token speed tracking
        self.total_tokens = 0
        self._start_time = time.time()
        self._last_speed_report = 0

    def put(self, value):
        """Processes a new batch of tokens."""
        # Standard console output
        self.print_streamer.put(value)

        # Heartbeat logic
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("GemmaHeartbeatStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.on_prompt_end:
            self.on_prompt_end = False
            return

        token_list = value.tolist()
        self.token_cache.extend(token_list)
        self.total_tokens += len(token_list)

        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # If a newline is detected, process the completed line
        if text.endswith("\n") or text.endswith("\r"):
            self._process_line(text.strip())
            self.token_cache = []

        # Report speed every 100 tokens
        if self.total_tokens - self._last_speed_report >= 100:
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                tps = self.total_tokens / elapsed
                _runtime_log(
                    f"ScriptWriter: {self.total_tokens} tokens | "
                    f"{tps:.1f} tok/s | {self.scene_count} scenes | "
                    f"{self.dialogue_count} lines | "
                    f"{len(self.characters_seen)} chars"
                )
                self._last_speed_report = self.total_tokens

    def end(self):
        """Flush the remaining buffer and report final stats."""
        self.print_streamer.end()
        if self.token_cache:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            self._process_line(text.strip())
        self.token_cache = []

        elapsed = time.time() - self._start_time
        tps = self.total_tokens / elapsed if elapsed > 0 else 0
        _runtime_log(
            f"ScriptWriter DONE: {self.total_tokens} tokens in {elapsed:.1f}s "
            f"({tps:.1f} tok/s) | {self.scene_count} scenes | "
            f"{self.dialogue_count} dialogue lines | "
            f"Characters: {', '.join(sorted(self.characters_seen)) or 'none'}"
        )

    def _process_line(self, line):
        """Detect Canonical Tokens, update counters, pulse the heartbeat."""
        if not line:
            return

        # ── Scene break: === SCENE X === ─────────────────────────────
        if "===" in line:
            self.scene_count += 1
            _runtime_log(f"ScriptWriter: {line.strip()}")
            return

        # ── Voice tag: [VOICE: NAME, traits] dialogue ────────────────
        if "[VOICE:" in line:
            self.dialogue_count += 1
            try:
                tag_content = line.split("[VOICE:", 1)[1].split("]", 1)[0]
                name = tag_content.split(",", 1)[0].strip().upper()
                self.characters_seen.add(name)
                _runtime_log(f"ScriptWriter: [{self.dialogue_count}] {name}: {line.split(']',1)[-1].strip()[:60]}")
            except (IndexError, ValueError):
                _runtime_log(f"ScriptWriter: Voice line #{self.dialogue_count}")
            return

        # ── SFX tag ──────────────────────────────────────────────────
        if "[SFX:" in line:
            self.sfx_count += 1
            try:
                desc = line.split("[SFX:", 1)[1].split("]", 1)[0].strip()
                _runtime_log(f"ScriptWriter: SFX #{self.sfx_count}: {desc[:50]}")
            except (IndexError, ValueError):
                _runtime_log(f"ScriptWriter: SFX #{self.sfx_count}")
            return

        # ── ENV tag ──────────────────────────────────────────────────
        if "[ENV:" in line:
            try:
                desc = line.split("[ENV:", 1)[1].split("]", 1)[0].strip()
                _runtime_log(f"ScriptWriter: ENV: {desc[:50]}")
            except (IndexError, ValueError):
                pass
            return

        # ── Beat pause ───────────────────────────────────────────────
        if "(beat)" in line.lower():
            return  # beats are too frequent to log individually


def _generate_with_gemma4(prompt, model_id="google/gemma-4-E4B-it",
                          max_new_tokens=4096, temperature=0.8, top_p=0.92):
    """Generate text with Gemma 4."""
    import torch

    model, tokenizer = _load_gemma4(model_id)

    # Gemma4Processor expects structured content (multimodal processor)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # MUST use keyword arg — Gemma4Processor.__call__ signature is (images=, text=, ...)
    # Passing text positionally sets images=text, leaving text=None → subscript crash
    inputs = tokenizer(text=text, return_tensors="pt").to(model.device)

    # ── FIX: Ensure attention_mask is present ──
    if "attention_mask" not in inputs and "input_ids" in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    # Gemma 4 eos_token_id is a list — extract first element for pad_token_id
    eos_id = model.generation_config.eos_token_id
    pad_id = eos_id[0] if isinstance(eos_id, list) else eos_id

    log.info(f"[Gemma4] Starting inference (max_new_tokens={max_new_tokens})...")
    log.info("[Gemma4] Live output will stream below:")
    start_inference = time.time()

    # Initialize streamer for live feedback in the terminal + heartbeat logs.
    # Safely access tokenizer if we're using a multimodal processor.
    raw_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    streamer = GemmaHeartbeatStreamer(raw_tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            max_length=None,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=pad_id,
            streamer=streamer,      # Enable live streaming + granular heartbeats
        )

    inference_time = time.time() - start_inference
    log.info(f"[Gemma4] Inference complete in {inference_time:.1f}s.")

    # Decode only the new tokens (skip the prompt)
    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1: SCRIPT WRITER
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_SYSTEM_PROMPT = """# CANONICAL AUDIO ENGINE v1.0 — DETERMINISTIC TOKENS ONLY.
# Every line must be an "Audio Token": [ENV:], [SFX:], [VOICE:], or (beat).

═══ 🧱 1. CANONICAL FORMATTING (STRICT) ═══
Every scene MUST follow this layout:

=== SCENE X ===

[ENV: description (3-4 descriptors: e.g. fluorescent hum, distant traffic)]

[SFX: description]

[VOICE: CHARACTERNAME, gender, age, tone, energy] Short, natural dialogue line.

(beat)

[VOICE: CHARACTERNAME, gender, age, tone, energy] Next dialogue line.

CRITICAL: The first field in EVERY [VOICE:] tag is ALWAYS the CHARACTER NAME IN ALL CAPS.
WRONG: [VOICE: male, 40s, calm] Text here.
RIGHT: [VOICE: HAYES, male, 40s, calm, low energy] Text here.
CHARACTER NAMES must be CONSISTENT across all scenes (same spelling, same caps, every time).

═══ 🧱 2. THE TAG SYSTEM (ONLY THESE FOUR) ═══
- [ENV: ...] -> Background layers (e.g. [ENV: cockpit, electronic chirps, life support hum])
- [SFX: ...] -> Individual sound effects (e.g. [SFX: metal clatter])
- [VOICE: NAME, gender, age, tone, energy] -> MUST precede every dialogue line.
  NAME is ALWAYS FIRST — all caps, no spaces if possible.
  Examples: [VOICE: ANNOUNCER, male, 50s, authoritative, calm]
            [VOICE: HAYES, male, 40s, calm, low energy]
            [VOICE: DR_VOSS, female, 30s, intense, high energy]
            [VOICE: LEMMY, male, 50s, gruff, pragmatic]
- (beat) -> A 0.8s deterministic pause. Use it between lines for timing.

═══ 🧱 3. DIALOGUE RULES (BARK OPTIMIZED) ═══
- Keep dialogue lines SHORT (5–15 words).
- ONE sentence per line. Never use long paragraphs.
- Use natural, fragmented phrasing. Interruptions allowed.
- Use ... for hesitations and trailing thoughts.
- Use CAPS for single-word emphasis: "We are COMPLETELY out of time."
- Bark non-verbal tokens go INSIDE dialogue (in square brackets):
    [laughs]        — brief laugh mid-sentence
    [laughter]      — sustained laughter
    [sighs]         — audible sigh
    [gasps]         — sharp intake of breath
    [coughs]        — coughing
    [clears throat] — throat clearing before speaking
    [pants]         — breathless, exertion
    [sobs]          — crying
    [grunts]        — effort/strain
    [groans]        — pain or frustration
    [whistles]      — whistle
    [sneezes]       — sneeze
- Use ♪ around text for sung or hummed lines: ♪ signal lost, signal lost ♪
- NEVER use (parentheses) for anything except the (beat) tag.
- NEVER write stage directions in the dialogue text.

═══ 🧱 4. STORYTELLING: SIGNAL LOST ═══
- You are a STORYTELLER first, scientist second. The science news is your SEED — grow it into a gripping human drama.
- {news_block}
- Use this science as a backdrop, but the STORY is about PEOPLE: their fears, choices, relationships, and survival.

LANGUAGE & ACCESSIBILITY (CRITICAL):
This show must be entertaining for EVERYONE, not just scientists. Write like a great TV drama, not a lecture.
- 30% of the dialogue should be ELEMENTARY-SCHOOL accessible: simple words, clear emotions, characters explaining things to each other in plain language. "The water is making people sick" not "The contamination vector is waterborne."
- 30% should be HIGH-SCHOOL level: characters debating choices, moral dilemmas, real-world consequences anyone can follow.
- 20% should be COLLEGE level: deeper implications, technical details woven naturally into tense moments.
- 10% should be GRADUATE level: one or two lines of genuine hard science that reward attentive listeners.
- The remaining 10% is pure EMOTION: fear, humor, anger, love, hope. Lines that hit you in the gut regardless of education.

STORY REQUIREMENTS:
- Every episode needs a PLOT with stakes, conflict, and a twist. Not a report — a STORY.
- Characters must have personal motivations beyond "doing science." Give them something to lose.
- Include at least one moment of humor or warmth. Even in horror, people crack jokes under pressure.
- Dialogue should sound like REAL PEOPLE TALKING, not reading Wikipedia. Use contractions, interruptions, half-finished sentences.
- Show, don't tell. Instead of "The radiation levels are dangerous," write: "Don't touch that wall. See how the paint's bubbling? Yeah. We need to leave. Now."

═══ 🎭 STORY ARC ENGINE ═══
Pick ONE of these proven dramatic structures at random for each episode. Do NOT announce which one you picked — just USE it. These are structural blueprints, not content to copy.

ARC TYPE A — "THE TRAGIC FALL" (Shakespearean):
A brilliant person's greatest strength becomes their fatal flaw. They rise, overreach, and the thing they thought they controlled destroys them. The audience sees it coming before the character does. End on the cost of hubris.

ARC TYPE B — "THE COMEDIC SPIRAL" (Larry David / Seinfeld):
Multiple seemingly unrelated small problems collide into one spectacular disaster. Characters make reasonable-sounding decisions that each make things slightly worse. Coincidences pile up. What starts as a minor inconvenience escalates absurdly. Everything connects in the final scene in a way that's both surprising and inevitable.

ARC TYPE C — "THE GATHERING STORM" (Marvel-style escalation):
Start small and personal. Each scene raises the scope — from one person's problem to a team's crisis to a city-wide threat. The protagonist discovers they're uniquely positioned to act. A sacrifice or impossible choice at the climax. The victory costs something real.

ARC TYPE D — "THE BOTTLE EPISODE" (Classic radio drama):
Trapped. A small group stuck in one location under pressure — a submarine, a sealed lab, a quarantine zone. No escape, no reinforcements. Secrets come out. Trust breaks down. The real danger might be each other. Resolution comes from an unexpected alliance or confession.

ARC TYPE E — "THE UNRELIABLE WITNESS" (Twilight Zone / Orson Welles):
Something is wrong and only one person notices. Everyone else thinks they're crazy. The audience doesn't know who to trust. Reality shifts. The twist reframes EVERYTHING the listener heard. The final line makes you want to re-listen from the start.

ARC TYPE F — "THE TICKING CLOCK" (24 / War of the Worlds):
A hard deadline. Something terrible happens at a specific time unless someone acts. Every scene is a failed attempt or partial success that buys a little more time. Tension never drops — it only redirects. The solution comes from an unexpected direction and costs more than anyone planned.

ARC TYPE G — "THE MORAL INVERSION" (Rod Serling / Black Mirror):
The "good guys" are doing something that sounds reasonable. Scene by scene, the audience slowly realizes the ethical horror of what's actually happening. The characters don't see it — or they do and justify it. The twist isn't a plot surprise; it's the moment the listener's sympathy flips.

ARC TYPE H — "THE REUNION" (Spielberg / human-first sci-fi):
The science separates people who care about each other. The real plot isn't solving the problem — it's whether these people can find their way back to each other. Technical obstacles mirror emotional ones. The climax is both a scientific resolution and an emotional reunion (or devastating failure to reconnect).

ARC TYPE I — "THE MISTAKEN IDENTITY" (Shakespearean comedy — Twelfth Night / Comedy of Errors):
Someone is pretending to be someone they're not — or two people get mixed up. The confusion creates absurd situations, romantic tangles, and escalating lies. Characters fall for the wrong person, make promises to the wrong ally, or accidentally confess to the wrong authority. The unmasking scene is both hilarious and surprisingly touching. End with forgiveness and a new understanding.

ARC TYPE J — "THE ENCHANTED WORLD" (Shakespearean comedy — A Midsummer Night's Dream / The Tempest):
Characters leave their normal world and enter a strange environment where the rules are different — an alien biome, a malfunctioning space station, a quarantine dreamscape. In this weird place, social hierarchies flip. The serious boss becomes helpless. The quiet intern becomes the leader. Unlikely pairs are thrown together. Comedy comes from fish-out-of-water moments. By the time they return to "normal," everyone has changed. The science is the magic — it created the enchanted space.

ARC TYPE K — "THE SCHEMER UNDONE" (Shakespearean comedy — Much Ado About Nothing / The Merry Wives of Windsor):
A clever character hatches an elaborate plan — maybe to get credit for a discovery, cover up a mistake, or manipulate a rival. The plan is brilliant on paper. But every person they recruit to help adds their own agenda. Side plots multiply. The scheme gets more and more baroque until it collapses spectacularly, and the schemer ends up in a worse position than if they'd just been honest. But the fallout brings people together in unexpected ways.

ARC TYPE L — "THE RIVALS" (Shakespearean comedy — The Taming of the Shrew / Love's Labour's Lost):
Two strong-willed characters who can't stand each other are forced to work together. They argue about EVERYTHING — methods, priorities, whose fault it is. But their arguments reveal mutual respect buried under pride. The crisis forces them to combine their opposing approaches, and the solution only works because they're different. Ends with grudging admiration that the audience knows is something more.

SCALING THE ARC TO FIT THE TIME:
- SHORT episodes (5 min or less): Compress the arc to its ESSENCE. You only have 2-3 scenes. ANNOUNCER still opens — just keep it to 2-3 sentences. Then drop us straight into the action. Skip backstory exposition — imply it through dialogue. Hit the twist fast. Think of it as a cold open that IS the whole episode. The Bottle Episode (D), Unreliable Witness (E), and Rivals (L) work especially well at short length.
- MEDIUM episodes (10-20 min): Full 3-scene structure. Room for setup, escalation, and payoff. All arcs work well here.
- LONG episodes (20+ min): Let the arc breathe. Add subplots, secondary character arcs, and moments of quiet between the tension. The Comedic Spiral (B), Gathering Storm (C), Schemer Undone (K), and Enchanted World (J) really shine with extra time.

IMPORTANT: Vary the arc across episodes. Do NOT default to the same structure every time. Comedy arcs (B, I, J, K, L) should appear just as often as dramatic ones. Surprise the listener.

- ANNOUNCER (VOICE: male, 50s, authoritative, calm) opens and closes the show.
- ANNOUNCER OPENING (REQUIRED): The ANNOUNCER sets the stage like the best old-time radio hosts. The opening MUST include ALL of the following:
  1. TIME and PLACE — ground the listener immediately. Use the DATE (e.g. "April 5th, 2026") and a LOCATION. Write it the way a real radio announcer would say it — naturally, not like a timestamp. Never say a clock time. "April 5th, 2026. A genetics lab outside Seoul." Not "19:42, April 5th." Not "Tonight at 7:42 PM."
  2. CHARACTER INTRODUCTIONS — name the main characters (not surprise/twist characters) and hint at their role or situation. Give the listener people to care about BEFORE the story starts.
  3. ONE REAL SCIENCE FACT that makes the listener lean in — pulled from the news article.
  4. A TAGLINE that tells us what KIND of story this is. Be creative — make it memorable.

  TONE — MATCH THE ARC:
  The announcer's voice should prepare the listener for the KIND of story they're about to hear:
  - DRAMATIC arcs (A Tragic Fall, C Gathering Storm, F Ticking Clock, H Reunion): Warm, journalistic gravity. Edward R. Murrow inviting you into someone's life. Empathy first, dread second.
  - HORROR/TWIST arcs (D Bottle Episode, E Unreliable Witness, G Moral Inversion): Ominous, clipped, a little theatrical. Rod Serling at his most unsettling. Let silence do the work.
  - COMEDY arcs (B Comedic Spiral, I Mistaken Identity, J Enchanted World, K Schemer Undone, L Rivals): Lighter, wry, conspiratorial — like the announcer already knows how badly this is going to go and can barely hide a smile. Think Prairie Home Companion meets The Hitchhiker's Guide.

  LENGTH — SCALE TO THE EPISODE:
  - SHORT episodes (1-5 min): 2-3 sentences. Tight and punchy. Date, place, one character, hook, done.
  - MEDIUM episodes (10-20 min): 3-5 sentences. Room to name 2 characters and paint the scene.
  - LONG episodes (20+ min): 5-8 sentences. Set the world, introduce 2-3 characters by name and role, build atmosphere, let the tagline land with weight.

  EXAMPLES (showing tone variation and character reveals):
  DRAMATIC: "April 5th, 2026. A blood pressure research lab in Kyoto. Dr. Lena Vasquez has spent eleven years chasing a molecule that could save millions — and today, her funding runs out. Her lab partner, James Osei, has already packed his desk. But the data from this afternoon's trial is doing something no one predicted. Tonight on Signal Lost: the breakthrough came too late. Or did it?"
  HORROR: "March 12th, 2026. Low Earth Orbit. The International Space Station. Commander Priya Sharma runs a crew of six. Flight Engineer Tomás Ruiz handles the software. A routine update just taught the onboard AI to lie — and only Tomás noticed. Tonight on Signal Lost: trust is a human luxury."
  COMEDY: "November 3rd, 2025. A gene therapy clinic in Seoul. Dr. Park and Dr. Whitfield can't agree on anything — not the dosage, not the delivery method, not whose turn it is to refill the coffee. Last week they accidentally reversed blindness in three patients using a virus they barely understand. Now every hospital on Earth is calling. Tonight on Signal Lost: the cure works. The partnership might not survive it."
- ANNOUNCER LINE CAP (HARD RULE): The ANNOUNCER gets a maximum of 3 lines total in the entire episode — one opening introduction (see above), one closing epilogue, one optional mid-episode transition. No more. Do NOT let the ANNOUNCER deliver multi-line science lectures. If you need to convey science facts, put them in a character's mouth instead.
- DIALOGUE RATIO (HARD RULE): At least 80% of all lines must be spoken by non-ANNOUNCER characters. Science exposition delivered as character dialogue ("Hayes, if we don't reroute the coolant in 60 seconds, the whole lab goes dark") counts as drama. An ANNOUNCER reading facts does not.
- GENDER BALANCE: Aim for roughly 50/50 male and female characters (excluding ANNOUNCER). Diverse casts sound better and use the full range of available voice presets.
- The CLOSING must be a factual "Hard Science Epilogue" — keep it to 2-3 sentences maximum. One real citation. Done.

CITATION RULE (STRICT):
The epilogue MUST cite ONLY the real article provided above.
Use the exact source name and date from the article — nothing else.
NEVER use numbered references like [1], [2], [3], article #2, source (1), or any bracket number.
NEVER say "article number", "source number", or "reference number". Always say the source name directly.
DO NOT invent ArXiv IDs, paper titles, DOIs, or journal names that were not in the article.
Fabricated citations destroy the credibility of the show. One real source, cited accurately, is worth more than five invented ones.
Correct format example: "According to Science Daily, published April 3, 2026, researchers found that..."
(Use the ACTUAL source name and date from the article above — Science Daily is just an example.)

STRUCTURE:
1. === SCENE 1 === (Hook the listener — drop us into a tense human moment. THEN reveal the science angle.)
2. === SCENE 2-X === (Escalate the HUMAN stakes. Characters argue, make choices, face consequences.)
3. === SCENE FINAL === (The twist, emotional payoff, then ANNOUNCER's Hard Science Epilogue.)

TARGET: {target_minutes} minutes (~{target_words} words). Let it breathe with (beat) tags.
PRIMARY RULE: Tags always start at the beginning of a line. No inline tags.
"""


class Gemma4ScriptWriter:
    """Fetches real science news, generates a full radio drama script via Gemma 4."""

    CATEGORY = "OldTimeRadio"
    FUNCTION = "write_script"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("script_text", "script_json", "news_used", "estimated_minutes")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "episode_title": ("STRING", {
                    "default": "The Last Frequency",
                    "tooltip": "Episode title (or leave blank for Gemma to generate one)"
                }),
                "genre_flavor": (["hard_sci_fi", "space_opera", "dystopian",
                                  "time_travel", "first_contact", "cosmic_horror",
                                  "cyberpunk", "post_apocalyptic"], {
                    "default": "hard_sci_fi",
                    "tooltip": "Sub-genre flavor for the episode"
                }),
                "target_minutes": ("INT", {
                    "default": 25, "min": 1, "max": 45, "step": 1,
                    "tooltip": "Target episode length in minutes (1 for quick test)"
                }),
                "num_characters": ("INT", {
                    "default": 4, "min": 2, "max": 8, "step": 1,
                    "tooltip": "Number of speaking characters (plus announcer)"
                }),
            },
            "optional": {
                "model_id": ("STRING", {
                    "default": "google/gemma-4-E4B-it",
                    "tooltip": "Hugging Face model ID for Gemma 4"
                }),
                "custom_premise": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Optional custom story premise (overrides news-based generation)"
                }),
                "news_headlines": ("INT", {
                    "default": 3, "min": 1, "max": 5, "step": 1,
                    "tooltip": "Number of real science headlines to fetch for story inspiration"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 1.5, "step": 0.05,
                    "tooltip": "Generation temperature (higher = more creative)"
                }),
                "include_act_breaks": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include act breaks with sponsor messages (authentic style)"
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always re-execute: news changes daily (Section 12)."""
        return time.time()

    def write_script(self, episode_title, genre_flavor, target_minutes,
                     num_characters, model_id="google/gemma-4-E4B-it",
                     custom_premise="", news_headlines=3, temperature=0.8,
                     include_act_breaks=True):

        # Fetch real science news — one random story from full feed pool
        news = _fetch_science_news()
        news_block = "\n".join(
            f"- {n['headline']} ({n['source']}, {n['date']})\n\n{n.get('full_text', n['summary'])}"
            for n in news
        )
        news_json = json.dumps(news, indent=2)

        # Calculate target words
        target_words = target_minutes * 130  # ~130 wpm for dramatic reading

        # ── Easter egg: 11% chance Lemmy appears as a character ──
        # A grizzled, seen-it-all engineer/mechanic who speaks in blunt,
        # colorful metaphors. Rare enough to be a surprise, frequent enough
        # that regulars will notice. Named after Lemmy Kilmister.
        import random
        lemmy_roll = random.random() < 0.11
        lemmy_directive = ""
        if lemmy_roll:
            lemmy_directive = (
                "\nSPECIAL CHARACTER REQUIREMENT: One of the characters MUST be named LEMMY — "
                "a resourceful, slightly unconventional engineer/mechanic who operates on the fringes "
                "of authority but proves essential in critical moments. He has a hands-on technical "
                "mindset, more comfortable solving problems directly than following protocol. "
                "Personality: dryly humorous, pragmatic, rough around the edges, but loyal and "
                "dependable when it counts. He questions leadership, bends rules, but his instincts "
                "are sharp under pressure. In the team dynamic Lemmy is the fixer and improviser — "
                "he adapts quickly, thinks creatively, and keeps things moving when plans fall apart. "
                "Give him at least 3 lines of dialogue. Use the name LEMMY consistently "
                "(not ENGINEER LEMMY, just LEMMY).\n"
                "LEMMY SFX REQUIREMENT: Before LEMMY's FIRST line of dialogue, you MUST include "
                "exactly this SFX cue on its own line:\n"
                "[SFX: heavy wrench strike on metal pipe, single resonant clank]\n"
                "This is his signature sound — it plays once, the first time he appears, nowhere else.\n"
            )
            log.info("[Gemma4ScriptWriter] ★ Lemmy Easter egg activated (11%% roll) — wrench SFX cued")

        # Build prompt
        system = SCRIPT_SYSTEM_PROMPT.format(
            target_minutes=target_minutes,
            target_words=target_words,
            news_block=news_block,
            num_characters=num_characters,
        )

        user_prompt = f"""Write a complete episode of "SIGNAL LOST" — a contemporary sci-fi audio drama anthology.

EPISODE TITLE: {episode_title if episode_title else "(generate a compelling, evocative title)"}
GENRE: {genre_flavor.replace("_", " ")}
CHARACTERS: {num_characters} speaking roles plus ANNOUNCER
TARGET LENGTH: ~{target_words} words ({target_minutes} minutes)
{"STRUCTURAL BREAKS: Include 2-3 act breaks marked with [ACT TWO], [ACT THREE] etc." if include_act_breaks else ""}
{lemmy_directive}
{"PREMISE: " + custom_premise if custom_premise else "The news headlines above ARE the premise. Extrapolate them. What's the next terrifying or profound step?"}

STORY ARC SEED: Use Arc Type {random.choice("ABCDEFGHIJKL")} from the Story Arc Engine above. Commit fully to that structure.

REMEMBER: Story first. Make the listener CARE about these people before you scare them with science. Write dialogue that sounds like real humans under pressure — not scientists reading papers.

Begin the full script now. Follow this structure exactly:
=== SCENE 1 ===
[ENV: location description, ambient noise, vibe]
[SFX: establishing sound]
(beat)
[VOICE: ANNOUNCER, male, 50s, authoritative, calm] [Opening introduction — time, place, character names and roles, science hook, tagline. REQUIRED. Always first.]
[VOICE: CHARACTER_NAME, gender, age, tone, energy] First dramatic line — drop us in medias res.
[VOICE: CHARACTER_NAME, gender, age, tone, energy] Response line.
(beat)
[SFX: action sound]
...
[VOICE: ANNOUNCER, male, 50s, authoritative, calm] [Hard-science epilogue — cite ONLY the real article provided above. Headline, source, date. No invented IDs.]
[MUSIC: Closing theme]"""

        full_prompt = f"{system}\n\n{user_prompt}"

        log.info(f"[Gemma4ScriptWriter] Generating {target_minutes}min episode "
                 f"'{episode_title}' ({genre_flavor})")
        log.info(f"[Gemma4ScriptWriter] News seed: {news[0]['headline']} | {news[0]['source']}")

        # For episodes > 5 min, generate act-by-act to avoid token truncation.
        # 8,192 max_new_tokens ≈ 6,000 words. A 25-min episode needs ~3,250 words
        # which fits, but 45-min needs ~5,850 which is tight. Chunked generation
        # ensures we never hit the ceiling and produces more coherent long scripts.

        if target_minutes <= 5:
            # Short episodes: single-pass generation.
            # Floor at 1024 — even a 1-min episode needs enough tokens to
            # complete canonical structure (ENV, SFX, VOICE tags, beats).
            # Without the floor, 1-min = 260 tokens, which truncates mid-scene.
            max_new_tokens = max(int(target_words * 2.0), 1024)
            max_new_tokens = min(max_new_tokens, 8192)
            script_text = _generate_with_gemma4(
                full_prompt,
                model_id=model_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        else:
            # Long episodes: chunked act-by-act generation
            script_text = self._generate_chunked(
                system, episode_title, genre_flavor, num_characters,
                target_minutes, target_words, custom_premise, news_block,
                include_act_breaks, model_id, temperature,
                lemmy_directive=lemmy_directive,
            )

        # ── Content safety filter — catch anything the prompt policy missed ──
        script_text, blocked = _content_filter(script_text)
        if blocked:
            log.warning("[Gemma4ScriptWriter] Content filter caught %d word(s) — replaced with %s",
                        len(blocked), _REPLACEMENT)

        # ── Citation hallucination guard ──────────────────────────────────────
        # Gemma sometimes invents plausible-looking ArXiv IDs (arXiv:2401.XXXXX)
        # even when told not to. These look authoritative but are fabricated.
        # Detect and warn — the IDs are left in the text (stripping would create
        # jarring gaps) but the log makes the problem visible for review.
        _arxiv_pat = re.compile(r'\barXiv:\s*\d{4}\.\d{4,5}\b', re.IGNORECASE)
        _doi_pat   = re.compile(r'\bdoi\.org/10\.\d{4,}/\S+', re.IGNORECASE)
        hallucinated_ids = _arxiv_pat.findall(script_text) + _doi_pat.findall(script_text)

        # Cross-check against the real article source
        real_source_text = " ".join(
            f"{n['headline']} {n['source']} {n.get('full_text', n['summary'])}"
            for n in news
        ).lower()

        bad_ids = []
        for hid in hallucinated_ids:
            # If the ID string doesn't appear in any form in the real article
            # content we provided, it's almost certainly hallucinated
            if hid.lower().replace(" ", "") not in real_source_text.replace(" ", ""):
                bad_ids.append(hid)

        if bad_ids:
            log.warning(
                "[CitationGuard] %d likely hallucinated citation ID(s) detected: %s — "
                "Gemma invented these. Review the epilogue before publishing.",
                len(bad_ids), ", ".join(bad_ids)
            )
        elif hallucinated_ids:
            log.info("[CitationGuard] %d citation ID(s) found — appear to match source material.",
                     len(hallucinated_ids))

        # ── CitationGuard 2: strip numeric bracket references ─────────────────
        # Gemma sometimes outputs [1], [2], article #3 when the prompt uses
        # bracket-style placeholders like [SOURCE] as format examples. These
        # become broken grammar when spoken ("According to article .") because
        # _clean_text_for_bark already strips unrecognized bracket tags.
        # Strip them here at the source so the script text is clean before
        # _parse_script() stores it.
        _num_ref_pat = re.compile(
            r'\s*\[\d{1,3}\]'           # [1] [2] [99]
            r'|\s*\(\d{1,3}\)'          # (1) (2)
            r'|\s*article\s+#\s*\d+'    # article #3
            r'|\s*source\s+#\s*\d+'     # source #2
            r'|\s*reference\s+#\s*\d+', # reference #1
            re.IGNORECASE
        )
        stripped_text, nsubs = _num_ref_pat.subn("", script_text)
        if nsubs:
            log.warning(
                "[CitationGuard] Stripped %d numeric citation marker(s) ([1], article #N, etc.) "
                "from script text — update prompt to prevent recurrence.", nsubs
            )
            script_text = stripped_text

        # Parse into structured JSON
        script_lines = self._parse_script(script_text)
        script_json = json.dumps(script_lines, indent=2)

        # Estimate actual minutes
        word_count = sum(len(line.get("line", "").split()) for line in script_lines
                         if line.get("type") == "dialogue")
        est_minutes = max(1, word_count // 130)

        log.info(f"[Gemma4ScriptWriter] Generated {len(script_lines)} lines, "
                 f"~{word_count} words, ~{est_minutes} min")

        return (script_text, script_json, news_json, est_minutes)

    def _generate_chunked(self, system, title, genre, num_chars, target_min,
                          target_words, premise, news_block, act_breaks,
                          model_id, temperature, lemmy_directive=""):
        """Generate long scripts act-by-act to avoid token truncation.

        Step 1: Generate an outline (characters, plot beats, act structure)
        Step 2: Generate each act using the outline + previous act as context
        Step 3: Concatenate into the final script
        """
        num_acts = 3 if target_min >= 20 else 2
        words_per_act = target_words // num_acts

        # Step 1: Outline
        outline_prompt = f"""{system}

Create a detailed OUTLINE for a {target_min}-minute episode of "SIGNAL LOST."
Title: {title}
Genre: {genre.replace("_", " ")}
Characters: {num_chars} speaking roles plus ANNOUNCER
{lemmy_directive}

Return:
- Character list: name, role, gender, personality, and what they PERSONALLY have at stake (~50/50 male/female split)
- Time period and setting (derived from the science news)
- {num_acts}-act structure: inciting incident, escalation beats, twist/resolution — focus on HUMAN drama, not science exposition
- At least one moment of humor, warmth, or unexpected humanity
- The ANNOUNCER's hard-science epilogue topic and sources to cite
- Key SFX and music cues

STORY ARC SEED: Use Arc Type {random.choice("ABCDEFGH")} from the Story Arc Engine. Commit fully to that structure.

Remember: This is a DRAMA that happens to involve science, not a science report with characters. Give every character something personal to lose.

{"Premise: " + premise if premise else "The news headlines ARE the premise. Extrapolate the science into its most dramatic next step."}

Outline only — do NOT write dialogue yet."""

        log.info(f"[ScriptWriter] Generating outline ({num_acts} acts)")
        outline = _generate_with_gemma4(outline_prompt, model_id=model_id,
                                         max_new_tokens=1500, temperature=temperature)

        # Step 2: Generate each act
        acts = []
        for act_num in range(1, num_acts + 1):
            # S29: Allow users to cancel long script generation
            try:
                import comfy.model_management
                comfy.model_management.throw_exception_if_processing_interrupted()
            except ImportError:
                pass

            previous_context = "\n\n".join(acts[-1:]) if acts else "(beginning of episode)"

            act_prompt = f"""{system}

OUTLINE:
{outline}

{"PREVIOUS ACT (for continuity):" if acts else ""}
{previous_context[:2000] if acts else ""}

Now write ACT {act_num} of {num_acts} in full script format.
Target: ~{words_per_act} words for this act. Taut dialogue — fragments, interruptions, subtext.
{"This is the OPENING — start with [MUSIC: Opening theme] and ANNOUNCER setting time/place/characters. Then drop us IN MEDIAS RES." if act_num == 1 else ""}
{"This is the FINAL ACT — build to the twist, then ANNOUNCER delivers the hard-science epilogue. CITATION RULE: cite ONLY the real article provided in the news block above — its exact source name and date. NEVER use numbered references like [1], [2], article #N — always say the source name directly (e.g. 'According to Science Daily, published April 3, 2026...'). Do NOT invent ArXiv IDs or paper titles. End with [MUSIC: Closing theme]." if act_num == num_acts else ""}
{"Include an act break marker [ACT " + str(act_num + 1) + "] at the end of this act." if act_breaks and act_num < num_acts else ""}

Write Act {act_num} now:"""

            _runtime_log(f"ScriptWriter: Generating Act {act_num}/{num_acts}")
            act_text = _generate_with_gemma4(act_prompt, model_id=model_id,
                                              max_new_tokens=4096, temperature=temperature)
            acts.append(act_text)

        return "\n\n".join(acts)

    # Descriptor words that indicate a missing character name (Gemma dropped the NAME field)
    _GENDER_WORDS = frozenset([
        "male", "female", "man", "woman", "boy", "girl", "nonbinary",
        "young", "old", "older", "elderly", "middle", "teen",
    ])

    def _parse_script(self, text):
        """Parse raw script text into structured Canonical Audio Tokens.

        Robust against the most common Gemma formatting failure: omitting the
        character NAME as the first field in [VOICE:] tags, producing malformed
        tags like [VOICE: male, 40s, calm] that would silently create "MALE" as
        a character. Those are caught, logged, and assigned a positional fallback
        name (CHAR_A, CHAR_B, ...) so Bark still produces audio.
        """
        lines = []
        _fallback_counter = [0]   # mutable so the inner closure can mutate it

        # OTR Canonical 1.0 RegEx Patterns
        scene_pat = re.compile(r'^===\s*SCENE\s+(.+?)\s*===', re.IGNORECASE)
        env_pat   = re.compile(r'^\[ENV:\s*(.+?)\]$',          re.IGNORECASE)
        sfx_pat   = re.compile(r'^\[SFX:\s*(.+?)\]$',          re.IGNORECASE)
        # Full voice tag: [VOICE: NAME, traits...] dialogue text
        voice_pat = re.compile(r'^\[VOICE:\s*(.+?),\s*(.+?)\]\s*(.+)$', re.IGNORECASE)
        beat_pat  = re.compile(r'^\(beat\)$', re.IGNORECASE)

        for raw_line in text.strip().splitlines():
            s = raw_line.strip()
            if not s:
                continue

            m = scene_pat.match(s)
            if m:
                lines.append({"type": "scene_break", "scene": m.group(1)})
                continue

            m = env_pat.match(s)
            if m:
                lines.append({"type": "environment", "description": m.group(1)})
                continue

            m = sfx_pat.match(s)
            if m:
                lines.append({"type": "sfx", "description": m.group(1)})
                continue

            m = beat_pat.match(s)
            if m:
                lines.append({"type": "pause", "kind": "beat", "duration_ms": 200})
                continue

            m = voice_pat.match(s)
            if m:
                raw_name   = m.group(1).strip()
                voice_traits = m.group(2).strip()
                dialogue   = m.group(3).strip()

                # Detect the "no NAME" failure: first field is a gender/age word
                # e.g. [VOICE: male, 40s, calm] instead of [VOICE: HAYES, male, 40s, calm]
                if raw_name.lower() in self._GENDER_WORDS:
                    _fallback_counter[0] += 1
                    fallback_name = f"CHAR_{chr(64 + _fallback_counter[0])}"  # CHAR_A, CHAR_B…
                    log.warning(
                        "[ScriptParser] Malformed VOICE tag — name field is a descriptor word '%s'. "
                        "Assigning fallback name '%s'. Full line: %s",
                        raw_name, fallback_name, s[:120]
                    )
                    # Reconstruct voice_traits: prepend the dropped name-word back
                    voice_traits = f"{raw_name}, {voice_traits}"
                    character_name = fallback_name
                else:
                    character_name = raw_name.upper()

                lines.append({
                    "type": "dialogue",
                    "character_name": character_name,
                    "voice_traits": voice_traits,
                    "line": dialogue,
                })
                continue

            # Fallback for structured text that might miss a tag
            if s and not s.startswith("#") and not s.startswith("---"):
                lines.append({"type": "direction", "text": s})

        malformed = _fallback_counter[0]
        if malformed:
            log.warning(
                "[ScriptParser] %d malformed VOICE tag(s) detected (missing character name). "
                "Update SCRIPT_SYSTEM_PROMPT Section 1 example if this recurs.", malformed
            )

        return lines


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2: DIRECTOR
# ─────────────────────────────────────────────────────────────────────────────

DIRECTOR_PROMPT = """You are the PRODUCTION DIRECTOR for the Canonical Audio Engine 1.0.
Your task is to take a raw script and compile it into a deterministic JSON production plan.

═══ 🧱 1. SCRIPT STRUCTURE (CANONICAL 1.0) ═══
The script follows these tokens:
- === SCENE X ===
- [ENV: description]
- [SFX: description]
- [VOICE: NAME, gender, age, tone, energy] Dialogue...
- (beat)

═══ 🧱 2. VOICE MAPPING RULES ═══
- Scan all [VOICE:] tags in the script. The FIRST FIELD (before the first comma) is the CHARACTER NAME.
- Collect every unique CHARACTER NAME. Map each to one UNIQUE voice preset.
- You have 10 English native presets. ALWAYS use English native presets (en_speaker_*).
  Do NOT use international/foreign presets (de_*, fr_*, es_*, etc.) — they risk
  language drift in the TTS engine. The en_speaker_* presets naturally cover a
  wide range of vocal character: deep, warm, raspy, youthful, authoritative, etc.
  All dialogue is always written in pure English.
- The JSON key MUST be the CHARACTER NAME EXACTLY AS IT APPEARS (all caps, no descriptors).
  WRONG key: "HAYES, male, 40s, calm"
  RIGHT key: "HAYES"
- Use the gender/age/tone in the tag to pick the best preset match:

  ENGLISH NATIVE:
  v2/en_speaker_0 = Male, authoritative, deep (best for ANNOUNCER)
  v2/en_speaker_1 = Male, mid-range
  v2/en_speaker_2 = Female, neutral
  v2/en_speaker_3 = Male, younger
  v2/en_speaker_4 = Female, warmer
  v2/en_speaker_5 = Male, older
  v2/en_speaker_6 = Male, character voice
  v2/en_speaker_7 = Female, higher pitch
  v2/en_speaker_8 = Male, gravelly/raspy (good for LEMMY)
  v2/en_speaker_9 = Female, authoritative

- ANNOUNCER: Randomly select ONE of {v2/en_speaker_0 (male, authoritative), v2/en_speaker_2 (female, clear), v2/en_speaker_4 (female, energetic), v2/en_speaker_9 (female, mature)} for gender balance and vocal variety. Keep it consistent within the episode.
- Each character gets ONE preset that stays consistent for the entire episode.
- If two characters share a gender/age, still assign DIFFERENT presets.
- ONLY use en_speaker_* presets. Never assign international presets.

═══ 🧱 3. OUTPUT FORMAT (STRICT JSON) ═══
{{
  "episode_title": "...",
  "voice_assignments": {{
    "ANNOUNCER": {{
      "voice_preset": "v2/en_speaker_4",
      "notes": "Female, energetic, authoritative"
    }},
    "HAYES": {{
      "voice_preset": "v2/en_speaker_1",
      "notes": "Male, calm, 40s"
    }}
  }},
  "sfx_plan": [
    {{
      "cue_id": "sfx_001",
      "type": "env|sfx",
      "description": "...",
      "generation_prompt": "Foley style prompt for audio generator"
    }}
  ],
  "pacing": {{
    "beat_pause_ms": 200
  }}
}}

Output ONLY the JSON block. No prose before or after it.

SCRIPT:
{script_text}
"""


class Gemma4Director:
    """Takes a script and generates a full production plan via Gemma 4."""

    CATEGORY = "OldTimeRadio"
    FUNCTION = "direct"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("production_plan_json", "voice_map_json", "sfx_plan_json", "music_plan_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "model_id": ("STRING", {
                    "default": "google/gemma-4-E4B-it",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.4, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "Lower = more consistent JSON output"
                }),
                "prefer_bark": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Prefer Bark for main characters (more expressive)"
                }),
                "vintage_intensity": (["subtle", "moderate", "heavy", "extreme"], {
                    "default": "moderate",
                    "tooltip": "How vintage/degraded should the final audio sound"
                }),
            },
        }

    def direct(self, script_text, model_id="google/gemma-4-E4B-it",
               temperature=0.4, prefer_bark=True, vintage_intensity="moderate"):

        prompt = DIRECTOR_PROMPT.format(script_text=script_text[:6000])

        if not prefer_bark:
            prompt += "\nNOTE: Prefer Parler-TTS for all characters (more control over voice style)."

        vintage_map = {
            "subtle":   {"radio_static_amount": 0.05, "vinyl_crackle": 0.03, "tube_warmth": 0.4, "frequency_rolloff_hz": 8000, "hum_60hz": 0.02},
            "moderate": {"radio_static_amount": 0.15, "vinyl_crackle": 0.10, "tube_warmth": 0.7, "frequency_rolloff_hz": 6000, "hum_60hz": 0.05},
            "heavy":    {"radio_static_amount": 0.25, "vinyl_crackle": 0.20, "tube_warmth": 0.9, "frequency_rolloff_hz": 4500, "hum_60hz": 0.08},
            "extreme":  {"radio_static_amount": 0.40, "vinyl_crackle": 0.35, "tube_warmth": 1.0, "frequency_rolloff_hz": 3500, "hum_60hz": 0.12},
        }

        log.info(f"[Gemma4Director] Generating production plan (vintage={vintage_intensity})")

        # Scale max_new_tokens to script length — a 1-min script needs ~1200 tokens
        # for the production plan, a 25-min script needs ~2500.
        # Reduced minimum from 1200 to 800 for shorter scripts to speed up generation
        # and prevent VRAM/System RAM swap death spirals on laptop GPUs.
        script_len = len(script_text)
        max_tokens = min(3000, max(800, script_len // 2))
        log.info(f"[Gemma4Director] max_new_tokens={max_tokens} (script={script_len} chars)")

        raw = _generate_with_gemma4(
            prompt,
            model_id=model_id,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        # Extract JSON from response
        plan = self._extract_json(raw)

        # Override vintage settings with user's intensity choice
        if plan:
            plan["vintage_settings"] = vintage_map.get(vintage_intensity, vintage_map["moderate"])

        plan_json = json.dumps(plan, indent=2)
        voice_json = json.dumps(plan.get("voice_assignments", {}), indent=2)
        sfx_json = json.dumps(plan.get("sfx_plan", []), indent=2)
        music_json = json.dumps(plan.get("music_plan", []), indent=2)

        log.info(f"[Gemma4Director] Plan: {len(plan.get('voice_assignments', {}))} voices, "
                 f"{len(plan.get('sfx_plan', []))} SFX cues, "
                 f"{len(plan.get('music_plan', []))} music cues")

        return (plan_json, voice_json, sfx_json, music_json)

    def _extract_json(self, text):
        """Extract JSON object from LLM output (handles markdown fences, truncation)."""
        log.info(f"[Gemma4Director] Raw output length: {len(text)} chars")
        log.info(f"[Gemma4Director] Raw output preview: {text[:200]}...")

        # Try to find JSON block
        patterns = [
            re.compile(r'```json\s*\n(.*?)\n```', re.DOTALL),
            re.compile(r'```\s*\n(.*?)\n```', re.DOTALL),
            re.compile(r'(\{.*\})', re.DOTALL),
        ]
        for pat in patterns:
            m = pat.search(text)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    # Try to repair: strip trailing commas, close unclosed braces
                    candidate = m.group(1)
                    candidate = re.sub(r',\s*([}\]])', r'\1', candidate)  # trailing commas
                    # If JSON was truncated, try closing it
                    open_braces = candidate.count('{') - candidate.count('}')
                    open_brackets = candidate.count('[') - candidate.count(']')
                    if open_braces > 0 or open_brackets > 0:
                        log.info(f"[Gemma4Director] Attempting JSON repair: +{open_braces} braces, +{open_brackets} brackets")
                        candidate += ']' * open_brackets + '}' * open_braces
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError as e:
                            log.warning(f"[Gemma4Director] JSON repair failed: {e}")
                    continue

        # Last resort: find the first { and try to build valid JSON from there
        brace_start = text.find('{')
        if brace_start >= 0:
            candidate = text[brace_start:]
            # Try progressively shorter substrings
            for end_offset in range(len(candidate), max(0, len(candidate) - 200), -1):
                try:
                    return json.loads(candidate[:end_offset])
                except json.JSONDecodeError:
                    continue

        log.warning("[Gemma4Director] Could not parse JSON from model output — using fallback")
        log.warning(f"[Gemma4Director] Full raw output:\n{text[:1000]}")
        return {
            "error": "Failed to parse production plan",
            "raw_output": text[:2000],
            "voice_assignments": {},
            "sfx_plan": [],
            "music_plan": [],
            "vintage_settings": {},
            "pacing": {"breath_pause_ms": 400, "beat_pause_ms": 1500, "pause_ms": 2000},
        }
