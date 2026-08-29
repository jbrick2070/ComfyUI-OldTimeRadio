r"""
OTR Orchestrator - Script Writer + Ledger Writer for "SIGNAL LOST"
===================================================================

Two nodes:
  1. LLMScriptWriter - Fetches real daily science news via RSS, feeds
     it to an LLM to generate a full audio drama script. Contemporary
     sci-fi anthology format (Black Mirror / NPR Invisibilia / Arrival).
     News-as-spine: real headlines become the inciting incident,
     extrapolated to dramatic extremes. Includes a hard-science
     epilogue citing real sources (ArXiv, Nature, etc.).

  2. LPL writer + cast helpers - Generates the L3 ledger directly
     (cast, lines, meta.visual_plan, meta.style) via the
     LedgerScriptWriter path. Cast-lock invariants run at writer
     exit; downstream consumers read the ledger via
     FreezeCascade.script_json fanout.

The legacy LLMDirector class was removed in voice-path-cleanbreak
S2 (commit 249bc06). Voice and video paths share the L3 ledger as
the single source of truth; there is no Director-shape projection
anywhere in active code.

LLM runs via transformers (local GPU). Content safety filter
catches profanity/NSFW that slips past the prompt policy.

v1.0  2026-04-04  Jeffrey Brick
v2.0  2026-05-13  voice-path-cleanbreak S23.2 (docstring scrub)
"""

import json
import logging
import os
import random
import re
import time

# The Lemmy coin flip does NOT live here. `_LEMMY_RNG` / `_LEMMY_HISTORY` (and
# the `SystemRandom` import that existed only for them) were removed
# 2026-08-28: definition-only, with no reader anywhere. The live roll is in
# `config/cast_pools.py` -- a different module object, so these were never the
# same RNG the episode actually used.
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

# Project State (v1.4 Theme C) - series bible for cross-episode consistency.
# (lean-mean order 5, 2026-08-23) `from .project_state import ProjectState`
# was here -- imported and never used, the only repo reference to that module.
# The node and module are retired; the import went with them.
# Per-phase VRAM telemetry (v1.4 Theme C). CUDA-absent safe.
from ._vram_log import vram_snapshot, vram_reset_peak
# (`force_vram_offload` was dropped from this import 2026-08-28: zero AST
# loads in this module -- an import is not a use. The function itself lives
# on in _vram_log and its real callers.)

# Canonical OTR paths -- single source of truth for output locations.
# (director_raw_dump_dir was deleted in voice-path-cleanbreak S23.1
# with no live consumer remaining; no replacement import needed.)


# Lazy heavy imports (Section 8) - torch, numpy, transformers inside methods/classes only

log = logging.getLogger("OTR")

# BaseStreamer for custom heartbeat logic.
# Graceful stub allows importing this module in test environments without
# a GPU or transformers installed - ScriptParser and pure-logic tests work fine;
# actual LLM generation will raise ImportError at call time as expected.
try:
    from transformers.generation.streamers import BaseStreamer, TextStreamer
except ImportError:
    class BaseStreamer:  # type: ignore[no-redef]
        """Stub - transformers not installed in this environment."""
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
    except Exception:
        # STAYS SILENT, DELIBERATELY, AND ONLY THIS ONE DOES. This function IS
        # the heartbeat logger, so reporting its own failure through itself
        # recurses, and a monitoring write must never kill an episode.
        # Narrowed from a bare `except:` so a KeyboardInterrupt / SystemExit
        # during the write propagates instead of being swallowed by the log.
        pass

# (lean-mean 2026-08-22: the FIRST _truncate_at_sentence_boundary /
# _tail_at_sentence_boundary bodies lived here and were permanently shadowed by
# a later same-named pair ~2200 lines down -- which was itself uncalled. All
# four are deleted. tests/test_no_duplicate_top_level_defs.py now refuses the
# pattern, so it cannot come back silently the way it did here twice.)

# (rip-sfx 2026-08-06: the [SFX:]-emitting _inject_scene_transitions body that
# lived here was dead code -- permanently shadowed by a later same-named
# function, itself also uncalled -- and both are deleted.)


# -----------------------------------------------------------------------------
# Phase 3c: WALL-CLOCK TIMEOUT WRAPPER
# Heavy LLM phases (Open-Close outlines, Critique, Revision) can hang if
# LLM stalls on a malformed prompt or GPU goes sideways. We run the
# call in a worker thread and bound it with a wall-clock budget. On timeout
# the thread is left to drain in the background (Gemma generation is not
# cancellable mid-token) but the caller gets control back via TimeoutError
# and the pipeline can fall back to its last known-good artifact.
# -----------------------------------------------------------------------------
class _LLMTimeout(Exception):
    """Raised when an LLM phase exceeds its wall-clock budget."""
    pass


class _LLMTimeoutWorkflowPause(_LLMTimeout):
    """S22.1 (IMP-26): raised when an LLM timeout occurs AND the next
    workflow stage cannot safely run with an orphan worker still on
    GPU. ComfyUI catches this at the node boundary and halts the
    queue cleanly.

    Subclasses ``_LLMTimeout`` so existing handlers that catch the
    base class still match -- they just see a more specific subtype
    now. New downstream consumers can branch on this class for
    graceful UI handling (e.g., a "rerun" prompt).

    The cache invalidation in ``_run_with_timeout`` (via
    `_otr_model_loader.invalidate_cache_no_gpu_teardown`) handles
    the LLM -> LLM case (next LLM phase forces a fresh load from
    disk). Raising this class handles the LLM -> visual case where the
    next stage is FLUX/LTX/HuMo and would race the orphan's still-
    running CUDA kernels. The only safe move is to halt the workflow
    so the orphan finishes naturally without anything else trying
    to touch the GPU.

    Assumption: ComfyUI's node-execution layer surfaces uncaught
    exceptions as queue halts. Stable since the 2025 unified-
    execution refactor; if a future ComfyUI version swallows the
    exception, this assumption needs revisiting.
    """
    pass


import threading
# `_TIMEOUT_CTX = threading.local()` was REMOVED 2026-08-28: its ONLY reader
# was the deleted GemmaHeartbeatStreamer's put(), so the writes and cleanup in
# `_run_with_timeout` had become write-only ceremony. The deadline the REAL
# transports check is loader-owned -- `set_generation_deadline()` below, read
# back by `_DeadlineStoppingCriteria` and the GGUF backend -- and that path is
# untouched.

def _run_with_timeout(fn, timeout_sec, phase_label="LLM"):
    """Run fn() in a worker thread with a wall-clock timeout.

    Returns fn's return value on success.
    Raises _LLMTimeout if the budget is exceeded.
    Re-raises any exception fn raised.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    from . import _otr_model_loader as _otr_loader_mod
    vram_reset_peak(phase_label)

    # ONE absolute deadline, computed BEFORE submit and shared by the worker
    # and the parent. Previously the worker computed its own
    # `time.time() + timeout_sec` AFTER being scheduled, while the parent's
    # future.result(timeout=timeout_sec) started counting at submission -- so
    # the worker's deadline outlived the parent's timeout by however long the
    # executor took to start it, and an abandoned worker got a grace period
    # nobody intended. monotonic, not time.time(): an epoch clock can step.
    deadline = time.monotonic() + timeout_sec

    def _worker():
        # The loader-owned deadline is what the real transports check: the
        # transformers closures via _DeadlineStoppingCriteria, and the GGUF
        # backend via get_generation_deadline() + conditional streaming.
        # (A thread-local mirror of this deadline was removed 2026-08-28
        # with its only reader, the legacy GemmaHeartbeatStreamer.)
        # Both now read the SAME monotonic value installed here.
        #
        # Not best-effort: a guard whose install can silently no-op is
        # worse than none (it claims protection that is not there), so a
        # failure here fails the call loudly rather than swallowing it.
        #
        # Installation moved INSIDE the try so the finally owns cleanup for
        # everything it sets -- previously an exception between the two
        # assignments could leave one installed with no owner.
        try:
            _otr_loader_mod.set_generation_deadline(deadline)
            # A worker scheduled AFTER the budget already expired must not
            # start at all. This is the check that covers the dominant
            # overrun case: request_slot (a cold model load, tens of seconds
            # for a ~12 GB GGUF) runs inside fn(), and NO deadline mechanism
            # on any lane can interrupt a load once it is under way.
            if time.monotonic() > deadline:
                raise _otr_loader_mod.GenerationDeadlineExceededError(
                    f"{phase_label}: {timeout_sec}s budget already expired "
                    f"before the worker was scheduled; refusing to start"
                )
            return fn()
        finally:
            _otr_loader_mod.set_generation_deadline(None)

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"otr-{phase_label}")

    try:
        future = executor.submit(_worker)
        try:
            res = future.result(
                timeout=max(0.0, deadline - time.monotonic()),
            )
            # LATE-RESULT RECHECK. Future.result() returns a finished future
            # whenever it is finished when the waiter reacquires the
            # condition; it performs no independent elapsed-time postcheck
            # (CPython concurrent/futures/_base.py). So a worker that
            # finished AFTER the absolute deadline but before we observed it
            # would otherwise be accepted as a clean success -- no cache
            # invalidation, no workflow pause. That is the same
            # accepted-but-wrong-artifact class PBUG-20260825-04 fixed inside
            # the transformers closure, and it is transport-agnostic, so it
            # belongs HERE rather than being re-solved per lane.
            if time.monotonic() > deadline:
                raise _otr_loader_mod.GenerationDeadlineExceededError(
                    f"{phase_label}: worker returned after the {timeout_sec}s "
                    f"budget expired; result discarded rather than accepted late"
                )
            vram_snapshot(phase_label)
            return res
        except (FuturesTimeout, _otr_loader_mod.GenerationDeadlineExceededError) as _timeout_exc:
            # GenerationDeadlineExceededError means the worker's own
            # generate() call noticed the deadline and raised instead of
            # returning truncated text -- see PBUG-20260825-04. Routed
            # through the exact same recovery path as FuturesTimeout so a
            # deadline hit landing right before/after this future.result()
            # call can never look like a clean (but truncated) success.
            _runtime_log(f"TIMEOUT: {phase_label} exceeded {timeout_sec}s wall-clock budget")
            log.warning("[Timeout] %s phase exceeded %ds - halting the "
                        "workflow (%s)",
                        phase_label, timeout_sec, type(_timeout_exc).__name__)
            vram_snapshot(f"{phase_label}_timeout")

            # BUG-LOCAL-111 + BUG-LOCAL-228 fix: timeout-recovery cache
            # invalidation.
            #
            # When FuturesTimeout fires, the worker thread is still running
            # an LLM forward pass on GPU. Python cannot safely terminate
            # threads, and `executor.shutdown(wait=False)` below does NOT
            # kill the worker -- it keeps churning until its forward pass
            # completes naturally (could be 30-60+ more seconds for a 16K
            # prompt). Result: the GPU has in-flight kernels the main
            # thread doesn't control. The cached model instance thinks
            # it's idle but the orphan is still mutating its tensors.
            # The NEXT phase that calls model.cpu() / any CUDA op
            # collides with the orphan's stale ops and Python aborts
            # with `cudaErrorIllegalAddress`.
            #
            # The recovery path invalidates the canonical
            # `_otr_model_loader.LLM_CACHE` dict references in-place
            # via `invalidate_cache_no_gpu_teardown` -- dict-only
            # invalidator, NO GPU calls. Orphan thread keeps its
            # model reference in the worker's stack frame and exits
            # naturally; the next `request_slot` forces a fresh load.
            try:
                # `invalidate_cache_no_gpu_teardown` (NOT `unload_llm`)
                # is the GPU-safe path here. `unload_llm` would call
                # `model.to("cpu")` + `torch.cuda.empty_cache()` while
                # the orphan worker thread is still executing CUDA
                # kernels on the cached model -- which is exactly the
                # `cudaErrorIllegalAddress` failure mode. The helper
                # invalidates the cache dict references WITHOUT
                # touching the GPU; the orphan thread holds the model
                # in its stack frame and exits naturally.
                # See BUG-LOCAL-228.
                _otr_loader_mod.invalidate_cache_no_gpu_teardown()
                _runtime_log(
                    f"TIMEOUT_RECOVERY: LLM_CACHE invalidated (GPU "
                    f"untouched; orphan {phase_label} worker keeps "
                    f"its model reference until natural completion). "
                    f"Next request_slot forces a fresh load."
                )
            except Exception as _recovery_exc:  # noqa: BLE001
                log.warning(
                    "[Timeout] cache invalidation failed (next phase may "
                    "still crash on stale CUDA state): %s",
                    _recovery_exc,
                )

            # S22.1 (IMP-26): raise the workflow-pause subclass so
            # ComfyUI halts the queue before the next stage races
            # the orphan worker's still-running CUDA kernels. The
            # cache invalidation above handles LLM -> LLM; this
            # raise handles LLM -> visual (FLUX / LTX / HuMo).
            raise _LLMTimeoutWorkflowPause(
                f"{phase_label} exceeded {timeout_sec}s; orphan "
                f"worker still on GPU. Halting workflow to prevent "
                f"the next visual stage from racing the orphan's "
                f"CUDA kernels. Re-run the workflow; the cache "
                f"invalidation above guarantees the next attempt "
                f"loads fresh."
            ) from _timeout_exc
    finally:
        # Don't wait for the orphaned worker - let it drain in the background.
        # Combined with the cache invalidation above, the orphan completes
        # in its own time WITHOUT the next phase trying to reuse its tensors.
        executor.shutdown(wait=False)


# -----------------------------------------------------------------------------
# THE CAST-CONSOLIDATION CLUSTER WAS REMOVED 2026-08-28.
#
# Four functions (_norm_cast_key, _cast_names_should_merge, and the two
# _consolidate_similar_cast_rows* entry points) that merged near-duplicate
# cast rows -- LLOYD vs LLOYD KAPOOR, STANLEY vs STANLEARY. They fixed a
# REAL bug (BUG-LOCAL-071/098, live on two consecutive runs in April 2026),
# and the banner above them claimed the LPL writer's cast-lock path used
# them. IT NEVER DID. Their only caller was the deleted LLMDirector.direct();
# that whole legacy class was deleted on 2026-05-12 (249bc06c, Director
# retirement).
#
# THE DEFECT IS NOT JUST UNOBSERVED, IT IS STRUCTURALLY IMPOSSIBLE NOW.
# The old bug needed cast rows DERIVED from LLM dialogue tags; today the
# cast is locked FIRST and the LLM is constrained to it, behind six
# independent guards -- pool names against a taken_names set, duplicate
# rejection in the cast validator, RuntimeErrors on duplicate names and on
# a count mismatch, an OutlineFailedError reroll for invented speakers, and
# a bare char_id_by_name subscript that raises rather than minting a row.
#
# MEASURED BEFORE DELETING: 1,987 frozen ledgers scanned with the shipped
# merge rule -- ZERO hits, zero dangling char_id refs, zero duplicate names
# or ids. A deliberately looser heuristic surfaced 57 pairs, and every one
# was two real characters (mothers and daughters, siblings, LAB TECHNICIAN
# 1/2) with different genders and different voices.
#
# AND WIRING IT WOULD HAVE BROKEN THE BUILD: a merge drops a cast row, so
# non_announcer_count falls below the requested num_characters and the
# assertion at OTR_LedgerScriptWriter.py:3999-4005 raises 'Cast lock count
# mismatch' -- every render where it fired would die. The generalizable
# lesson survives as BUG_BIBLE legacy_id BUG-LOCAL-068.


# THE BARK PRESET-HEALTH CLUSTER WAS REMOVED 2026-08-28 -- the one C2 item
# that was NOT ordinary dead code, resolved by verdict after an adversarial
# review (kibitz-runs/2026-08-28-c2-bark-health/).
#
# `_bark_test_presets` / `_bark_health_check` / `_bark_health_check_for_cast`
# were LIVE until the Director retirement (249bc06c, 2026-05-12) deleted
# their only call site, silently orphaning a real safety check: nothing on
# the current path validated that a Bark preset produces audible audio, and
# the adversarial pass CONFIRMED a finite, nonempty, silent tensor passes
# every downstream contract (pack, sequence, enhance, master, mux,
# obs_publish) -- none of which tests audibility.
#
# THE SAFETY MIGRATED BEFORE THE DELETION, to the right seam:
# BarkEngine.generate_voice now rejects empty / nonfinite /
# peak-below-1e-4 output with BarkSilentOutputError, on every live
# production Bark render, BEFORE downstream spend. Never a preset remap --
# the deleted cluster's remapping behaviour is exactly what the no-fallback
# rip forbids reintroducing.
#
# The tables went with it: `_VOICE_PROFILES` was a semantic duplicate of
# config/cast_pools.py; `_ANNOUNCER_PRESETS` and `_LEMMY_PROFILE` were
# stale, DIVERGENT copies of definitions that live elsewhere -- keeping a
# wrong copy of a voice table is how a settled voice regresses.







# -----------------------------------------------------------------------------
# LOG CLEANUP - compliant fixes handle most warnings at the source.
# These catch residual library noise from urllib3/httpx cache checks.
# -----------------------------------------------------------------------------
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# PROCEDURAL CHARACTER GENERATOR - name, age, gender, demeanor, accent, voice
# All traits derived deterministically from episode seed + character index.
# LEMMY stays LEMMY with fixed traits. ANNOUNCER stays ANNOUNCER.
#
# BARK TTS ACCENT RULES (per Suno documentation):
#   - Foreign preset + pure English text = English spoken with that accent
#   - en_speaker_* = neutral American/British English
#   - de_speaker_* = English with German accent
#   - fr_speaker_* = English with French accent
#   - es_speaker_* = English with Spanish accent  ... etc.
#   - ALL text is ALWAYS pure ASCII English (enforced by ASCII sanitizer
#     in batch_bark_generator.py) - this prevents language drift
#   - Temperature capped at 0.55 for international presets (0.5 first lines)
# -----------------------------------------------------------------------------

# The `_FIRST_NAMES` / `_LAST_NAMES` pools were removed 2026-08-28:
# definition-only, with no reader in production or tests. The LIVE name
# pools are `config/cast_pools.py` (FIRST_NAMES_BY_GENDER /
# FIRST_NAMES_BY_GENRE / LAST_NAMES), which is what casting actually draws
# from -- these were a stale second copy, and their sibling trait pools had
# already been removed for the same reason.

# The procedural trait pools (_GENDERS / _AGE_BRACKETS / _DEMEANORS /
# _VOICE_TRAITS) were removed 2026-08-28: definition-only, zero loads --
# the cast-roll path they described no longer draws from this module.








# -----------------------------------------------------------------------------
# NEWS FETCHER - pulls real science headlines to seed the story
# -----------------------------------------------------------------------------

SCIENCE_NEWS_FEEDS = [
    # -- Open-access: full article text fetchable, no paywall --
    "https://www.sciencedaily.com/rss/all.xml",           # Best: full articles, open
    "https://www.eurekalert.org/rss/technology_engineering.xml",  # Press releases, open
    "https://www.eurekalert.org/rss/space.xml",           # Press releases, open
    "https://www.eurekalert.org/rss/biology.xml",         # Press releases, open
    "https://www.eurekalert.org/rss/chemistry_physics.xml", # Press releases, open
    "https://www.eurekalert.org/rss/earth_environment.xml", # Press releases, open
    # -- Government / institutional (fully open) --
    "https://www.nasa.gov/rss/dyn/breaking_news.rss",     # NASA, open
    "https://www.nih.gov/news-events/news-releases.xml",  # NIH, open
    "https://www.nsf.gov/rss/rss_www_news.xml",           # NSF, open
    # -- UCLA Newsroom (open-access institutional research) --
    "https://newsroom.ucla.edu/cats/health_+_behavior.xml",      # Best: full content:encoded in RSS
    "https://newsroom.ucla.edu/cats/science_+_technology.xml",   # Open-access, URL scrape works
    "https://newsroom.ucla.edu/cats/environment_+_climate.xml",  # Open-access, URL scrape works
    # -- Open journalism (full text accessible) --
    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",  # BBC, open
    "https://feeds.arstechnica.com/arstechnica/science",  # Ars, open
    "https://theconversation.com/us/science/rss",         # The Conversation, open
    "https://cosmosmagazine.com/feed/",                   # Cosmos, open
    # -- MIT News (open-access university research feeds) --
    "https://news.mit.edu/topic/mitartificial-intelligence2-rss.xml",  # MIT AI, open
    "https://news.mit.edu/topic/mitmachine-learning-rss.xml",          # MIT machine learning, open
    "https://news.mit.edu/topic/mitcomputers-rss.xml",                 # MIT CSAIL / computer science, open
    "https://news.mit.edu/topic/mitrobotics-rss.xml",                  # MIT robotics, open
    "https://news.mit.edu/topic/mitquantum-computing-rss.xml",         # MIT quantum computing, open
    "https://news.mit.edu/rss/topic/neuroscience-neurology-and-cognitive-sciences",  # MIT brain + cognitive sci, open
    "https://news.mit.edu/topic/mitsynthetic-biology-rss.xml",         # MIT synthetic biology, open
    "https://news.mit.edu/topic/mitmedia-lab-0-rss.xml",               # MIT Media Lab, open
    # -- Carnegie Mellon (open-access university research feeds) --
    "https://www.ri.cmu.edu/feed/",                       # CMU Robotics Institute, open
    "https://www.cs.cmu.edu/news/feed.rss",               # CMU School of Computer Science, open
    "https://blog.ml.cmu.edu/feed/",                      # ML@CMU research blog (low-volume), open
    "https://hcii.cmu.edu/taxonomy/term/72/feed",         # CMU HCII human-computer interaction, open
    "https://www.sei.cmu.edu/news/feeds/latest/rss/",     # CMU SEI software engineering / AI / security, open
]


#: Set (once) to the ImportError text when the HTML body scraper cannot load.
#: Its ONLY reader is the log-once guard below -- it keeps a missing
#: BeautifulSoup from re-reporting itself on every article in a run. It does
#: NOT reach the source-floor failure message, which an earlier version of this
#: comment claimed; that message still names the feeds. Empty = available.
_BODY_SCRAPER_UNAVAILABLE: str = ""


def _fetch_full_article(url, timeout=20):
    """Fetch the full text of a science article from its URL.

    Fetches through the bounded seam (`_otr_feed_fetch`) and uses
    BeautifulSoup to strip HTML boilerplate and extract the article body.
    Returns all extracted text admitted by the secure 2 MiB fetch seam so the
    story engine gets methodology, findings, implications, and the article
    tail - not just the RSS teaser or an arbitrary local slice.

    Wave 5: the fetch is https-only, size-capped, redirect-capped and
    address-checked. The two failure classes are NOT the same thing and are
    handled differently on purpose. A `FeedFetchUnavailable` (paywall, bot
    block, 404, timeout) still returns "" so the caller degrades to the next
    candidate exactly as before. A `FeedFetchRefused` -- our own bound tripped,
    e.g. a redirect into the private network or a 40 MB body -- PROPAGATES.
    That is a defect in the URL or the configuration, and swallowing it is how
    this function spent an unknown number of episodes silently returning ""
    for a missing bs4 (see the scar below).

    The scraper tries a cascade of CSS selectors before falling back to
    the full document, so it handles sites that don't use semantic
    <article>/<main> tags (e.g. UCLA Newsroom, institutional press pages).
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        # A MISSING PACKAGE MUST NOT LOOK LIKE A PAYWALL.
        #
        # This used to `return ""`, silently. The caller then logged "scrape
        # blocked - RSS summary" and fell back to the RSS teaser, and the v4
        # source floor eventually failed the run with "No science RSS candidate
        # met the v4 source floor ... after inspecting 10 candidates" -- an error
        # that blames the FEEDS for a package that was never installed.
        #
        # Found 2026-07-14: bs4 was absent from the ComfyUI venv, so EVERY
        # science article body had been "" for as long as that was true. Every
        # science-sourced episode was written from a ~120-character headline
        # teaser instead of the methodology and findings this function exists to
        # fetch (a live probe after installing it returned 2,041 and 6,708
        # characters from the same feed, in 0.3s). The degradation was total,
        # permanent, and invisible.
        #
        # So say it once, LOUDLY, and record it so the floor's own failure
        # message can name the real cause instead of the feeds.
        global _BODY_SCRAPER_UNAVAILABLE
        if not _BODY_SCRAPER_UNAVAILABLE:
            _BODY_SCRAPER_UNAVAILABLE = str(exc)
            log.error(
                "[NewsFetcher] ARTICLE BODY SCRAPING IS DISABLED: %s. Every "
                "article will fall back to its RSS teaser (~100-300 chars), so "
                "the v4 source floor (>=400 chars) can only ever be met by the "
                "few feeds that publish full content:encoded. Stories will be "
                "written from headlines, not findings. Fix: "
                "pip install beautifulsoup4",
                exc,
            )
        return ""

    from ._otr_feed_fetch import (
        FeedFetchRefused, FeedFetchUnavailable, fetch_article,
    )

    try:
        document = fetch_article(url, deadline_s=timeout)
    except FeedFetchRefused:
        raise                      # our own bound -- never swallowed
    except FeedFetchUnavailable as exc:
        log.debug("[NewsFetcher] article body unavailable (%s): %s",
                  exc.reason, url)
        return ""

    try:
        soup = BeautifulSoup(document.text, "html.parser")

        # Strip boilerplate - nav, ads, footer, sidebar, scripts
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "form", "noscript", "iframe"]):
            tag.decompose()

        # Cascade of content selectors - most specific to least.
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
            body = soup  # last resort - full stripped document

        # Extract paragraphs AND headings - h2/h3 carry section context
        # (methodology, implications, researcher quotes) that's often the
        # richest science content buried past the lede.
        content_tags = body.find_all(["p", "h2", "h3"])
        text = " ".join(tag.get_text(" ", strip=True) for tag in content_tags)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception:
        return ""


# 2026-04-29: News-history persistence path. Stores recently-used article
# URLs so the curator skips them on the next run.
#
# BUG-LOCAL-090 (2026-05-04 EVENING): moved from <repo>/config/news_history.json
# to <output>/otr/state/news_history.json. The repo is code; this is
# per-machine runtime state. Living under output/ aligns with where every
# other persistent OTR state lives (episodes/, obs/, etc.) and keeps the
# repo working tree clean. The legacy path is read-only -- on first run
# after migration the loader picks up legacy entries, the next save writes
# only to the new path, and from then on legacy is dead but harmless.
try:
    from . import _otr_paths as _OTR_PATHS  # type: ignore
    _NEWS_HISTORY_PATH = str(_OTR_PATHS.otr_state_dir() / "news_history.json")
except Exception:  # noqa: BLE001 -- defensive at import time
    _NEWS_HISTORY_PATH = os.path.join(
        os.path.expanduser("~"), ".otr_state", "news_history.json",
    )
_NEWS_HISTORY_LEGACY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "news_history.json",
)
_NEWS_HISTORY_MAX_ENTRIES = 200  # rolling window; oldest entries drop off
# 2026-05-04 (BUG-LOCAL-090): only block URLs used within this many days.
# Older entries are kept on disk for audit but no longer filter the pool,
# so a 5-day-old headline is fair game again. Without this, RSS feeds that
# rotate slowly (43-headline pool with 200-entry history) get filtered to
# zero and the fallback has to restore the unfiltered pool every run.
_NEWS_HISTORY_FILTER_DAYS = 5


def _read_news_history_file(path: str) -> list:
    """Read and JSON-parse the news_history file at ``path``. Returns
    the raw list (or empty list on any error). Used by both the new
    canonical path and the BUG-LOCAL-090 legacy migration fallback."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    except Exception:  # noqa: BLE001 -- best-effort
        return []


def _load_news_history() -> set[str]:
    """Return set of recently-used article URLs (within
    ``_NEWS_HISTORY_FILTER_DAYS`` days).

    Used to filter the candidate pool so back-to-back runs don't pick the
    same RSS feed top story. Entries older than the TTL window are kept
    on disk (for audit) but excluded from the active filter set so a
    headline can recycle into the pool after enough time has passed.

    BUG-LOCAL-090 migration: the canonical path is
    ``<output>/otr/state/news_history.json``. If the new path is missing
    or empty, fall back to the legacy ``<repo>/config/news_history.json``
    so a user's existing history carries forward on the first post-fix
    run. The next save writes only to the new path, after which legacy
    becomes stale-but-harmless.

    Failures return an empty set -- the dedup is best-effort, never
    blocks.
    """
    data = _read_news_history_file(_NEWS_HISTORY_PATH)
    if not data:
        # First-run fallback: pick up legacy entries from the
        # pre-BUG-090 path so the user keeps their dedup window.
        data = _read_news_history_file(_NEWS_HISTORY_LEGACY_PATH)

    cutoff = datetime.now() - timedelta(days=_NEWS_HISTORY_FILTER_DAYS)
    fresh: set[str] = set()
    for entry in data or []:
        url = (entry or {}).get("url")
        if not url:
            continue
        ts = (entry or {}).get("timestamp") or ""
        try:
            entry_dt = datetime.fromisoformat(ts) if ts else None
        except (TypeError, ValueError):
            entry_dt = None
        # Missing or unparseable timestamps -> treat as fresh (safer to
        # filter them once than to surface a same-day repeat).
        if entry_dt is None or entry_dt >= cutoff:
            fresh.add(url)
    return fresh


def _record_news_usage(url: str, headline: str) -> None:
    """Append (url, headline, timestamp) to news_history.json.

    Cap at _NEWS_HISTORY_MAX_ENTRIES rolling. Older entries drop off so the
    file never grows unbounded but recent picks are remembered.

    BUG-LOCAL-090 migration: writes go to the new canonical path
    (``<output>/otr/state/news_history.json``). On first save after
    migration, if the new path is empty/missing but legacy entries
    exist, the legacy list is loaded as the seed so the user's dedup
    window carries forward.
    """
    if not url:
        return
    try:
        # Read existing entries from new path; fall back to legacy if
        # new is empty/missing (one-time migration carry-forward).
        data = _read_news_history_file(_NEWS_HISTORY_PATH)
        if not data:
            data = _read_news_history_file(_NEWS_HISTORY_LEGACY_PATH)
        data.append({
            "url":          str(url),
            "headline":     str(headline)[:240],
            "timestamp":    datetime.now().isoformat(timespec="seconds"),
        })
        if len(data) > _NEWS_HISTORY_MAX_ENTRIES:
            data = data[-_NEWS_HISTORY_MAX_ENTRIES:]
        os.makedirs(os.path.dirname(_NEWS_HISTORY_PATH), exist_ok=True)
        with open(_NEWS_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info("[NewsFetcher] Recorded usage: %s ... (%d total entries)",
                 headline[:60], len(data))
    except Exception as exc:  # noqa: BLE001 -- best-effort
        log.warning("[NewsFetcher] Failed to record news_history: %s", exc)


def _llm_rank_news_candidates(
    pool: list[dict],
    model_id: str = "mistralai/Mistral-Nemo-Instruct-2407",
    top_k: int = 5,
    load_config=None,
    policy=None,
) -> list[dict]:
    """Use the LLM to rank news headlines for genre-fit, return top_k.

    Cheap LLM call: short prompt (43 headlines x ~100 chars = ~5K chars),
    short response (just indices), temp=0.0 for deterministic ranking.
    Returns the top_k highest-ranked candidates ordered by LLM preference.

    On any failure (LLM unavailable, parse error, etc.) falls back to the
    original shuffled-order top_k. The downstream body-fetch loop still
    works -- LLM ranking is an enhancement, not a hard requirement.
    """
    if len(pool) <= top_k:
        return list(pool)
    try:
        # Trim to first 30 candidates to keep prompt bounded; pool is
        # already shuffled so no systematic bias.
        candidates = pool[:30]
        headline_list = "\n".join(
            f"{i + 1}. {(p.get('headline') or '').strip()[:160]}"
            for i, p in enumerate(candidates)
        )
        prompt = (
            f"You are picking news headlines for a radio drama episode. "
            f"From the numbered list below, choose the {top_k} "
            f"headlines with the strongest narrative potential -- prefer "
            f"specific events, mysteries, breakthroughs, or human stakes "
            f"over generic announcements or PR pieces.\n\n"
            f"Return ONLY the chosen indices, comma-separated, no other text. "
            f"Example: 3,7,12,18,22\n\n"
            f"Headlines:\n{headline_list}\n\n"
            f"Top {top_k} indices:"
        )
        # 2026-04-29: 65-second wall-clock budget for the curation LLM call
        # (Jeffrey requested: "give it 65 secs to do the search"). The
        # ranker is a short-output call (~64 tokens of indices) so under
        # normal conditions it returns in 5-15 sec. The 65s budget is a
        # ceiling, not a target -- it bounds the worst-case where prompt
        # processing on a cold cache or a 12B model takes longer than
        # expected. On timeout, _run_with_timeout raises
        # _LLMTimeoutWorkflowPause, caught below by its own dedicated
        # `except _LLMTimeoutWorkflowPause: raise` BEFORE the broad
        # except -- it halts the workflow rather than falling back to
        # shuffle order (PBUG-20260825-04: silently falling back here let
        # the main thread start ANOTHER LLM load while this phase's orphan
        # worker was still alive on GPU). The cache-invalidation we added
        # in BUG-LOCAL-111 (commit 27e54e9) also fires here so the orphan
        # worker doesn't poison the next phase's CUDA state.
        def _do_rank_call():
            # 2026-04-29 fix: transformers rejects temperature=0.0 with
            # "must be strictly positive". For greedy-deterministic
            # ranking, use a tiny positive (0.05) -- effectively argmax
            # but passes the validator. The ranking output is just a
            # comma-separated list of indices so any low-temp value
            # produces stable picks.
            #
            # LLM slot: technical -- structured short-output ranking
            # task. Caller composes the chat message; make_generate_fn
            # bakes top_p=0.92 (the canonical surface does not expose
            # per-call top_p override -- acceptable for a stochastic-
            # argmax ranker at temperature=0.05).
            from . import _otr_model_loader as _OTRML
            cache_entry = _OTRML.request_slot(
                "technical", model_id, policy=policy, load_config=load_config,
            )
            gen_fn = _OTRML.make_generate_fn(cache_entry)
            return gen_fn(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.05,
                max_new_tokens=64,
            )
        response = _run_with_timeout(
            _do_rank_call,
            timeout_sec=65,
            phase_label="NewsCuration",
        )
        # Parse: extract integers, dedupe, cap at top_k
        seen = set()
        indices: list[int] = []
        for tok in re.split(r"[^\d]+", str(response or "")):
            if not tok:
                continue
            try:
                idx = int(tok) - 1  # 1-indexed in prompt -> 0-indexed
            except ValueError:
                continue
            if 0 <= idx < len(candidates) and idx not in seen:
                seen.add(idx)
                indices.append(idx)
            if len(indices) >= top_k:
                break
        if not indices:
            if load_config is not None:
                raise RuntimeError(
                    "[NewsFetcher] local GGUF news ranking returned no "
                    f"parseable indices (response={str(response)[:120]!r}); "
                    "refusing to silently fall back to shuffle order "
                    "(operator directive: no local-LM fallbacks)."
                )
            log.warning("[NewsFetcher] LLM ranking returned no parseable indices "
                        "(response=%r) - falling back to shuffle order",
                        str(response)[:120])
            return list(pool[:top_k])
        ranked = [candidates[i] for i in indices]
        log.info("[NewsFetcher] LLM-ranked top %d candidates:", len(ranked))
        for r in ranked:
            log.info("[NewsFetcher]   - %s", (r.get("headline") or "")[:80])
        return ranked
    except _LLMTimeoutWorkflowPause:
        # 2026-08-25: this subtype's own docstring says "ComfyUI catches
        # this at the node boundary and halts the queue cleanly" -- but the
        # broad `except Exception` below used to catch it here FIRST and
        # (when load_config is None) silently fall back to shuffle order,
        # letting the main thread immediately start ANOTHER LLM load while
        # this phase's orphan worker is still alive on GPU (generation is
        # not cancellable mid-token -- _run_with_timeout abandons it, it
        # does not stop it). That is exactly the window PBUG-20260825-04
        # was found in. Re-raise unconditionally so the pause always
        # reaches the node boundary, regardless of load_config.
        raise
    except Exception as exc:  # noqa: BLE001 -- enhancement for non-GGUF lanes only
        if load_config is not None:
            log.error(
                "[NewsFetcher] local GGUF news ranking failed (%s); failing "
                "loud (operator directive: no local-LM fallbacks)", exc,
            )
            raise
        log.warning("[NewsFetcher] LLM ranking failed (%s) - falling back to "
                    "shuffle order", exc)
        return list(pool[:top_k])


def _llm_rerank_with_bodies(
    candidates_with_body: list[dict],
    model_id: str = "mistralai/Mistral-Nemo-Instruct-2407",
    load_config=None,
    policy=None,
) -> list[dict]:
    """Body-aware second-pass news rank ("Option B / 65s budget").

    Phase-1 ranking (`_llm_rank_news_candidates`) operates on headlines
    only -- ~160 chars each. This pass feeds the LLM the first ~800
    chars of each candidate's actual article body so the pick is based
    on narrative bones, not the catchy title. Returns the input list
    re-ordered (best first). On a non-timeout failure (parse error, LLM
    unavailable) returns the input list unchanged so the caller's normal
    fallback walk still works -- body re-rank is an enhancement, never a
    blocker. On a WALL-CLOCK TIMEOUT specifically, `_LLMTimeoutWorkflowPause`
    is re-raised rather than swallowed into that fallback (PBUG-20260825-04:
    silently falling back here let the main thread start ANOTHER LLM load
    while this phase's orphan worker was still alive on GPU).

    Designed to fit inside the 65-second news-curation wall-clock
    budget alongside Phase 1: Phase 1 ~10-15s + parallel body-fetch
    ~5-10s + this re-rank ~25-40s ~= 50s total.
    """
    if len(candidates_with_body) <= 1:
        return list(candidates_with_body)
    try:
        blocks = []
        for i, c in enumerate(candidates_with_body):
            headline = (c.get("headline") or "").strip()[:160]
            body = (c.get("full_text") or c.get("summary") or "").strip()
            body_preview = _body_rerank_preview(body).replace("\n", " ")
            blocks.append(
                f"{i + 1}. HEADLINE: {headline}\n   ARTICLE: {body_preview}"
            )
        text = "\n\n".join(blocks)
        prompt = (
            f"You are picking ONE news story to seed a radio drama. "
            f"You have already shortlisted {len(candidates_with_body)} "
            f"candidates by headline. Now you can read each article body. "
            f"Choose the SINGLE story with the strongest narrative bones "
            f"for an audio drama: specific human stakes, "
            f"mystery, scientific breakthrough, or vivid scene potential. "
            f"Avoid press releases, funding announcements, and generic "
            f"'researchers find X' filler.\n\n"
            f"Return ONLY the chosen index, no other text. Example: 3\n\n"
            f"Candidates:\n{text}\n\n"
            f"Best index:"
        )

        def _do_rerank_call():
            # Same temperature=0.05 trick as headline rank: transformers
            # rejects 0.0; tiny positive value is effectively argmax.
            #
            # LLM slot: technical -- single-index body rerank.
            from . import _otr_model_loader as _OTRML
            cache_entry = _OTRML.request_slot(
                "technical", model_id, policy=policy, load_config=load_config,
            )
            gen_fn = _OTRML.make_generate_fn(cache_entry)
            return gen_fn(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.05,
                max_new_tokens=8,
            )

        response = _run_with_timeout(
            _do_rerank_call,
            timeout_sec=40,
            phase_label="NewsCurationDeep",
        )
        m = re.search(r"\d+", str(response or ""))
        if not m:
            if load_config is not None:
                raise RuntimeError(
                    "[NewsFetcher] local GGUF body re-rank returned no "
                    f"parseable index (response={str(response)[:120]!r}); "
                    "refusing to silently keep headline order "
                    "(operator directive: no local-LM fallbacks)."
                )
            log.warning(
                "[NewsFetcher] body re-rank returned no parseable index "
                "(response=%r) - keeping headline order",
                str(response)[:120],
            )
            return list(candidates_with_body)
        idx = int(m.group(0)) - 1
        if not (0 <= idx < len(candidates_with_body)):
            if load_config is not None:
                raise RuntimeError(
                    f"[NewsFetcher] local GGUF body re-rank index {idx + 1} out "
                    f"of range (have {len(candidates_with_body)}); refusing to "
                    "silently keep headline order (operator directive: no "
                    "local-LM fallbacks)."
                )
            log.warning(
                "[NewsFetcher] body re-rank index %d out of range "
                "(have %d) - keeping headline order",
                idx + 1, len(candidates_with_body),
            )
            return list(candidates_with_body)
        chosen = candidates_with_body[idx]
        rest = [c for i, c in enumerate(candidates_with_body) if i != idx]
        log.info(
            "[NewsFetcher] body re-rank chose #%d: %s",
            idx + 1, (chosen.get("headline") or "")[:80],
        )
        return [chosen] + rest
    except _LLMTimeoutWorkflowPause:
        # 2026-08-25: this subtype's own docstring says "ComfyUI catches
        # this at the node boundary and halts the queue cleanly" -- but the
        # broad `except Exception` below used to catch it here FIRST and
        # (when load_config is None) silently fall back to shuffle order,
        # letting the main thread immediately start ANOTHER LLM load while
        # this phase's orphan worker is still alive on GPU (generation is
        # not cancellable mid-token -- _run_with_timeout abandons it, it
        # does not stop it). That is exactly the window PBUG-20260825-04
        # was found in. Re-raise unconditionally so the pause always
        # reaches the node boundary, regardless of load_config.
        raise
    except Exception as exc:  # noqa: BLE001 -- enhancement for non-GGUF lanes only
        if load_config is not None:
            log.error(
                "[NewsFetcher] local GGUF body re-rank failed (%s); failing "
                "loud (operator directive: no local-LM fallbacks)", exc,
            )
            raise
        log.warning(
            "[NewsFetcher] body re-rank failed (%s) - keeping headline order",
            exc,
        )
        return list(candidates_with_body)


_RSS_FRAGMENT_BLOCK_OR_BREAK_RE = re.compile(
    r"</?(?:address|article|aside|blockquote|br|dd|details|dialog|div|dl|dt|"
    r"fieldset|figcaption|figure|footer|form|h[1-6]|header|hgroup|hr|li|main|"
    r"""nav|ol|p|pre|section|table|tbody|td|tfoot|th|thead|tr|ul)(?=[\s/>])"""
    r"""(?:[^"'<>]|"[^"]*"|'[^']*')*>""",
    re.IGNORECASE,
)
_RSS_FRAGMENT_ANY_TAG_RE = re.compile(
    r"""<(?:[^"'<>]|"[^"]*"|'[^']*')*>"""
)


def _extract_rss_fragment_text(fragment: str) -> str:
    """Extract text from an inline RSS HTML fragment without fusing blocks.

    Only explicit block/break tags create a separator. Remaining inline tags
    are removed without one so ``H<sub>2</sub>O`` and
    ``anti-<em>microbial</em>`` retain their literal spelling. HTML entity
    spellings are deliberately left untouched for the downstream coordinate
    normalizer.
    """
    text = fragment or ""
    text = _RSS_FRAGMENT_BLOCK_OR_BREAK_RE.sub(" ", text)
    text = _RSS_FRAGMENT_ANY_TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _select_rss_content(entry) -> tuple[str, int | None, int]:
    """Choose the longest usable RSS ``content`` alternative.

    Raw list positions are preserved for the source receipt. Malformed rows
    still count, but cannot displace a usable alternative. A non-list top-level
    value is not an alternatives collection and is treated as absent.
    """
    raw = entry.get("content", []) if hasattr(entry, "get") else []
    if not isinstance(raw, list):
        return "", None, 0
    best_text = ""
    best_index: int | None = None
    for index, row in enumerate(raw):
        if not hasattr(row, "get"):
            continue
        value = row.get("value")
        if not isinstance(value, str):
            continue
        try:
            extracted = _extract_rss_fragment_text(value)
        except Exception as exc:  # one malformed alternative cannot hide later rows
            log.debug(
                "[NewsFetcher] RSS content alternative %d was unusable: %s",
                index, exc,
            )
            continue
        if extracted and len(extracted) > len(best_text):
            best_text = extracted
            best_index = index
    return best_text, best_index, len(raw)


def _select_news_body(candidate: dict, fetched_body: str) -> tuple[str, str]:
    """Choose the most complete clean body with stable route tie-breaking."""
    rss = str(candidate.get("rss_full") or "")
    article = str(fetched_body or "")
    summary = str(candidate.get("summary") or "")
    # max() keeps the first member on ties: RSS, then linked article, then
    # summary. The route for a selected summary records whether a URL existed.
    choices = (
        (rss, "rss_full"),
        (article, "url_scrape"),
        (
            summary,
            "summary_fallback" if candidate.get("link") else "summary_only",
        ),
    )
    return max(choices, key=lambda item: len(item[0]))


def _body_rerank_preview(text: str, limit: int = 800) -> str:
    """Return a bounded head/middle/tail view without modifying source text."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("body preview limit must be a positive integer")
    body = str(text or "")
    if len(body) <= limit:
        return body
    separator = " ... "
    if limit <= 2 * len(separator) + 2:
        return body[:limit]
    available = limit - 2 * len(separator)
    head_chars = available // 3
    middle_chars = available // 3
    tail_chars = available - head_chars - middle_chars
    middle_start = max(0, (len(body) - middle_chars) // 2)
    return separator.join((
        body[:head_chars],
        body[middle_start:middle_start + middle_chars],
        body[-tail_chars:],
    ))


def _fetch_science_news(max_feeds=10,  # kept: max_feeds is API stability arg; current body iterates the full feed list. Wiring is a future feature, not a cleanbreak target
                         model_id=None,
                         *, load_config=None, policy=None):
    # `optimization_profile` was removed from this signature 2026-08-28. It
    # was threaded three levels deep -- here, then into the two LLM news-rank
    # helpers -- and NEITHER receiver ever read it (AST-verified: zero Load
    # references, zero onward forwards). The identically named writer widget
    # that really does select a quantization profile is a different value on a
    # different path and is untouched.
    """Fetch science stories from multiple RSS feeds in parallel.

    2026-04-29: now also (a) filters out previously-used URLs via
    config/news_history.json, (b) calls the LLM to rank remaining
    candidates by narrative fit, and (c) records the chosen article to
    history after selection.

    Style-engine consolidation (2026-07-05): this stage runs BEFORE the
    single style engine (which needs script_brief, not yet produced) --
    ranking is style-agnostic by design; the `style` parameter and its
    hardcoded "mission_control_procedural" fallback are removed entirely.

    Original fast-path behaviour (shuffle + first-with-enough-body) is
    preserved when model_id is None or LLM ranking fails -- the dedup
    still works regardless. Shipped behind model_id so legacy callers
    without it fall back to the simple path.

    Uses ThreadPoolExecutor to hit all feeds simultaneously, dramatically
    reducing the wait time when feeds are slow or unresponsive. Each feed
    has its own timeout.
    """
    try:
        import feedparser
    except ImportError:
        msg = (
            "-==================================================================-\n"
            "-  CRITICAL: feedparser is missing.                              -\n"
            "-  Run `pip install feedparser` to enable live science news.     -\n"
            "-  The OTR ScriptWriter REQUIRES real headlines - no fallback.   -\n"
            "-==================================================================-"
        )
        log.error(msg)
        raise ImportError(msg)

    from ._otr_feed_fetch import FeedFetchRefused, fetch_feed

    def _fetch_single_feed(feed_url):
        data = []
        try:
            # Wave 5: the bytes arrive through the bounded seam and feedparser
            # is handed a STRING. It never touches the network here.
            #
            # What this replaced: `feedparser.parse(feed_url)` -- which does its
            # own unbounded urllib fetch -- wrapped in a PROCESS-GLOBAL
            # socket.setdefaulttimeout(7). That global was set and restored by
            # every worker in a ~30-wide thread pool concurrently, so the
            # timeout any given feed actually ran under was whatever another
            # thread had most recently installed. It was never a per-feed
            # timeout; it only looked like one.
            feed = feedparser.parse(fetch_feed(feed_url).text)

            for entry in feed.entries[:6]:
                title = entry.get("title", "").strip()
                if not title:
                    continue


                rss_full, rss_content_index, rss_content_count = (
                    _select_rss_content(entry)
                )
                summary = entry.get("summary", "").strip()
                summary = re.sub(r'<[^>]+>', '', summary).strip()
                data.append({
                    "headline": title,
                    "summary": summary,
                    "rss_full": rss_full,
                    "source": feed.feed.get("title", feed_url.split("/")[2]),
                    "date": entry.get("published", str(datetime.now().date())),
                    "link": entry.get("link", ""),
                    "_rss_content_index": rss_content_index,
                    "_rss_content_count": rss_content_count,
                })
            return data
        except FeedFetchRefused:
            # A shipped or client-authored feed row that is http://, points at
            # a private address, or serves something that is not a feed is a
            # CONFIGURATION defect, not a flaky feed. Let it out of the pool
            # worker so `future.result()` re-raises it and the run fails loud.
            raise
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
        log.error("[NewsFetcher] ALL feeds failed - check network connectivity")
        raise RuntimeError(
            "No science headlines could be fetched. Check your internet connection. "
            "The OTR ScriptWriter requires live RSS feeds to generate scripts."
        )

    # 2026-04-29: history-aware deduplication + LLM-curated ranking.
    # 1) drop any candidate whose URL is in news_history.json (back-to-
    #    back runs no longer pick the same Orion Flywheel article).
    # 2) shuffle the remaining pool to break feed-order bias.
    # 3) optionally call the LLM to rank top 5 by narrative fit for the
    #    requested style. This step adds ~10-30s of LLM time but
    #    the LLM is the same one NewsSummary will load anyway, so the
    #    NewsSummary phase that follows hits a cache HIT instead of
    #    paying the load cost twice.
    # 2026-04-29 BUG-LOCAL-112: history-wipe restoration. The previous
    # implementation had a comment admitting the reset was a "no-op" --
    # if every URL in the fresh fetch was already in news_history.json,
    # the filter emptied `pool` and the fall-through logged a warning
    # but never restored the pool. Result: body-fetch saw 0 candidates,
    # writer fell back to no-news, news-seeded plot was lost.
    # Real fix: stash the unfiltered pool before filtering, restore it
    # if the filter wipes everything. The history dedup still wins on
    # the typical day; the reset only fires when every fresh headline
    # is in history (which means the rolling cap is too small for the
    # user's run cadence and we'd rather repeat than starve).
    unfiltered_pool = list(pool)
    used_urls = _load_news_history()
    if used_urls:
        before = len(pool)
        pool = [
            p for p in pool
            if not (p.get("link") and p["link"] in used_urls)
        ]
        dropped = before - len(pool)
        if dropped:
            log.info(
                "[NewsFetcher] Filtered %d previously-used candidate(s) via "
                "news_history (%d remaining of %d)",
                dropped, len(pool), before,
            )
    if not pool:
        # All N candidates were already used. Restore the unfiltered
        # pool -- better to pick a recent repeat than to starve the
        # writer with zero news context.
        log.warning(
            "[NewsFetcher] All %d candidate(s) filtered out by history -- "
            "restoring unfiltered pool so the writer still gets a real "
            "article (history dedup will catch up as new headlines come "
            "in; consider raising the rolling cap if this happens often)",
            len(unfiltered_pool),
        )
        pool = list(unfiltered_pool)
        used_urls = set()

    random.shuffle(pool)

    # LLM rank: only if model_id provided and pool is non-trivial. This
    # spends one short LLM call up-front to pick narrative-fit
    # candidates. The LLM stays warm for NewsSummary which fires next.
    if model_id and len(pool) > 5:
        ranked = _llm_rank_news_candidates(
            pool,
            model_id=model_id,
            top_k=5,
            load_config=load_config,
            policy=policy,
        )
        # Put LLM-ranked picks at the front of the pool; everything
        # else stays as a fallback in case all 5 ranked picks have
        # thin bodies.
        ranked_links = {r.get("link") for r in ranked if r.get("link")}
        non_ranked = [p for p in pool if p.get("link") not in ranked_links]
        pool = ranked + non_ranked

    # 2026-04-29 Option B: parallel body-fetch + LLM body-aware re-rank.
    # Old behavior: serial walk-the-list, break at first candidate above
    # the content floor. Time spent: only as long as candidate-1 fetch
    # took. New behavior: body-fetch ALL top-N in parallel (network-
    # bound, fast), then ask the LLM to re-pick using actual article
    # text instead of just the headline. Total budget ~50s, comfortably
    # under the 65s news-curation ceiling. The LLM stays warm for
    # NewsSummary which fires next, so the re-rank's GPU time is not
    # wasted.
    #
    # Thin content (<400 chars) gives the writer too little to
    # extrapolate from -- the story ends up generic rather than
    # grounded in real science. Anything below the floor is excluded
    # from the re-rank pool.
    CONTENT_FLOOR = 400
    MAX_ATTEMPTS = 5

    def _resolve_body(candidate: dict) -> dict:
        """Body resolver for one candidate. Pure; thread-safe."""
        out = dict(candidate)
        fetched_body = (
            _fetch_full_article(out["link"], timeout=5)
            if out.get("link") else ""
        )
        out["full_text"], out["_body_source"] = _select_news_body(
            out, fetched_body,
        )
        log.info(
            "[NewsFetcher] [%s] selected %s body: %d chars",
            (out.get("headline") or "")[:50],
            out["_body_source"],
            len(out["full_text"]),
        )
        return out

    attempts = pool[:MAX_ATTEMPTS]
    log.info(
        "[NewsFetcher] Body-fetching top %d candidate(s) in parallel...",
        len(attempts),
    )
    body_start = time.time()
    # Cap workers at len(attempts) -- ThreadPoolExecutor errors on max_workers=0.
    with ThreadPoolExecutor(max_workers=max(1, len(attempts))) as ex:
        fetched = list(ex.map(_resolve_body, attempts))
    log.info(
        "[NewsFetcher] Body-fetch complete in %.2fs",
        time.time() - body_start,
    )

    rich = [
        candidate for candidate in fetched
        if len(candidate.get("full_text", "")) >= CONTENT_FLOOR
    ]

    if rich:
        log.info(
            "[NewsFetcher] %d/%d candidate(s) passed content floor "
            "(>=%d chars%s) -> body re-rank",
            len(rich), len(fetched), CONTENT_FLOOR, "",
        )
        if model_id and len(rich) > 1:
            rich = _llm_rerank_with_bodies(
                rich,
                model_id=model_id,
                load_config=load_config,
                policy=policy,
            )
        chosen = rich[0]
    else:
        # All candidates thin - take the richest available so the run
        # continues. Better a thin real story than a hard fail.
        chosen = max(
            fetched,
            key=lambda x: len(x.get("full_text", x.get("summary", ""))),
        )
        chosen.setdefault("full_text", chosen.get("summary", ""))
        log.warning(
            "[NewsFetcher] All %d candidate(s) were thin - using richest "
            "available (%d chars): %s",
            len(fetched), len(chosen["full_text"]),
            chosen.get("headline", "")[:60],
        )

    # Record selection so back-to-back runs don't repeat. Best-effort;
    # logged warning on failure, never blocks generation.
    try:
        _record_news_usage(
            url=chosen.get("link", ""),
            headline=chosen.get("headline", ""),
        )
    except Exception as _hist_exc:  # noqa: BLE001
        log.warning("[NewsFetcher] history record failed (non-fatal): %s",
                    _hist_exc)

    return [chosen]


# -----------------------------------------------------------------------------
# LLM INFERENCE WRAPPER
# -----------------------------------------------------------------------------

# ── Token Budget Ratios ──────────────────────────────────────────────────────
# target_words * ratio = max_new_tokens. Different content types tokenize at
# different rates. Radio drama is dialogue-dominant (~60% character lines),
# so structural overhead (VOICE tags, SFX, ENV, scene headers) is lower than
# a screenplay or narration-heavy format.
#
# Breakdown for dialogue-dominant OTR scripts:
#   tokenizer overhead:  ~1.3 tokens per English word
#   script markup:       ~1.2x (VOICE/SFX/ENV tags, scene headers, beats)
#   combined:            1.3 * 1.2 = 1.56 → round to 1.6
#
# THE LAST TWO RATIOS WENT THE SAME WAY (2026-08-28). Three siblings were
# removed on 2026-08-22 for being written and never read; re-checked at HEAD,
# `_TOKEN_RATIO_DIALOGUE` and `_TOKEN_RATIO_ACT_CHUNK` had no reader either --
# `tests/test_core.py` re-declares the literals rather than importing them, so
# the constants proved nothing about the code. The arithmetic above is kept as
# the RECORD of where 1.6 and 2.0 came from, should a caller ever want them.


# The dialogue-name normalizer's REGEXES were removed 2026-08-28 with the
# same reasoning as the class below: `_RE_LLM_DIALOGUE_NAME`,
# `_BRACKET_STRUCTURAL_TOKENS` and `_RE_LLM_BRACKET_NAME_DIALOGUE` were
# definition-only after their sole consumer went, while the block around them
# still described live normalization in the present tense. The BUG-023 and
# BUG-LOCAL-063 histories they documented are preserved in the bug log; a
# comment that claims a rewrite nothing performs is worse than no comment.


# `GemmaHeartbeatStreamer` WAS REMOVED 2026-08-28: it was never once
# CONSTRUCTED -- zero `GemmaHeartbeatStreamer(` call sites in the repo.
# `_normalize_dialogue_names` went with it, its only caller being the
# dead class's own `_process_line`.
#
# BOTH ITS JOBS HAVE NAMED LIVE REPLACEMENTS, which is why this is a
# deletion rather than a gap: deadline enforcement is
# `_DeadlineStoppingCriteria` in `_otr_model_loader.py`, wired into the
# real generate() calls; the dashboard heartbeat is
# `_otr_writer_heartbeat.WriterHeartbeatStreamer`, wired at two live
# call sites (commit 2894b852, 'you can watch the model write again').
#
# ONE CAPABILITY HAS NO REPLACEMENT and is named here rather than lost
# quietly: its `live_ledger` / `_emit_partial_ledger` partial-ledger
# streaming. Since the class was never constructed, that capability was
# already inert -- deleting it loses nothing that was not already lost.


# ── Scene inventory (diagnostic instrumentation) ────────────────
# Extracts the list of scene tokens from a script so the orchestrator
# can log scene counts at every pipeline checkpoint. A scene leak in
# any pass (WORD_EXTEND, ANNOUNCER, FORMAT_NORM, GRAMMARIAN, PARSE)
# shows up as a count drop in the soak log, localizing the bug.
#
# BUG-LOCAL-026 fix: restrict the scene-number capture to digits only.
# The previous pattern (\S+?) matched literals like "FINAL" that the
# creative LLM emits as a closing-scene marker ('=== SCENE FINAL ==='),
# inflating scene counts and fooling the FORMAT_NORM skip heuristic.
# Any '=== SCENE FINAL ===' is promoted to 'END' (terminator) below.


# ── Name cleanup (fuzzy match against canonical cast) ────────────
# BUG-020 fix: Under maximum chaos, LLMs hallucinate variant spellings
# (NEMEO_SIRIKIT instead of NEMO SIRIKIT). This pure-Python pass reads
# the canonical cast from config/episode_cast.txt and fuzzy-matches
# every CHARACTER: line against the roster. No LLM call, no VRAM cost.

# Register the LLM unloader with the VRAM Power Wash system so that
# force_vram_offload() at node entry points also evicts the LLM.
# `register_vram_cleanup`'s caller (`force_vram_offload` in
# `_vram_log.py`) already wraps each callback invocation in
# `try/except: pass`, so the callback contract is "no-arg callable"
# only -- no wrapper required.
from ._vram_log import register_vram_cleanup
from . import _otr_model_loader as _otr_loader_mod

register_vram_cleanup(_otr_loader_mod.unload_llm)




# -----------------------------------------------------------------------------
# (lean-mean 2026-08-22) The "v1.4 Theme B sentence-boundary truncation helpers"
# block lived here: _SENTENCE_END_CHARS, _BOUNDARY_SCAN_WINDOW, and the SECOND
# _truncate_at_sentence_boundary / _tail_at_sentence_boundary bodies.
#
# All of it is gone. The pair here won the module-level rebinding against an
# earlier same-named pair ~2200 lines up, and then had zero callers anywhere in
# the repo -- so BOTH copies were unreachable, by two different mechanisms, and
# the two constants existed only to serve them. Removing the functions orphaned
# the constants in the same pass, which is why a dead-symbol sweep is iterative.
#
# The live rule for where a shortened line may end is
# nodes/_otr_shared/text_tails.py, and the shadowing pattern itself is now
# refused by tests/test_no_duplicate_top_level_defs.py.
# -----------------------------------------------------------------------------


# `_CURRENT_LLM_MODEL` was removed 2026-08-28. Its comment described a
# model-memory-inheritance mechanism -- the next phase reusing whatever the
# Script Writer loaded -- and no such code existed: the name was assigned once
# and read nowhere. Slot residency is owned by `_otr_model_loader.request_slot`
# and the writer's `_SlotScheduler`, which is where that behaviour really is.


# ============================================================================
# SHARED INFERENCE ENGINE
# Both nodes call this loader. It caches the model in VRAM and tracks the peak
# memory watermark for diagnostics.
# ============================================================================


# (lean-mean 2026-08-22: the SECOND, shadowing _tail_at_sentence_boundary body
# was here -- same story as its sibling above: it won the rebinding and had no
# caller. Deleted.)


# (rip-sfx 2026-08-06: the second, shadowing _inject_scene_transitions body and
# its _SCENE_MARKER_RE / _HANDOFF_CUE_RE globals were deleted here -- the name
# won the binding but had zero callers, which is not survival.)


# -----------------------------------------------------------------------------
# NODE 1: SCRIPT WRITER
# -----------------------------------------------------------------------------

# THE CANON LOADER WAS REMOVED 2026-08-28 (dead-code audit).
#
# `_CANON_PATH` and `_load_canon_for_writer()` had NO caller. The
# function's own docstring recorded that its consumer had already been
# deleted and marked itself orphan-pending-audit; this is that audit. The
# canon file itself (docs/OTR-CANON.md) is untouched -- only the unused
# loader is gone, so re-wiring it later means writing a caller, not
# recovering a file.


# ============================================================================
# 2026-05-15 Sprint C C2b: SCAFFOLDING_PREAMBLE + SCRIPT_SYSTEM_PROMPT
# constants deleted. Both were orphaned during the eec4718 LPL extraction
# sprint and had zero live consumers across nodes/, visual/, scripts/, and
# tests/ (verified via 8-search audit at C2b: getattr / glob-import /
# substring-content / alt-name / git log -S / writer-callsite / __all__ /
# tests). The current ledger pipeline composes prompts from per-phase
# system constants in _otr_outline, _otr_line_composer,
# _otr_ledger_reviewer, and _otr_period_prompts.
#
# No back-compat shim. Per the standing directive (no legacy back-compat),
# orphan constants from a deleted pipeline are deleted, not preserved.
# Future contributors who try `from .story_orchestrator import
# SCRIPT_SYSTEM_PROMPT` will get AttributeError -- intentional, so dead
# wirings fail loud.
#
# Broader orphan-constant sweep of this file (3000+ lines, gutted across
# multiple sprints) deferred to Sprint G per the no-scope-creep rule.
# Each candidate gets its own 8-search audit before deletion.
# ============================================================================


# ---------------------------------------------------------------------------
# 2026-05-10: LegacyLLMScriptWriter shim removed alongside
# nodes/_otr_legacy_writer.py. The v2.0 canonical writer is
# OTR_LedgerScriptWriter (LPL). Any caller that still does
# `from .story_orchestrator import LLMScriptWriter` will now raise
# AttributeError -- intentional, so stale wirings fail loudly instead of
# silently using a dead code path.
# ---------------------------------------------------------------------------
