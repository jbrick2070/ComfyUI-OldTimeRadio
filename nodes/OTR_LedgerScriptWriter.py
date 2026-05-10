"""nodes/OTR_LedgerScriptWriter.py

Integration ComfyUI node for the v2.0 LPL (Local Per-Line Ledger)
script writer. Replaces the legacy LLMScriptWriter for new
workflows; legacy stays untouched until Phase 3 wiring lands.

Pipeline inside the FUNCTION method:
    1. load_llm via _otr_model_loader        -> cache_entry
    2. build truncating generate_fn          -> generate_fn
    3. _otr_outline.generate_outline         -> validated Outline
    4. word-budget integration check (warn)
    5. production_ledger.new_ledger          -> pending_<ts> workspace
    6. _otr_canon.write_episode_canon        -> episode root
    7. per-beat loop:
         - voiced roles  -> _otr_line_composer.compose_line
         - non-voiced    -> use beat.sfx_cue / beat.intent verbatim
    8. set_cast / set_lines + speaker_role post-patch
    9. assemble script_text with [VOICE:] / [SFX:] tokens
   10. assemble news_json (1-element JSON array)
   11. polish_pass no-op (logs and skips)
   12. led.save()
   13. return 4-tuple

All heavy work is lazy. No model loads at import. No GPU at import.

Output contract (STRICT, mirrors legacy LLMScriptWriter at
story_orchestrator.py:4310-4311):
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("script_text", "script_json", "news_used",
                    "estimated_minutes")

Notable v2.0 contract divergences from legacy LLMScriptWriter:
    - script_json (slot 1) is a JSON dump of the production ledger
      ``led.data`` dict, NOT the legacy script_lines token tree.
      The ledger IS the canonical truth in v2.0; downstream
      consumers in Phase 3 read ledger fields directly.
    - news_used (slot 2) is built from the user-supplied news_seed
      packed into a 1-element JSON array matching the legacy
      article-dict shape (story_orchestrator.py:5141-5283).

Hard rules followed:
    - Writes ONLY under otr/episodes/<episode_id>/. Three-bucket
      file-layout contract.
    - No edits to any shipped file (_otr_outline, _otr_canon,
      _otr_line_composer, _otr_model_loader, production_ledger,
      _otr_ledger, _otr_paths, story_orchestrator).
    - No model loads / GPU at import.
    - UTF-8 no BOM. Safe-for-work content. No "dummy" naming.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger("OTR")


__all__ = ["OTR_LedgerScriptWriter"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOICED_ROLES = {"character", "announcer"}
"""Speaker roles that produce spoken dialogue. These trigger an LLM
compose_line call. Other roles (music_*, sfx) skip the LLM."""

NON_VOICED_ROLES = {"music_open", "music_close", "music_inter", "sfx"}
"""Speaker roles that render as [SFX: ...] tokens with no LLM call."""

DEFAULT_TRAITS = "neutral"
"""Fallback traits string when a beat has no mood. Mirrors the
'traits = beat.mood or "neutral"' rule from the kickoff prompt."""

DEFAULT_MODEL_ID = "mistralai/Mistral-Nemo-Instruct-2407"
"""Default story-writing LLM. Validated production path (cleared
BUG-061/062/063 format hardening)."""

LAST_LINES_WINDOW = 3
"""Rolling context window size for compose_line. Each character /
announcer beat appends to the window; non-voiced beats do not."""

WORD_BUDGET_RATIO_LO = 0.7
WORD_BUDGET_RATIO_HI = 1.3
"""Acceptable band for sum(beat.target_words) / target_words. Outside
this band logs WARNING but does not fail the run."""

WORDS_PER_SECOND_BUDGET = 2.5
"""Word budget heuristic for outline planning (radio ~150 wpm)."""

WORDS_PER_MINUTE_ESTIMATE = 140
"""Word-per-minute estimate for est_minutes (slot 3). Mirrors legacy
at story_orchestrator.py:6584."""


# ---------------------------------------------------------------------------
# Truncating generate_fn wrapper (mirrors story_orchestrator.py:3506-3527)
# ---------------------------------------------------------------------------


def _build_truncating_generate_fn(cache_entry: dict):
    """Return a generate_fn (messages, *, temperature, max_new_tokens)
    that left-truncates oversized prompts before model.generate.

    Cap math: max_input_tokens = max(64, context_cap - max_new_tokens).
    Truncation is left-side (drops oldest tokens, preserves most
    recent context -- matches legacy behavior at
    story_orchestrator.py:3506-3527).

    Logs at WARNING level when truncation fires; line format mirrors
    legacy 'PROMPT_GUARD: Truncated X -> Y tokens (context_cap=Z,
    max_new_tokens=N)'.

    This wrapper rebuilds the body of _otr_model_loader.make_generate_fn
    rather than wrapping it -- the truncation must happen between
    tokenization and model.generate, which is mid-body. We keep the
    loader file untouched (HARD RULE 2) and reimplement the small
    chat-template adapter here.
    """
    model = cache_entry["model"]
    tokenizer = cache_entry["tokenizer"]
    context_cap = int(cache_entry.get("context_cap") or 8192)

    def generate_fn(messages, *, temperature, max_new_tokens):
        import torch  # local import; never load torch at module import
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        max_input_tokens = max(64, context_cap - int(max_new_tokens))
        input_len = inputs["input_ids"].shape[-1]
        if input_len > max_input_tokens:
            trunc = input_len - max_input_tokens
            inputs["input_ids"] = inputs["input_ids"][:, trunc:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, trunc:]
            log.warning(
                "[OTR_LedgerScriptWriter] PROMPT_GUARD: Truncated "
                "%d -> %d tokens (context_cap=%d, max_new_tokens=%d)",
                input_len, max_input_tokens, context_cap, max_new_tokens,
            )

        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=float(temperature),
                top_p=0.92,
                max_new_tokens=int(max_new_tokens),
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(
            out[0][prompt_len:], skip_special_tokens=True,
        )

    return generate_fn


# ---------------------------------------------------------------------------
# Pure helpers (testable without model load)
# ---------------------------------------------------------------------------


def _normalize_inputs(
    news_seed: str,
    style_hint: str,
    target_seconds,
    cast_size,
) -> tuple:
    """Validate + normalize the four required user inputs.

    Returns ``(news_seed, style_hint, target_seconds, cast_size,
    target_words)``. Raises ``RuntimeError`` on empty news_seed.
    Out-of-range numerics get clamped.

    Pulled out as a free helper so the self-test can exercise it
    without loading a model.
    """
    news_seed = (news_seed or "").strip()
    if not news_seed:
        raise RuntimeError(
            "[OTR_LedgerScriptWriter] news_seed is empty. v2.0 has "
            "no RSS fallback; supply a 1-3 sentence factual seed."
        )
    style_hint = (style_hint or "").strip() or "psychological slow-burn"
    target_seconds = max(10, min(600, int(target_seconds)))
    cast_size = max(1, min(6, int(cast_size)))
    target_words = max(5, int(target_seconds * WORDS_PER_SECOND_BUDGET))
    return news_seed, style_hint, target_seconds, cast_size, target_words


def _build_cast_rows(cast_names) -> tuple:
    """Build legacy-schema cast rows + a name->char_id index from
    a list of ALL-CAPS character names.

    Returns ``(cast_rows, char_id_by_name)``. char_id is ``c01``,
    ``c02``, ... in the order the names appear in ``cast_names``.
    """
    cast_rows = []
    char_id_by_name = {}
    for i, name in enumerate(cast_names):
        cid = f"c{i + 1:02d}"
        cast_rows.append({
            "char_id":      cid,
            "name":         name,
            "description":  None,
            "gender":       None,
            "voice_preset": None,
            "line_count":   0,
            "word_count":   0,
        })
        char_id_by_name[name] = cid
    return cast_rows, char_id_by_name


def _build_news_payload(outline, news_seed: str) -> str:
    """Build the slot-2 news_used JSON string.

    1-element JSON array matching legacy article shape
    (story_orchestrator.py:5141-5283 + RECON 4(b)).
    """
    news = [{
        "headline":  outline.title,
        "summary":   outline.premise[:500],
        "full_text": news_seed,
        "source":    "User Seed",
        "date":      datetime.now().date().isoformat(),
        "link":      "",
    }]
    return json.dumps(news, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------


class OTR_LedgerScriptWriter:
    """v2.0 LPL script writer. ComfyUI integration node.

    Wires the four shipped LPL modules (_otr_outline, _otr_canon,
    _otr_line_composer, _otr_model_loader) plus production_ledger
    into the legacy 4-slot output contract.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "news_seed": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Real science story or factual seed (1-3 "
                        "sentences). The outline will extrapolate "
                        "dramatically from this. Leave blank to fail "
                        "loudly -- there is no RSS fallback in v2.0."
                    ),
                }),
                "style_hint": ("STRING", {
                    "multiline": False,
                    "default": "psychological slow-burn",
                    "tooltip": (
                        "Free-form style descriptor. The local LLM "
                        "uses its own trained dialogue register; this "
                        "is just a tonal nudge. Examples: 'pulp "
                        "adventure', 'noir thriller', 'hard sci-fi "
                        "procedural', 'rust-belt cyber-noir'."
                    ),
                }),
                "target_seconds": ("INT", {
                    "default": 90, "min": 10, "max": 600, "step": 5,
                    "tooltip": (
                        "Target episode length in seconds. Word "
                        "budget = target_seconds * 2.5 (radio "
                        "~150 wpm)."
                    ),
                }),
                "cast_size": ("INT", {
                    "default": 2, "min": 1, "max": 6, "step": 1,
                    "tooltip": (
                        "Number of speaking characters in the outline."
                    ),
                }),
            },
            "optional": {
                "model_id": ([
                    "mistralai/Mistral-Nemo-Instruct-2407",
                    "google/gemma-4-E2B-it",
                    "google/gemma-4-E4B-it",
                    "Qwen/Qwen2.5-14B-Instruct [ALPHA]",
                    "Nitral-AI/Captain-Eris_Violet-V0.420-12B (EXPERIMENTAL)",
                    "inflatebot/MN-12B-Mag-Mell-R1 (EXPERIMENTAL)",
                ], {
                    "default": DEFAULT_MODEL_ID,
                    "tooltip": (
                        "Hugging Face model ID. Mistral-Nemo is the "
                        "validated production default. Suffix tags "
                        "([ALPHA], (EXPERIMENTAL)) are stripped by "
                        "the loader before HF lookup."
                    ),
                }),
                "polish_pass": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "v2.0 MVP no-op. The full-ledger continuity-"
                        "drift rewrite is deferred to v2.1 once the "
                        "50-run soak proves the per-line path stable. "
                        "Setting True logs an explicit notice but "
                        "does not change output."
                    ),
                }),
            },
        }

    CATEGORY = "OldTimeRadio"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = (
        "script_text", "script_json", "news_used", "estimated_minutes",
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Mirror legacy LLMScriptWriter: always re-execute. The seed
        # may change, the model may have warmed, etc.
        import time as _t
        return _t.time()

    # ------------------------------------------------------------------
    # FUNCTION method
    # ------------------------------------------------------------------

    def run(
        self,
        news_seed,
        style_hint,
        target_seconds,
        cast_size,
        model_id=DEFAULT_MODEL_ID,
        polish_pass=False,
    ):
        """Generate a v2.0 LPL script. See module docstring for pipeline."""

        # --- Validate + normalize inputs ----------------------------
        news_seed, style_hint, target_seconds, cast_size, target_words = (
            _normalize_inputs(news_seed, style_hint, target_seconds, cast_size)
        )

        log.info(
            "[OTR_LedgerScriptWriter] start: model=%r, target_seconds=%d, "
            "cast_size=%d, target_words=%d, style_hint=%r, polish_pass=%s",
            model_id, target_seconds, cast_size, target_words,
            style_hint, polish_pass,
        )

        # --- Late imports (no GPU / no model loads at module import) ---
        from . import _otr_outline as _OTRO
        from . import _otr_canon as _OTRC
        from . import _otr_line_composer as _OTRLC
        from . import _otr_model_loader as _OTRML
        from . import _otr_ledger as _OTRL
        from . import production_ledger as _PL

        # --- A. Load LLM + build truncating generate_fn -------------
        cache_entry = _OTRML.load_llm(
            model_id, device="cuda", optimization_profile="Standard",
        )
        generate_fn = _build_truncating_generate_fn(cache_entry)

        # --- B. Generate validated outline --------------------------
        outline_req = _OTRO.OutlineRequest(
            news_seed=news_seed,
            style_hint=style_hint,
            cast_size=cast_size,
            target_seconds=target_seconds,
            target_words=target_words,
        )
        outline = _OTRO.generate_outline(generate_fn, outline_req)

        # --- C. Word-budget integration check (WARN, do not fail) ---
        beat_word_sum = sum(b.target_words for b in outline.beats)
        ratio = beat_word_sum / max(1, target_words)
        if not (WORD_BUDGET_RATIO_LO <= ratio <= WORD_BUDGET_RATIO_HI):
            log.warning(
                "[OTR_LedgerScriptWriter] WORD_BUDGET_DRIFT: outline "
                "beats sum to %d words, target %d (ratio=%.2f); "
                "proceeding anyway",
                beat_word_sum, target_words, ratio,
            )

        # --- D. Create per-episode workspace via new_ledger ---------
        # new_ledger() with episode_id=None creates a pending_<ts> slug
        # and the on-disk audio/ dir. Do NOT bypass this two-step
        # pattern -- SignalLostVideo later calls rename_episode().
        led = _PL.new_ledger(episode_id=None)
        episode_id = led.episode_id           # pending_<YYYYMMDD_HHMMSS>
        audio_dir = Path(led.out_dir)         # otr/episodes/<ep>/audio/
        episode_root = audio_dir.parent       # otr/episodes/<ep>/
        # episode_root was created by Ledger.__init__ via os.makedirs
        # on out_dir; sibling subdirs (stills/, portraits/, ...) are
        # created by their respective producer nodes when they run.

        # --- E. Stamp episode_canon at episode root -----------------
        canon = _OTRC.episode_canon_from_outline_dict({
            "title":       outline.title,
            "premise":     outline.premise,
            "setting":     outline.setting,
            "time_of_day": outline.time_of_day,
            # sound_palette intentionally omitted (Outline schema
            # doesn't carry it; canon defaults to []).
        })
        _OTRC.write_episode_canon(episode_root, canon)
        canon_header = _OTRC.render_episode_canon_header(canon)
        log.info(
            "[OTR_LedgerScriptWriter] episode_canon stamped at %s",
            episode_root / _OTRC.EPISODE_CANON_FILENAME,
        )

        # --- F. Build cast rows + char_id index ---------------------
        cast_rows, char_id_by_name = _build_cast_rows(outline.cast)
        led.set_cast(cast_rows)

        # --- G. Per-beat loop ---------------------------------------
        line_rows: list = []
        script_text_parts: list = []
        last_lines: list = []  # rolling window of LAST_LINES_WINDOW

        for beat in outline.beats:
            traits = (beat.mood or "").strip() or DEFAULT_TRAITS
            cleaned: str
            cid: str
            token: str

            if beat.speaker_role == "character":
                line_req = _OTRLC.LineRequest(
                    speaker=beat.speaker,
                    intent=beat.intent,
                    mood=beat.mood,
                    target_words=beat.target_words,
                    canon_header=canon_header,
                    last_lines=list(last_lines),
                )
                cleaned = _OTRLC.compose_line(generate_fn, line_req)
                cid = char_id_by_name[beat.speaker]
                token = f"[VOICE: {beat.speaker}, {traits}] {cleaned}"

                last_lines.append((beat.speaker, cleaned))
                if len(last_lines) > LAST_LINES_WINDOW:
                    last_lines.pop(0)

            elif beat.speaker_role == "announcer":
                line_req = _OTRLC.LineRequest(
                    speaker="ANNOUNCER",
                    intent=beat.intent,
                    mood=beat.mood,
                    target_words=beat.target_words,
                    canon_header=canon_header,
                    last_lines=list(last_lines),
                )
                cleaned = _OTRLC.compose_line(generate_fn, line_req)
                cid = "announcer"
                token = f"[VOICE: ANNOUNCER, {traits}] {cleaned}"

                last_lines.append(("ANNOUNCER", cleaned))
                if len(last_lines) > LAST_LINES_WINDOW:
                    last_lines.pop(0)

            elif beat.speaker_role in NON_VOICED_ROLES:
                # No LLM call. Use sfx_cue if present, else intent.
                cleaned = (beat.sfx_cue or beat.intent or "").strip()
                cid = beat.speaker_role
                token = f"[SFX: {cleaned}]"
                # Non-voiced beats do NOT push into last_lines (they
                # add no spoken context for the next compose call).

            else:
                # Defensive: an unknown role slipped past the schema.
                # Log and skip; do not raise -- the outline already
                # validated, so this would be a code drift between
                # _otr_outline.SpeakerRole and what we handle here.
                log.warning(
                    "[OTR_LedgerScriptWriter] unknown speaker_role %r "
                    "on beat %s; skipping",
                    beat.speaker_role, beat.beat_id,
                )
                continue

            line_rows.append({
                # Legacy schema fields (pass through set_lines):
                "line_id":       beat.beat_id,
                "shot_id":       None,
                "beat_id":       beat.beat_id,
                "char_id":       cid,
                "text":          cleaned,
                "traits":        traits,
                "boundary":      None,
                "bark_wav_path": None,
                "start_s":       None,
                "dur_s":         None,
                # Internal-only -- NOT passed to set_lines:
                "_speaker_role": beat.speaker_role,
            })
            script_text_parts.append(token)

        # --- H. set_lines + post-patch additive speaker_role --------
        # set_lines normalizes the legacy schema fields. The additive
        # speaker_role key is stamped per-row via patch_line_fields
        # immediately after, since set_lines drops unknown keys.
        led.set_lines([
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in line_rows
        ])
        for r in line_rows:
            _OTRL.patch_line_fields(
                led.data, r["line_id"],
                {"speaker_role": r["_speaker_role"]},
            )

        # --- I. Assemble return values ------------------------------
        script_text = "\n\n".join(script_text_parts)

        # script_json shape: v2.0 contract divergence from legacy.
        # Legacy emits json.dumps(script_lines) (token tree consumed
        # by LLMDirector). v2.0 LPL emits the canonical ledger dict
        # (kickoff §6.7b: "Validated ledger JSON string"). Phase 3
        # wiring will need to bridge any remaining consumer drift.
        script_json = json.dumps(led.data, indent=2, ensure_ascii=False)

        news_json = _build_news_payload(outline, news_seed)

        # est_minutes: float-with-1-decimal, 140 wpm (matches legacy
        # at story_orchestrator.py:6584). ComfyUI's INT type tag
        # silently coerces -- RECON 4 confirmed this is the on-wire
        # contract regardless of the type tag.
        actual_word_count = sum(
            int(r.get("word_count") or 0) for r in led.data["lines"]
        )
        est_minutes = max(
            1, round(actual_word_count / WORDS_PER_MINUTE_ESTIMATE, 1),
        )

        # --- J. Polish pass no-op (MVP) -----------------------------
        if polish_pass:
            log.info(
                "[OTR_LedgerScriptWriter] polish_pass=True received but "
                "is a no-op for v2.0 MVP; full-ledger rewrite deferred "
                "to v2.1"
            )

        # --- K. Save ledger -----------------------------------------
        saved_path = led.save()
        log.info(
            "[OTR_LedgerScriptWriter] DONE: episode_id=%s, lines=%d, "
            "words=%d, est_minutes=%s, ledger=%s",
            episode_id, len(led.data["lines"]), actual_word_count,
            est_minutes, saved_path,
        )

        return (script_text, script_json, news_json, est_minutes)


# ---------------------------------------------------------------------------
# Self-test (no-model smoke)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import traceback

    failures: list = []

    # 1. Class instantiation.
    try:
        cls = OTR_LedgerScriptWriter
        obj = cls()
        assert obj is not None
        print("[1/5] PASS: class instantiation")
    except Exception:
        failures.append(("1/5 class instantiation", traceback.format_exc()))
        print("[1/5] FAIL: class instantiation")

    # 2. INPUT_TYPES schema introspection.
    try:
        spec = cls.INPUT_TYPES()
        assert "required" in spec, "missing required block"
        assert "optional" in spec, "missing optional block"
        for k in ("news_seed", "style_hint", "target_seconds", "cast_size"):
            assert k in spec["required"], f"required missing key: {k}"
        for k in ("model_id", "polish_pass"):
            assert k in spec["optional"], f"optional missing key: {k}"
        # news_seed is multiline STRING
        ns_type, ns_meta = spec["required"]["news_seed"]
        assert ns_type == "STRING"
        assert ns_meta.get("multiline") is True
        # target_seconds INT clamps
        ts_type, ts_meta = spec["required"]["target_seconds"]
        assert ts_type == "INT"
        assert ts_meta["min"] == 10 and ts_meta["max"] == 600
        # cast_size INT clamps
        cs_type, cs_meta = spec["required"]["cast_size"]
        assert cs_type == "INT"
        assert cs_meta["min"] == 1 and cs_meta["max"] == 6
        # model_id dropdown is hardcoded list of 6 entries
        model_choices, _ = spec["optional"]["model_id"]
        assert isinstance(model_choices, list) and len(model_choices) == 6
        assert model_choices[0] == "mistralai/Mistral-Nemo-Instruct-2407"
        # polish_pass BOOLEAN, default False
        pp_type, pp_meta = spec["optional"]["polish_pass"]
        assert pp_type == "BOOLEAN"
        assert pp_meta["default"] is False
        print("[2/5] PASS: INPUT_TYPES schema")
    except Exception:
        failures.append(("2/5 INPUT_TYPES", traceback.format_exc()))
        print("[2/5] FAIL: INPUT_TYPES schema")

    # 3. Locked output contract.
    try:
        assert cls.RETURN_TYPES == ("STRING", "STRING", "STRING", "INT"), \
            f"RETURN_TYPES drift: {cls.RETURN_TYPES}"
        assert cls.RETURN_NAMES == (
            "script_text", "script_json", "news_used", "estimated_minutes",
        ), f"RETURN_NAMES drift: {cls.RETURN_NAMES}"
        assert cls.FUNCTION == "run"
        assert cls.CATEGORY == "OldTimeRadio"
        print("[3/5] PASS: output contract")
    except Exception:
        failures.append(("3/5 output contract", traceback.format_exc()))
        print("[3/5] FAIL: output contract")

    # 4. _build_truncating_generate_fn returns a callable.
    try:
        fake_cache = {
            "model": None,
            "tokenizer": None,
            "context_cap": 8192,
        }
        gen = _build_truncating_generate_fn(fake_cache)
        assert callable(gen), "wrapper did not return a callable"
        # Do NOT call gen() -- that would dereference the None model.
        print("[4/5] PASS: truncating generate_fn build")
    except Exception:
        failures.append(("4/5 generate_fn build", traceback.format_exc()))
        print("[4/5] FAIL: truncating generate_fn build")

    # 5. Validation paths fire (uses the free helper, no model load).
    try:
        # 5a. Empty news_seed raises RuntimeError.
        raised = False
        try:
            _normalize_inputs("", "noir", 90, 2)
        except RuntimeError:
            raised = True
        assert raised, "empty news_seed did not raise RuntimeError"

        # 5b. Whitespace-only news_seed also raises.
        raised = False
        try:
            _normalize_inputs("   \n  ", "noir", 90, 2)
        except RuntimeError:
            raised = True
        assert raised, "whitespace news_seed did not raise RuntimeError"

        # 5c. target_seconds high clamp (700 -> 600).
        _, _, ts, _, tw = _normalize_inputs("seed text", "noir", 700, 2)
        assert ts == 600, f"target_seconds high-clamp failed: {ts}"
        # target_words derives from clamped target_seconds.
        assert tw == int(600 * WORDS_PER_SECOND_BUDGET), \
            f"target_words derive failed: {tw}"

        # 5d. target_seconds low clamp (5 -> 10).
        _, _, ts, _, _ = _normalize_inputs("seed text", "noir", 5, 2)
        assert ts == 10, f"target_seconds low-clamp failed: {ts}"

        # 5e. cast_size high clamp (10 -> 6).
        _, _, _, cs, _ = _normalize_inputs("seed text", "noir", 90, 10)
        assert cs == 6, f"cast_size high-clamp failed: {cs}"

        # 5f. cast_size low clamp (0 -> 1).
        _, _, _, cs, _ = _normalize_inputs("seed text", "noir", 90, 0)
        assert cs == 1, f"cast_size low-clamp failed: {cs}"

        # 5g. style_hint default fallback when blank.
        _, sh, _, _, _ = _normalize_inputs("seed text", "", 90, 2)
        assert sh == "psychological slow-burn", \
            f"style_hint default failed: {sh!r}"

        print("[5/5] PASS: validation + clamping (7 sub-cases)")
    except Exception:
        failures.append(("5/5 validation", traceback.format_exc()))
        print("[5/5] FAIL: validation")

    # Summary.
    if not failures:
        print("\nSELF-TEST PASS: 5/5")
        sys.exit(0)
    else:
        print(f"\nSELF-TEST FAIL: {len(failures)} of 5")
        for name, tb in failures:
            print(f"\n--- {name} ---\n{tb}")
        sys.exit(1)
