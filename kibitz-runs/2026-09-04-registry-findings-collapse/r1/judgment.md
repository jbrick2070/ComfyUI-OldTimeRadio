# r1 judgment -- registry-findings-collapse (driver: Claude / Opus 5)

Roster, as it happened: Codex CLI lane QUOTA-HELD (rate limit; retry window
~10:35 PDT; its `codex.md` is an event stream, not a review -- discarded);
Antigravity CLI `Gemini 3.8 Flash (High)` returned a real review; the Codex
seat is filled by a Sonnet 5 substitute; Cursor reviews by the operator's
manual paste. First fan-out died at spawn (WinError 206: the regenerated
overlay pushed the prompt past the Windows command-line limit); rerun with
`--no-profiles`, the overlay path named in the prompt instead. Driver anchor
written before the fan-out (`driver_anchor.md`).

## Antigravity (r1), grounded

| # | claim | verdict | disposition |
|---|---|---|---|
| MF1 | the `credential-access` tag follows the RULE, so one owner CONCENTRATES it on `env.py` rather than removing it; the plan contradicts itself ("no credential tags" vs "lands on the owner once") | CONFIRMED (by the payload: the tag rides on `OTR_TEST_MODE` and `OTR_WRITER_HEARTBEAT_EVERY` reads, i.e. rule-driven) | plan corrected: `env.py` keeps `system-modification` (1) and `credential-access` (1); the owner stays dynamic -- no hardcoded credential names, no catalog inside it |
| MF2 | `gpu_residency.py` ALREADY tries `psutil.pid_exists` first (`:73-74`); line 82 is the ctypes FALLBACK -- the plan's "replace with psutil" inverts the code | CONFIRMED (read `:66-100`; `psutil` is in ComfyUI core `requirements.txt:20`) | fix is: delete the ctypes fallback block, keep psutil then the existing `os.kill(pid, 0)` path |
| MF3 | `prestartup_script.py` DOES insert the package dir into `sys.path` (`:90-91`) and carries FIVE env mutations (four `setdefault` + `HF_HOME`), not one; "cannot import the package" is false | CONFIRMED | narrative corrected. It STAYS inline by decision, not inability: it must run before transformers and stay decoupled from node loading. One per-file finding remains (+1 in the floor) |
| MF4 | five unaccounted mutation sites: `_otr_workflow_validator.py:450` (`OTR_SNAPSHOT_HASH`), `_otr_hf_env.py:126` (`HF_HUB_CACHE`), `hf_token.py:99` (`setdefault HUGGING_FACE_HUB_TOKEN`), root `__init__.py:51` (`setdefault HF_HUB_DISABLE_TELEMETRY`), `OTR_LedgerScriptWriter.py:1755` (`environ.pop`) -- `pin()` cannot express `pop` or `setdefault` | CONFIRMED by grep (12 mutation sites in all across nodes/ and the two roots) | the owner's surface is `get / pin / setdefault / unpin`; the guard covers `environ[...]=`, `setdefault`, `pop`, `putenv`, `unsetenv`, `getenv` |
| MF5a | a transparent pass-through `proc.py` reads as an evasion shim to a human reviewer; an executable allowlist (ffmpeg, ffprobe, git, sidecar interpreters, blender) makes it a boundary | PLAUSIBLE, a genuine design question | carried to r2 as THE question for item B; the driver's lean is yes -- it is the "one owner" shape applied to what may run, and it strengthens the review request instead of looking like a dodge |
| MF5b | `production_ledger.py:222` uses `subprocess.check_output`; the proposed grep proxy would miss it | CONFIRMED | `proc.py` covers `run`, `Popen`, `check_output`; the guard flags every `subprocess.` attribute outside the owner |
| SF1 | `$media_list_assign_indirect` may match the `cmd = [` SHAPE, not the literal -- the argv helper may clear nothing | PLAUSIBLE (the pattern's regex is private) | verify-at-build: the next published scan is the receipt; do not promise the six |
| SF2 | batch by subsystem, not alphabetically | ACCEPTED | `_otr_shared` -> audio engines -> video/upscale engines -> top-level nodes + roots; singletons and strings first |
| SF3 | do not unify requests / urllib / raw socket into one HTTP owner for a yield of four | ACCEPTED | item C shrinks to a minimal helper for the two `urllib.request.urlopen` sites; `requests` (two backends) and the RSS fetcher's bounded-connect socket stay as NAMED exceptions with their reason -- the allowlist-with-reason shape `test_output_root_single_owner.py` already uses |
| OPT1 | chunked sha256 also fixes a whole-file RAM spike on multi-hundred-MB checkpoints | CONFIRMED | folded into D |
| OPT2 | verify no test asserts the `ffprobe -count_frames` wording | CHECKED: none does (the one grep hit is a helper's name) | E is safe |
| CUT1 | the declared knob catalog | ACCEPTED (converges with the driver anchor's MF2: spelling-only migration) | cut from this plan; `otr_env.get(name, default)` returns `str | None` exactly like `os.environ.get` |
| CUT2 | typed getters | ACCEPTED | cut; casts stay at the call site, untouched |
| CUT3 | a second owner for streams | ACCEPTED | `proc.popen()` returns the standard `Popen` |

## Driver anchor, kept

MF1 (owner reads `os.environ` live, caches nothing -- `monkeypatch.setenv`
in hundreds of tests and the launchers' pins depend on it), MF2 (spelling-only
migration; no cast or default moves), MF3 (guard covers every mutation
spelling), MF4 (`proc.py` is not merged into the ffmpeg owner), MF5 (the
guard's shipping order: it ships LAST with the final batch, or as a ratchet
with an asserted, shrinking offender count per batch -- r2 picks). All stand;
none contradicted by the panel.

## Sonnet r1 seat (substitute for Codex), grounded

Verified every per-rule count against the payload exactly (103/103, 35/20,
12/3, 5/5, three singletons, all tag totals).

| # | claim | verdict | disposition |
|---|---|---|---|
| MF1 | `python_command_injection_risk` is per SITE (`otr_credits_roll.py` carries four findings for four `subprocess.run` lines); a process owner with one `run`, one `Popen` and one `check_output` nets THREE, not two; "about 5" must be conditional on one call site per primitive and re-verified by a real scan | CONFIRMED | floor recomputed below; the number quoted in the review request comes from the next published scan, never from the plan |
| MF2 | the plan's own precedent breaks its guard on day one: `_otr_shared/ffmpeg.py:71` and `_otr_shared/ffprobe.py:172` read `os.environ` directly and are not in any batch; the scan predates the ffmpeg owner, so today's count is 104 | CONFIRMED (grep proxy: 104 files) | both named in batch 1 (`_otr_shared`); `env.py` is a leaf, `ffmpeg.py` imports it, no cycle; `tests/test_ffmpeg_single_resolution.py` keys on the `OTR_FFMPEG` constant and on `which()`, so it should hold -- run, not assume |
| SF3 | the RSS fetcher's socket is SSRF hardening -- resolve, reject private ranges, connect to the VALIDATED address, then TLS-wrap pinned to it, to defeat DNS rebinding between check and connect -- not "a bounded connect"; question 6 has one answer | CONFIRMED (`_otr_feed_fetch.py:_connect` docstring: "pinned to the address we validated") | it stays a NAMED exception with that reason; the urllib helper must never become an alternate path for feed URLs |
| SF4 | the plan does not say whether this is a mechanical rename or a per-site behaviour audit; the ffmpeg precedent earned its keep by finding drift, and a mechanical pass could reintroduce "silent default wins" if two sites read one knob with different defaults | CONFIRMED as a gap | decided: the MIGRATION is spelling-only (driver anchor MF2). A separate, earlier step lists every knob read with more than one default or cast across sites (a grep is enough) and hands that list to a design review -- the audit is its own item with its own receipts, never smuggled into the rename |
| SF5 | "four hand-rolled bool parsers" undersells it: three named `_bool_env` functions plus dozens of inline `== "1"` checks; typed getters are closer to the one defensible answer | PARTLY: the count is CONFIRMED; the conclusion is REJECTED for this arc | agy cut typed getters, the anchor says spelling-only; the fragmentation is REAL and is logged as the follow-on item that the default-drift list feeds -- after the rename has shipped and stayed green |
| SF6 | `HF_TOKEN` is read three times, not two; sidecar-venv launches are three, not four | CONFIRMED | wording fixed |
| OPT | keep psutil's core-requirement status explicit in the request | ACCEPTED (it IS in core `requirements.txt:20`; say so) | |
| CUT1 | the knob catalog | ACCEPTED (third lane to say it) | cut |

## Cursor (manual paste by the operator), grounded

Headline counts confirmed against the payload. Three collapses challenged:

| # | claim | verdict | disposition |
|---|---|---|---|
| D/OpenProcess | psutil is tried first; the ctypes block is the Windows fail-safe; deleting it changes lock-steal safety | CONFIRMED, and worse than stated: the path behind it is `os.kill(pid, 0)`, which on Windows TERMINATES (CPython calls `TerminateProcess` for any non-console signal) | D rewritten: on Windows without psutil return True (do not steal); `os.kill` probe on POSIX only; `OpenProcess` still goes |
| C/RSS socket | folding it into a generic HTTP owner is an SSRF regression | CONFIRMED (third lane) | named exception with the reason; the helper never serves feed URLs |
| C/stdlib-only | a helper wrapping `requests` cannot be stdlib-only | CONFIRMED | C is urllib-only; `requests` sites are named exceptions |
| A/knob catalog | a catalog that lists `OTR_FFMPEG` trips `test_ffmpeg_single_resolution.py` (the constant outside the owner) | CONFIRMED | catalog cut (all three lanes); `env.py` never spells that constant or calls `which()` |
| acceptance grep | `subprocess.` is the wrong proxy: `check_output`, `PIPE`, type annotations stay red | CONFIRMED | guard is an AST rule on execution CALLS, not attributes; the grep proxy is retired |
| credential-access | retract "no tags"; a generic `get(name)` keeps the tag on the owner | CONFIRMED (same as agy MF1) | retracted in r1 final |
| floor | about 7 if the unsafe merges are refused | CONSISTENT with the driver's ~9 (Cursor counts the urllib helper and prestartup differently); the number the request quotes is the next scan's | |

## Revised floor, conditional and re-verified by the next scan

env 1 (owner) + 1 (`prestartup_script.py`, by decision) + proc 3 (one site
each of `run`, `Popen`, `check_output`) + network 3 named exceptions (two
`requests` backends, the SSRF-hardened socket) + 1 (the urllib helper) +
singletons 0 + strings 0 = **about 9 findings**, all `info`, from 158 (104
today). Not Active -- that is zero or the manual review -- but a one-screen
review, and the `credential-access` tag on one file instead of eleven.
