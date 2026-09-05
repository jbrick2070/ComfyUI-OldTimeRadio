Both `Shell` and `WebFetch` were rejected this session, so I could not run the section-0B pull or re-read the live scan record. The assessment below is grounded against the working tree (snapshot showed `v2.0-alpha` in sync with origin) and the guards that ratchet the shipped surface.

## The twelve, mapped to files

The count decomposes exactly onto the three guards in `tests/`, so it is not a mystery to investigate; it is a floor that was designed.

**4 env reads** (`python_environment_manipulation`, fires per FILE) -- the `ALLOWED | PENDING` set in `tests/test_env_single_owner.py`:
- `nodes/_otr_shared/env.py` -- the owner. Irreducible by definition.
- `prestartup_script.py` -- runs before the pack is a package; nothing to import an owner from.
- `nodes/_otr_audio_engines/eng_indextts2.py` -- byte-hashed in `RUNTIME_FINGERPRINT_SOURCES`; migrating it demoted the Lemmy voice route (6 tests red on 09-04). Operator RULED closed (`6a5fec90`).
- `nodes/_otr_writer_heartbeat.py` -- leaf by contract (`test_writer_heartbeat_is_visible.py` forbids any `from ._otr` import). Removing the finding means weakening a safeguard; operator refused that trade.

**5 network** (`NETWORK_CALLERS` in `tests/test_network_call_sites_are_named.py`): `_otr_comfy_backend.py`, `_otr_openrouter_backend.py` (`requests`, one streams), `_otr_google_api/client.py`, `_otr_shared/cloud_media_invoke.py` (`urlopen`), `_otr_feed_fetch.py` (raw `socket` + `http.client`). The r1 panel cut the unified-HTTP-owner idea on purpose: merging three incompatible transport layers to save four `info` lines puts risk on the streaming path. This is a register, not a collapse.

**3 subprocess** (`python_command_injection_risk`, fires per SITE): `proc.py:161` (`run`) + `proc.py:168` (`popen`) + `eng_indextts2.py:214` (fingerprint-locked, same ruling as above). `_pipeline.py`'s `import subprocess` is only `except subprocess.TimeoutExpired` and correctly does not count.

## Assessment

There is no code action here. Every one of the twelve sits behind either a product fact (ffmpeg is a subprocess; a video renderer cannot reach zero) or an operator ruling already recorded in the guards' `BLOCKED` tables. The record is also explicit that the count was never the gate: alpha.13/.14 were banned for attacker-reachable RCE classes (closed in alpha.20), and 0 of 102 policy-v0.2 approvals had a clean scan -- 31 shelled out via subprocess. Re-opening the number would be the mistake three earlier sessions made.

The one shave still technically available -- loading `env.py` by path from `prestartup_script.py` to drop 4 to 3 -- buys one `info` line for an import hack in the pre-package boot path. Not worth it, and it changes nothing about Active.

## What actually moves this

Only the operator: post `docs/2026-09-05-registry-review-request-ALPHA20.md` (the post body between the rules, ~430 words) at [Comfy-Org/registry-backend issues](https://github.com/Comfy-Org/registry-backend/issues). Every earlier draft is DO-NOT-SEND (they name alpha.19, which ships the file read). No further `pyproject.toml` bump.
