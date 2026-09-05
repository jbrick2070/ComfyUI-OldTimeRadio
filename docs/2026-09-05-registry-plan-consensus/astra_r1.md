**P1 is confirmed; the punch list is incomplete. Rank: D > C > A > B.** Read-only review of HEAD `02ca6d5a` and extracted alpha.19; no files changed or attack paths executed. References below are HEAD unless marked **A19**.

The supplied policy evidence says **“Node-widget taint keeps full severity … unconfined write = INSECURE”** (`policy_v02_examples.md:127`, Rule 11). JSON encoding and `forceInput=True` do not establish trusted provenance.

**P1 — CONFIRMED: unconfined writes.**

| Source → sink | Evidence |
|---|---|
| CaptionBurn `output_path` | `nodes/otr_caption_burn.py:350` STRING → `:415` remote-only check → `:441` **`out = output_path.strip() or self._default_out(video_path)`** → `:273` derived ASS destination → `_otr_captions.py:507` `write_text`; FFmpeg output at `otr_caption_burn.py:288`. |
| MasterAudioMux `output_path` | `nodes/otr_master_audio_mux.py:1064` STRING → `:1317` remote-only check → `:1325` supplied output → `:405` FFmpeg output argument, executed at `:414`. |
| SilentComposite `output_path` | `nodes/otr_silent_composite.py:1536` STRING → `:1752` remote-only check → `:1753` supplied output → `:1459` FFmpeg destination; additional JSON write at `:1482`. |

These match the quoted OCIO/SHARPPredict Rule-11 verdicts. The paths remain open in alpha.19.

Two driver claims are false:

- `_otr_paths.py` **has containment**: `:302` `_validate_contract`; `:308` **`p.resolve().relative_to(base)`**. Explicit overrides bypass it.
- CaptionBurn’s default is **not necessarily inside episodes**: `otr_caption_burn.py:383–385` writes beside the caller-selected input video.

**Fix:** admit explicit **and derived** destinations at node entry, before directory creation, ASS construction or rendering. Use resolved containment under trusted output roots; retain operator environment configuration for extra roots. Preserve canonical filenames, ASS sidecars, master-audio copying and OBS publication.

**P2 — split ruling: CONFIRMED raw LFI in alpha.19; UNSURE blanket classification of media/JSON reads.**

- **A19 raw LFI:** `otr_image_gen_dispatcher.py:2229` STRING `script_json` → `:1555` **`im.get("pool_path") or im.get("path")`** → `:1557` existence-only check → `:635` **`shutil.copyfile(str(src_path), dst)`**, where `dst` is an episode `.png`. No decoding verifies image content. This matches `policy_v02_examples.md:139`: **“arbitrary file read (LFI) … attacker-reachable via unauthenticated /prompt.”**
- **Media disclosure is real:** `otr_master_audio_mux.py:1032` `master_audio_path` → `:402–404` `"-i", master_audio_path`, audio mapping and stream copying. An outside-root audio file can become output content. Conversely, this does not establish arbitrary text-file disclosure: FFmpeg must accept the input format. The supplied one-line LFI verdict does not settle that distinction.
- **Validator “JSON parsed and echoed” is refuted:** `_otr_workflow_validator.py:110` reads JSON, but `:545–555` returns counts/path/status, not its contents. Errors can reveal structure.
- **Replay is constrained:** `production_ledger.py:621–643` requires manifest schema, relative entries, sizes and hashes; HEAD `:654–659` adds ledger/master membership. `replay_from` is not simply an arbitrary secret-file selector. Hashes prove consistency, not trusted authorship.

**Fix for confirmed LFI:** ship the image-cache repair and validate source/destination provenance before copying. Preserve legitimate cache reuse and episode materialization. HEAD `_trusted_pool_source` uses `abspath/commonpath` (`otr_image_gen_dispatcher.py:647–651`), so its containment claim still needs a symlink/junction case; `abspath` is not `realpath`.

For broader reads, define admitted media/workflow/replay roots at entry boundaries, preserving canonical workflows, imported bundles and operator reference paths. The missing evidence is the applicable policy distinction or a concrete additional disclosure chain—not another byte diff.

**P3 — REFUTED as the quoted arbitrary-path/side-effect class; exposure confirmed.**

`__init__.py:465` registers GET; `:483` selects **`in_flight_ledger_path()`**; `:491–500` returns the selected ledger and **`"fullpath": latest`**. No request parameter chooses the file, and the handler does not mutate it.

Removing wildcard CORS is not authentication. Reachable clients still obtain episode data. The supplied verdicts do not establish this fixed-resource GET alone as bannable.

The render POSTs are **disabled by default, not authenticated**: `__init__.py:594`. With the environment flag enabled, unauthenticated side effects remain at `:597–606` and `:610–621`. Do not describe them as universally closed.

**P4 — REFUTED as framed.**

No `drawtext`, `movie=` or `amovie=` appears in the extracted alpha.19 Python tree.

Actual chain: `_otr_captions.py:346/:441` escapes text into ASS dialogue → `:499–507` writes ASS → `otr_caption_burn.py:127` checks the filename → `:283` constructs `ass=<checked basename>,fps=...`. The blacklist at `:100` is **`set(",;:=[]'\\")`**. Credits use Pillow.

Do not invent a drawtext repair or introduce episode-content filtering. P1 remains: the ASS write precedes filename/filter validation.

**P5 — UNSURE as a ban class; widget-derived environment mutation CONFIRMED.**

- `_otr_workflow_validator.py:263` STRING `profile_id` → `:375` profile load → `:452` **`otr_env.pin("OTR_ACTIVE_PROFILE", profile_id)`**.
- `:458–461` hashes the selected workflow and pins `OTR_SNAPSHOT_HASH`.
- `OTR_LedgerScriptWriter.py:2508` scaffold selection → `:2996` helper → `:1731/:1733` pins `OTR_ENABLE_STYLE_GRAMMAR` to `"1"`/`"0"`. Restoration occurs on another `auto` invocation, not in `finally`.
- Actual sink: `_otr_shared/env.py:77`, **`os.environ[name] = value`**.

“All constants/configuration” is false. However, names are fixed and values constrained/derived; no supplied verdict establishes these metadata/grammar mutations as independently bannable. The missing evidence is an applicable environment rule or dangerous downstream consumer.

**P6 — the twelve scanner sites.**

All ten containing files are byte-identical between alpha.19 and HEAD. “REFUTED” below concerns the alleged attacker-controlled ban path, not the existence of the operation.

| Site | Source → sink; ruling |
|---|---|
| `prestartup_script.py:60` | `__file__`-derived directory → `HF_HOME` assignment. **REFUTED:** boot configuration. |
| `nodes/_otr_writer_heartbeat.py:61` | Environment read → bounded interval. **REFUTED:** not mutation/execution. |
| `nodes/_otr_audio_engines/eng_indextts2.py:176` | Operator environment/default → interpreter path. **REFUTED:** no widget selection. |
| `nodes/_otr_shared/env.py:77` | Callers → environment assignment. **UNSURE**, P5. |
| `nodes/_otr_comfy_backend.py:384` | `_chat_url():409–413` environment/constants → `requests.post`. **REFUTED:** body may be tainted; host is not widget-selected. |
| `nodes/_otr_feed_fetch.py:249` | Feed configuration/fetched links → socket. **REFUTED:** HTTPS at `:374`, public-address validation at `:206`, connection to validated address at `:252`, bounded reads at `:265`. |
| `nodes/_otr_openrouter_backend.py:1011` | Environment base at `:1038` → catalog GET. **REFUTED:** no widget-selected host. |
| `nodes/_otr_google_api/client.py:191` | Environment/constant base at `:179` + API resource → `urlopen`. **REFUTED:** no demonstrated widget-host path. |
| `nodes/_otr_shared/cloud_media_invoke.py:578` | Provider-returned URL at `:611–613` → download. **REFUTED as demonstrated widget SSRF.** No destination/redirect guard; provider-output controllability would reopen it. |
| `nodes/_otr_audio_engines/eng_indextts2.py:214` | Trusted interpreter/worker configuration → **`[py, worker, "--model-dir", model_dir]`** at `:209` → Popen. **REFUTED:** no widget RCE. |
| `nodes/_otr_shared/proc.py:161` | Guarded argv → subprocess.run. **REFUTED:** string argv, shell and executable replacement refused at `:125/:142/:150`. |
| `nodes/_otr_shared/proc.py:168` | Same guards → Popen. **REFUTED**, same qualification. |

The gateway does not cure attacker-controlled file arguments. SAFE verdicts explicitly accepting “subprocess: ffprobe” and “Subprocess: ffmpeg” support retaining the render path.

**P7+ FOUND — additional blockers.**

| Item | Evidence, classification and fix |
|---|---|
| **P7: More unconfined sibling writes — CONFIRMED, Rule 11** | Blend: `otr_post_upscale_procgen_blend.py:927` caller source → `:959` **`src.parent / …`** → `:963` `shutil.copy2` on bypass. Suffix sanitization at `:953` does not confine the directory; bypass requires no media decoding. Credits: `otr_credits_roll.py:1564–1567` derives sibling destinations from `video_path`; writers execute at `:1580–1583`, given usable ledger/layout. **Fix:** admit derived destinations at `blend`/`roll` entry. Preserve bypass copies, credits receipts and canonical placement. |
| **P8: Voice JSON paths → writes — CONFIRMED, Rule 11** | `_otr_voice_node_common.py:354` STRING ledger → `:932–947` parsed metadata → `:643` **`paths.get("ledger_path")`** → `:654` JSON read → `:662` save → `_otr_ledger.py:519` **`os.replace(tmp_name, target)`**. Requires an existing dictionary JSON and a cloud-cache or local policy-route receipt (`:1410/:1439/:1523`); not every ordinary local line. Separately, `meta.paths.audio_dir` → `:398–400` cache directory → `_otr_audio_cache.py:258–281/:391–394` directory/audio/JSON writes on cache-enabled profiles. **Fix:** establish current-ledger identity and admit nested destinations at voice entry. Preserve reload/merge, atomic saves, cache keys, operator cache roots and qualified receipts. |
| **P9: Remaining UNC egress — CONFIRMED** | **Voice fingerprint:** `_otr_voice_node_common.py:729–734` caller `cast[].voice_route` → `:797–800` `ref_path` → `_otr_audio_engines/base.py:140–141` absolute passthrough → `_otr_voice_route.py:102` `open`. **Blend:** `scopes_mp4_path` STRING at `:858` omitted from guard `:925`, resolved/statted at `:929–930`. **Validator:** `profile_id` → `_otr_shared/capability_profiles.py:324` **`os.path.join(d, f"{profile_id}.json")`** → `:325/:336` stat/open before shape checks. **Manifests:** SilentComposite `:1756` parses caller JSON → `:680–697` trusts clip paths → `:1415–1417` stats; SceneAwareScopes `:443/:458` → `:261–266` stats/probes. These match the supplied **“egress / SSRF … attacker-reachable via … node widget”** class. **Fix:** reject remote values before resolution/stat/hash in every callback, including `IS_CHANGED`; validate consumed nested paths and profile identifiers. Preserve local bank references, frame directories, plates and 4060 profile IDs. |
| **P10: Identifier-to-output traversal — CONFIRMED, with directory prerequisites** | `otr_video_render_batch.py:465` STRING engine → `:543` **`"node_single_%s.json" % engine`** → `:560–561` write. Unknown engines return failure reports (`render_driver.py:6885–6890`), so engine failure does not prevent writing. SceneAwareScopes `:460` caller episode ID → `:557` filename → `:564` encoder. Traversal requires suitable existing intermediate directories. **Fix:** validate identifiers and final resolved destinations at entry; preserve engine identity, diagnostic reports and scopes output. |
| **P11: Arbitrary mkdir — operation CONFIRMED; standalone ban severity UNSURE** | `scene_sequencer.py:796` STRING `output_dir` → `:910` **`os.makedirs(output_dir, exist_ok=True)`**. No later audio write uses that variable. It permits directory creation/UNC access and can supply P10’s directory prerequisite. **Fix:** remove the unused side effect or admit its destination at `sequence`; preserve positional workflow compatibility. |

P7–P11 remain at HEAD and in alpha.19’s relevant code. Alpha.19 additionally lacks the portrait `episode_id` traversal guard now at `_otr_shared/portrait_ledger.py:83–92`, and CreditsRoll’s remote guard.

**Plan ranking and sequence: D > C > A > B.**

1. **D — recommended:** coder closes the complete boundary inventory above, including pre-execution callbacks and JSON-carried paths; distinguishes uncertain policy cases from demonstrated chains. Review/test the resulting package, not just the scanner’s twelve lines.
2. **Coder:** preserve the three byte-hashed files; verify canonical wiring/widget order, required regression suite and Bug Bible, Windows path/junction cases, and 5080/4060 behavior. Any live run loads canonical JSON and verifies episode/OBS assets. Commit and push the green code chunk.
3. **Operator:** bump/publish alpha.20. **Coder:** verify the downloaded artifact against the reviewed commit and inspect its actual scan record.
4. **C:** operator posts a short factual note identifying that verified version and fixes. Avoid claiming every surface is closed while P9 remains.
5. **A:** acceptable only with D’s release checks; weaker because it assumes the review queue will act promptly.
6. **B:** last. The existing request promotes alpha.19 despite its confirmed raw copier and unconfined writes.

The driver overstates the process evidence: identical `verify-deep` wording does not prove automation, same-day examples do not establish a review SLA, and absent public replies do not prove issues have no effect. Code repair is necessary; automatic, fast activation is not established.