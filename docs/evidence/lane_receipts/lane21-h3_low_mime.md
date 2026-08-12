# VIDEO_LANE_PREFLIGHT receipt -- lane 21, `h3_low_mime` (the standalone runner)

`VIDEO_LANE_PREFLIGHT receipt: h3_low_mime | 2026-08-12 | deliverables
output/otr/episodes/_mime/ | verdict PASS`

**THE LAST PACKET, AND THE ONLY ONE THAT KEEPS ITS AUDIO.** Lane 21 has no
preflight-matrix row and no gates to flip, because it registers nothing. It is a
script.

## Why it is a runner and not an engine

Operator ruling 2026-08-10 (spec P3). **r3 proved a registered engine cannot be
kept OUT of the episode dropdowns** -- `role_compat` grants every role the full
input vocabulary and the dropdown IS the registry -- so a mime engine in the
registry is immediately selectable for ordinary beats. And a mime beat's audio
does not exist until its video renders, which the audio-first master freeze
cannot absorb without a phase inversion (the beat's duration would come from the
script target rather than TTS samples, its video would render EARLY, and its stem
would join assembly pre-freeze). That is sacred-path surgery with its own spec.

So this build ships the capability as a runner and the dropdown entry follows
later, behind a real standalone-only boundary, built on the pieces this runner
proves. `tests/test_h3_mime_runner.py` holds the not-registered invariant from
both sides: the registry does not contain it, and importing the runner does not
change the registry or the public menu.

## THE AUDIO LAW, INVERTED -- and why that is not a V-1 violation

Every other H3 path in this repo proves SILENCE. Lanes 19 and 20 never even
decode the audio latent, and `canonicalize` ffprobes their emitted file to PROVE
`has_audio: False`.

**This runner is the G5.2 exemption: the clip's invented score IS its audio.**
Its graph adds the half the registered lanes drop -- `VAEDecodeAudio` off the
SAME sampled latent the picture comes from, into `CreateVideo`, into
`SaveVideo`. So the assert here is the opposite of every other lane's: the
delivered mp4 MUST carry exactly one audio stream, and a silent mime render is a
FAILURE rather than a quiet success.

V-1 is untouched because **these outputs never enter episode assembly.** V-1
says only `OTR_MasterAudioMux` adds audio to an EPISODE. These are runner
deliverables. If a mix ever graduates toward episodes it re-enters through a
fresh ear gate and its own spec, not through this script.

## Two decisions this lane makes that no other lane may copy

**1. It delivers at the MODEL's 24 fps, un-converted.** Lanes 19/20 remap their
frames to the 25 fps canvas because they hand clips to a 25 fps timeline. This
clip never reaches that timeline, and resampling the picture while its generated
score stays at the model's timebase is exactly how a soundtrack desyncs from the
images it was scored for. Native rate, both streams, one timebase.

**2. It accepts lengths BELOW the trained band.** The registered lanes publish
only model 124..362 because a beat that renders below it renders badly. The
lab's two passing mime legs are model **f90** (3.750 s) -- legal on the 17k+5
grid, below that floor, and both scored well. A mime clip is a short authored
moment, not a beat. The runner takes the whole legal grid and SAYS when it is
below the band rather than refusing, and the receipt records
`below_trained_band` so no reading of it can mistake one for the other.

## G8.1 solo run

| Item | Value |
|---|---|
| Boot | the named **`h3`** contract: Sage-free, `--disable-pinned-memory`, `--reserve-vram 12` |
| Command | `otr_h3_mime_runner.py --portrait <png> --seconds 3.75 --seed 43 --stem-wav --mux-tts <wav>` |
| Prompt id | `b5071307-3cc6-4499-ad24-768859cafb63` |
| Target -> grid | 3.750 s asked -> model **f90** -> 3.750 s delivered, exactly |
| Wall time | **155.4 s** -- faster than lanes 19/20 (~240 s) because f90 is a shorter render than their f124, on the same stack |
| Canvas PROBED | **864x480** |
| Rate PROBED | **24/1** -- the model's own, deliberately NOT the 25 fps canvas |
| Frames / duration | **90 frames, 3.750000 s** = 90/24 exactly |
| **Audio PROBED** | **`nb_streams=2`: one video + ONE AAC audio stream, 32 kHz stereo** -- the inversion this lane exists for |
| Score stem | `lane21_mime_smoke.flac` -- **FLAC, lossless**, 32 kHz stereo, 3.750 s, from the SAME decode (no second sampler pass) |
| Review mix | `lane21_mime_smoke_with_voice_REVIEW.mp4` -- voice over the score ducked -9 dB, picture stream COPIED not re-encoded, 3.750 s |
| Originals | both preserved untouched; the mix is a THIRD file |
| Receipt | `lane21_mime_smoke.receipt.json`, written beside the deliverables |
| Deliverables path | `output/otr/episodes/_mime/` -- durable, never tmp (CLAUDE.md section 6) |

## THREE LIVE FAILURES BEFORE THE PASS, and all three were the same lesson

Every one was an API-serialization mismatch, and none was visible to a CPU test.

1. **The portrait staged to the wrong machine's input directory.**
   `wrapper_bridge.stage_into_comfy_input` asks `folder_paths` for the input dir
   -- correct INSIDE the ComfyUI process, where every registered adapter runs.
   This is a separate process: `folder_paths` does not import, the helper falls
   back to a path derived from THIS repo's location, and the file landed in
   `Documents\ComfyUI\input` while the server reads
   `ComfyUI-Installs\ComfyUI\ComfyUI\input`. The graph then failed validation
   against a filename that was not there. Fixed by uploading through the
   server's own `POST /upload/image`, which is right by construction on any
   install layout.
2. **`SaveAudioAdvanced.format` is a `DynamicCombo`, so the API graph takes the
   flat SELECTOR STRING** (`"flac"`), not the dict `execute` receives. The
   executor expands a DynamicCombo exactly as it expands an Autogrow -- selector
   plus dotted sub-inputs, reassembled before the call. Passing the dict is what
   an in-process call would do, and it failed with "missing 1 required
   positional argument: 'format'". **This is lane 20's lesson pointing the other
   way:** there, the in-process call needed the dict the API serializes dotted.
3. **`SaveVideo` reports its mp4 under the `images` key** (with
   `animated: [true]`), not `videos`. The runner keyed on the history key name
   and reported "no video output" for a render that had just succeeded. Now
   classified by FILE EXTENSION, because the key names are a UI convention and
   this one does not mean what it says.

**The general form, recorded as L27:** a graph that is correct in-process is not
automatically correct over the API, and vice versa. The boundary has its own
rules -- where files live, how dynamic inputs serialize, what the history calls
things -- and none of them can be checked by a test that never crosses it.

## THREE DEFECTS THE POST-CODING QA FOUND, all mine, all fixed before the push

1. **A failing `--mux-tts` would have destroyed the receipt.** The mux ran AFTER
   the clip was validated and copied to its durable path but BEFORE the receipt
   was written, with a bare `subprocess.check_call`. A missing ffmpeg or a
   malformed voice raised, and a valid audio-carrying deliverable was left on
   disk with NO receipt beside it -- silently breaking the one thing this
   runner's docstring promises. Now `write_review_mix` never raises: it returns
   `(path, error)`, the error is recorded IN the receipt, and `main` exits **2**
   (distinct from the hard failures' 1) so a missing review copy is loud without
   being fatal to the clip.
2. **Deliverables could be silently overwritten.** The default name carries the
   seed and a one-SECOND timestamp, so two runs with the same seed inside one
   second collide -- and an explicit `--name` collides on every repeat. All
   three writes went straight to `open(..., "wb")`. Now `existing_deliverables`
   refuses by name unless `--overwrite` is passed.
3. **More than one video row from `/history` was silently narrowed** by taking
   `[0]`. This graph has exactly one `SaveVideo`, so a longer list means the
   prompt did something the runner does not model; it now refuses and names the
   count, symmetric with the stream asserts beside it.

**And a fourth, self-inflicted, worth recording because it is L26 AGAIN:** the
first tests written for fixes 1 and 2 were LEXICAL -- `inspect.getsource` plus a
substring, one of them containing a tautological `or True` that could never
fail. Written minutes after L26 was added to the ledger. They were replaced by
BEHAVIOURAL ones: `write_review_mix` is pointed at a nonexistent voice file and
must return an error rather than raise, and `existing_deliverables` has each
file created under it in turn. The lesson needed applying four times in one
session before it stuck.

## Deliberately NOT done here

**No dropdown entry, no `mime overrules TTS/music` wiring, no phase inversion.**
All three are the LATER spec this runner is the test bed for.

**No quality claim about the score.** The clip is machine-proven -- right canvas,
right length, right rate, and a real audio stream from the same generation as the
picture. Whether the music is *good* is the operator's ear, exactly as the lab's
own mime receipts record ("HUMAN VISUAL OK; formal ear fields pending").

**No VRAM number claimed.** The runner does not carry a `VramPeakProbe` (it
drives the server over HTTP rather than in-process), so it reports none rather
than an `nvidia-smi` sample that would be a lower bound (the 2026-08-11 NET
ruling forbids seeding a row from one).

**The `--stem-wav` flag writes FLAC, not WAV.** The name is the spec's; the
format is lossless FLAC because the stem is meant to be reusable material and
FLAC is smaller at identical fidelity. Flagged rather than silently renamed.

## ONE THING THE OPERATOR OWES A RULING ON

`CLAUDE.md` section 0 requires every API/headless run to load
`workflows/otr_canonical.json`, and section 0A carves out exactly two bench
runners -- **"No other runner."** This script submits its own H3 graph, so it
fits neither.

It is authorized by the LATER and more specific ruling of 2026-08-10 (spec P3,
"ships THIS build as a STANDALONE RUNNER ... rendering ONE self-scored mp4 into
a durable deliverables directory with its own receipt"), which is a decision
about this exact script. It also borrows the carve-out's DISCIPLINE: every node
class confirmed live in `/object_info` before submit, the asset resolved from
`/history` and never a glob, and the result ffprobe'd.

**But the carve-out's "no other runner" sentence should be amended to name this
script.** Written up here rather than silently widened, because quietly
expanding a boundary is exactly what that sentence exists to prevent.
