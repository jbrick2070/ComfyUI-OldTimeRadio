# Correction comment for Comfy-Org/registry-backend#224

**PREPARED, NOT POSTED.** The original post is already public; this corrects two
claims in it that our own audit found to be wrong. Posting is yours.

**Why post at all rather than quietly fix it:** the post asks a reviewer to
trust our account of what the pack executes. One of its sentences is false and
is checkable in thirty seconds by grepping the zip. A reviewer who finds it
themselves discounts everything else we wrote; a publisher who corrects it
first is worth listening to. That asymmetry is the whole argument.

---

Correcting two claims in my own post above, both found by a follow-up audit of
the published `2.0.0-alpha.20` archive rather than the source tree.

**1. "Every `subprocess` call goes through one gateway" is wrong.** It is very
nearly true and I should have written what was actually measured. Every ffmpeg
and ffprobe spawn does go through `nodes/_otr_shared/proc.py`, which refuses
`shell=True`, refuses a string argv, and allowlists `argv[0]`. But
`nodes/_otr_audio_engines/eng_indextts2.py:214` calls `subprocess.Popen`
directly and bypasses that gateway. Its argv is a fixed
`[interpreter, worker.py, --model-dir, ...]` shape whose elements come from
the operator's environment configuration, never from a widget or a request,
and the file is content-hashed by the pack's own voice-route
fingerprint so it cannot be edited without demoting the route -- which is also
precisely why it was not migrated. That is a reason, not an exemption, and the
sentence as published overstated the guarantee.

**2. "no longer returns an absolute path" is overstated** for the one
unauthenticated route we still register, `GET /otr/latest_ledger`. The outer
`fullpath` field and the raw exception text were removed in alpha.20, and that
part is accurate. But the response still returns the ledger document itself,
and that document is full of absolute paths -- 75 of them in this morning's
episode ledger: the nine directory and file paths under `meta.paths`
(`ledger_path`, `episode_root`, `audio_dir`, the stills / portraits / videos /
composited dirs, `obs_dir`, `obs_final`), every still's `path` and cache
`pool_path`, the music-cue WAVs, and the final audio, video and publication
targets. So the route still discloses the operator's directory tree to anyone
who can reach it. The route takes no request parameter, has no side effect,
and reads a server-chosen file -- but "no absolute paths" was not a correct
description and I would rather say so than have it read as one.

**Two packaging defects the same audit turned up, both fixed for the next
version:**

- `scripts/otr_mesh_stage_blender.py` is required at runtime by
  `nodes/_otr_video_engines/eng_mesh_stage.py` and was excluded from the
  published archive, so the mesh/3D lane could not start on a registry install.
- `scripts/_otr_indextts2_install.ps1` shipped in the archive and bootstrapped a
  dependency with `irm https://astral.sh/uv/install.ps1 | iex`. Fetching and
  executing a remote script has no business in a package under security review,
  regardless of how well-known the host is. It now refuses and prints the manual
  install command instead. The weights-download helper that installer calls at
  its last step (`scripts/_otr_idx_download_weights.py`, a `snapshot_download`
  wrapper) was excluded the same way, so the one installer we did ship pointed
  at a file that was not there; it ships beside the installer now.

alpha.20's scan record stands at 12 `info` findings. The next version adds two
small helper scripts to the archive, so I will read and state its record when
it lands rather than predict it here. I am flagging these because they are the
kind of thing a reviewer should not have to find, and because the first two
mean my summary above was more confident than the code justified.

---

## Notes for the operator (not part of the post)

Reviewed 2026-09-05 against the real tree before posting. Every claim in the
post was checked against a file:

* Claim 1: `nodes/_otr_audio_engines/eng_indextts2.py:209-216` builds
  `[py, worker, "--model-dir", model_dir]` (+ `--fp16`) from
  `OTR_INDEXTTS2_VENV` / `OTR_INDEXTTS2_WORKER` / `OTR_INDEXTTS2_DIR` and calls
  `subprocess.Popen` directly. It is the ONLY direct spawn under `nodes/` and
  `__init__.py`; every other spawn imports `otr_proc` (18 files). The six
  `scripts/` files that ship contain no spawn at all. The file is listed in
  `_otr_voice_route.RUNTIME_FINGERPRINT_SOURCES["indextts2"]`.
* Claim 2: `__init__.py:465-517` returns `"ledger": ledger` with the whole
  document; `fullpath` and the exception text are gone. `_otr_ledger.py:262-290`
  writes nine absolute paths into `meta.paths`; a grep of this morning's
  `signal_lost_the_caretakers_clause_20260905_061224` ledger counts **75**
  values beginning `C:`. The draft's "three keys" enumeration was corrected to
  that measurement -- a reviewer who curls the route would have seen 75 and
  read "three" as another understatement.
* Packaging: `eng_mesh_stage.py:84` joins `scripts/otr_mesh_stage_blender.py`;
  `.comfyignore` now negates it, and the script has zero subprocess / exec /
  eval / network / `os.environ` hits. `_otr_indextts2_install.ps1` now throws
  when `uv` is absent instead of piping the remote installer.
* The "does not change the scan record" sentence was softened to a promise to
  read the next record. Reason: `.comfyignore` ALSO re-includes
  `scripts/_otr_idx_download_weights.py`, which reads `os.environ` twice
  (`:70`, `:73`) and calls `huggingface_hub.snapshot_download`. On a per-file
  env rule that is at least one new `info` finding, so "still 12" was a
  prediction the post should not make. **Open decision, yours:** ship the helper
  (current tree; the record moves, and the post now says so), or keep it out
  and have the installer do the weights step through the venv's own
  `hf download` (no new Python file, no new finding, but the helper's size /
  manifest validation would have to move into the .ps1).
