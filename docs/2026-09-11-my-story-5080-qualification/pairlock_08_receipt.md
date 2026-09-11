# Canonical recovery attempt08: Gemma treatment/frame collision

Full canonical on7b41a7fd19ae919044fb397ac0435d3184bd00dc failed during
treatment, prompt8c575ba7-94d1-40b7-b9e8-b43ffc9c9a73. Server351.85seconds,
runner352seconds. Installed google/gemma-4-12b-it in both writer slots,
NF4/SDPA; same dinner source, one act, two-character hint, blank byline,
otr_w45_still_pan and sampling as07. The actual graph differs only in the
two writer model selections; root and Luna independently checked it. Code,
qualified file hashes and canonical stayed unchanged throughout.

## Prequeue and terminal evidence

The first dry-run used the bare Gemma ID and failed COMBO validation before
queuing or generating anything. Root corrected only the temporary wrapper's
input to the exact observed dropdown label:
google/gemma-4-12b-it (23.9 GB, nv16 nv24).
Prequeue request/logs are retained separately; this is not a ninth story attempt.
The committed proposed wrapper remains historical; pairlock_08_wrapper_source.txt
is the actual corrected wrapper used. No production code or test changed.

The full run loaded the real workflows/otr_canonical.json through the shipped
runner, with no replay, alternate graph or partial target. Terminal history,
runner/server/watchdog logs, actual request/prompt and failed ledger are saved.
Ledger: C:/Users/jeffr/Documents/ComfyUI/output/otr/episodes/pending_20260911_064311/audio/pending_20260911_064311_ledger.json
SHA256:7fd01add3a33a4f50c63313eb9e952397c08e96b2dcaa08f0d4c3b508cd6fe49

No audio/video/publication exists for this attempt. There are zero authored
ledger rows, no accepted treatment/act/cast, no freeze and no media dispatch.
History has no episode media outputs, so the wrapper's media-correlation check
correctly remains false and exits after preserving all evidence. The server log
explicitly records this episode's skeleton creation and final failure save at
lines222-223/300 within the single queued run. Do not use the prior episode's
media or interpret a latest-ledger candidate alone as a successful correlation.

## Measured model behavior

Gemma loaded its real local Gemma4Unified checkpoint under Transformers5.10.4;
preflight resolved the23,919,549,408byte model blob. Live snapshot uses the
loaded decoder's native131072 context, not a claim based on catalog8192. The
source-interpretation correction records2178prompt tokens and fits=true.
All four generations ended naturally on106 with EOS IDs[1,106,50]; returned
tokens638/638/801/800. No capacity refusal, OOM, decoder truncation or provider
failure is recorded. Two author-P0/P1 operations plus one source correction
produce four actual calls:1P0,1P0-source,2P1. P1 allows at most3 but the existing
typed-repair ladder ends after the failed second attempt; no budget reset.

P0 and its source rewrite preserve girlfriend_mention as required, but include
Announcer as required speaking named_cast and explain planned3 as including
the non-diegetic frame. P0 omits explicit mother-response/current-appreciation
requirement rows; full raw source remains intact. Counts requested1act/2dramatic
characters, proposed1/3; accepted/actual counts remain null after failure.

Both complete P1 drafts include Jeffrey, Mother and ANNOUNCER. Both begin act1
turns with an ANNOUNCER opening and set global ending to an ANNOUNCER closing
thought. Their local ending_state instead correctly keeps Jeffrey and Mother
at the shared meal. The requested girlfriend mention, Mother's warm response
and current appreciation are represented in planned dramatic turns, not yet
realized dialogue. Attempt2 changes punctuation but repeats all three frame
leaks. Existing post-validation fails twice with:
ANNOUNCER is reserved for the frame; give story characters distinct names.
P1 source correction never runs because authoring never reaches acceptance.

## Grounded diagnosis and next work

Root and Terra independently traced phase ownership. P0 names people who should
have lines without a frame carve-out. P1's cast instruction excludes ANNOUNCER,
but the treatment's other fields do not clearly exclude frame material. The
repair error asks for distinct character names although Jeffrey and Mother are
already distinct; the problem is frame ownership. P3 StoryFrame and assembly
already own the announcer and its open/close/coda. P4 creates c01 independently.
Merely deleting the cast entry would leave frame turns and an ANNOUNCER global
ending, which the final-act endpoint now foregrounds. Do not blindly strip,
rename, alias or add a source gate. Review the existing P0/P1/P3 boundary and
make the current repair rewrite the actual dramatic treatment within its budget.
No implementation is yet claimed for this new failure. Source fidelity remains
unqualified; planned correct facts do not count as a published story.

After archival, root selectively stopped verified ComfyUI processes37700/34028.
Port8000 is empty; GPU returned to2591then2628MiB desktop usage, versus2628
before boot. Watchdog DONE/RESULT FAIL reports termination, not success. No code
edit during generation. Mac/4060 held/no contact; RunPod no auth/no rental.

Eight full canonical attempts across revisions/families: four writing failures,
four source-defective publications, zero source-qualified. FourQwen/threeNemo/
oneGemma. Preserve all failures; no next generation before the ownership repair
is scoped, coded, reviewed by Sonnet and regression-qualified. Broader source,
Jeffrey/Codex act/model stress, Original credits and actual listening remain
open. GO_FORWARD is the sole remaining-work queue.
