# Finished-diff QA: shared grammar state and MyStory P1 connection

Read actual Windows git diff against e07476eb; do not edit. Six production files:
OTR_LedgerScriptWriter, _otr_constrained_generate, _otr_decode_guard,
_otr_generation_budget, _otr_structured_call, _otr_my_story. Four test files
extend actual constraints, decode guard, shared repairs and MyStory runner.
Full design arc converged in my-story-cross-machine/r4. This is one CLI QA.

Expected: cache only tokenizer identity/preprocessing, drop warm by_schema.
Fresh parser/prefix for each ACTUAL model.generate including min_p retry; no
stateful closure attributes. LMFE config0 before construction AND after prefix
builder, retaining alphabet and explicit maxItems. Real root/nested25items and
explicit2/25caps tested. Same closures/retries/errors have fresh histories;
weakref tests prove no resident history or dynamic-schema retention. min_p warning
uses str(exc) so buffered log records do not retain its generation traceback.

ProviderCapacityMessages marks unbounded JSON fields. Both bound transports
snapshot before normalization; None disables only open-string tracking, preserves
cycle detection. Default bound calls keep their guard. Long varying field and
actual cycle controls, normalization marker and true halt-reason/evidence tests.
Generic string repair inherits original custom list subtype and instance attrs;
plain-string behavior stays. No bank import in shared helpers.

Only MyStory P1 binds the existing _otr_bind_schema(StoryTreatment) once locally;
all three attempts use that callable. P2/P3 remain original. Failed transport
raw_completion is durable SEPARATE from raw_output; final proposal counts only
inspect actual returned raw_output. Binder-once/retry/failclosed and saved ledger
evidence tests cover it; existing scheduler tests cover call accounting.

Focused197passed. Full running against tmp/my_story_diagnostics_verified_full.xml
baseline14135pass/53fail/183skip/1xfail. No new known-fail quarantine. Native
capacity correction is NEXT chunk: MIN_OUTPUT_TOKENS remains64, no new fit probe
yet. No adaptive organization, source checker, cleaner edits or Mac fix claimed.
No workflow surface change; actual canonical hash unchanged. No live media now.

Ground concrete bugs; do not demand the later A2/F1/C1 work in this diff. Static
M1 is not proved cause/cure of Mac OS kill, whose current MyStory cleanup is
unbound. PBUG-20260910-05 is the existing live binder-routing failure; any Bible
promotion must be for that live-backed contract, not a fabricated M1 production
incident. Ignore inherited diff.txt/diff_utf8.txt and unrelated review artifacts.
