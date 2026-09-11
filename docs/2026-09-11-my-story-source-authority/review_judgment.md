# Finished-code QA judgment

Actual Sonnet5 reviewed the final production/test diff after the last test
revision. Its complete response/manifest remain in sonnet_qa/. Cost about
USD0.0640. Terra independently read the same real files and found no must-fix.
Root finds no demonstrated introduced defect requiring another code revision.
This is not a claim that Sonnet returned an unqualified approval: its findings
and the grounded dispositions are below. No source/test code changed after QA.

| Sonnet observation | Root grounding and disposition |
|---|---|
| Blank-ending cases duplicate | They exercise different required outcomes: whitespace global with nonempty LOCAL FALLBACK versus both global/local empty. The former must retain a target; the latter must stay allowed. Neither is an unused production endpoint. Keep both. |
| Three scope branches duplicate boilerplate | Maintainability consideration accepted. Only two are final scopes; one names a supplied global ending and one permits absent target. Three short static strings keep conditions/data out of system instruction and remain readable. No extra template/helper justified. |
| Future caller scope silently overwritten | Current caller passes journal/scheduler/model only. _pass_act owns the derived scope and receives a private kwargs dict; no shared mutation. Reject an assertion/new refusal for a hypothetical unsupported caller. |
| GLOBAL assertion decorative | It is a fixture sentinel: every nonblank global fixture includes GLOBAL. Interpolating that dynamic ending into scope would make the test fail. Exact target transport is tested separately. This establishes those actual cases, not universal injection immunity. |
| Marker-string suffix assertions fragile/self-referential | Test splits at the last actual static ACT SCOPE delimiter; the marker inside ending precedes it. It then asserts the complete expected endpoint suffix and final Write act N tail, not mere text presence. Static scope must reach correction once, and be byte-identical across cast cases. Formatting changes may properly require updating this prompt-contract test; no nondeterministic parsing observed. |
| Nine/eighteen call counts are brittle | Deliberate fixed-budget contract: six act authors plus interpretation/treatment/frame = nine authors, each one valid correction = eighteen calls. No interstitial model pass. User explicitly forbids unnoticed extra calls; a new pass must trigger review and this test. |
| Bible should say treatment.ending was never provided | Factually wrong: it was already present inside full treatment JSON. The defect is which endpoint was foregrounded. Existing11.39 addition states that accurately and names live06; current receipt provides exact files/diff. Do not invent a transport loss/certain model cause. |
| No check scope survives multiline endpoint | Actual author/corrector captured scopes, target suffix, branch obligations and identical must_speak-axis scopes are asserted in the multiline fixture. These prove routing/preservation in those cases, not semantic response to free-form prose. |
| Other phases need a kwarg-level mock | Real runner's final correction system/context proves act instructions do not reach those phases; effect at the actual slot is stronger than a duplicate mocked-boundary assertion. Default empty instruction is explicit in _call. No demonstrated gap. |

First focused execution found41 test assertion errors: tests incorrectly
expected scope to be the last text in the system prompt, but the real shared
structured owner appends its schema contract. Assertions now verify one exact
scope copy. Production code did not change for this fixture correction.
Final focused suite:246 pass. All45 newly exercised assertions fail against
untouched production1e079681 (temporary final-test overlay then restored).
Full suite/Bible/canonical evidence is recorded separately; no inherited
failure is relabeled green, and no quarantine is added.

R1-R4 actual Opus5/Gemini3.1 Pro review artifacts are under roundtable/.
The R4 Opus response hit its configured token allowance and is incomplete;
root does not infer a missing verdict. Driver convergence is grounded judgment,
not unanimous panel consensus. Cursor previously timed out twice and produced
no review. Do not spend another loop to manufacture assent.

Remaining limit: correct prompt transport and actual correction application do
not compel the model to fix source omissions. The next full canonical episode
must be inspected; a no-op can still be a bad model judgment. No fresh source
qualification or image-fidelity result is claimed by these offline checks.
