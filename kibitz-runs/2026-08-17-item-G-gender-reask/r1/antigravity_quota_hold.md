# Antigravity Quota Hold

Antigravity failed on quota/credit/rate-limit usage. Suggested retry after 1h (2026-08-17T15:36-07:00).

Kibitz detected provider quota, credit, or rate-limit markers for this lane.
The active driver should acknowledge this to the user and ask when to retry.
If the user does not choose a time, use the suggested retry window above.

To change the built-in retry window, set `KIBITZ_QUOTA_RETRY_AFTER`
(examples: `30m`, `1h`, `4h`, `1d`) before rerunning Kibitz.

## Diagnostic

Antigravity quota/backend exhaustion detected in recent CLI log C:\Users\jeffr\.gemini\antigravity-cli\log\cli-20260817_141301.log:
"code": 429,
"message": "Resource has been exhausted (e.g. check quota).",
"status": "RESOURCE_EXHAUSTED"
ERROR: logging before google.Init: E0817 14:13:02.915837       1 cli_setting_manager.go:841] failed to propagate telemetry setting: RESOURCE_EXHAUSTED (code 429): Resource has been exhausted (e.g. check quota).
