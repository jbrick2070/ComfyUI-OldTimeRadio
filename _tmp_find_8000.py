"""Find the :8000 Comfy process without printing secrets."""
from __future__ import annotations

import subprocess


def main() -> int:
    raw = subprocess.check_output(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Where-Object { $_.CommandLine -match '8000' } | "
                "Select-Object ProcessId, CommandLine | "
                "Format-List"
            ),
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in raw.splitlines():
        # Never echo a line that might carry an API key.
        if "api_key" in line.lower() or "OTR_COMFY" in line:
            print("CommandLine: [redacted env-ish line]")
            continue
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
