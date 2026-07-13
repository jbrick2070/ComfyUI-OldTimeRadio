<#
  Positive ownership selectors for the canonical OTR headless harness.

  A ComfyUI process is safe for the harness to stop only when it carries the
  canonical headless model-path configuration.  Port number alone is not
  ownership: the operator may have an interactive ComfyUI server on any port.
#>

function Test-OtrHeadlessServerCommand {
    param([AllowEmptyString()][string]$CommandLine)

    if (-not $CommandLine) { return $false }
    return (
        $CommandLine -match '(?i)(?:^|[\\/])main\.py(?:$|[\s"])' -and
        $CommandLine -match '(?i)--port\s+\d+\b' -and
        $CommandLine -match '(?i)_otr_headless_model_paths\.yaml'
    )
}


function Test-OtrCanonicalRunnerCommand {
    param([AllowEmptyString()][string]$CommandLine)

    if (-not $CommandLine) { return $false }
    return $CommandLine -match '(?i)(?:^|[\\/])otr_canonical_api_run\.py(?:$|[\s"])'
}


Export-ModuleMember -Function Test-OtrHeadlessServerCommand, Test-OtrCanonicalRunnerCommand
