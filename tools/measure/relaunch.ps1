# Stop everything, then start exactly one X-Plane.
#
# WHY THIS EXISTS
#
# Ad-hoc relaunches written inline kept leaving two sims running. The pattern
# that fails is
#
#     Stop-Process -Force ; Start-Sleep -Seconds 2 ; Start-Process ...
#
# and it fails two ways at once. Two seconds is not reliably long enough for
# X-Plane to exit, so the new instance starts while the old one is still up;
# and the launcher SHELL - a cmd.exe running dev-run.cmd, plus the
# resume-flight.ps1 it spawns - survives independently of the sim, so those
# accumulate across every relaunch until something restarts a sim nobody asked
# for.
#
# Two instances is not merely untidy. They fight over the layer trace, the
# settings file and the shared memory, so every measurement taken afterwards is
# describing an unknown mixture of two runs - which has already invalidated
# results in this project more than once.
#
# So: kill the shells first, kill the sims, POLL until the count is actually
# zero rather than sleeping and hoping, and refuse to start if it never
# reaches zero.
#
#   .\tools\measure\relaunch.ps1              normal
#   .\tools\measure\relaunch.ps1 -Validate    with the Vulkan validation layer
#   .\tools\measure\relaunch.ps1 -NoStart     stop everything and leave it down

param(
    [switch]$Validate,
    [switch]$NoStart,
    [switch]$KeepTrace
)

$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent

# ---- shells first: they are what can start a sim behind our back
$shells = @(Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -like '*dev-run*' })
foreach ($s in $shells) { Stop-Process -Id $s.ProcessId -Force -ErrorAction SilentlyContinue }

$resume = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -like '*resume-flight*' })
foreach ($s in $resume) { Stop-Process -Id $s.ProcessId -Force -ErrorAction SilentlyContinue }

# ---- then the sims, and WAIT for zero rather than assuming
Get-Process X-Plane -ErrorAction SilentlyContinue | Stop-Process -Force
$n = 0
while ((@(Get-Process X-Plane -ErrorAction SilentlyContinue)).Count -gt 0 -and $n -lt 120) {
    Start-Sleep -Milliseconds 500
    $n++
}
$left = @(Get-Process X-Plane -ErrorAction SilentlyContinue).Count
if ($left -gt 0) {
    throw "$left X-Plane instance(s) would not exit - refusing to start another"
}
Write-Host "  all stopped ($($shells.Count) shell(s), waited $([int]($n/2))s)"

if ($NoStart) { return }

# The trace never truncates, so anything read after this would otherwise be a
# mixture of this run and every previous one.
if (-not $KeepTrace) {
    $t = "$env:TEMP\taa_layer.txt"
    if (Test-Path $t) { Remove-Item $t -Force }
    $v = "$env:TEMP\mv_validation.txt"
    if (Test-Path $v) { Remove-Item $v -Force }
}

$args = @('/c', "`"$root\dev-run.cmd`"")
if ($Validate) { $args += 'validate' }
Start-Process -FilePath 'cmd.exe' -ArgumentList $args

# Confirm exactly one came up, rather than reporting success on faith.
$n = 0
while ((@(Get-Process X-Plane -ErrorAction SilentlyContinue)).Count -lt 1 -and $n -lt 120) {
    Start-Sleep -Milliseconds 500
    $n++
}
$up = @(Get-Process X-Plane -ErrorAction SilentlyContinue).Count
if ($up -eq 1) { Write-Host "  started 1 X-Plane$(if ($Validate) { ' (validation ON)' })" }
else           { Write-Host "  WARNING: $up X-Plane instances are running" }
