# test.ps1 - one entry point for building, flying, and reading results.
#
# Everything a test session needs, encoded once - including the lessons that
# cost whole days when they were missing:
#   * TAA_VELOCITY must be EXACTLY "1" or the velocity pass is disarmed and
#     every field reads zero ("observation only" in the trace).
#   * The live file (%TEMP%\taa_live.ini) overrides env; check it before
#     blaming code.
#   * %TEMP%\taa_dump_every.txt overrides TAA_VELOCITY_DUMP; a stale 0 there
#     silently disables the readback.
#   * Exactly ONE plugin copy may exist under Resources\plugins\MotionVectors -
#     a nested duplicate publishes a dead share and zeroes the panel.
#
# Usage:
#   .\test.ps1 build                  # build + install everything
#   .\test.ps1 launch                 # start XP with layer + velocity armed
#   .\test.ps1 launch -Taa            # ...and TAA resolve enabled with viz off
#   .\test.ps1 launch -Viz 2          # ...TAA on, viz heatmap (1=dir 2=mag)
#   .\test.ps1 baseline               # start XP with NO layer at all
#   .\test.ps1 kill                   # stop the sim
#   .\test.ps1 status                 # process, share, armed?, gate, coverage
#   .\test.ps1 trace [-Match GATE]    # tail the layer trace (optional filter)
#   .\test.ps1 live key value         # set a live-file key (applies in-sim)
#   .\test.ps1 park                   # TAA off, viz off, dump off - safe state

param(
    [Parameter(Position = 0)] [string]$Action = "status",
    [Parameter(Position = 1)] [string]$Key,
    [Parameter(Position = 2)] [string]$Value,
    [switch]$Taa,
    [int]$Viz = -1,
    [string]$Match
)

$root  = Split-Path -Parent $MyInvocation.MyCommand.Path   # MotionVectors\
$xp    = Split-Path -Parent $root                          # X-Plane 12\
$live  = Join-Path $env:TEMP "taa_live.ini"
$trace = Join-Path $env:TEMP "taa_layer.txt"

function Set-LiveKey([string]$k, [string]$v) {
    if (-not (Test-Path $live)) { Set-Content $live "" -Encoding ascii }
    $c = Get-Content $live
    if ($c -match "^$([regex]::Escape($k))=") {
        $c -replace "^$([regex]::Escape($k))=.*", "$k=$v" | Set-Content $live -Encoding ascii
    } else {
        Add-Content $live "$k=$v"
    }
    Write-Host "live: $k=$v"
}

switch ($Action) {

    "build" {
        & (Join-Path $root "build.ps1")
    }

    "launch" {
        $env:VK_LAYER_PATH           = Join-Path $root "build\vklayer"
        $env:VK_LOADER_LAYERS_ENABLE = "VK_LAYER_mv"
        $env:TAA_LAYER_TRACE         = "1"
        $env:TAA_VELOCITY            = "1"     # exactly "1" - see header
        if (Test-Path $trace) { Clear-Content $trace }
        if ($Taa -or $Viz -ge 0) { Set-LiveKey "taa.enable" "1" } else { Set-LiveKey "taa.enable" "0" }
        if ($Viz -ge 0) { Set-LiveKey "taa.viz" "$Viz" } else { Set-LiveKey "taa.viz" "0" }
        Start-Process -FilePath (Join-Path $xp "X-Plane.exe") -WorkingDirectory $xp
        Start-Sleep 8
        if (Get-Process -Name "X-Plane" -ErrorAction SilentlyContinue) {
            Write-Host "X-Plane up: layer + velocity armed$(if ($Taa -or $Viz -ge 0) { ', TAA on' })$(if ($Viz -ge 0) { ", viz=$Viz" })"
        } else { Write-Host "X-Plane did not start" }
    }

    "baseline" {
        Remove-Item Env:VK_LAYER_PATH, Env:VK_LOADER_LAYERS_ENABLE -ErrorAction SilentlyContinue
        Start-Process -FilePath (Join-Path $xp "X-Plane.exe") -WorkingDirectory $xp
        Start-Sleep 8
        if (Get-Process -Name "X-Plane" -ErrorAction SilentlyContinue) {
            Write-Host "X-Plane up: NO layer (baseline)"
        } else { Write-Host "X-Plane did not start" }
    }

    "kill" {
        Get-Process -Name "X-Plane" -ErrorAction SilentlyContinue | Stop-Process -Force
        Write-Host "killed"
    }

    "status" {
        $p = Get-Process -Name "X-Plane" -ErrorAction SilentlyContinue
        if ($p) { Write-Host "sim: PID $($p.Id) ($([int](((Get-Date) - $p.StartTime).TotalMinutes)) min)" }
        else    { Write-Host "sim: not running" }
        # One nested-duplicate check saves a day of ghost-share debugging.
        $dup = Join-Path $xp "Resources\plugins\MotionVectors\MotionVectors"
        if (Test-Path $dup) { Write-Host "WARNING: duplicate plugin at $dup - dead share incoming" }
        if (Test-Path $trace) {
            foreach ($pat in @("VEL: velocity pass", "SHARE: attached",
                               "MV STICKY", "MV COVERAGE", "gateDepth=")) {
                $l = Select-String -Path $trace -Pattern $pat | Select-Object -Last 1
                if ($l) { Write-Host $l.Line }
            }
        } else { Write-Host "no trace file yet" }
    }

    "trace" {
        if (-not (Test-Path $trace)) { Write-Host "no trace"; break }
        if ($Match) { Select-String -Path $trace -Pattern $Match | Select-Object -Last 20 | ForEach-Object { $_.Line } }
        else        { Get-Content $trace -Tail 25 }
    }

    "live" {
        if (-not $Key) { Get-Content $live; break }
        Set-LiveKey $Key $Value
    }

    "park" {
        Set-LiveKey "taa.enable" "0"
        Set-LiveKey "taa.viz" "0"
        Set-Content (Join-Path $env:TEMP "taa_dump_every.txt") "0" -Encoding ascii
        Write-Host "parked: TAA off, viz off, readback off"
    }

    default { Write-Host "actions: build launch [-Taa] [-Viz n] baseline kill status trace [-Match x] live <k> <v> park" }
}
