# ONE LAUNCH, EVERY ANSWER.
#
# WHY THIS EXISTS
#
# Every measurement session this project has run has been assembled by hand,
# and the same four mistakes have invalidated results over and over:
#
#   1. MEASURING A STALE TRACE. %TEMP%\taa_layer.txt never truncates, so
#      grepping it returns whatever a previous session left behind. Findings
#      were reported three times from a previous run's lines.
#   2. MEASURING A SETTING THAT WAS NEVER APPLIED. The layer holds the ini
#      open; a write can fail silently, and the capture then measures the old
#      value. Every write here is verified by reading it back.
#   3. BLOCK ORDERING. Configs measured one after another in blocks are
#      invalidated by anything that changes mid-run - a view change did
#      exactly that and produced "167% detail retained", which is impossible.
#      Configs are INTERLEAVED so drift hits all of them equally.
#   4. MEASURING WITH AN INSTRUMENT THAT CONTAMINATES. viz modes 4, 5 and 6
#      imageStore into the history buffer, so with any of them enabled the
#      history IS the previous frame's heatmap and every history-derived
#      number saturates by construction. This bench never enables them.
#
# So this drives the whole session itself: it kills the sim, clears the trace,
# launches, waits for the scene, applies each configuration with read-back
# verification, captures interleaved, and hands bench.py a set of files that
# are known to describe one run.
#
#   .\tools\measure\bench.ps1              full bench, ~4 minutes
#   .\tools\measure\bench.ps1 -Quick       parked measurements only, ~90 s
#   .\tools\measure\bench.ps1 -NoLaunch    use the sim already running
#
# The motion section needs the camera to move and this script cannot fly the
# aeroplane. It prompts, then DETECTS whether motion actually happened. If it
# did not, that section is reported VOID rather than printed as a number - the
# distinction between "measured zero" and "could not measure" is the one this
# project has most often lost.

param(
    [switch]$Quick,
    [switch]$NoLaunch,
    [int]$PanSeconds = 40
)

$ErrorActionPreference = 'Stop'
$root  = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$live  = "$env:TEMP\taa_live.ini"
$trace = "$env:TEMP\taa_layer.txt"
$shots = "$env:TEMP\bench"

function Say($s)  { Write-Host $s }
function Head($s) { Write-Host ""; Write-Host "== $s" -ForegroundColor Cyan }

# ---- SET A KEY AND PROVE IT LANDED.
#
# The layer keeps the file open, so a write can lose the race. Retrying until a
# read-back agrees is the difference between measuring the configuration you
# asked for and measuring the one that was already there.
function Set-Key($k, $v) {
    for ($t = 0; $t -lt 30; $t++) {
        try {
            $c = Get-Content $live
            $pat = "^" + [regex]::Escape($k) + "="
            if ($c -match $pat) { $c = $c -replace ($pat + ".*"), "$k=$v" }
            else                { $c += "$k=$v" }
            Set-Content -Path $live -Value $c -Encoding utf8
            $back = Get-Content $live | Where-Object { $_ -match $pat }
            if ($back -eq "$k=$v") { return }
        } catch { Start-Sleep -Milliseconds 100 }
    }
    throw "could not set $k=$v - the layer is holding the settings file"
}

function Shot($name) {
    & "$root\xp-shot.ps1" -Out "$shots\$name.png" -Width 1400 | Out-Null
}

function Wait-Scene($timeoutSec) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        if ((Test-Path $trace) -and
            (Select-String -Path $trace -Pattern 'TAA GATE: scenePass=1' -Quiet)) {
            return $true
        }
        Start-Sleep -Seconds 3
    }
    return $false
}

# ---------------------------------------------------------------- launch
if (Test-Path $shots) { Remove-Item $shots -Recurse -Force }
New-Item -ItemType Directory -Force $shots | Out-Null

if (-not $NoLaunch) {
    Head "launching a clean run"
    Get-Process X-Plane -ErrorAction SilentlyContinue | Stop-Process -Force
    $n = 0
    while ((@(Get-Process X-Plane -ErrorAction SilentlyContinue)).Count -gt 0 -and $n -lt 120) {
        Start-Sleep -Milliseconds 500; $n++
    }
    # The trace is cleared HERE, so everything bench.py reads belongs to this
    # run and nothing else. This is mistake 1 from the header, closed.
    if (Test-Path $trace) { Remove-Item $trace -Force }
    & "$root\tools\measure\settings.ps1" -Restore | Out-Null
    Start-Process -FilePath "cmd.exe" -ArgumentList '/c', "`"$root\dev-run.cmd`""
    Say "  waiting for the scene (up to 6 minutes)"
    if (-not (Wait-Scene 360)) { throw "the scene never came up - nothing can be measured" }
    Say "  scene is up"
    Start-Sleep -Seconds 8      # let the resolve settle before measuring
} else {
    if (-not (Wait-Scene 10)) { throw "no scene in the running sim" }
}

# ---------------------------------------------------------------- parked
# Interleaved rather than blocked: five rounds of on/off rather than all the
# on frames then all the off frames. Anything that drifts - light, cloud, a
# view change - then hits both configurations equally instead of landing
# entirely on one and inventing a difference.
Head "parked image quality (interleaved, 5 rounds)"
Set-Key 'taa.viz' '0'

# ---- WAIT FOR STILLNESS, DO NOT ASSUME IT.
#
# The first run of this bench measured "parked" image quality while the CAMERA
# was moving between captures - consecutive frames differed by 33 of 255, and
# the section still reported a flicker ratio as though the scene were static.
# The comparison stays fair because the configurations are interleaved, but
# the thresholds are calibrated for a still camera and the label was a lie.
#
# So the same rule as the motion section, inverted: measure, do not assume.
Write-Host "  HOLD THE CAMERA STILL" -ForegroundColor Yellow
Write-Host "  (waiting for a steady frame, up to 2 minutes)"
$still = $false
$stillDeadline = (Get-Date).AddMinutes(2)
while ((Get-Date) -lt $stillDeadline) {
    Shot "still_a"
    Start-Sleep -Milliseconds 450
    Shot "still_b"
    $d = & python (Join-Path $PSScriptRoot "bench.py") --framediff `
             "$shots\still_a.png" "$shots\still_b.png"
    if ($LASTEXITCODE -eq 0 -and [double]$d -lt 2.0) {
        Say ("  camera is steady (frame delta {0})" -f $d)
        $still = $true
        break
    }
}
if (-not $still) {
    Say "  camera never settled - the parked section will be labelled as moving"
}
for ($r = 0; $r -lt 5; $r++) {
    Set-Key 'taa.enable' '1'
    Start-Sleep -Milliseconds 1600
    for ($i = 0; $i -lt 3; $i++) { Shot "park_on_${r}_$i" }
    Set-Key 'taa.enable' '0'
    Start-Sleep -Milliseconds 1600
    for ($i = 0; $i -lt 3; $i++) { Shot "park_off_${r}_$i" }
    Say "  round $($r + 1) of 5"
}
Set-Key 'taa.enable' '1'

# ---------------------------------------------------------------- motion
if (-not $Quick) {
    Head "velocity field (needs camera motion)"

    # ---- WAIT FOR MOTION, DO NOT ANNOUNCE AND HOPE.
    #
    # The first version printed PAN NOW and slept. That fails whenever the
    # console is not being watched - and when this is launched in the
    # background its output goes to a file, so the operator never sees the
    # prompt at all. The run then captures 33 frames of a parked aeroplane and
    # the section comes back VOID, which is honest but useless.
    #
    # So the bench watches for the camera to actually move and starts the real
    # capture only then. It cannot fly the aeroplane, but it can wait for it.
    Write-Host "  PAN NOW - hold an arrow key" -ForegroundColor Yellow
    Write-Host "  (waiting for the camera to move, up to 3 minutes)"
    Set-Key 'taa.viz' '0'
    Start-Sleep -Milliseconds 800

    $moved = $false
    $probeDeadline = (Get-Date).AddMinutes(3)
    while ((Get-Date) -lt $probeDeadline) {
        Shot "probe_a"
        Start-Sleep -Milliseconds 450
        Shot "probe_b"
        $delta = & python (Join-Path $PSScriptRoot "bench.py") --framediff `
                     "$shots\probe_a.png" "$shots\probe_b.png"
        if ($LASTEXITCODE -eq 0 -and [double]$delta -gt 2.0) {
            Say ("  motion detected (frame delta {0}) - capturing" -f $delta)
            $moved = $true
            break
        }
    }
    if (-not $moved) {
        Say "  no motion after 3 minutes - velocity section will report VOID"
    }

    # viz=2 is VIZ_MAGNITUDE, which reads the velocity texture. It does not
    # touch history, so unlike viz 4/5/6 it cannot contaminate what it measures.
    Set-Key 'taa.viz' '2'
    Start-Sleep -Milliseconds 1200
    $frames = [int]($PanSeconds / 1.2)
    for ($i = 0; $i -lt $frames; $i++) { Shot "mag_$i" }

    Set-Key 'taa.viz' '0'
    Set-Key 'taa.enable' '0'
    Start-Sleep -Milliseconds 1200
    for ($i = 0; $i -lt 10; $i++) { Shot "flow_off_$i" }
    Set-Key 'taa.enable' '1'
    Say "  motion capture done - you can stop panning"
}

# ---------------------------------------------------------------- vram
Head "vram report"
Set-Key 'vram.report' '1'
Start-Sleep -Seconds 4

# ---------------------------------------------------------------- analyse
Head "analysis"
Copy-Item $trace "$shots\trace.txt" -Force
& python "$root\tools\measure\bench.py" $shots
exit $LASTEXITCODE
