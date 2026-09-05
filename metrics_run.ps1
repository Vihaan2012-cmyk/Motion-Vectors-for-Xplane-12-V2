# metrics_run.ps1 - one measured X-Plane run.
#
# Applies a set of live-ini keys, launches X-Plane with the layer armed, soaks
# for a fixed time, kills it, and returns the aggregate MET line the metrics
# module wrote. This is the unit that sweep.ps1 calls once per configuration;
# run it by hand to sanity-check a single config or to calibrate the depth cut.
#
#   .\metrics_run.ps1 -Seconds 90
#   .\metrics_run.ps1 -Set "taa.contact=0,taa.ao=0" -Seconds 90
#   .\metrics_run.ps1 -Label baseline -Seconds 120
#
# NOTE ON DLSS-D. taaRecordResolve returns early when RR produces the frame, so
# the metrics dispatch - which sits at the tail of the TAA composite - is never
# reached with taa.dlssd=1. This script therefore forces taa.dlssd=0 unless you
# pass -AllowDlssd, so a sweep cannot silently produce "no data" rows and read
# them as "this config was stable".
param(
    [string]$Set = "",
    [string]$Label = "run",
    [int]$Seconds = 90,
    [int]$Report = 300,
    [int]$Windows = 3,
    [switch]$AllowDlssd,
    [switch]$KeepShim
)
$ErrorActionPreference = "Stop"
$MV = $PSScriptRoot
$XP = Split-Path $MV -Parent

function Stop-XP {
    Get-Process "X-Plane*" -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 1500
}

# ---- live ini ---------------------------------------------------------------
# Rewrite in place: the layer polls this file, and every key we do not name
# keeps whatever the previous run left, which is what makes a one-at-a-time
# sweep meaningful.
function Set-IniKeys([hashtable]$kv) {
    $ini = Join-Path $env:TEMP "taa_live.ini"
    if (-not (Test-Path $ini)) { throw "no live ini at $ini - launch X-Plane once first" }
    $lines = @(Get-Content $ini)
    foreach ($k in $kv.Keys) {
        $v = $kv[$k]
        $pat = '^' + [regex]::Escape($k) + '\s*='
        $hit = $false
        for ($i = 0; $i -lt $lines.Count; $i++) {
            if ($lines[$i] -match $pat) { $lines[$i] = "$k=$v"; $hit = $true }
        }
        if (-not $hit) { $lines += "$k=$v" }
    }
    Set-Content -Path $ini -Value $lines -Encoding utf8
}

Stop-XP

# The Streamline shim stays out of a measurement run unless asked for: it arms
# SL, which changes device creation, and a metric that moves because of that is
# not a metric about the setting under test.
if (-not $KeepShim) {
    foreach ($f in @("vulkan-1.dll", "vk_real_mv.dll")) {
        $p = Join-Path $XP $f
        if (Test-Path $p) { Remove-Item $p -Force -ErrorAction SilentlyContinue }
    }
}

$keys = @{
    "taa.metrics"        = 1
    "taa.metrics_report" = $Report
}
if (-not $AllowDlssd) { $keys["taa.dlssd"] = 0 }
if ($Set) {
    foreach ($pair in $Set.Split(",")) {
        $t = $pair.Trim()
        if (-not $t) { continue }
        $eq = $t.IndexOf("=")
        if ($eq -lt 1) { throw "bad -Set entry '$t' (want key=value)" }
        $keys[$t.Substring(0, $eq).Trim()] = $t.Substring($eq + 1).Trim()
    }
}
Set-IniKeys $keys

$out = Join-Path $env:TEMP "mv_metrics.txt"
if (Test-Path $out) { Remove-Item $out -Force -ErrorAction SilentlyContinue }

$env:VK_LAYER_PATH           = Join-Path $MV "build\vklayer"
$env:VK_LOADER_LAYERS_ENABLE = "VK_LAYER_mv"
$env:TAA_VELOCITY            = "1"
$env:TAA_LAYER_TRACE         = "1"
Remove-Item Env:VK_LAYER_SETTINGS_PATH -ErrorAction SilentlyContinue
Set-Content -Path (Join-Path $env:TEMP "taa_dump_every.txt") -Value "0" -Encoding ascii

$trace = Join-Path $env:TEMP "taa_layer.txt"
for ($i = 0; $i -lt 40 -and (Test-Path $trace); $i++) {
    Remove-Item $trace -Force -ErrorAction SilentlyContinue
    if (Test-Path $trace) { Start-Sleep -Milliseconds 250 }
}

Write-Host ("[{0}] {1}" -f $Label, (($keys.GetEnumerator() | Sort-Object Name |
    ForEach-Object { "$($_.Name)=$($_.Value)" }) -join " ")) -ForegroundColor Cyan

$p = Start-Process -FilePath (Join-Path $XP "X-Plane.exe") -WorkingDirectory $XP -PassThru
$deadline = (Get-Date).AddSeconds($Seconds)
$reports = 0
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 3
    if ($p.HasExited) {
        Write-Host ("  X-Plane exited early, code {0}" -f $p.ExitCode) -ForegroundColor Yellow
        break
    }
    # Wait for $Windows report windows, then stop early rather than burn the
    # rest of the soak. Measured: window 1 -> 2 still dropped 22% as scenery
    # streamed in, so two is not enough; the default of three puts the load-in
    # transient two full windows behind the number that gets reported.
    if (Test-Path $out) {
        $reports = @(Select-String -Path $out -Pattern "^MET " -ErrorAction SilentlyContinue).Count
        if ($reports -ge $Windows) { Write-Host ("  {0} windows closed - stopping early" -f $reports) -ForegroundColor Green; break }
    }
}
Stop-XP

if (-not (Test-Path $out)) {
    Write-Host "  NO METRICS OUTPUT - the dispatch never ran." -ForegroundColor Red
    Write-Host "  Check taa_layer.txt for 'METRICS:' lines; the usual causes are" -ForegroundColor Red
    Write-Host "  the layer not loading, or the resolve returning early." -ForegroundColor Red
    exit 1
}

# The LAST MET line is the answer: earlier windows still carry the load-in.
$met = @(Select-String -Path $out -Pattern "^MET " | ForEach-Object { $_.Line })
if ($met.Count -eq 0) { Write-Host "  metrics file exists but has no MET line." -ForegroundColor Red; exit 1 }
Write-Host ("  windows: {0}" -f $met.Count) -ForegroundColor DarkGray
Write-Host ("RESULT[{0}] {1}" -f $Label, $met[-1]) -ForegroundColor Green
Get-Content $out
