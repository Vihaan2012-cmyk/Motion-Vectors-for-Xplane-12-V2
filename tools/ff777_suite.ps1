# ff777_suite.ps1 - one-command diagnostic flight for the FlightFactor 777.
#
# Arms everything the 777 investigation needs, launches X-Plane, waits for it to
# close, then collects and analyses. You fly: ~10 s parked, ~90 s fast and low in
# the cockpit view with a few turns, ~90 s the same in an external view. Done.
#
#   powershell -ExecutionPolicy Bypass -File tools\ff777_suite.ps1
#
# What it arms for the session (all undone when the sim closes):
#   TAA_KEEP_ON_777=1     the resolve runs on the 777 instead of standing down
#   TAA_DATUM_CHECK=1     the plugin's body/world matrix self-checks in its log
#   TAA_MV_PID=1          every injected draw tags the velocity target with its
#                         pipeline id (in the depth channel; depth comes from the
#                         capture's own depth plane instead)
#   TAA_MV_DIAG=<diag>    the pid table lands beside the shaders
#   TAA_DUMP_SHADERS=<diag>\shaders   every shader module as <hash>.spv + pipelines.csv
#   nn.capture=1, every 60 frames a 4-frame burst with all planes, tag ff777
#
# Afterwards: the layer trace and plugin log are copied into <diag>, every .spv is
# disassembled with spirv-dis when the Vulkan SDK is present, and
# tools\nn\diag777.py writes <diag>\report.md.
#
# Copyright (C) 2026 MotionVectors contributors. SPDX-License-Identifier: GPL-3.0-or-later

$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent                       # MotionVectors
$xp   = Split-Path $root -Parent                               # X-Plane 12
$stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$diag = "E:\NNCap\ff777_diag_$stamp"
$capRoot = "E:\NNCap"
New-Item -ItemType Directory -Force "$diag\shaders" | Out-Null

if (Get-Process -Name "X-Plane" -ErrorAction SilentlyContinue) {
    Write-Host "X-Plane is running - close it first." -ForegroundColor Yellow; exit 1
}

# A staged layer build (MV_LAYER_OUT) becomes live now that nothing holds the DLL.
$live = "$root\build\vklayer\VkLayer_mv.dll"; $staged = "$root\build\vklayer\VkLayer_mv.staged.dll"
if ((Test-Path $staged) -and ((Get-Item $staged).LastWriteTime -gt (Get-Item $live).LastWriteTime)) {
    Copy-Item $staged $live -Force; Write-Host "  staged layer swapped in ($((Get-Item $live).LastWriteTime))"
}

# ---- live ini: the capture profile. Keys are replaced in place or appended.
$ini = Join-Path $env:TEMP "taa_live.ini"
$want = @{
    "nn.capture" = "1"; "nn.capture_every" = "60"; "nn.capture_burst" = "4"; "nn.capture_gbuf" = "1";
    "nn.capture_dir" = $capRoot; "nn.capture_tag" = "ff777"; "nn.capture_max_gb" = "80";
    "nn.sweep_time" = "0"; "taa.nn" = "0"; "taa.fg" = "0"
}
$lines = @(); if (Test-Path $ini) { $lines = Get-Content $ini }
$seen = @{}
$out = foreach ($l in $lines) {
    $m = [regex]::Match($l, '^([A-Za-z0-9_.]+)=')
    if ($m.Success -and $want.ContainsKey($m.Groups[1].Value)) { $seen[$m.Groups[1].Value] = $true; "$($m.Groups[1].Value)=$($want[$m.Groups[1].Value])" } else { $l }
}
foreach ($k in $want.Keys) { if (-not $seen[$k]) { $out += "$k=$($want[$k])" } }
Set-Content $ini $out -Encoding ASCII
Write-Host "  live ini armed: capture every 60 frames, 4-frame bursts, tag ff777 -> $capRoot"

# ---- environment for the sim process
$env:TAA_KEEP_ON_777  = "1"
$env:TAA_DATUM_CHECK  = "1"
$env:TAA_MV_PID       = "1"
$env:TAA_MV_DIAG      = $diag
$env:TAA_DUMP_SHADERS = "$diag\shaders"
$env:TAA_GPU_TIMING   = "0"

$before = @(Get-ChildItem "$env:TEMP\taa_layer_*.txt" | Select-Object -ExpandProperty Name)
Write-Host ""
Write-Host "Launching X-Plane. Load the 777, then:" -ForegroundColor Cyan
Write-Host "   ~10 s parked, ~90 s fast+low with turns in the COCKPIT view, ~90 s the same EXTERNAL, then quit the sim." -ForegroundColor Cyan
$p = Start-Process -FilePath "$xp\X-Plane.exe" -WorkingDirectory $xp -PassThru
$p.WaitForExit()
Write-Host "X-Plane closed. Collecting..."

# ---- undo the capture profile
$lines = Get-Content $ini
$lines = $lines -replace '^nn\.capture=1', 'nn.capture=0'
Set-Content $ini $lines -Encoding ASCII

# ---- trace + plugin log
Get-ChildItem "$env:TEMP\taa_layer_*.txt" | Where-Object { $before -notcontains $_.Name } | ForEach-Object { Copy-Item $_.FullName $diag -Force; Write-Host "  trace: $($_.Name)" }
$plog = "$xp\Resources\plugins\MotionVectors\MotionVectors.txt"
if (Test-Path $plog) { Copy-Item $plog "$diag\MotionVectors_plugin.txt" -Force }
Copy-Item "$xp\Log.txt" "$diag\XPlane_Log.txt" -Force -ErrorAction SilentlyContinue

# ---- disassemble the shader dump
$sdk = (Get-ChildItem "C:\VulkanSDK\*" -Directory -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1).FullName
$dis = if ($sdk) { Join-Path $sdk "Bin\spirv-dis.exe" } else { $null }
$n = 0
if ($dis -and (Test-Path $dis)) {
    foreach ($f in Get-ChildItem "$diag\shaders\*.spv") {
        & $dis $f.FullName -o ($f.FullName -replace '\.spv$', '.txt') 2>$null
        $n++
    }
}
Write-Host "  shaders: $n modules dumped and disassembled in $diag\shaders"

# ---- the newest ff777 capture session
$sess = Get-ChildItem "$capRoot\*_ff777" -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
if ($sess) {
    Write-Host "  capture: $($sess.FullName) ($((Get-ChildItem $sess.FullName -Directory).Count) frames)"
    $py = "C:\Users\bansa\AppData\Local\Python\pythoncore-3.14-64\python.exe"
    if (-not (Test-Path $py)) { $py = "python" }
    & $py "$root\tools\nn\diag777.py" --session $sess.FullName --diag $diag --xplane $xp
    Write-Host ""
    Write-Host "Report: $diag\report.md" -ForegroundColor Green
} else {
    Write-Host "  no ff777 capture session found under $capRoot - was the 777 loaded and the resolve running?" -ForegroundColor Yellow
}
