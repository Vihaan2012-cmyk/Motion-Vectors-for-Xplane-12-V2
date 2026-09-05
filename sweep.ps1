# sweep.ps1 - rank MotionVectors configurations by measured instability.
#
# Answers one question: which setting, or which combination of settings, leaves
# the image least stable? It does that by measuring, not by reasoning.
#
# EVERY RUN FLIPS ONE BIT FROM A KNOWN STATE. Each toggle carries its baseline
# value and its flipped value, and EVERY run writes ALL toggles to baseline
# before applying its own flip. metrics_run.ps1 only rewrites the keys it is
# handed, so without this a flip from run N would leak into run N+1 and the
# whole table would be measuring accumulated state, not one setting. The first
# version of this script had exactly that bug, and also "turned off" four
# features that were already off - four runs of pure noise.
#
# THE NOISE FLOOR COMES FIRST. The baseline is run -BaselineRuns times before
# anything is toggled; the spread across those identical runs is the floor.
# Any toggle whose effect is smaller than that is reported as NO EFFECT rather
# than ranked. Measured 2026-09-05: three baselines spanned 0.0045.
#
# NOTHING IS LOST. Every result is appended to sweep-results.jsonl as it
# lands. -Resume skips configs already measured, so a killed sweep picks up
# where it stopped, and -CombosOnly / -TableOnly work from the file.
#
#   .\sweep.ps1 -NoCombos                 # phase 1 + every single flip -> table
#   .\sweep.ps1 -CombosOnly -Resume       # then every PAIR of movers
#   .\sweep.ps1 -TableOnly                # re-render the table from the file
#   .\sweep.ps1 -Only "taa.ao,taa.gi"     # a subset
param(
    [int]$Seconds = 200,
    [int]$Report = 200,
    [int]$BaselineRuns = 3,
    [switch]$NoCombos,
    [switch]$CombosOnly,
    [switch]$TableOnly,
    [switch]$Resume,
    [string]$Only = "",
    # Measured on the calibration run: reverse-Z, cleared/far = 0, and the
    # cockpit region of the engine depth buffer reads as cleared, so this one
    # cut removes sky AND shell and leaves the world seen through the windows.
    [string]$Mask = "taa.metrics_mask=2,taa.metrics_far=0.0001",
    [string]$ResultsFile = ""
)
$ErrorActionPreference = "Stop"
$MV = $PSScriptRoot
if (-not $ResultsFile) { $ResultsFile = Join-Path $MV "sweep-results.jsonl" }

# ---- what gets dissected ----------------------------------------------------
# base = the value in the live ini on 2026-09-05 (or the code default where the
# key was absent); flip = the other state. Strengths flip to their code
# default, which is what "on" means for them.
$AllToggles = @(
    @{ k = "taa.contact";             base = "0";   flip = "0.6";  note = "contact shadows (shipped strength)" },
    @{ k = "taa.ao";                  base = "0";   flip = "0.5";  note = "ambient occlusion (strength)" },
    @{ k = "taa.sharpen";             base = "0";   flip = "0.5";  note = "post-resolve sharpen (shipped strength)" },
    @{ k = "taa.gi";                  base = "0";   flip = "1";    note = "screen-space GI" },
    @{ k = "taa.reactive";            base = "0";   flip = "1";    note = "reactive mask" },
    @{ k = "taa.engine_depth";        base = "0";   flip = "1";    note = "engine depth for the resolve" },
    @{ k = "taa.clear_after_resolve"; base = "0";   flip = "1";    note = "clear after resolve" },
    @{ k = "taa.taau";                base = "0";   flip = "1";    note = "temporal upscaling" },
    @{ k = "taa.dilate";              base = "1";   flip = "0";    note = "velocity dilation" },
    @{ k = "taa.hist_catmull";        base = "1";   flip = "0";    note = "Catmull-Rom history fetch" },
    @{ k = "taa.novec_reproject";     base = "1";   flip = "0";    note = "reproject where no vectors" },
    @{ k = "taa.box_mod";             base = "1";   flip = "0";    note = "neighbourhood box modulation" },
    @{ k = "taa.unjitter";            base = "1";   flip = "0";    note = "unjitter shift" },
    @{ k = "taa.cr_unjitter";         base = "1";   flip = "0";    note = "Catmull-Rom unjitter fetch (code default on)" },
    @{ k = "taa.fg";                  base = "1";   flip = "0";    note = "frame generation" },
    @{ k = "taa.nearfield_m";         base = "5.0"; flip = "0";    note = "near-field select (0 disarms)" }
)
$toggles = $AllToggles
if ($Only) {
    $want = $Only.Split(",") | ForEach-Object { $_.Trim() }
    $toggles = @($AllToggles | Where-Object { $want -contains $_.k })
    if ($toggles.Count -eq 0) { throw "-Only matched none of the known toggles" }
}

# Every run starts from this: all toggles at baseline plus the mask.
$BaseState = (($AllToggles | ForEach-Object { "$($_.k)=$($_.base)" }) -join ",") + "," + $Mask
function Set-For([string[]]$flips) {
    # Later entries win in metrics_run's Set-IniKeys, so flips go after base.
    if ($flips.Count -eq 0) { return $BaseState }
    return $BaseState + "," + ($flips -join ",")
}

# ---- persistence ------------------------------------------------------------
function Load-Results {
    if (-not (Test-Path $ResultsFile)) { return @() }
    $rows = @()
    foreach ($line in Get-Content $ResultsFile) {
        $t = $line.Trim()
        if ($t) { $rows += (ConvertFrom-Json $t) }
    }
    return $rows
}
function Save-Result($h) {
    $o = New-Object PSObject
    foreach ($k in $h.Keys) { $o | Add-Member -MemberType NoteProperty -Name $k -Value $h[$k] }
    Add-Content -Path $ResultsFile -Value (ConvertTo-Json $o -Compress) -Encoding utf8
}
function Find-Result($rows, [string]$key) {
    foreach ($r in $rows) { if ($r.key -eq $key) { return $r } }
    return $null
}

# ---- run one config ---------------------------------------------------------
function Invoke-Config([string]$key, [string]$phase, [string]$set) {
    $args = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
              (Join-Path $MV "metrics_run.ps1"),
              "-Label", $key, "-Seconds", $Seconds, "-Report", $Report, "-Set", $set)
    $raw = & powershell.exe @args 2>&1 | Out-String
    $m = [regex]::Match($raw, '(?m)^RESULT\[[^\]]*\]\s+MET\s+(.*)$')
    if (-not $m.Success) {
        Write-Host ("  {0}: NO RESULT" -f $key) -ForegroundColor Red
        return $null
    }
    $h = @{ key = $key; phase = $phase; set = $set; when = (Get-Date -Format "s") }
    foreach ($tok in $m.Groups[1].Value.Trim().Split(" ")) {
        $eq = $tok.IndexOf("=")
        if ($eq -gt 0) { $h[$tok.Substring(0, $eq)] = $tok.Substring($eq + 1) }
    }
    Save-Result $h
    Write-Host ("  {0,-40} instab {1,8:N4}  stat {2,9:N6}  edge {3,9:N6}  flick {4,9:N6}  novec {5,7:N4}" -f
        $key, [double]$h["instab"], [double]$h["stat"], [double]$h["edge"], [double]$h["flicker"], [double]$h["novec"]) -ForegroundColor Gray
    return (ConvertFrom-Json (ConvertTo-Json $h -Compress))
}

# Run-or-reuse: with -Resume a config already in the file is not re-measured.
# NB: not "Measure" - that is PowerShell's alias for Measure-Object, and an
# alias outranks a function, so a function by that name is never called.
function Measure-Config([string]$key, [string]$phase, [string]$set) {
    if ($Resume -or $CombosOnly -or $TableOnly) {
        $have = Find-Result (Load-Results) $key
        if ($have) {
            Write-Host ("  {0,-40} (from file) instab {1,8:N4}" -f $key, [double]$have.instab) -ForegroundColor DarkGray
            return $have
        }
    }
    if ($TableOnly) { return $null }
    return Invoke-Config $key $phase $set
}

# ---- phases -----------------------------------------------------------------
if (-not $CombosOnly -and -not $TableOnly) {
    Write-Host ""
    Write-Host "================ PHASE 1: NOISE FLOOR ================" -ForegroundColor Yellow
    Write-Host ("baseline x{0}, {1}s each" -f $BaselineRuns, $Seconds) -ForegroundColor DarkGray
    for ($i = 1; $i -le $BaselineRuns; $i++) { [void](Measure-Config ("baseline#" + $i) "baseline" (Set-For @())) }

    Write-Host ""
    Write-Host "================ PHASE 2: ONE FLIP AT A TIME ================" -ForegroundColor Yellow
    foreach ($t in $toggles) {
        $key = "$($t.k)=$($t.flip)"
        [void](Measure-Config $key "single" (Set-For @($key)))
    }
}

$rows = Load-Results
$base = @($rows | Where-Object { $_.phase -eq "baseline" })
if ($base.Count -lt 2) { throw "need at least 2 baseline rows in $ResultsFile to establish a noise floor" }
$baseVals = $base | ForEach-Object { [double]$_.instab }
$baseMean = ($baseVals | Measure-Object -Average).Average
$noise = [Math]::Max((($baseVals | Measure-Object -Maximum).Maximum - ($baseVals | Measure-Object -Minimum).Minimum), 1e-9)

$singles = @($rows | Where-Object { $_.phase -eq "single" })
$movers = @($singles | Where-Object { [Math]::Abs([double]$_.instab - $baseMean) -gt $noise } |
            Sort-Object { -[double]$_.instab })

if (-not $NoCombos -and -not $TableOnly) {
    Write-Host ""
    Write-Host "================ PHASE 3: EVERY PAIR OF MOVERS ================" -ForegroundColor Yellow
    $n = $movers.Count
    Write-Host ("{0} movers beat the noise floor -> {1} pairs" -f $n, ($n * ($n - 1) / 2)) -ForegroundColor DarkGray
    for ($i = 0; $i -lt $n; $i++) {
        for ($j = $i + 1; $j -lt $n; $j++) {
            $a = $movers[$i].key; $b = $movers[$j].key
            [void](Measure-Config ("$a + $b") "pair" (Set-For @($a, $b)))
        }
    }
    $rows = Load-Results
}

# ---- the table --------------------------------------------------------------
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$md = Join-Path $MV ("sweep-table-" + $stamp + ".md")
$sb = New-Object System.Text.StringBuilder
function W([string]$s) { [void]$sb.AppendLine($s) }

W "# MotionVectors stability sweep"
W ""
W ('Generated {0}. {1}s per run, {2}-frame windows, third window reported. Mask: `{3}`.' -f (Get-Date), $Seconds, $Report, $Mask)
W ""
W ("**Baseline instability {0:N4}, noise floor {1:N4}** (spread over {2} identical runs). " -f $baseMean, $noise, $base.Count)
W 'A row is only WORSE or better if its delta clears the floor; otherwise it is `-` and the sweep will not pretend to rank it.'
W ""
W "INSTAB = 2000*static + 1000*edge + 100*flicker. Lower is more stable. Weights are a judgement; every component is in the table so you can rank on one instead."
W ""
W "| config | phase | instab | delta | verdict | static | edge | flicker | trms | lap | sharp | stair | novec | vel | sign |"
W "|---|---|---:|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|"

$ranked = @($rows | Sort-Object { -[double]$_.instab })
foreach ($r in $ranked) {
    $d = [double]$r.instab - $baseMean
    $verdict = "-"
    if ($r.phase -ne "baseline" -and [Math]::Abs($d) -gt $noise) { if ($d -gt 0) { $verdict = "WORSE" } else { $verdict = "better" } }
    W ("| {0} | {1} | {2:N4} | {3:+0.0000;-0.0000} | {4} | {5:N6} | {6:N6} | {7:N6} | {8:N6} | {9:N5} | {10:N5} | {11:N3} | {12:N4} | {13:N2} | {14} |" -f
       $r.key, $r.phase, [double]$r.instab, $d, $verdict, [double]$r.stat, [double]$r.edge, [double]$r.flicker,
       [double]$r.trms, [double]$r.lap, [double]$r.sharp, [double]$r.stair, [double]$r.novec, [double]$r.vel, $r.sign)
}
W ""
if ($movers.Count -gt 0) {
    W ('**Least stable single flip: `{0}`** at {1:N4}, {2:+0.0000} over baseline.' -f $movers[0].key, [double]$movers[0].instab, ([double]$movers[0].instab - $baseMean))
    W ""
    W ('Movers (cleared the floor): ' + (($movers | ForEach-Object { '`' + $_.key + '`' }) -join ', '))
} else {
    W "**No single flip moved instability beyond the noise floor.** Either these settings do not affect stability in this scene, or the runs are too short - raise -Seconds before concluding."
}
$pairs = @($rows | Where-Object { $_.phase -eq "pair" } | Sort-Object { -[double]$_.instab })
if ($pairs.Count -gt 0) {
    W ""
    W ('**Least stable pair: `{0}`** at {1:N4}.' -f $pairs[0].key, [double]$pairs[0].instab)
}
W ""
W "Columns: static = mean |delta| on unmoved pixels (pure instability); edge = |delta| weighted by local contrast (what the eye catches); flicker = fraction of pixels reversing delta sign at a 2-LSB threshold; trms = RMS residual; lap = Laplacian energy (aliasing proxy, higher = more high-frequency detail and more jaggies); sharp = gradient energy (a config cannot win by going soft); stair = intra-quad / neighbourhood variance; novec = fraction of pixels with no velocity written; vel = mean vector length in px; sign = which velocity convention reprojected better."
W ""
W "Limits: parked cockpit autoload, so ghost / disocclusion / sign are uninformative (no motion). Reverse-Z was measured, not assumed (54.6% of pixels below 0.0001)."

[IO.File]::WriteAllText($md, $sb.ToString())
Write-Host ""
Get-Content $md
Write-Host ""
Write-Host ("table: {0}" -f $md) -ForegroundColor Green
Write-Host ("results: {0}" -f $ResultsFile) -ForegroundColor Green
