# sweep.ps1 - rank MotionVectors configurations by measured instability.
#
# Answers one question: which setting, or which combination of settings, leaves
# the image least stable? It does that by measuring, not by reasoning.
#
# THE NOISE FLOOR COMES FIRST. The baseline config is run -BaselineRuns times
# before anything is toggled, and the spread across those identical runs is the
# noise floor. X-Plane streams scenery, the weather moves, the sun moves; two
# identical runs do NOT produce identical numbers. Any toggle whose effect is
# smaller than that spread is reported as NO EFFECT rather than ranked, because
# ranking it would be ranking noise. This is the whole reason the sweep is
# trustworthy and it is not optional.
#
# PHASE 2 IS EARNED. Only toggles that moved instability beyond the noise floor
# go into the combination phase - there is no point pairing two settings that
# individually did nothing. Combinations are pairs of the movers, worst-first.
#
# COCKPIT AUTOLOAD. X-Plane loads straight into the cockpit, so every run starts
# with a loading screen and a few seconds of scenery streaming that would swamp
# any real difference. metrics_run.ps1 waits for the SECOND measurement window
# to close and reports that one, so the load-in transient is never in the data.
#
#   .\sweep.ps1                          # full sweep
#   .\sweep.ps1 -Quick                   # the six settings most likely to matter
#   .\sweep.ps1 -Seconds 150             # longer per run, tighter noise floor
#   .\sweep.ps1 -Only "taa.contact,taa.ao"
param(
    [int]$Seconds = 150,
    [int]$Report = 200,
    [int]$BaselineRuns = 3,
    [switch]$Quick,
    [switch]$NoCombos,
    [string]$Only = "",
    [string]$Baseline = ""
)
$ErrorActionPreference = "Stop"
$MV = $PSScriptRoot

# ---- what gets dissected ----------------------------------------------------
# Each entry is one binary flip against the baseline. Names are the live-ini
# keys the layer actually reads.
$AllToggles = @(
    @{ k = "taa.contact";            off = 0; note = "contact shadows" },
    @{ k = "taa.ao";                 off = 0; note = "ambient occlusion" },
    @{ k = "taa.sharpen";            off = 0; note = "post-resolve sharpen" },
    @{ k = "taa.gi";                 off = 0; note = "screen-space GI" },
    @{ k = "taa.dilate";             off = 0; note = "velocity dilation" },
    @{ k = "taa.hist_catmull";       off = 0; note = "Catmull-Rom history fetch" },
    @{ k = "taa.novec_reproject";    off = 0; note = "reproject where no vectors" },
    @{ k = "taa.engine_depth";       off = 0; note = "engine depth for the resolve" },
    @{ k = "taa.box_mod";            off = 0; note = "neighbourhood box modulation" },
    @{ k = "taa.reactive";           off = 0; note = "reactive mask" },
    @{ k = "taa.cr_unjitter";        off = 0; note = "Catmull-Rom unjitter" },
    @{ k = "taa.clear_after_resolve"; off = 0; note = "clear after resolve" },
    @{ k = "taa.taau";               off = 0; note = "temporal upscaling" },
    @{ k = "taa.unjitter";           off = 0; note = "unjitter shift" },
    @{ k = "taa.nearfield_m";        off = 0; note = "near-field select (0 disarms)" }
)
$QuickKeys = @("taa.contact", "taa.ao", "taa.sharpen", "taa.gi", "taa.dilate", "taa.hist_catmull")

$toggles = $AllToggles
if ($Quick) { $toggles = $AllToggles | Where-Object { $QuickKeys -contains $_.k } }
if ($Only) {
    $want = $Only.Split(",") | ForEach-Object { $_.Trim() }
    $toggles = $AllToggles | Where-Object { $want -contains $_.k }
    if ($toggles.Count -eq 0) { throw "-Only matched none of the known toggles" }
}

# ---- run one config, return its parsed MET fields ---------------------------
function Invoke-Config([string]$label, [string]$set) {
    $args = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
              (Join-Path $MV "metrics_run.ps1"),
              "-Label", $label, "-Seconds", $Seconds, "-Report", $Report)
    if ($set) { $args += @("-Set", $set) }
    $raw = & powershell.exe @args 2>&1 | Out-String
    $m = [regex]::Match($raw, '(?m)^RESULT\[[^\]]*\]\s+MET\s+(.*)$')
    if (-not $m.Success) {
        Write-Host ("  {0}: NO RESULT" -f $label) -ForegroundColor Red
        return $null
    }
    $h = @{}
    foreach ($tok in $m.Groups[1].Value.Trim().Split(" ")) {
        $eq = $tok.IndexOf("=")
        if ($eq -gt 0) { $h[$tok.Substring(0, $eq)] = $tok.Substring($eq + 1) }
    }
    $h["_label"] = $label
    $h["_set"]   = $set
    Write-Host ("  {0,-34} instab {1,9:N4}  stat {2,9:N6}  edge {3,9:N6}  flick {4,9:N6}" -f
        $label, [double]$h["instab"], [double]$h["stat"], [double]$h["edge"], [double]$h["flicker"]) -ForegroundColor Gray
    return $h
}

$results = @()
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outFile = Join-Path $MV ("sweep-" + $stamp + ".txt")

Write-Host ""
Write-Host "================ PHASE 1: NOISE FLOOR ================" -ForegroundColor Yellow
Write-Host ("baseline x{0}, {1}s each" -f $BaselineRuns, $Seconds) -ForegroundColor DarkGray
$base = @()
for ($i = 1; $i -le $BaselineRuns; $i++) {
    $r = Invoke-Config ("baseline#" + $i) $Baseline
    if ($r) { $base += $r }
}
if ($base.Count -lt 2) { throw "need at least 2 successful baseline runs to establish a noise floor" }

$baseVals = $base | ForEach-Object { [double]$_["instab"] }
$baseMean = ($baseVals | Measure-Object -Average).Average
$baseMin  = ($baseVals | Measure-Object -Minimum).Minimum
$baseMax  = ($baseVals | Measure-Object -Maximum).Maximum
# The floor is the observed spread across identical runs. A toggle has to beat
# that to count as having done anything at all.
$noise = [Math]::Max($baseMax - $baseMin, 1e-9)
Write-Host ("  baseline instab mean {0:N4}  spread {1:N4}  (NOISE FLOOR)" -f $baseMean, $noise) -ForegroundColor Cyan

Write-Host ""
Write-Host "================ PHASE 2: ONE AT A TIME ================" -ForegroundColor Yellow
foreach ($t in $toggles) {
    $set = if ($Baseline) { "$Baseline,$($t.k)=$($t.off)" } else { "$($t.k)=$($t.off)" }
    $r = Invoke-Config ($t.k + "=" + $t.off) $set
    if ($r) {
        $r["_key"]   = $t.k
        $r["_note"]  = $t.note
        $r["_delta"] = [double]$r["instab"] - $baseMean
        $results += $r
    }
}

$movers = @($results | Where-Object { [Math]::Abs($_["_delta"]) -gt $noise } |
            Sort-Object { -[double]$_["instab"] })

if (-not $NoCombos -and $movers.Count -ge 2) {
    Write-Host ""
    Write-Host "================ PHASE 3: COMBINATIONS ================" -ForegroundColor Yellow
    Write-Host ("pairing the {0} settings that beat the noise floor" -f $movers.Count) -ForegroundColor DarkGray
    $top = @($movers | Select-Object -First 5)
    for ($i = 0; $i -lt $top.Count; $i++) {
        for ($j = $i + 1; $j -lt $top.Count; $j++) {
            $a = $top[$i]; $b = $top[$j]
            $pair = "$($a['_key'])=0,$($b['_key'])=0"
            $set = if ($Baseline) { "$Baseline,$pair" } else { $pair }
            $r = Invoke-Config ("$($a['_key'])+$($b['_key'])") $set
            if ($r) {
                $r["_key"]   = "$($a['_key'])+$($b['_key'])"
                $r["_note"]  = "combination"
                $r["_delta"] = [double]$r["instab"] - $baseMean
                $results += $r
            }
        }
    }
}

# ---- the report -------------------------------------------------------------
$sb = New-Object System.Text.StringBuilder
function W([string]$s, [string]$c = "Gray") { [void]$sb.AppendLine($s); Write-Host $s -ForegroundColor $c }

W ""
W "===================== SWEEP RESULT =====================" "Yellow"
W ("generated {0}   {1}s per run   report window {2} frames" -f (Get-Date), $Seconds, $Report)
W ("baseline instab {0:N4}   noise floor {1:N4} (spread over {2} identical runs)" -f $baseMean, $noise, $base.Count)
W ""
W "Ranked WORST FIRST by instability. 'delta' is against baseline; anything"
W "inside the noise floor is marked '-' because the sweep cannot tell it from"
W "run-to-run variation and will not pretend otherwise."
W ""
W ("{0,-34} {1,10} {2,10} {3,10} {4,10} {5,10} {6,8}" -f "config", "instab", "delta", "static", "edge", "flicker", "verdict")
W ("{0,-34} {1,10} {2,10} {3,10} {4,10} {5,10} {6,8}" -f ("-"*34), ("-"*10), ("-"*10), ("-"*10), ("-"*10), ("-"*10), ("-"*8))

$ranked = @($results | Sort-Object { -[double]$_["instab"] })
foreach ($r in $ranked) {
    $d = [double]$r["_delta"]
    $verdict = if ([Math]::Abs($d) -le $noise) { "-" } elseif ($d -gt 0) { "WORSE" } else { "better" }
    W ("{0,-34} {1,10:N4} {2,+10:N4} {3,10:N6} {4,10:N6} {5,10:N6} {6,8}" -f
       $r["_key"], [double]$r["instab"], $d, [double]$r["stat"], [double]$r["edge"], [double]$r["flicker"], $verdict)
}

W ""
if ($movers.Count -gt 0) {
    $worst = $movers[0]
    W ("LEAST STABLE: {0}  (instab {1:N4}, {2:N4} above baseline)" -f
       $worst["_key"], [double]$worst["instab"], [double]$worst["_delta"]) "Red"
} else {
    W "No configuration moved instability beyond the noise floor." "Green"
    W "Either these settings genuinely do not affect stability in this scene, or"
    W "the runs are too short - raise -Seconds and re-run before concluding."
}
W ""
W "Sharpness is reported per-config in mv_metrics.txt: a config that wins on"
W "stability while losing sharpness has traded detail for calm, which is not"
W "the same as being better. Check both before acting on this ranking."

[IO.File]::WriteAllText($outFile, $sb.ToString())
Write-Host ""
Write-Host ("written: {0}" -f $outFile) -ForegroundColor Green
