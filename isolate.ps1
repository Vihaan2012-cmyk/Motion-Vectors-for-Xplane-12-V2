# isolate.ps1 - test ONE open issue at a time.
#
# Every artifact chased in this project has cost more time to ATTRIBUTE than to
# fix. The black speckle took four wrong hypotheses before one setting - contact
# off, then ao off - settled it in two looks. The seam survived four features
# being disabled one by one and turned out not to be any of them.
#
# So the presets below are not convenience. Each one leaves exactly one suspect
# running, and says what a pass and a failure look like, because "it still looks
# wrong" is not a result if three things were on.
#
# These write %TEMP%\taa_live.ini, which the layer re-reads every ~15 frames.
# No restart, no rebuild - change preset while flying and watch.
#
#   .\isolate.ps1 ao          apply a preset
#   .\isolate.ps1             list them
#   .\isolate.ps1 -Show       print the live values without changing anything

param([string]$Preset = "", [switch]$Show)

$ini = Join-Path $env:TEMP "taa_live.ini"
if (-not (Test-Path $ini)) { Write-Host "no live ini at $ini" -Foreground Red; exit 1 }

# key -> value per preset. Anything not listed is left exactly as it is, so a
# preset never silently resets a knob you were mid-experiment on.
$presets = [ordered]@{
  "baseline" = @{ v = @{ "taa.enable"="0" }
                  why = "Mod fully out of the picture. THE reference frame."
                  look = "Whatever is still wrong here is X-Plane's, not ours. Run this FIRST for any new artifact." }

  "taa"      = @{ v = @{ "taa.enable"="1"; "taa.ao"="0.0"; "taa.contact"="0.0"; "taa.sharpen"="0.0"; "taa.gi"="0"; "taa.taau"="0"; "taa.fg"="0" }
                  why = "The resolve alone - reprojection and temporal blend, nothing layered on."
                  look = "Aliasing/shimmer HERE is the accumulation problem. Read: ACCUM LEDGER (BOTH unchanged %)." }

  "ao"       = @{ v = @{ "taa.enable"="1"; "taa.ao"="0.5"; "taa.contact"="0.0"; "taa.sharpen"="0.0"; "taa.gi"="0" }
                  why = "TAA + ambient occlusion. Tests the far-edge ramp (2026-08-30)."
                  look = "Moving black speckle on BRIGHT surfaces = not fixed. Watch >500 frames; it recurred on that period." }

  "contact"  = @{ v = @{ "taa.enable"="1"; "taa.ao"="0.0"; "taa.contact"="0.6"; "taa.sharpen"="0.0"; "taa.gi"="0" }
                  why = "TAA + contact shadows. Tests the bias/far ramps."
                  look = "Same speckle signature. Sun must be up and low - no sun, no contact shadows, no test." }

  "sharpen"  = @{ v = @{ "taa.enable"="1"; "taa.ao"="0.0"; "taa.contact"="0.0"; "taa.sharpen"="0.5"; "taa.gi"="0" }
                  why = "TAA + sharpen only. The remaining suspect for dark overshoot on high-contrast edges."
                  look = "Dark fringing beside bright edges = unsharp ringing, not an occlusion march." }

  "gi"       = @{ v = @{ "taa.enable"="1"; "taa.ao"="0.0"; "taa.contact"="0.0"; "taa.sharpen"="0.0"; "taa.gi"="1" }
                  why = "SSGI. First flight 2026-08-30: looks good. Still the newest path here."
                  look = "Trace must say 'GI: geometry from the ENGINE' - the fallback is a visibly worse image and says so. Watch disocclusions for popping (the history reject ramp is tuned by taa.gi_reject)." }

  "taau"     = @{ v = @{ "taa.enable"="1"; "taa.taau"="1"; "taa.ao"="0.0"; "taa.contact"="0.0"; "taa.gi"="0" }
                  why = "Temporal upsampling. ALSO NEVER RUN."
                  look = "taa.taau_split=0.5 puts upscaled left / native right in one frame - the seam IS the comparison." }

  "fg"       = @{ v = @{ "taa.enable"="1"; "taa.fg"="1"; "taa.ao"="0.0"; "taa.contact"="0.0"; "taa.gi"="0" }
                  why = "Frame generation. Exercises the proxy double-destroy fix."
                  look = "Read: FG LEDGER. 'ADDRESS COLLISIONS > 0' proves the reuse the fix was written for. Trigger a resolution change." }

  "fsr"      = @{ v = @{ "taa.enable"="1"; "fsr.replace"="1"; "fsr.backend_fsr3"="1"; "taa.ao"="0.0"; "taa.contact"="0.0"; "taa.gi"="0" }
                  why = "FSR3 upscaling. Built and linked all along - switched OFF in the stability sweep to reclaim ~1 GB VRAM, not because it failed."
                  look = "First run through the temporal core: its MV scale and jitter now come from tadapt::toFsr3, not the call site. Smearing that TAA alone does not have means the adapter's units are wrong - check fsr.mv_x/mv_y." }

  "tuned"    = @{ v = @{ "taa.enable"="1"; "taa.ao"="0.5"; "taa.contact"="0.6"; "taa.sharpen"="0.5"; "taa.gi"="0"; "taa.taau"="0" }
                  why = "The normal shipping-ish configuration."
                  look = "Not a test. Where you go back to when you are done isolating." }
}

function Show-Live {
    $keys = @("taa.enable","taa.ao","taa.contact","taa.sharpen","taa.gi","taa.taau","taa.fg","taa.viz","taa.unlock")
    Write-Host "`nlive values:" -Foreground Cyan
    foreach ($k in $keys) {
        $m = Select-String -Path $ini -Pattern ("^" + [regex]::Escape($k) + "=(.*)$")
        if ($m) { "  {0,-14} {1}" -f $k, $m.Matches[0].Groups[1].Value }
    }
}

if ($Show) { Show-Live; exit 0 }

if (-not $Preset -or -not $presets.Contains($Preset)) {
    if ($Preset) { Write-Host "unknown preset '$Preset'" -Foreground Red }
    Write-Host "`nisolate one issue at a time:`n" -Foreground Cyan
    foreach ($k in $presets.Keys) { "  {0,-10} {1}" -f $k, $presets[$k].why }
    Write-Host "`n  .\isolate.ps1 <name>     apply"
    Write-Host "  .\isolate.ps1 -Show      current live values`n"
    Show-Live
    exit 0
}

$p = $presets[$Preset]
$text = Get-Content $ini -Raw
foreach ($k in $p.v.Keys) {
    $pat = "(?m)^" + [regex]::Escape($k) + "=.*$"
    $rep = $k + "=" + $p.v[$k]
    if ($text -match $pat) { $text = [regex]::Replace($text, $pat, $rep) }
    else                   { $text = $text.TrimEnd() + "`n" + $rep + "`n" }
}
Set-Content -Path $ini -Value $text -Encoding utf8 -NoNewline

Write-Host "`n[$Preset] $($p.why)" -Foreground Green
Write-Host "LOOK FOR: $($p.look)" -Foreground Yellow
Write-Host "`nchanged:" -Foreground Cyan
foreach ($k in $p.v.Keys) { "  {0,-14} -> {1}" -f $k, $p.v[$k] }
Write-Host "`nlive in ~15 frames. Everything not listed was left alone.`n"
