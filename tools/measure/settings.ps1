# Install, save or show the live settings.
#
# WHY THIS EXISTS
#
# The live settings live in %TEMP%\taa_live.ini. That is outside the repo, so
# no folder backup captures it and no git checkout restores it - and it is the
# file that actually governs the picture.
#
# The consequences were expensive. Three builds were restored and compared in
# one evening while this file quietly held probe values from hours earlier, so
# the comparison measured the settings rather than the builds and concluded
# nothing. Deleting it is worse: the layer regenerates it EMPTY, every value
# falls back to a compiled default, and the tuning disappears with no message.
#
#   .\tools\measure\settings.ps1 -Restore   config/taa_live.ini -> %TEMP%
#   .\tools\measure\settings.ps1 -Save      %TEMP% -> config/taa_live.ini
#   .\tools\measure\settings.ps1 -Show      what is live right now
#   .\tools\measure\settings.ps1 -Diff      live vs the saved known-good
param(
    [switch]$Restore,
    [switch]$Save,
    [switch]$Show,
    [switch]$Diff
)

$root  = "d:\Steam Games\steamapps\common\X-Plane 12\MotionVectors"
$saved = Join-Path $root "config\taa_live.ini"
$live  = "$env:TEMP\taa_live.ini"

function Pairs($path) {
    $h = @{}
    if (Test-Path $path) {
        foreach ($l in Get-Content $path) {
            if ($l -match '^\s*#' -or $l -notmatch '=') { continue }
            $k, $v = $l -split '=', 2
            $h[$k.Trim()] = $v.Trim()
        }
    }
    return $h
}

if ($Restore) {
    if (-not (Test-Path $saved)) { throw "no saved settings at $saved" }
    # Keep whatever is live now, so a restore is never a one-way door.
    if (Test-Path $live) {
        $bak = "$live.before-restore"
        Copy-Item $live $bak -Force
        Write-Host "  previous live settings kept at $bak"
    }
    Copy-Item $saved $live -Force
    $n = (Pairs $live).Count
    Write-Host "  restored $n settings to $live"
    Write-Host "  the layer re-reads this within a few frames - no restart needed"
}
elseif ($Save) {
    if (-not (Test-Path $live)) { throw "no live settings at $live" }
    $n = (Pairs $live).Count
    if ($n -lt 5) {
        throw ("the live file holds only $n settings - it is empty or " +
               "regenerated, and saving it would destroy the known-good set")
    }
    Copy-Item $live $saved -Force
    Write-Host "  saved $n settings to $saved (commit it, and it travels with every backup)"
}
elseif ($Diff) {
    $a = Pairs $saved
    $b = Pairs $live
    $keys = ($a.Keys + $b.Keys) | Sort-Object -Unique
    $same = 0
    foreach ($k in $keys) {
        $x = $a[$k]; $y = $b[$k]
        if ($x -eq $y) { $same++; continue }
        if ($null -eq $x) { Write-Host ("  LIVE-ONLY  {0,-24} = {1}" -f $k, $y) }
        elseif ($null -eq $y) { Write-Host ("  MISSING    {0,-24} saved={1}" -f $k, $x) }
        else { Write-Host ("  DIFFERS    {0,-24} saved={1}  live={2}" -f $k, $x, $y) }
    }
    Write-Host "  $same of $($keys.Count) identical"
}
else {
    $b = Pairs $live
    if ($b.Count -eq 0) {
        Write-Host "  live settings MISSING or EMPTY - every value is a compiled default"
        Write-Host "  run with -Restore to put the known-good set back"
    } else {
        Write-Host "  $($b.Count) live settings:"
        foreach ($k in ($b.Keys | Sort-Object)) { Write-Host ("    {0,-24} = {1}" -f $k, $b[$k]) }
    }
}
