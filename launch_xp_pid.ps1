# PID-mode launch: every velocity texel carries the id of the VERTEX SHADER that
# wrote it (channel B), and the layer writes the id -> module-hash table to
# %TEMP%\mv_pid_table_<pid>.txt with one PIPE line per patched pipeline.
#
# This is the "who wrote that motion" run. Alpha keeps its 0/1 select, so
# blended draws (decals, glass) are measured exactly as they ship - but channel
# B is a tag, not depth, so AO / contact shadows are meaningless while it runs.
# Fly the same spot for ~30 s, then read:
#
#   %TEMP%\mv_diag_N.txt         -> MOTION BY VERTEX SHADER  (which pid moves)
#   %TEMP%\mv_pid_table_<pid>.txt -> vs <pid> <hash>          (which module)
#                                    PIPE vs=<pid> ...        (its layout/blend)
#
# then TAA_MV_DUMP_HASH=<hash> on a later run writes that module's SPIR-V.
param(
    # Module hash from mv_pid_table_<pid>.txt (the "vs <pid> <hash>" line of the
    # writer you want to read). The layer writes that module's original and
    # patched SPIR-V to %TEMP%\mv_vs_<pid>.spv / mv_vs_<pid>_patched.spv.
    [string]$DumpHash = "",
    [switch]$RawPolicy
)
Set-Location $PSScriptRoot
$env:TAA_MV_PID  = "1"
$env:TAA_MV_DIAG = $env:TEMP
# Keep the shipped family/additive policy APPLIED in the readback run unless
# -RawPolicy: an additive full-screen overlay in the cockpit otherwise blends
# its tag over every texel and the census sees nothing underneath.
if ($RawPolicy) { Remove-Item Env:TAA_MV_PID_POLICY -ErrorAction SilentlyContinue } else { $env:TAA_MV_PID_POLICY = "1" }
if ($DumpHash) { $env:TAA_MV_DUMP_HASH = $DumpHash; Write-Host "will dump module $DumpHash" -ForegroundColor Cyan }
else { Remove-Item Env:TAA_MV_DUMP_HASH -ErrorAction SilentlyContinue }
Write-Host "PID mode: tag in channel B, table -> $env:TEMP\mv_pid_table_<pid>.txt" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "launch_xp.ps1")
# Arm the periodic readback (one report every 240 frames, four reports max).
# AFTER the plain launcher: it resets this file to 0 so a plain run never
# carries the arm, and the layer only reads it once frames are flowing.
Set-Content -Path (Join-Path $env:TEMP "taa_dump_every.txt") -Value "240" -Encoding ascii
