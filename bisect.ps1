# Autonomous 4K-load crash bisect.
# Baseline (no layer) loads the Felis 4K situation clean; the layer crashes it.
# Each config launches X-Plane straight into the saved situation, watches for
# DEVICE_LOST / process death, records a verdict, kills, moves on.
$ErrorActionPreference = "Continue"
$xp   = "d:\Steam Games\steamapps\common\X-Plane 12"
$sit  = "$xp\Output\situations\Felis 742 GE XP12 Situation.sit"
$log  = "$xp\Log.txt"
$out  = "$xp\MotionVectors\bisect_results.txt"
$layerPath = "$xp\MotionVectors\build\vklayer"

# Two rounds per config: the crash is intermittent, one clean pass proves little.
$configs = @(
    @{ name = "A_velOFF_vramON";  vel = $null; hooks = $null },  # velocity dormant, vram hooks on (crashed today)
    @{ name = "B_velON_vramOFF";  vel = "1";   hooks = "0"   },  # velocity live, vram hooks stripped
    @{ name = "C_velOFF_vramOFF"; vel = $null; hooks = "0"   },  # layer skeleton only
    @{ name = "D_velON_vramON";   vel = "1";   hooks = $null }   # everything (crashed today)
)

Add-Content $out "==== bisect started $(Get-Date) ===="
foreach ($round in 1..2) {
  foreach ($cfg in $configs) {
    # Environment for this config.
    $env:VK_LAYER_PATH = $layerPath
    $env:VK_LOADER_LAYERS_ENABLE = "VK_LAYER_mv"
    $env:TAA_LAYER_TRACE = "1"
    if ($cfg.vel)   { $env:TAA_VELOCITY = $cfg.vel }   else { Remove-Item Env:TAA_VELOCITY   -ErrorAction SilentlyContinue }
    if ($cfg.hooks) { $env:TAA_VRAM_HOOKS = $cfg.hooks } else { Remove-Item Env:TAA_VRAM_HOOKS -ErrorAction SilentlyContinue }

    Get-Process -Name "X-Plane" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep 5
    $t0 = Get-Date
    Start-Process -FilePath "$xp\X-Plane.exe" -WorkingDirectory $xp -ArgumentList "--load_sit=`"$sit`""

    # Watch up to 8 minutes. Crash windows today were 2:00-3:30 into load.
    $verdict = $null
    while (((Get-Date) - $t0).TotalMinutes -lt 8) {
        Start-Sleep 15
        $alive = Get-Process -Name "X-Plane" -ErrorAction SilentlyContinue
        $fresh = (Test-Path $log) -and ((Get-Item $log).LastWriteTime -gt $t0)
        if (-not $alive) {
            if ($fresh -and (Select-String -Path $log -Pattern "DEVICE_LOST" -Quiet)) { $verdict = "CRASH DEVICE_LOST" }
            elseif ($fresh) { $verdict = "DIED no-vulkan-error (CTD)" }
            else { $verdict = "DIED before log" }
            break
        }
    }
    if (-not $verdict) {
        # Survived 8 minutes. Confirm the situation actually loaded, not a menu idle.
        $loaded = (Select-String -Path $log -Pattern "B742GE|Felis" -Quiet)
        $verdict = if ($loaded) { "PASS survived 8min in-situation" } else { "INVALID sim idled at menu (sit never loaded)" }
    }
    Add-Content $out "$(Get-Date -Format HH:mm:ss) round$round $($cfg.name): $verdict"
    Get-Process -Name "X-Plane" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep 5
  }
}
Add-Content $out "==== bisect finished $(Get-Date) ===="
