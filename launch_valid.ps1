# X-Plane with the Khronos validation layer, for diagnosing frame generation.
#
# Guessing at a Vulkan fault costs a sim launch per attempt. Validation names
# the object and the VUID directly - it is what found the three layout
# mismatches behind the original FSR3 corruption in this project, and it is the
# right instrument for a crash whose cause is not obvious from the trace.
#
# Synchronization validation is ON: frame generation runs work on queues the
# engine does not own, which is exactly the case where a missing barrier or a
# queue-ownership mistake shows up as a crash rather than a wrong pixel.
#
# NOTE: validation is slow. Frame rate under this is not a measurement of
# anything, and a hang here may be validation's serialisation rather than the
# bug - which is itself informative, since the original FSR3 race SURVIVED
# under validation for exactly that reason.
param([int]$Fg = 1)
$root = (Get-Location).Path
$xp   = Split-Path $root -Parent
$vk   = "C:\VulkanSDK\1.4.357.0"

$env:VK_LAYER_PATH           = (Join-Path $root "build\vklayer") + ";$vk\Bin"
$env:VK_LOADER_LAYERS_ENABLE = "VK_LAYER_mv,VK_LAYER_KHRONOS_validation"
$env:VK_LAYER_SETTINGS_PATH  = "$env:TEMP\vk_layer_settings.txt"
$env:TAA_LAYER_TRACE         = "1"
$env:TAA_VELOCITY            = "1"
$env:TAA_FG                  = "$Fg"

# Same clean-file discipline as launch_xp.ps1: both of these are opened in
# append mode, and two runs in one file read as one process doing something
# impossible. That cost two wrong diagnoses here.
foreach ($f in @("$env:TEMP\taa_layer.txt", "$env:TEMP\vk_validation.txt")) {
    for ($i = 0; $i -lt 40 -and (Test-Path $f); $i++) {
        Remove-Item $f -Force -ErrorAction SilentlyContinue
        if (Test-Path $f) { Start-Sleep -Milliseconds 250 }
    }
    if (Test-Path $f) { Write-Host "WARNING: $f still locked - it will contain more than one run" -ForegroundColor Yellow }
}

Start-Process -FilePath (Join-Path $xp "X-Plane.exe") -WorkingDirectory $xp
Write-Host "launched with validation, TAA_FG=$Fg"
