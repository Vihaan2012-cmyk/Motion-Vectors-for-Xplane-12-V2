# Build the drag-and-drop release zip.
#
# WHY THIS SHAPE
#
# The zip mirrors the X-Plane 12 folder, so extracting it INTO the X-Plane 12
# directory merges every file into the place it has to live. There is no
# installer, no registry write, and nothing outside the sim's own folder.
#
# That works because every path in the mod is already relative:
#
#   VkLayer_mv.json  says library_path ".\VkLayer_mv.dll", and the Vulkan
#                    loader resolves that against the manifest's own directory.
#   the launcher     finds itself with GetModuleFileName, accepts X-Plane.exe
#                    either beside it or one level up, and points
#                    VK_LAYER_PATH at its own directory.
#
# So the layer folder can sit anywhere inside the sim and still work.
#
# WHY A LAUNCHER IS REQUIRED AT ALL. This is an EXPLICIT Vulkan layer, which
# the loader only loads when VK_LAYER_PATH and VK_LOADER_LAYERS_ENABLE are set
# in the process environment. A plugin cannot set them: by the time X-Plane
# loads plugins, the Vulkan instance already exists. The alternative is
# registering an implicit layer in the registry, which is a machine-wide change
# that affects every Vulkan application - the opposite of portable. So the sim
# is started through MotionVectorsLauncher.exe, which sets the two variables
# and then starts X-Plane.exe with whatever arguments it was given.
#
#   .\package.ps1              build dist\MotionVectors-<version>.zip
#   .\package.ps1 -NoBuild     package whatever is already in build\

param([switch]$NoBuild)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$ver  = '0.0.17'

if (-not $NoBuild) {
    Write-Host "Building..."
    # The Qt settings launcher is optional and its toolchain fails on this
    # machine. That must not block a release, so the build's exit status is
    # ignored here and the artefacts that actually matter are checked below.
    # In a CHILD PROCESS. g++ writing to stderr surfaces as a NativeCommandError
    # that PowerShell 5.1 treats as terminating regardless of
    # ErrorActionPreference here, so an optional component failing to build
    # would kill the packaging run. A separate process cannot do that; its exit
    # status is ignored and the artefact check below is what actually gates.
    # No 2>&1. In PowerShell 5.1 redirecting a native command's stderr wraps
    # every line in an ErrorRecord, which is terminating under
    # ErrorActionPreference Stop - so the redirect itself was what killed the
    # run, not the compiler. try/catch covers the rest.
    try {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$root\build.ps1" |
            Out-Null
    } catch {
        Write-Host "  (build reported errors; checking artefacts below)"
    }
}

# The layer DLL and the plugin are the two artefacts that must exist; the Qt
# launcher is optional and its build is allowed to fail without blocking a
# release, so it is checked separately below rather than gating this.
$need = @(
    "$root\build\vklayer\VkLayer_mv.dll",
    "$root\build\vklayer\VkLayer_mv.json",
    "$root\build\MotionVectors.xpl",
    "$root\build\MotionVectorsLauncher.exe"
)
foreach ($f in $need) {
    if (-not (Test-Path $f)) { throw "missing build artefact: $f" }
}

$stage = Join-Path $root "dist\stage"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }

# ---- the tree, laid out exactly as it must land inside X-Plane 12
$dirs = @(
    "$stage\MotionVectors",
    "$stage\Resources\plugins\MotionVectors\64",
    "$stage\Resources\plugins\FlyWithLua\Scripts"
)
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d | Out-Null }

# Layer + launcher live together: the launcher points VK_LAYER_PATH at its own
# directory, so the manifest and the DLL have to be beside it.
Copy-Item "$root\build\vklayer\VkLayer_mv.dll"  "$stage\MotionVectors\"
Copy-Item "$root\build\vklayer\VkLayer_mv.json" "$stage\MotionVectors\"
Copy-Item "$root\build\MotionVectorsLauncher.exe" "$stage\MotionVectors\"
if (Test-Path "$root\build\qt\MotionVectors.exe") {
    Copy-Item "$root\build\qt\MotionVectors.exe" "$stage\MotionVectors\MotionVectorsSettings.exe"
}

# X-Plane requires the plugin at Resources/plugins/<name>/64/win.xpl
Copy-Item "$root\build\MotionVectors.xpl" "$stage\Resources\plugins\MotionVectors\64\win.xpl"

# The panel is a FlyWithLua script. Shipped, but optional - the mod runs without
# it; it only provides the in-sim controls.
if (Test-Path "$root\lua\MotionVectors.lua") {
    Copy-Item "$root\lua\MotionVectors.lua" "$stage\Resources\plugins\FlyWithLua\Scripts\"
}

# The tuned settings, for reference. NOT required: every value here is also a
# compiled default, so a fresh install with no settings file behaves the same.
# It is included so a user can see what the knobs are and edit them.
if (Test-Path "$root\config\taa_live.ini") {
    Copy-Item "$root\config\taa_live.ini" "$stage\MotionVectors\taa_live.ini.reference"
}

# GPL-3.0 requires the licence to accompany the binaries, so it ships in the
# zip rather than living only in the repository.
Copy-Item "$root\LICENSE" "$stage\LICENSE.txt"

# The README is a FILE, not a here-string. Embedding it here meant every
# Windows path in it went through two layers of escaping on the way in, and
# "\64\win.xpl" and "%TEMP%\taa_live.ini" arrived as "4\win.xpl" and a literal
# tab. An install guide whose paths are wrong is worse than none.
$readmeSrc = Join-Path $root "packaging\READ ME FIRST.txt"
if (-not (Test-Path $readmeSrc)) { throw "missing $readmeSrc" }
(Get-Content $readmeSrc -Raw).Replace('@VER@', $ver) |
    Set-Content -Path "$stage\READ ME FIRST.txt" -Encoding utf8


$dist = Join-Path $root "dist"
$zip  = Join-Path $dist "MotionVectors-$ver.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path "$stage\*" -DestinationPath $zip -CompressionLevel Optimal

Write-Host ""
Write-Host "Package: $zip"
Write-Host ("  {0:N2} MB" -f ((Get-Item $zip).Length / 1MB))
Write-Host ""
Get-ChildItem $stage -Recurse -File | ForEach-Object {
    Write-Host ("  " + $_.FullName.Substring($stage.Length + 1))
}
