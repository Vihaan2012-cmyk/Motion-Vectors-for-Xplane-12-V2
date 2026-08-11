@echo off
rem Development launcher.
rem
rem The shipped launcher is a small exe that sets the two loader variables and
rem nothing else. This one adds the debugging switches and starts the sim
rem straight into a saved situation, so a build can be tested without anyone
rem picking an aircraft and waiting for the world to load.
rem
rem --load_smo is the ONLY option that bypasses the flight configuration screen.
rem --load_acf, --load_apt, --go_to_apt and --new_flight_json are all parsed and
rem all only pre-fill it, and no XPLM callback runs while it is up. That was
rem established by trying every one of them.

setlocal
set "XPROOT=%~dp0.."
set "LAYERDIR=%~dp0build\vklayer"

if not exist "%XPROOT%\X-Plane.exe" (
    echo ERROR: X-Plane.exe not found at "%XPROOT%"
    exit /b 1
)
if not exist "%LAYERDIR%\VkLayer_mv.json" (
    echo ERROR: layer manifest missing - run build.ps1 first
    exit /b 1
)

rem EXPLICIT layer: the loader only loads it because these are set, and they are
rem set for this process alone. Nothing else on the machine is affected.
rem VK_INSTANCE_LAYERS is the old spelling and is ignored silently by loader
rem 1.3.234 and newer.
set "VK_LAYER_PATH=%LAYERDIR%"
set "VK_LOADER_LAYERS_ENABLE=VK_LAYER_mv"

rem Layer log. Off in normal use - it writes from the render path.
set "TAA_LAYER_TRACE=1"

rem Straight into the saved situation.
set "XPARGS=--load_smo=Output/situations/Cirrus SR-22 Replay.sit"

echo Layer:     %LAYERDIR%
echo Situation: %XPARGS%
start "" /D "%XPROOT%" "%XPROOT%\X-Plane.exe" %XPARGS%
endlocal
