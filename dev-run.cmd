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

rem Measure the velocity field every N frames. This is the acceptance gate for
rem the whole project, so it is on for debugging - but it reads back 31.9 MB at
rem 4K, which is why it stays off in a release build rather than defaulting on
rem like the injection does.
set "TAA_VELOCITY_DUMP=20"

rem Drive the camera through a known yaw and pitch once the flight settles, and
rem compare the vectors against the pixel displacement that motion must
rem produce. This is what turns "does it look right" into a number, and it is
rem the only way to measure without someone flying.
set "TAA_SELFTEST=1"

rem Load the situation after all. It comes up in replay, which is not ideal -
rem but replay still renders, and the self-test drives the CAMERA rather than
rem the aeroplane, so the measurement is unaffected. Without it the sim sits on
rem the configuration screen and nothing can be measured at all.

rem Measure the velocity field every N frames. This is the acceptance gate for
rem the project, so it is on for debugging - but it reads back 31.9 MB at 4K,
rem which is why it stays off in a release build rather than defaulting on the
rem way the injection does.

rem NO --load_smo. The saved situation was captured from a replay, so loading it
rem puts the sim straight back into replay rather than a live flight. Start
rem normally and fly it.


echo Layer:     %LAYERDIR%
start "" /D "%XPROOT%" "%XPROOT%\X-Plane.exe"
endlocal
