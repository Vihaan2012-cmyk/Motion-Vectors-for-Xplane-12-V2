@echo off
rem Development launcher.
rem
rem The shipped launcher is a small exe that sets the two loader variables and
rem nothing else. This one adds the debugging switches and then clicks RESUME
rem LAST FLIGHT on the menu, so a build can be measured without anyone present.
rem
rem No command line reaches a LIVE flight. --load_smo is the only switch that
rem bypasses the menu at all - --load_acf, --load_apt, --go_to_apt and
rem --new_flight_json are parsed but only pre-fill it - and the saved situation
rem was captured during a replay, so it always returns to one. A replay flies
rem the aeroplane and drags the camera with it, which is the worst case for this
rem measurement: once the camera translates, near geometry moves further than
rem far geometry and a single expected displacement no longer fits the frame.
rem Resuming gives a PARKED flight, where a camera rotation moves every depth
rem alike.

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
rem TAA is now switched from the panel; this only forces it on for debugging.
rem set "TAA_RESOLVE=1"






set "TAA_MV_IMAGE=C:/Users/bansa/AppData/Local/Temp/mvimg"
set "TAA_MV_M5=2.0"
set "TAA_MV_DIAG=C:/Users/bansa/AppData/Local/Temp/mvdiag"
set "TAA_EXT_TEST=1"
rem set "TAA_SELFTEST=1"

rem Experiment switches are passed THROUGH this script, not set in it. setlocal
rem scopes what is set here, but variables already in the environment survive -
rem so TAA_MV_IDENTITY and friends set by the caller reach X-Plane. The layer
rem prints what it actually received on its first present; check that line
rem before believing any run made through one of them.

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

rem Wait for the menu, then resume. The script polls for the window itself, so
rem the delay here only has to cover the sim getting far enough to draw one.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0resume-flight.ps1"
endlocal
