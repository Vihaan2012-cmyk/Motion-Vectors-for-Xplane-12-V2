Motion Vectors for X-Plane 12 - install
=======================================

1. Drag the CONTENTS of this folder into your X-Plane 12 folder
   (the one containing X-Plane.exe). Merge when asked.

2. Start X-Plane through:  MotionVectors\MotionVectorsLauncher.exe
   Launching from the Steam menu does NOT load the mod - unless you do
   the one-time Steam setup below.

Steam one-time setup (optional, recommended)
--------------------------------------------
Steam -> X-Plane 12 -> Properties -> Launch Options, paste:

    "<your X-Plane folder>\MotionVectors\MotionVectorsLauncher.exe" %command%

After that, launching from Steam loads the mod automatically.

Is it working?
--------------
The in-sim panel (FlyWithLua required) shows a status line. If it says the
layer never loaded, you launched without the launcher, or antivirus blocked
MotionVectors\VkLayer_mv.dll - unblock it (right-click -> Properties ->
Unblock) or add an exclusion; the launcher also un-blocks it automatically
on every start.
