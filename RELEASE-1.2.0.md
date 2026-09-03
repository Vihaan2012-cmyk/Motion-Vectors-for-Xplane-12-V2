# Motion Vectors 1.2.0

Zip only, no installer. Drag the contents into your X-Plane 12 folder and
start the sim through `MotionVectors\MotionVectorsLauncher.exe` (or the Steam
launch option in README.txt).

## What changed since 1.1.5

**Vectors by shader family.** The layer now knows which of X-Plane's shipped
shaders draw what, by hashing the modules in `Resources/shaders/bin/spv.zip`.
Terrain, vegetation, ocean, sky and object surfaces write motion; overlays,
UI, fonts, manipulators, rain and effects are masked or write zero; ground
families never take the near-field (body-frame) path. This is what removed the
"racing" apron markings and the smearing cockpit text: a per-vertex select was
mixing body-frame and world-frame previous positions across one triangle.

**Cockpit steady.** Geometry drawn after the resolve was still being jittered
and never un-jittered; one jitter is now latched per frame and post-resolve
draws get none. Near-field is armed in cockpit views only (1000/1017/1026);
armed in the chase view it froze fuselage vertices while the camera orbited.

**Frame generation and window changes.** Moving the window to another monitor,
opening a second window (IOS) or a fullscreen-exclusive toggle used to crash in
the driver: X-Plane creates a new swapchain on the same surface and the FFX
proxy kept presenting retired images. The proxy is now torn down first; frame
generation stays off for the rest of that session. A resolution change no
longer copies the old extent into a smaller target.

**Thin taxi lines.** `taa.lod_bias` ships at -0.5 so fragmented line
markings survive anti-aliasing.

**Panel.** MFG, DLAA and XeSS rows removed; SSGI toggle added (`taa.gi`);
`taa.body_zero` and `taa.nearfield_pass` exposed.

**Diagnostics.** Live-ini edits to shipped-locked keys are now traced as
`IGNORED` (set `taa.unlock=1` to override). `launch_xp_pid.ps1` gives a
per-vertex-shader motion table. `TAA_DATUM_CHECK=1` logs the aircraft datum's
motion by every reprojection path once a second in external views.

## Known issue

Fine livery detail (rivets, window rows, small text) can flicker in the
external chase view. It is not a vector defect: the airframe's vector was
measured exact, and the resolve accepts those pixels as valid history. The
remaining suspect is the history clamp on sub-pixel detail. Opt-in
`taa.chase_body=1` reprojects the airframe in the body frame in the chase view
(measured correct in flight); it is off by default because scenery within the
same radius takes the airframe's motion while taxiing.

## Contact shadows

Ship at `taa.contact=0.6`. The periodic-dot regression seen during development
was a stale cascade region; regions are now frame-stamped and the fix is
verified in flight.
