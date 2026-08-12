# Motion Vectors 0.0.14 — TAA quality, and the aeroplane finally moves

## Verified in powered flight

Every previous release measured a parked aircraft. `paused=0 groundspeed=0.00
throttle=0.00 parkbrake=1.00` — the whole suite verified rotation and a scripted
camera slide, and nothing else. Real flight translates metres per frame, an order
of magnitude beyond anything the scripted phases produce.

There is now a FLY phase. It releases the parking brake, opens the throttle, and
leaves the camera to X-Plane, so the reprojection runs through the same path a
user's frames take rather than through a synthetic override:

    groundspeed=10.75 m/s throttle=1.00 parkbrake=0.00
    groundspeed=13.72 m/s throttle=1.00 parkbrake=0.00
    groundspeed=16.25 m/s throttle=0.00 parkbrake=1.00   <- restored

**Epipolar residual under real motion: 0.021 – 0.174 px** over ~400,000 samples
per frame.

That is sub-pixel, and about fifty times what the scripted phases measure
(0.000–0.003 px). Worth stating rather than rounding away: real flight brings a
spinning propeller, turning wheels, compressing suspension and airframe
vibration, none of which is rigid with the camera and none of which a
camera-only reprojection can predict. A camera-only field *cannot* reach 0.003 px
against non-rigid geometry, so this is close to the floor for the design.

## TAA quality

Kept, each verified by capture:

- **YCoCg variance clipping.** A min/max box over 3×3 is defined by its two most
  extreme texels, so one specular highlight opens it wide enough to admit history
  from a different surface. `mean ± 1.5σ` describes where the neighbourhood
  actually lives, and YCoCg puts luminance on its own axis — where nearly all of
  a neighbourhood's variation is.
- **Clip toward the mean**, not clamp per channel, which changes hue on the way.
- **Catmull-Rom history, bounded by its own taps.** Reprojection lands between
  texels every frame, so bilinear's softening compounds until the history is
  visibly blurrier than the frame it came from. The negative lobes overshoot on
  hard edges; bounding the result to the range of the five taps removes that
  exactly where it exists and nowhere else.
- **Motion-adaptive blend** — more of the current frame when the history is being
  dragged furthest, which is also when the eye can least see the extra noise.

## The heuristic that had to go

Taking the longest velocity of the 3×3 is the standard trick for stopping a
moving silhouette from trailing. It assumes the longest vector belongs to the
*nearest* surface — true when one object crosses a static background, false
against thin geometry like the canopy frame and the wipers, where one 3×3 holds
wildly different velocities and the maximum belongs to whichever fragment was
fastest rather than to the surface being shaded.

A capture named it: a dense speckled band across the top of the frame, exactly
where thin cockpit structure meets sky, clean everywhere else. I first read it as
Catmull-Rom ringing on HDR contrast and fixed the ringing — a real fix, kept —
and the band survived. What identified it was the **shape**: a band follows
geometry; ringing follows edges everywhere in the frame.

Doing it properly needs depth, to take the closest rather than the fastest.
Nothing binds depth to this pass, so it uses the pixel's own velocity: correct
for every pixel, at the cost of some trailing on moving silhouettes — the smaller
and far more local error.
