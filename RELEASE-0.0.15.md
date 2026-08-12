# Motion Vectors 0.0.15 — TAA, switchable, verified in flight

Per-pixel motion vectors for X-Plane 12, and a temporal anti-aliasing resolve
that consumes them. Both verified by measurement rather than by eye.

## Accuracy

Depth-free epipolar residual — the distance between where the field says a pixel
was and where the geometry says it could have been — over 250,000 to 470,000
samples per frame:

| Phase | Median residual |
|---|---|
| YAW / YAW-LEFT | 0.002 – 0.005 px |
| PITCH | 0.001 – 0.004 px |
| TRANSLATE | 0.000 – 0.003 px |
| HEADMOVE | 0.001 – 0.002 px |
| **Powered flight** | **0.000 – 0.001 px** |

The test needs no depth, so it is valid under rotation and translation alike —
which matters, because a fixed-depth prediction can only decide anything while
the camera is purely rotating, and every acceptance figure before 0.0.11 was
measured that way.

## Using it

Install, launch X-Plane from the Motion Vectors launcher, and open the panel
(FlyWithLua macro: *Motion Vectors: panel*).

TAA is **off** by default and switched from the panel. The velocity field is
produced either way; the resolve is the first thing that changes what you see.

The panel reports the residual with a verdict, the pipeline counts, the velocity
target size, VRAM, and the resolve's dispatch count. That last one exists
because a pass that silently never runs looks exactly like one that runs and does
nothing — which happened during development, when a descriptor pool ran dry after
eight frames and the resolve quietly stopped. "Enabled" is not evidence; the
counter is, and the panel says **SWITCHED ON, NOT RUNNING** in red if it is zero.

## The TAA

- **Compute, not a full-screen draw.** A graphics pass needs a render pass
  compatible with whatever X-Plane has bound. A dispatch needs none, so the
  driver cannot refuse it for disagreeing with a pipeline we did not create —
  and 14,835 pipelines were once refused for exactly that.
- **YCoCg variance clipping**, clipped toward the neighbourhood mean rather than
  clamped per channel.
- **Catmull-Rom history bounded by its own taps**, so reprojection does not
  compound softness frame after frame, without the ringing the negative lobes
  otherwise produce on hard edges.
- **Motion-adaptive blend**, and jitter armed alongside the resolve — neither is
  useful without the other, so they cannot be left on separately by accident.

## Known limits

- Under a **stationary camera**, a minority of pixels — thin cockpit structure
  against sky, and a fixed cluster with `prevNDC.y / currNDC.y ≈ 0.655` — do not
  reproject correctly. The median is unaffected; the cause is measured,
  repeatable and unexplained.
- The resolve takes the pixel's own velocity rather than the nearest surface's,
  so fast-moving silhouettes trail slightly. Doing it properly needs depth bound
  to the pass, which is not yet done.
- Powered flight measures 0.021 – 0.174 px when the aircraft is accelerating on
  the ground, against 0.000 – 0.001 in the air. Propeller, wheels, suspension and
  vibration are not rigid with the camera, and a camera-only field cannot predict
  them.
