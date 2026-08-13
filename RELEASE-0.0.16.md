# Motion Vectors 0.0.16 — TAA removed, temporarily

**TAA is gone from this build.** It shipped in 0.0.13 and 0.0.15 and it was
broken. The motion vectors themselves are unchanged and still measure sub-pixel.

## Why

Every fault was in the consumer, not the field — which is exactly why the
acceptance suite missed all of it. The suite measures the velocity field, and
the field is fine.

What the captures showed:

- **The cockpit shakes violently** — on frames where the residual read
  `0.000 px SUB-PIXEL`.
- **Warping and tearing at a current-frame weight of 0.70.** At 0.70 the live
  frame is 70% of the result, so accumulation can barely smear anything. Only a
  systematic error survives that weight, which rules out the blend and points at
  where the history is coming from.
- **Dithered corruption on the aircraft in external view**, with the scenery
  clean in the same frame.
- **Vertical streaking** below the aircraft on the runway.
- One frame of **total corruption** — magenta stripes across the whole screen.

## The two leading causes, neither proven

1. The layer logs **two** distinct HDR scene targets and X-Plane alternates
   between them. The resolve picks one per frame; if the pick is off by one it
   reads the buffer X-Plane is *not* rendering into, so "current" is last frame.
   Blending last against older warps in a way no blend weight can suppress.
2. In external view the **aircraft is a moving object**. Its surfaces need the
   body-frame reprojection that already exists in the code (`bodyReproj`) and is
   never used — `bodyValid` reads 0. That explains corruption on the aircraft
   with clean scenery around it.

Both need proving before anything is rebuilt.

## Why removed rather than switched off

A resolve that can corrupt a whole frame is not something to leave one dataref
away from a user. Jitter goes with it — jitter without something accumulating it
makes high-contrast edges crawl, so the two only make sense together.

## What remains

Per-pixel motion vectors, verified by a depth-free epipolar residual across
rotation, pitch, translation, head movement and powered flight, and the panel
that reports it. That part is unaffected.

## The lesson

The field was verified to 0.003 px across eight phases and that told us nothing
about whether a consumer of it worked. **An acceptance suite that measures the
producer cannot certify the consumer** — and shipping the consumer on the
producer's evidence is how two releases went out broken.
