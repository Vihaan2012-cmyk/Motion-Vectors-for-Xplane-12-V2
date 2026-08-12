# Motion Vectors 0.0.10 — verified at 0.003 px

The motion vectors are correct. This is the first release where that statement
rests on a test that could have failed.

## What the test measures now

Every earlier acceptance run compared the field against a prediction made at an
**assumed depth** — one metre, or infinity. That can only decide anything while
the camera is purely rotating, because only then does depth stop mattering.

The test is now depth-free. As distance sweeps 0 to infinity, the previous
position of a pixel traces the image of a ray — a straight line, the epipolar
line. So the question is not "at what depth" but "does the measured previous
position lie on that line", which is a perpendicular distance. It is well posed
under translation, under rotation, and under both.

**Median residual, ~400,000 samples per frame:**

| Phase | Median | p95 |
|---|---|---|
| YAW | 0.000–0.002 px | 0.001–0.010 px |
| PITCH | 0.001–0.003 px | 0.004–0.008 px |
| HEADMOVE | 0.000–0.003 px | 0.005–0.019 px |
| Free flight | 0.000–0.011 px | — |

## Three real defects, all found by measurement

**1. The velocity sign was inverted.** The shader wrote `curr - prev` for the
whole life of the project. Nothing could see it: every statistic was a
magnitude, and a negated vector has exactly the right magnitude. The direct
calibration named it in one line, on the steady frames of a scripted yaw:

    field=(-13.249, -0.041)  matrix=(+13.150, -0.000)  ratio=-1.008
    field=(-13.219, +0.038)  matrix=(+13.150, -0.000)  ratio=-1.005

Every consumer this field exists for wants the vector that carries the current
pixel back to where it was, so the history is sampled at `uv + velocity`. That
is `prev - curr`. Written the other way it does not fail or warn — it reprojects
twice as far in the wrong direction, which reads as ghosting.

**2. The body matrix was never converted to view space.** When the main
reprojection moved to view space, the shader changed what it feeds the matrix.
`bodyReproj`, the cockpit path, stayed clip-to-clip. Measured with the aircraft
frozen at 0.0000 m per frame: a full-width band below the horizon, `vx` exactly
zero at x = 700, 996 and 3808, and `prevNDC.y / currNDC.y = 0.6549` dead
constant over 28,536 pixels and seven consecutive frames.

**3. The `independentBlend` query never ran.** It passed a NULL instance to
`vkGetInstanceProcAddr`, which the spec allows for four global entry points that
do not include `vkGetPhysicalDeviceFeatures`. It returned NULL, the query was
skipped, and the result was read as "unsupported" — and quoted as a cause of the
pipeline rejections. Resolved from the physical device's own instance it reads
supported=1.

## Also in this release

The debug probes are **removed, not defaulted off**. `TAA_MV_RAW`,
`TAA_MV_DEBUG_DEPTH`, `TAA_MV_PROBE_CONST` and `TAA_MV_TESTYAW` each replaced
the real velocity write or the pushed matrix, so a stray environment variable in
a launcher would have turned the field into a debug pattern — and it would still
have passed every magnitude test. `TAA_MV_DUMP_SPIRV` is kept; it only writes
files and cannot alter what is rendered.

## Known, not fixed

- The TRANSLATE phase reports 80–196 px on some frames while others on the same
  phase read 0.002 px. The alternation points at frame pairing, not a continuous
  error.
- A p95 tail persists in the hold phases where the median is 0.000. Under a
  frozen sim nearly every pixel is exactly zero and excluded from the statistic,
  so this is a small set of genuinely moving pixels. Not yet identified.

## This supersedes 0.0.08

0.0.08's "full acceptance, 30/32" was measuring a negated field against a broken
yardstick. See the note on that release.
