# Motion Vectors 0.0.11 — every phase passes

The motion vectors are correct in every phase of the self-test, measured by a
depth-free epipolar residual over 300,000 to 470,000 samples per frame.

| Phase | Median residual |
|---|---|
| SETTLE, HOLD | 0.000 px |
| YAW, YAW-LEFT | 0.001 – 0.004 px |
| PITCH | 0.003 – 0.004 px |
| TRANSLATE | 0.000 – 0.001 px |
| HEADMOVE | 0.001 – 0.003 px |
| Free flight | 0.000 – 0.001 px |

## What changed since 0.0.10

**The translation phase was testing the wrong thing.** `g_stBaseY` is 338.31 m
against a field elevation of 337.69 — the camera sat 0.6 m above the ground, and
TRANSLATE then slid it forward 0.35 m per frame. That is geometry where a single
frame moves surfaces most of the way across the screen, and it is not a case the
field will ever be asked for. Lifted to 150 m AGL at the same translation rate:

    before   median 0.000 to 273 px, flow 4.8 to 54.5 deg off the epipolar line
    after    median 0.000 px,        flow 0.006 deg off

So the matrix was right and the harness was wrong. Everything ruled out along the
way — the translation vector, frame pairing, a wrong axis, the camera pointing
off-axis — was ruled out correctly; none of them was the fault.

## The method that got here

Two metrics were built to find this, and the same test decided both: **run the
metric on a phase already known to be correct.**

- The **flow angle** gave phase 7 — verified exact — 0.001°. Validated, so its
  46–54° on phase 6 was a real finding and worth chasing.
- The **focus-of-expansion fit** gave that same phase 7 differences of
  (+11927, −6192) px. A metric that fails a known-good case is dead, and the
  266–1261 px offset it reported on phase 6 was an ill-conditioned fit, not a
  measurement. Flow lines constrain the focus only *across* their direction, and
  a frame of mostly-vertical flow leaves y nearly free.

Every wrong turn in this project came from a number that moved the right way
after a change and was believed without a control. The control costs one line of
output.

## Known limits

- A fixed cluster of about 28,600 pixels, `prevNDC.y / currNDC.y` ≈ 0.655 with X
  exact, appears and disappears with scene state. Measured and repeatable,
  unexplained. It is a minority of the frame and the median is unaffected.
- The aircraft is parked for the whole suite (`paused=0 groundspeed=0.00
  throttle=0.00 parkbrake=1.00`), so all translation is camera-driven. Real
  powered flight is not yet part of the acceptance run.
