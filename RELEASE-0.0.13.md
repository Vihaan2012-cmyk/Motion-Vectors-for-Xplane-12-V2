# Motion Vectors 0.0.13 — TAA

The velocity field has a consumer. A compute resolve reprojects the previous
frame through the field, clamps it to the current pixel's neighbourhood, and
blends. Jitter is armed alongside it.

Enable with `TAA_RESOLVE=1`. Off by default: the velocity field is the product,
and this is the first thing that changes what the user sees.

## The resolve

**Compute, not a full-screen draw.** A graphics pass needs a render pass
compatible with whatever X-Plane has bound, and the whole difficulty of the
velocity work was that X-Plane's pipelines are not ours to shape. A dispatch
needs no render pass, no framebuffer and no blend state, so the driver cannot
refuse it for disagreeing with a pipeline we did not create — and 14,835
pipelines were once refused for exactly that kind of disagreement.

**The history is sampled at `uv + velocity`,** because the field is
`prev - curr`. That is stated in the shader and again at the top of the header
rather than left to be inferred from a sign: the opposite convention does not
fail or warn, it reprojects twice as far in the wrong direction and reads as
ghosting. It was backwards in this project for its entire life until 0.0.10.

**Neighbourhood clamping** is an RGB AABB over the 3×3 — the loose version on
purpose. A wrong tight bound flickers; a loose bound only ghosts slightly.

## Jitter, finally armed

The layer has carried a jitter path for a long time behind this comment:

> It is armed for measurement, not for use, until a resolve exists to cancel it.

A resolve exists, so jitter is now tied to it. Neither can be left on without
the other by accident: jitter alone shifts the sample grid every frame with
nothing accumulating the result, which makes high-contrast edges crawl, and the
resolve alone smooths without ever gaining a sample the frame did not have.

## The bug that would have hidden

The resolve dispatched three times and stopped. **Silently** — a failed
descriptor allocation is a return code, not a crash. The pool held eight sets,
one was allocated per dispatch, none was ever freed.

It was caught in a single run only because the call site logs when the resolve
does not record. A pass that silently never runs is indistinguishable from one
that runs and has no effect. Eight sets are now allocated once and cycled.

## Also

- Shaders are written in GLSL and compiled to SPIR-V by `build.ps1` into an
  embedded header. The injected vertex and fragment patches must be assembled by
  hand because they are grafted into X-Plane's own modules; the resolve is ours
  start to finish, so the compiler can check it.
- The panel reports the residual, the pipeline counts, the velocity target size
  and the build version, and goes red the moment a pipeline is rejected.
- One missing dataref (`taaimpl/render_scale`, documented and never registered)
  took down the entire Lua engine. FlyWithLua reports a missing name above Lua,
  so `pcall` never saw it. The panel now asks `XPLMFindDataRef` first.

Verified: 2400 consecutive dispatches at 3840×2160, residual holding at 0.000 px.
