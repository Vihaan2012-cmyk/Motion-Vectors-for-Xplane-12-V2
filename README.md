# Motion Vectors for X-Plane 12

Per-pixel motion vectors for X-Plane 12, produced by a Vulkan layer and an XPLM
plugin.

X-Plane does not render a velocity buffer. This derives one: the plugin supplies
the camera's reprojection each frame, and the layer patches the sim's own vertex
and fragment shaders so every drawn pixel reports where it was in the previous
frame.

Also included are the VRAM systems — budget reporting, texture paging controls
and memory priority.

Correctness is measured, not judged by eye. The camera is driven through a known
yaw and pitch, and the vectors are compared against the pixel displacement that
motion must produce. A build is correct when that ratio is 1.

## Progress

- [x] Project set up, logging and measurement harness planned
- [x] Camera reprojection published each frame
- [x] Shader patching: previous-frame clip position per vertex
- [x] Velocity target written per fragment
- [ ] Measured ratio of 1 through yaw and pitch
- [x] VRAM systems
- [ ] Jitter

## Requirements

- X-Plane 12, Vulkan renderer
- Windows

## Install

Run the installer. It places the plugin in `Resources/plugins/` and registers
the Vulkan layer.

## Logging

Everything is written to a dedicated log beside the plugin, not to X-Plane's
`Log.txt`.

---

Work that consumes these vectors is in progress.

[minimal project](https://github.com/Vihaan2012-cmyk/Motion-Vectors-For-Xplane-12)
