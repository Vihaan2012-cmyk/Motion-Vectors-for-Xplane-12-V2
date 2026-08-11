# Motion Vectors for X-Plane 12

Per-pixel motion vectors for X-Plane 12, produced by a Vulkan layer and an XPLM
plugin.

X-Plane does not render a velocity buffer. This derives one: the plugin supplies
the camera's reprojection each frame, and the layer patches the sim's own vertex
and fragment shaders so every drawn pixel reports where it was in the previous
frame.

Also included are the VRAM systems — budget reporting, texture paging controls
and memory priority.

## Status

Motion vectors are the whole project until they are provably correct. Nothing is
built on top of them before that.

Correctness is measured, not judged by eye: the camera is driven through a known
yaw and pitch, and the vectors are compared against the pixel displacement that
motion must produce. A build is correct when that ratio is 1.

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

Later work may consume these vectors. That is not this project.

[minimal project](https://github.com/Vihaan2012-cmyk/Motion-Vectors-For-Xplane-12)
