# Motion Vectors for X-Plane 12

Temporal anti-aliasing for X-Plane 12, built on per-pixel motion vectors that
the sim does not provide.

X-Plane does not render a velocity buffer. This derives one: an XPLM plugin
publishes the camera's reprojection each frame, and a Vulkan layer patches the
sim's own vertex and fragment shaders as they compile, so every drawn pixel
reports where it was in the previous frame. A compute pass then accumulates
detail across frames using that field.

Also included is a VRAM manager — budget shaping, memory priorities, upload
pacing and a recycle pool — which exists because X-Plane has those mechanisms
and does not use them well under pressure.

Correctness is measured, not judged by eye. The velocity field is checked
against real image flow by normalised cross-correlation, which shares no
assumption with the matrix that produced it — the tooling for that is in
`tools/measure/`, with a self-test that verifies the instrument against known
answers before it is trusted.

## Install

Download the zip from [Releases](../../releases), drag its contents into your
X-Plane 12 folder, and let the folders merge. Then start the sim with
`MotionVectors\MotionVectorsLauncher.exe` instead of `X-Plane.exe`.

The launcher is required. This is an *explicit* Vulkan layer, and the loader
only enables one when `VK_LAYER_PATH` and `VK_LOADER_LAYERS_ENABLE` are set in
the process environment. A plugin cannot set them, because Vulkan is already
running by the time plugins load. The alternative — registering an implicit
layer in the registry — is a machine-wide change affecting every Vulkan
application, so it is deliberately not done.

Nothing is written outside your X-Plane folder. Uninstalling is deleting five
files.

## Requirements

- X-Plane 12 on the Vulkan renderer, not OpenGL
- Windows
- FlyWithLua, for the settings panel

## Performance

Roughly 30 fps in flight at 4K against 38 with the mod disabled, on the
development machine — about a fifth of the frame budget. One machine, one data
point.

## Settings

Every value is live-editable in `%TEMP%\taa_live.ini` while the sim is running;
the layer re-reads it within a few frames, so changes apply without a restart.
`MotionVectors\taa_live.ini.reference` in the install lists every key with its
shipped value.

| key | what it does |
|---|---|
| `taa.enable` | 0 disables the resolve entirely |
| `taa.alpha` | blend weight when still — lower keeps history longer |
| `taa.alpha_moving` | blend weight while moving, ramped by each pixel's speed |
| `taa.varclip` | neighbourhood clamp width — lower rejects more history |
| `taa.reactive` | transparency mask, off by default (see below) |
| `vram.*` | budget shaping, recycling, priorities, upload governor |

## Antivirus

Some scanners flag `MotionVectorsLauncher.exe`. It is a false positive, and the
reason is structural: the launcher sets two environment variables and starts
`X-Plane.exe`, which is the same shape as a loader, and it is small, new and
unsigned. Every such detection so far has been machine-learning generated —
Microsoft's `Wacatac.B!ml`, Symantec's `ML.Attribute.HighConfidence` — and none
names an actual malware family.

The launcher is 132 lines of source in this repository, and the built binary
imports exactly four Windows functions: `SetEnvironmentVariableA`,
`CreateProcessA`, and `LoadLibraryA`/`GetProcAddress` from the C runtime. No
networking, no registry, no file writes. Build it yourself if you would rather
not take that on trust.

## Known issues

`taa.reactive` is **off by default**. It exists to stop a propeller disc
accumulating, but on the test aircraft it measured 4.34× the temporal flicker
of no TAA at all while making no measurable difference to detail. Four faults
were found and fixed in its coverage path and none of them changed that number,
so the cause is not yet understood. If you fly a propeller aircraft and see
artefacts around the disc, set `taa.reactive=1` and please open an issue.

TAA remains measurably less temporally stable than running without it, though
no longer dramatically so.

## Building

```
.\build.ps1        layer, plugin, launcher, shader
.\package.ps1      the release zip in dist\
```

Needs a MinGW-w64 g++ and `glslangValidator`. The Qt settings launcher is
optional and its build is allowed to fail without blocking a release.

## Logging

Everything is written to a dedicated log beside the plugin, not to X-Plane's
`Log.txt`.

## Licence

GPL-3.0. A distributed fork or derivative must publish its source under the
same terms. See [LICENSE](LICENSE).
