# NN capture format (Stage 0 contract)

One directory per capture session, one sub-directory per captured frame.
Everything is raw little-endian, row-major, top row first, `[h][w][c]`.
No compression; a session at 2953x1661 costs ~55 MB per frame.

```
<session>/                       e.g. D:\NNCap\2026-09-08_2215_stock_KSFO\
  session.json                   written once: layer build, X-Plane version,
                                 display size, capture profile name, live-ini
                                 snapshot at session start
  frame_000123/
    manifest.json                per-frame metadata (schema below)
    color.bin                    f16 x4  RGBA   linear HDR scene colour, RENDER res,
                                            as the resolve sampled it (jittered raster)
    velocity.bin                 f16 x4  R16G16B16A16_SFLOAT velocity target:
                                            xy = NDC displacement curr->prev * 0.5
                                            (layer convention, see taa.comp mvReconstructVel),
                                            z = clip-w of the fragment (view distance),
                                            w = coverage; xy == +64/+64 px-equivalent sentinel = unwritten
    depth.bin                    f32 x1  R32_SFLOAT engine depth (depthcopy image), REVERSE-Z:
                                            dist = 1/(clip.z*d + clip.w) with gClipInfo from manifest
    normal.bin                   f16 x2  R16G16_SFLOAT engine normal, Lambert azimuthal (spheremap)
                                            encoding: n.xy = e*sqrt(1 - dot(e,e)/4), n.z = 1 - dot(e,e)/2
    resolved.bin                 f16 x4  the TAA resolve's output this frame (history write), RENDER res
    gbuf_0.bin ... gbuf_N.bin    the scene pass's colour attachments by index, formats in manifest
                                 (albedo / material / emissive are among these; identity is
                                 established by the validator, not assumed)
                                 Copied at the END of the G-buffer pass (not at the resolve):
                                 by resolve time X-Plane has re-laid-out / reused those targets
                                 and a copy there is scrambled (measured 2026-09-09).
    preview.bmp                  optional 24-bit preview of `resolved` (tonemapped x/(1+x)), for eyes
```

## manifest.json

```json
{
  "schema": 1,
  "frame": 12345,                       // g_frameCount (presented frames)
  "sim_time": 43210.5,                  // plugin simTime, seconds
  "render": {"w": 2953, "h": 1661, "layers": 1},
  "display": {"w": 3840, "h": 2160},
  "render_scale": 0.769,                // render.w / display.w
  "jitter": {"x": 0.375, "y": 0.0556, "index": 6, "phases": 8, "scale": 1.0},
                                        // x,y in PIXELS at render res, centred (+/-0.5);
                                        // Halton(2,3): x = halton(index+1,2)-0.5, y = halton(index+1,3)-0.5
                                        // scale = taa.jitter_scale actually applied (0 => unjittered)
  "proj": [16 floats, column-major, OpenGL convention, from the plugin],
  "world": [16 floats],
  "cam": {"x": 0, "y": 0, "z": 0},      // world position, metres
  "fov_deg": 65.0, "near": 0.1, "far": 524288.0, "reverse_z": 1, "infinite_far": 0,
  "view_type": 1026,
  "clip_info": [x, y, z, w],            // gClipInfo[0] from the tapped u_gbuffer_data block, or null
  "sun_view": [x, y, z],                // unit vector toward the sun, view space
  "planes": {
    "color":    {"format": "R16G16B16A16_SFLOAT", "w": 2953, "h": 1661, "c": 4, "dtype": "f16"},
    "velocity": {"format": "R16G16B16A16_SFLOAT", "w": 2953, "h": 1661, "c": 4, "dtype": "f16"},
    "depth":    {"format": "R32_SFLOAT",          "w": 2953, "h": 1661, "c": 1, "dtype": "f32"},
    "normal":   {"format": "R16G16_SFLOAT",       "w": 2953, "h": 1661, "c": 2, "dtype": "f16"},
    "resolved": {"format": "R16G16B16A16_SFLOAT", "w": 2953, "h": 1661, "c": 4, "dtype": "f16"},
    "gbuf_0":   {"format": "<vk format name>",     "w": 2953, "h": 1661, "c": 4, "dtype": "u8|f16|u32"}
  },
  "live": {"taa.alpha": "0.05", "taa.contact": "0.6", "...": "every taa.* key at capture time"}
}
```

Rules:
- A plane is present only if its `planes` entry exists. Readers never assume.
- `dtype` is one of `u8`, `f16`, `f32`, `u32` (raw bits of the Vulkan format;
  packed formats such as A2B10G10R10 are dumped as `u32` x1 and decoded by the reader).
- The writer never blocks the GPU: the copy is recorded into X-Plane's command
  buffer at the resolve, read back N frames later from a host-visible ring, and
  written from a worker thread. A dropped frame is logged, never stalled for.
- Python reader: `tools/nn/capture_io.py` — `load_frame(path) -> dict[str, np.ndarray] + meta`.

## Capture profiles

- `stock`: Custom Scenery reduced to Laminar packs only (`X-Plane Airports - *`,
  `X-Plane Landmarks - *`, `*GLOBAL_AIRPORTS*`), Laminar default aircraft only,
  `taa.jitter_scale` as the session demands (0 for ground-truth stills, 1 for
  temporal sequences). This is the "render base": X-Plane's own shaders on
  X-Plane's own content. The training set is built from this profile.
- `user`: whatever is installed. Supplement only, never the base.
