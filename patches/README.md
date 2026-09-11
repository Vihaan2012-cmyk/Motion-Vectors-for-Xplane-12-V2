# Vendored FidelityFX SDK patch

`third_party/` is ignored, so our changes to AMD's SDK live here as a patch against
FidelityFX-SDK commit `c6efa6b` (SDK 1.1.4, frame interpolation 1.1.3).

```
git clone https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK third_party/FidelityFX-SDK
git -C third_party/FidelityFX-SDK checkout c6efa6b
git -C third_party/FidelityFX-SDK apply ../../patches/ffx-sdk-c6efa6b-motionvectors.patch
```

What it carries: GCC fixes for the VK frame-interpolation swapchain, the layer's direct-call
resolver hooks in ffx_vk.cpp, and the trusted-vector / static-snap logic in the interpolation
pass (dilated MV binding 11, flags bits 16/17, epsilon byte). The regenerated pass blobs are
committed under build/ffx_shaders; tools/ffx_regen_fi.ps1 rebuilds them with the SDK's bundled
glslang 11.12.0 (the Vulkan SDK's glslang produces different, non-identical blobs).
