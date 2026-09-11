# Regenerate the four permutation headers of FFX's frame-interpolation MAIN pass with our
# own generator (build\ffx_permute.exe; AMD's needs MSVC). One run per variant:
#   base, _16bit (FFX_HALF=1), _wave64, _wave64_16bit.
# On the 4060 the _16bit header is the one that runs (fp16 supported, wave64 never forced);
# the wave64 twins are byte-identical to their non-wave64 counterparts for this pass.
#
# Gate (docs/superpowers/specs/2026-09-10-fg-trusted-vectors-design.md, section 7): from the
# UNMODIFIED sources this must reproduce digest 59247a0deabd3e29 (base, 19108 B) and
# d2e13637e46a9843 (16bit, 19140 B). Run with -OutDir $env:TEMP\ffx_gate for that check.
param([string]$OutDir = "")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if ($OutDir -eq "") { $OutDir = Join-Path $root "build\ffx_shaders" }
# AMD's bundled glslang (11.12.0), NOT the Vulkan SDK's (16.4.0): the headers in build\ffx_shaders
# were made with the bundled one and the two optimise differently (-Os), so only the bundled
# compiler reproduces the shipped digests. Measured 2026-09-10: SDK 16.4.0 gave a 232-byte-smaller
# blob with a different instruction mix from identical sources.
$G = Join-Path $root "third_party\FidelityFX-SDK\sdk\tools\binary_store\glslangValidator.exe"
$permute = Join-Path $root "build\ffx_permute.exe"
$src = Join-Path $root "third_party\FidelityFX-SDK\sdk\src\backends\vk\shaders\frameinterpolation\ffx_frameinterpolation_pass.glsl"
$inc = Join-Path $root "third_party\FidelityFX-SDK\sdk\include\FidelityFX\gpu"
foreach ($p in @($G, $permute, $src, $inc)) { if (-not (Test-Path $p)) { throw "missing: $p" } }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$opts = @("--permute", "FFX_FRAMEINTERPOLATION_OPTION_LOW_RES_MOTION_VECTORS",
          "--permute", "FFX_FRAMEINTERPOLATION_OPTION_JITTER_MOTION_VECTORS",
          "--permute", "FFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH")
$variants = @(
    @{ name = "ffx_frameinterpolation_pass";              defs = @() },
    @{ name = "ffx_frameinterpolation_pass_16bit";        defs = @("-D", "FFX_HALF=1") },
    @{ name = "ffx_frameinterpolation_pass_wave64";       defs = @("-D", "FFX_PREFER_WAVE64=1") },
    @{ name = "ffx_frameinterpolation_pass_wave64_16bit"; defs = @("-D", "FFX_HALF=1", "-D", "FFX_PREFER_WAVE64=1") })
foreach ($v in $variants) {
    $out = Join-Path $OutDir ($v.name + "_permutations.h")
    $a = @("--glslang", $G, "--src", $src, "--name", $v.name, "--out", $out, "-I", $inc, "-D", "FFX_GPU=1", "-D", "FFX_GLSL=1") + $v.defs + $opts
    & $permute @a
    if ($LASTEXITCODE -ne 0) { throw ("ffx_permute failed for " + $v.name) }
    $m = (Select-String -Path $out -Pattern ("g_" + $v.name + "_([0-9a-f]{16})_size = (\d+)") | Select-Object -First 1).Matches[0]
    Write-Host ("  {0,-45} digest={1} bytes={2}" -f $v.name, $m.Groups[1].Value, $m.Groups[2].Value)
}
