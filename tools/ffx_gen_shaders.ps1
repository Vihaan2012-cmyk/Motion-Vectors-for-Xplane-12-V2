# Generate every FidelityFX shader permutation header the Vulkan backend needs.
#
# WHY THIS IS A SCRIPT AND NOT CMAKE
#
# The SDK drives its generator from CMake, and that path needs AMD's ffx_sc.exe,
# which needs Visual Studio - see the header of tools/ffx_permute.cpp. This does
# the same job with tools/ffx_permute.exe and AMD's own bundled glslangValidator.
#
# THE ARGUMENTS ARE TRANSCRIBED, NOT CHOSEN
#
# Every define below is copied from the SDK's own CMakeCompile<Component>Shaders
# files. They are not tuning knobs - the backend packs a permutation key from
# exactly these options in exactly this order, so a difference here produces a
# header that compiles and then selects the wrong blob at runtime.
#
#   frameinterpolation  sdk/include/FidelityFX/gpu/frameinterpolation/CMakeCompileFrameinterpolationShaders.txt
#   fsr3upscaler        sdk/include/FidelityFX/gpu/fsr3upscaler/CMakeCompileFSR3UpscalerShaders.txt
#   opticalflow         sdk/include/FidelityFX/gpu/opticalflow/CMakeCompileOpticalflowShaders.txt
#
# THE FOUR VARIANTS PER PASS
#
# The blob accessors include <pass>, <pass>_wave64, <pass>_16bit and
# <pass>_wave64_16bit, so all four headers must exist or the backend will not
# compile. On the GLSL path the wave64 arguments are EMPTY - CMakeCompileShaders
# sets HLSL_WAVE64_ARGS only inside its HLSL branch - so wave64 differs from the
# base build in name alone. The real axis is FFX_HALF, which is 0 for the plain
# variants and 1 for the 16bit ones.
#
#   .\tools\ffx_gen_shaders.ps1              generate everything
#   .\tools\ffx_gen_shaders.ps1 -Component opticalflow
#
# Copyright (C) 2026 MotionVectors contributors
# SPDX-License-Identifier: GPL-3.0-or-later

param(
    [string]$Component = "all",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$sdk  = Join-Path $root "third_party\FidelityFX-SDK"

$permute = Join-Path $root "build\ffx_permute.exe"
$glslang = Join-Path $sdk  "sdk\tools\ffx_shader_compiler\libs\glslangValidator\bin\x64\glslangValidator.exe"
$gpuPath = Join-Path $sdk  "sdk\include\FidelityFX\gpu"
$shaderRoot = Join-Path $sdk "sdk\src\backends\vk\shaders"
$outDir  = Join-Path $root "build\ffx_shaders"

foreach ($p in @($permute, $glslang, $gpuPath, $shaderRoot)) {
    if (-not (Test-Path $p)) { throw "missing: $p" }
}
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# Transcribed from the SDK's CMakeCompile<Component>Shaders files. Order of
# Permute matters: it is the bit order of the permutation key.
$components = @{
    "frameinterpolation" = @{
        Base = @(
            "FFX_FRAMEINTERPOLATION_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0",
            "FFX_FRAMEINTERPOLATION_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0",
            "FFX_FRAMEINTERPOLATION_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1",
            "FFX_FRAMEINTERPOLATION_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0",
            "FFX_FRAMEINTERPOLATION_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2"
        )
        Permute = @(
            "FFX_FRAMEINTERPOLATION_OPTION_LOW_RES_MOTION_VECTORS",
            "FFX_FRAMEINTERPOLATION_OPTION_JITTER_MOTION_VECTORS",
            "FFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH"
        )
    }
    "fsr3upscaler" = @{
        Base = @(
            "FFX_FSR3UPSCALER_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0",
            "FFX_FSR3UPSCALER_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0",
            "FFX_FSR3UPSCALER_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1",
            "FFX_FSR3UPSCALER_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0",
            "FFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2"
        )
        Permute = @(
            "FFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE",
            "FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT",
            "FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS",
            "FFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS",
            "FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH",
            "FFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING"
        )
    }
    "opticalflow" = @{
        Base    = @()
        Permute = @("FFX_OPTICALFLOW_OPTION_HDR_COLOR_INPUT")
    }
}

$wanted = if ($Component -eq "all") { $components.Keys } else { @($Component) }
$totalHeaders = 0
$failed = @()
$sw = [System.Diagnostics.Stopwatch]::StartNew()

foreach ($comp in $wanted) {
    if (-not $components.ContainsKey($comp)) { throw "unknown component: $comp" }
    $cfg = $components[$comp]
    $srcDir = Join-Path $shaderRoot $comp
    $passes = Get-ChildItem -Path $srcDir -Filter "*.glsl" -ErrorAction SilentlyContinue
    if (-not $passes) { throw "no .glsl passes under $srcDir" }

    Write-Host ""
    Write-Host "==== $comp : $($passes.Count) passes, 2^$($cfg.Permute.Count) keys each ====" -ForegroundColor Cyan

    foreach ($pass in $passes) {
        $passName = [System.IO.Path]::GetFileNameWithoutExtension($pass.Name)

        # name suffix -> FFX_HALF. wave64 adds nothing on the GLSL path, but the
        # accessor includes the header by that name, so it must be produced.
        $variants = @(
            @{ Suffix = "";              Half = 0 },
            @{ Suffix = "_wave64";       Half = 0 },
            @{ Suffix = "_16bit";        Half = 1 },
            @{ Suffix = "_wave64_16bit"; Half = 1 }
        )

        foreach ($v in $variants) {
            $name   = "$passName$($v.Suffix)"
            $header = Join-Path $outDir "${name}_permutations.h"
            if ((Test-Path $header) -and (-not $Force)) { $totalHeaders++; continue }

            $args = @(
                "--glslang", $glslang,
                "--src",     $pass.FullName,
                "--name",    $name,
                "--out",     $header,
                "--tmp",     (Join-Path $env:TEMP "ffx_permute\$name"),
                "-I",        $gpuPath,
                "-I",        (Join-Path $gpuPath $comp),
                "-D",        "FFX_GLSL=1",
                "-D",        "FFX_GPU=1",
                "-D",        "FFX_HALF=$($v.Half)"
            )
            foreach ($d in $cfg.Base)    { $args += @("-D", $d) }
            foreach ($p in $cfg.Permute) { $args += @("--permute", $p) }

            & $permute @args
            if ($LASTEXITCODE -ne 0) {
                # Recorded and carried on rather than thrown: one pass failing
                # should not hide whether the other thirty work.
                Write-Host "  FAILED: $name" -ForegroundColor Red
                $failed += $name
            } else {
                $totalHeaders++
            }
        }
    }
}

$sw.Stop()
Write-Host ""
Write-Host "$totalHeaders header(s) in $([int]$sw.Elapsed.TotalSeconds)s -> $outDir" -ForegroundColor Green
if ($failed.Count -gt 0) {
    Write-Host "$($failed.Count) FAILED:" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}
