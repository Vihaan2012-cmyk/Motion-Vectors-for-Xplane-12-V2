# Offline unit tests for the pure-maths half of the crash destruction system.
#
# Nothing here needs X-Plane. Crash detection, grid classification, fragment
# integration, ground contact and the constraint solver are all arithmetic, and
# arithmetic that can be tested at a command line should never be tested by
# flying an aeroplane into the ground - which is slow, unrepeatable, and cannot
# run before a commit.
#
#   .\tests.ps1
#
# Exit code 0 when every check passed.

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$out  = Join-Path $root "build"
New-Item -ItemType Directory -Force $out | Out-Null

# -Wall -Wextra deliberately: these headers are also compiled into the plugin
# and the layer, and a warning here is cheaper than finding it inside a Vulkan
# callback.
& g++ -o "$out\test_destruct.exe" "$root\src\test_destruct.cpp" `
      -I"$root\src" -m64 -O1 -std=c++17 -Wall -Wextra
if ($LASTEXITCODE -ne 0) { throw "test build failed" }

& "$out\test_destruct.exe"
exit $LASTEXITCODE
