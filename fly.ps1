# Move the camera without anyone present.
#
# The diagnostic suite is only interesting under MOTION - a still camera makes
# the reprojection legitimately identity and every rate reads clean, which is
# exactly the reading that wasted a evening. This drags the mouse across the
# sim window with the left button held, which is how X-Plane pans the view in
# both cockpit and external views, then pauses so the accumulation settles
# before the next sweep.
#
# PostMessage rather than a synthetic click, for the reason resume-flight.ps1
# documents: a posted message goes to X-Plane's own queue and does not need the
# window in the foreground, so this survives builds and greps running alongside.
param([int]$Seconds = 240)

Add-Type @'
using System;
using System.Runtime.InteropServices;
public class FWin {
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, uint msg, IntPtr wp, IntPtr lp);
    [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
    public struct RECT { public int L, T, R, B; }
}
'@
[void][FWin]::SetProcessDPIAware()

$WM_MOUSEMOVE = 0x0200; $WM_LBUTTONDOWN = 0x0201; $WM_LBUTTONUP = 0x0202
$MK_LBUTTON = [IntPtr]1

$deadline = (Get-Date).AddSeconds($Seconds)
while ((Get-Date) -lt $deadline) {
    $p = Get-Process X-Plane -ErrorAction SilentlyContinue |
         Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
    if (-not $p) { Start-Sleep -Seconds 5; continue }
    $h = $p.MainWindowHandle
    $r = New-Object FWin+RECT
    [void][FWin]::GetClientRect($h, [ref]$r)
    $w = $r.R - $r.L; $ht = $r.B - $r.T
    if ($w -le 0) { Start-Sleep -Seconds 5; continue }

    # A sweep: press near the middle, drag sideways in steps, release. Each step
    # is its own posted move so the sim sees continuous motion rather than one
    # teleport - a single jump reads as a camera cut and resets history, which
    # is not the state worth measuring.
    foreach ($dir in @(1, -1)) {
        $cy = [int]($ht * 0.5)
        $x0 = [int]($w * 0.5)
        $lp0 = [IntPtr](($cy -shl 16) -bor $x0)
        [void][FWin]::PostMessage($h, $WM_MOUSEMOVE,   [IntPtr]::Zero, $lp0)
        [void][FWin]::PostMessage($h, $WM_LBUTTONDOWN, $MK_LBUTTON,    $lp0)
        for ($i = 1; $i -le 24; $i++) {
            $x = $x0 + $dir * [int]($w * 0.012 * $i)
            if ($x -lt 2) { $x = 2 }; if ($x -gt ($w - 2)) { $x = $w - 2 }
            $lp = [IntPtr](($cy -shl 16) -bor $x)
            [void][FWin]::PostMessage($h, $WM_MOUSEMOVE, $MK_LBUTTON, $lp)
            Start-Sleep -Milliseconds 90
        }
        [void][FWin]::PostMessage($h, $WM_LBUTTONUP, [IntPtr]::Zero, $lp0)
        # Let history accumulate while still: the suite wants both states.
        Start-Sleep -Seconds 4
    }
}
