# Dismiss the two panels that block an unattended launch.
#
# Killing the sim to unlock the layer DLL - which every bisect run does - looks
# exactly like a crash to X-Plane, so the NEXT launch meets SUBMIT CRASH REPORT
# and SAFE MODE. SAFE MODE is the dangerous one: its Plugins box is ticked by
# default, so accepting it starts a session with our plugin absent, which reads
# afterwards as "the layer stopped working" rather than as an unanswered dialog.
#
# Both answer to No Thanks. Clicking those points costs nothing when the panels
# are absent - on the menu they are empty sky above the tiles. Coordinates and
# the PostMessage approach come from resume-flight.ps1; see its header for why a
# posted click beats a synthetic one in an unattended loop.
param([int]$WatchSec = 120)

Add-Type @'
using System;
using System.Runtime.InteropServices;
public class DWin {
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, uint msg, IntPtr wp, IntPtr lp);
    [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
    public struct RECT { public int L, T, R, B; }
}
'@
[void][DWin]::SetProcessDPIAware()

$WM_MOUSEMOVE = 0x0200; $WM_LBUTTONDOWN = 0x0201; $WM_LBUTTONUP = 0x0202
$MK_LBUTTON = [IntPtr]1
$dismiss = @(
    @(0.5185, 0.5554),   # SUBMIT CRASH REPORT -> No Thanks
    @(0.5470, 0.5920)    # SAFE MODE           -> No Thanks
)

$deadline = (Get-Date).AddSeconds($WatchSec)
while ((Get-Date) -lt $deadline) {
    $proc = Get-Process X-Plane -ErrorAction SilentlyContinue |
            Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
    if ($proc) {
        $h = $proc.MainWindowHandle
        $r = New-Object DWin+RECT
        [void][DWin]::GetClientRect($h, [ref]$r)
        $w = $r.R - $r.L; $ht = $r.B - $r.T
        if ($w -gt 0) {
            foreach ($p in $dismiss) {
                $cx = [int]($w * $p[0]); $cy = [int]($ht * $p[1])
                $lp = [IntPtr](($cy -shl 16) -bor $cx)
                [void][DWin]::PostMessage($h, $WM_MOUSEMOVE,   [IntPtr]::Zero, $lp)
                Start-Sleep -Milliseconds 120
                [void][DWin]::PostMessage($h, $WM_LBUTTONDOWN, $MK_LBUTTON,    $lp)
                Start-Sleep -Milliseconds 90
                [void][DWin]::PostMessage($h, $WM_LBUTTONUP,   [IntPtr]::Zero, $lp)
                Start-Sleep -Milliseconds 300
            }
        }
    }
    Start-Sleep -Seconds 5
}
