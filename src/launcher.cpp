// MotionVectors for X-Plane 12 - temporal anti-aliasing from injected motion
// vectors, plus a VRAM manager.
//
// Copyright (C) 2026 Vihaan2012
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
// more details.
//
// You should have received a copy of the GNU General Public License along
// with this program. If not, see <https://www.gnu.org/licenses/>.

// Launcher: start X-Plane with the motion vector layer enabled, and only then.
//
// The alternative was registering the layer through the loader's ImplicitLayers
// key, which loads the DLL into EVERY Vulkan application on the machine. The
// layer declines to do anything outside X-Plane, but declining still means it
// was loaded, and that is a cost paid by unrelated software for no reason.
//
// An EXPLICIT layer is only loaded when something asks for it. Nothing asks
// unless these two variables are set, so setting them here and starting the sim
// scopes the whole thing to this one process:
//
//   VK_LAYER_PATH            where the loader looks for the manifest
//   VK_LOADER_LAYERS_ENABLE  which layers to turn on
//
// VK_INSTANCE_LAYERS is the older spelling and is ignored silently by loader
// 1.3.234 and newer, which is a good way to spend an afternoon wondering why
// nothing loaded.

#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>   // getenv, for the TEMP path the layer reads its config from

// Everything after our own arguments is handed to X-Plane untouched, so
// --load_smo and friends still work through the launcher.
static const char *kLayerName = "VK_LAYER_mv";

static void dirOf(const char *path, char *out, size_t n)
{
    strncpy(out, path, n - 1);
    out[n - 1] = 0;
    char *slash = strrchr(out, 0x5C);      // 0x5C is a backslash
    if (slash) *slash = 0;
}

static bool fileExists(const char *p)
{
    DWORD a = GetFileAttributesA(p);
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

int main(int argc, char **argv)
{
    char self[MAX_PATH] = {0};
    GetModuleFileNameA(nullptr, self, sizeof(self) - 1);

    // The launcher is installed inside the X-Plane folder, so the sim is beside
    // it and the layer is in our own directory.
    char here[MAX_PATH];
    dirOf(self, here, sizeof(here));

    char xplane[MAX_PATH];
    snprintf(xplane, sizeof(xplane), "%s\\X-Plane.exe", here);

    // Also accept being installed one level down, next to the layer.
    char layerDir[MAX_PATH];
    snprintf(layerDir, sizeof(layerDir), "%s", here);
    if (!fileExists(xplane)) {
        char up[MAX_PATH];
        dirOf(here, up, sizeof(up));
        snprintf(xplane, sizeof(xplane), "%s\\X-Plane.exe", up);
        if (!fileExists(xplane)) {
            MessageBoxA(nullptr,
                "X-Plane.exe was not found next to this launcher.\n\n"
                "Put the launcher in your X-Plane 12 folder, or in a folder "
                "directly inside it.",
                "Motion Vectors", MB_ICONERROR | MB_OK);
            return 1;
        }
    }

    char manifest[MAX_PATH];
    snprintf(manifest, sizeof(manifest), "%s\\VkLayer_mv.json", layerDir);
    if (!fileExists(manifest)) {
        char msg[MAX_PATH + 160];
        snprintf(msg, sizeof(msg),
                 "The layer manifest was not found:\n\n%s\n\n"
                 "X-Plane will start without motion vectors.", manifest);
        MessageBoxA(nullptr, msg, "Motion Vectors", MB_ICONWARNING | MB_OK);
    }

    SetEnvironmentVariableA("VK_LAYER_PATH", layerDir);
    SetEnvironmentVariableA("VK_LOADER_LAYERS_ENABLE", kLayerName);

    // ---- ACTUALLY TURN THE MOD ON.  (v1.1.0 shipped without this.)
    //
    // Attaching the layer is not enough. The layer only INJECTS motion vectors
    // when TAA_VELOCITY=1 (velArmed, layer.cpp), and only RESOLVES when
    // TAA_RESOLVE=1 or taa.enable=1 in the live config. The dev launch script
    // set TAA_VELOCITY; this launcher did not - so every v1.1.0 user got the
    // layer loaded but ZERO velocity, hence no TAA at all. Set both here so the
    // mod works out of the box even if the taa_live.ini seed below is skipped.
    SetEnvironmentVariableA("TAA_VELOCITY", "1");
    SetEnvironmentVariableA("TAA_RESOLVE",  "1");

    // ---- SEED THE TUNED SETTINGS ON FIRST RUN.
    //
    // The layer reads its live config from %TEMP%\taa_live.ini (getenv("TEMP")),
    // and writes a blank template there if none exists - so a fresh machine
    // would start on compiled defaults, not the tuning this build ships with.
    // Copy the packaged taa_live.ini (beside this launcher) into place, but ONLY
    // if the user has no file yet: existing tweaks are never overwritten, and the
    // shipped settings apply out of the box otherwise. Same TEMP resolution the
    // layer uses, so the launcher writes exactly the file the layer will read.
    {
        char shipped[MAX_PATH];
        snprintf(shipped, sizeof(shipped), "%s\\taa_live.ini", layerDir);
        const char *tmp = getenv("TEMP");
        if (tmp && *tmp && fileExists(shipped)) {
            char dst[MAX_PATH];
            snprintf(dst, sizeof(dst), "%s\\taa_live.ini", tmp);
            CopyFileA(shipped, dst, TRUE);   // TRUE = keep any existing user file
        }
    }

    // Rebuild the command line: our own path first, then anything the user
    // passed, quoted so paths with spaces survive.
    char cmd[8192];
    int n = snprintf(cmd, sizeof(cmd), "\"%s\"", xplane);
    for (int i = 1; i < argc && n < (int)sizeof(cmd) - 4; ++i)
        n += snprintf(cmd + n, sizeof(cmd) - n, " %s", argv[i]);

    char workdir[MAX_PATH];
    dirOf(xplane, workdir, sizeof(workdir));

    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof(si));
    memset(&pi, 0, sizeof(pi));
    si.cb = sizeof(si);

    if (!CreateProcessA(nullptr, cmd, nullptr, nullptr, FALSE,
                        0, nullptr, workdir, &si, &pi)) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Could not start X-Plane (error %lu).",
                 (unsigned long)GetLastError());
        MessageBoxA(nullptr, msg, "Motion Vectors", MB_ICONERROR | MB_OK);
        return 1;
    }

    // Exit immediately. Waiting would leave a process in the task list for no
    // purpose - the sim is independent of us once it is running.
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return 0;
}
