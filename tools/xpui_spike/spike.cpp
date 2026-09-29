// xpui_spike - THROWAWAY feasibility probe, not part of the mod.
//
// Question: does a plugin window render on X-Plane 12's start screen, before any
// flight is loaded? FlyWithLua cannot answer it (its scripts start with a flight);
// a compiled plugin is loaded at sim start, so it can - if the sim draws plugin
// windows over its own start screen at all. Nothing in the SDK headers says.
//
// Evidence goes to Log.txt, not to a screenshot:
//   "XPUI_SPIKE: draw #N (no flight loaded yet)"  - the window was drawn before a flight
//   "XPUI_SPIKE: flight loaded (plane/airport message)" - the first flight arrived
// If draw lines precede the flight line, the window renders on the start screen.
//
// Build (from this folder):
//   g++ -shared -o win.xpl spike.cpp -I<SDK>/CHeaders/XPLM -DIBM=1 -DXPLM200 -DXPLM210
//       -DXPLM300 -DXPLM301 -m64 -O2 -static -static-libgcc -static-libstdc++
//       -L<SDK>/Libraries/Win -lXPLM_64 -lopengl32
// Install: Resources/plugins/XPUISpike/64/win.xpl   Remove: delete that folder.

#include <windows.h>
#include <GL/gl.h>
#include <cstdio>
#include <cstring>
#include "XPLMDisplay.h"
#include "XPLMGraphics.h"
#include "XPLMPlugin.h"
#include "XPLMUtilities.h"

static XPLMWindowID g_win = nullptr;
static bool g_flight = false;
static unsigned long g_draws = 0;

static void logf(const char *fmt, unsigned long n)
{
    char b[256];
    snprintf(b, sizeof(b), fmt, n);
    XPLMDebugString(b);
}

static void draw(XPLMWindowID id, void *)
{
    ++g_draws;
    // Log the first few draws and then one per ~10 s at 60 fps, so the log stays short.
    if (g_draws <= 3 || (g_draws % 600) == 0)
        logf(g_flight ? "XPUI_SPIKE: draw #%lu (flight loaded)\n" : "XPUI_SPIKE: draw #%lu (no flight loaded yet)\n", g_draws);

    int l, t, r, b;
    XPLMGetWindowGeometry(id, &l, &t, &r, &b);
    XPLMSetGraphicsState(0, 0, 0, 0, 1, 0, 0);
    // --black-1 panel with a --lite-blue-3 header strip, X-Plane 12's own tokens
    glColor4f(0x15 / 255.0f, 0x19 / 255.0f, 0x1a / 255.0f, 0.94f);
    glBegin(GL_QUADS); glVertex2i(l, t); glVertex2i(r, t); glVertex2i(r, b); glVertex2i(l, b); glEnd();
    glColor4f(0x00 / 255.0f, 0x81 / 255.0f, 0xb3 / 255.0f, 1.0f);
    glBegin(GL_QUADS); glVertex2i(l, t); glVertex2i(r, t); glVertex2i(r, t - 28); glVertex2i(l, t - 28); glEnd();

    float white[3] = {1.0f, 1.0f, 1.0f}, grey[3] = {0.70f, 0.74f, 0.74f};
    XPLMDrawString(white, l + 12, t - 19, (char *)"FLIGHT CONFIGURATION (plugin spike)", nullptr, xplmFont_Proportional);
    XPLMDrawString(grey, l + 12, t - 52, (char *)"If you can read this on the start screen,", nullptr, xplmFont_Proportional);
    XPLMDrawString(grey, l + 12, t - 70, (char *)"the full UI can live here before a flight.", nullptr, xplmFont_Proportional);
    char n[64]; snprintf(n, sizeof(n), "frames drawn: %lu   flight loaded: %s", g_draws, g_flight ? "yes" : "no");
    XPLMDrawString(white, l + 12, t - 96, n, nullptr, xplmFont_Proportional);
}

static int  click(XPLMWindowID, int, int, XPLMMouseStatus, void *) { return 1; }
static void key(XPLMWindowID, char, XPLMKeyFlags, char, void *, int) {}
static XPLMCursorStatus cursor(XPLMWindowID, int, int, void *) { return xplm_CursorDefault; }
static int  wheel(XPLMWindowID, int, int, int, int, void *) { return 1; }

PLUGIN_API int XPluginStart(char *name, char *sig, char *desc)
{
    strcpy(name, "XP UI spike");
    strcpy(sig, "com.vihaan2012.xpuispike");
    strcpy(desc, "Throwaway probe: does a plugin window render on the X-Plane 12 start screen?");
    XPLMDebugString("XPUI_SPIKE: plugin started\n");
    return 1;
}

PLUGIN_API int XPluginEnable(void)
{
    int gl, gt, gr, gb;
    XPLMGetScreenBoundsGlobal(&gl, &gt, &gr, &gb);
    XPLMCreateWindow_t p;
    memset(&p, 0, sizeof(p));
    p.structSize = sizeof(p);
    p.left = gl + 60; p.top = gt - 120; p.right = gl + 480; p.bottom = gt - 240;
    p.visible = 1;
    p.drawWindowFunc = draw;
    p.handleMouseClickFunc = click;
    p.handleRightClickFunc = click;
    p.handleKeyFunc = key;
    p.handleCursorFunc = cursor;
    p.handleMouseWheelFunc = wheel;
    p.decorateAsFloatingWindow = xplm_WindowDecorationRoundRectangle;
    p.layer = xplm_WindowLayerFloatingWindows;
    g_win = XPLMCreateWindowEx(&p);
    XPLMSetWindowTitle(g_win, "XP UI spike");
    XPLMDebugString(g_win ? "XPUI_SPIKE: window created at enable (before any flight)\n" : "XPUI_SPIKE: window creation FAILED\n");
    return 1;
}

PLUGIN_API void XPluginDisable(void)
{
    if (g_win) { XPLMDestroyWindow(g_win); g_win = nullptr; }
}

PLUGIN_API void XPluginStop(void) {}

PLUGIN_API void XPluginReceiveMessage(XPLMPluginID, int msg, void *)
{
    if (!g_flight && (msg == XPLM_MSG_PLANE_LOADED || msg == XPLM_MSG_AIRPORT_LOADED)) {
        g_flight = true;
        logf("XPUI_SPIKE: flight loaded (plane/airport message) after %lu start-screen draws\n", g_draws);
    }
}
