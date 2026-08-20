--[[ ===========================================================================
  TAA FOR X-PLANE 12 - in-sim panel

  Laid out as a 1:1 replica of X-Plane's activation dialog: dark panel, a
  caption line, a wide read-only field with a red status box beside it,
  green-bordered action buttons on the right, two grey-bordered buttons at the
  bottom, and a log area filling the middle.

  Everything this panel writes goes to the SAME live file the layer reads
  (%TEMP%\taa_live.ini), which the layer re-reads every few frames. So the panel
  and any script are driving one piece of state and cannot disagree - the same
  rule the previous panel followed with datarefs.

  Readouts are at the BOTTOM, under the advanced settings, so the top of the
  window stays the user-facing surface.
=========================================================================== ]]

local have = {}

-- FlyWithLua's dataref() reports a missing name through a hard error rather
-- than a nil, and a quarantined script takes the panel with it. XPLMFindDataRef
-- just returns nil, which is the answer we actually want.
local function try_dataref(name, kind)
    if XPLMFindDataRef == nil then have[name] = false return false end
    if XPLMFindDataRef(name) == nil then have[name] = false return false end
    local var = name:gsub("/", "_"):gsub("^taaimpl_", "MV_"):gsub("^sim_", "SIM_")
    local ok = pcall(dataref, var, name, kind)
    have[name] = ok
    return ok
end

try_dataref("taaimpl/layer_attached",     "readonly")
try_dataref("taaimpl/render_scale",       "readonly")
try_dataref("taaimpl/viewport_w",         "readonly")
try_dataref("taaimpl/viewport_h",         "readonly")
try_dataref("taaimpl/jitter_phases",      "readonly")
try_dataref("taaimpl/moving_objects",     "readonly")
try_dataref("taaimpl/reverse_z",          "readonly")
try_dataref("taaimpl/lod_bias",           "writable")
try_dataref("taaimpl/vram_used_mb",       "readonly")
try_dataref("taaimpl/vram_budget_mb",     "readonly")
try_dataref("taaimpl/vram_total_mb",      "readonly")
try_dataref("taaimpl/velocity_mb",        "readonly")
try_dataref("taaimpl/pipelines_patched",  "readonly")
try_dataref("taaimpl/pipelines_rejected", "readonly")
try_dataref("sim/version/xplane_internal_version", "readonly")

local function get(name, default)
    if not have[name] then return default end
    local var = name:gsub("/", "_"):gsub("^taaimpl_", "MV_"):gsub("^sim_", "SIM_")
    local v = _G[var]
    if v == nil then return default end
    return v
end

--[[ ---------------------------------------------------------------------------
  Colour, taken from the reference dialog rather than invented.

  ImGui takes ABGR, so these read backwards from the usual hex.
--------------------------------------------------------------------------- ]]
local C_PANEL     = 0xCE181818   -- window body, translucent
                                 -- (ABGR: the leading D0 is alpha, so the
                                 -- scene shows through as the reference does)
local C_FIELD     = 0x7B202020   -- the wide read-only field
local C_FIELD_ED  = 0x80707070   -- its border
local C_TEXT      = 0xFFE8E8E8
local C_DIM       = 0xFF707070
local C_GREEN     = 0xFF5BD65B   -- the green-bordered buttons
local C_GREEN_BG  = 0x832A3F2A
local C_GREY_ED   = 0x90A0A0A0   -- muted border, same family
local C_GREY_BG   = 0x7B202020
local C_RED       = 0xFF4B4BFF
local C_RED_BG    = 0x90303A6B   -- the *** box, dark red in ABGR
local C_AMBER     = 0xFF3BA7FF
local C_LABEL     = 0xFFB0793B
local C_LOG       = 0x6A101010   -- log box, translucent like the panel

-- Version the panel reports, and the newest X-Plane this build has been tested
-- against. No network check: bump this constant per release.
local MOD_VERSION      = "0.0.16"
local MAX_TESTED_XP    = "12.43.11"

-- A style push that cannot quarantine the script.
--
-- FlyWithLua builds differ in which imgui.constant.Col entries they expose, and
-- indexing a missing one throws - which takes the whole panel down, exactly the
-- failure the predecessor's panel died of. So every push is guarded and the
-- pops are counted, never assumed.
local function pushcol(name, col)
    local C = imgui.constant and imgui.constant.Col
    if not C or C[name] == nil then return 0 end
    imgui.PushStyleColor(C[name], col)
    return 1
end
local function popcol(n) if n > 0 then imgui.PopStyleColor(n) end end

local function text(col, s)
    local n = pushcol("Text", col)
    imgui.TextUnformatted(s)
    popcol(n)
end

local function same(x) imgui.SameLine(x) end

local function row(label, value, col, indent)
    text(C_LABEL, string.format("  %s", label))
    same(indent or 210)
    text(col or C_TEXT, value)
end

local function heading(s)
    imgui.TextUnformatted("")
    text(C_AMBER, s)
    imgui.Separator()
end

-- A button drawn in the reference's style. `enabled` false greys it and
-- swallows the click, which is how the un-implemented backends and the
-- Continue button are shown without a separate code path.
local function styled_button(label, w, h, edge, bg, enabled)
    local e = enabled and edge or C_DIM
    local t = enabled and edge or C_DIM
    local n = 0
    n = n + pushcol("Button",        bg)
    n = n + pushcol("ButtonHovered", bg)
    n = n + pushcol("ButtonActive",  bg)
    n = n + pushcol("Border",        e)
    n = n + pushcol("Text",          t)
    local clicked = imgui.Button(label, w, h)
    popcol(n)
    return clicked and enabled
end

--[[ ---------------------------------------------------------------------------
  The live control file.

  The layer re-reads this every few frames, so writing a key here is how the
  panel changes anything. Read-modify-write of whole lines: the file carries
  comments explaining every key and losing them would make the file useless to
  anyone editing it by hand.
--------------------------------------------------------------------------- ]]
local function ini_path()
    local t = os.getenv("TEMP") or os.getenv("TMP") or "."
    return t .. "\\taa_live.ini"
end

local function ini_read()
    local f = io.open(ini_path(), "r")
    if not f then return nil end
    local s = f:read("*a")
    f:close()
    return s
end

local function ini_get(key, default)
    local s = ini_read()
    if not s then return default end
    local v = s:match("\n" .. key:gsub("%.", "%%.") .. "=([^\r\n]*)")
             or s:match("^" .. key:gsub("%.", "%%.") .. "=([^\r\n]*)")
    if v == nil then return default end
    return v
end

local function ini_set(key, value)
    local s = ini_read()
    if not s then return false end
    local pat = key:gsub("%.", "%%.")
    local line = key .. "=" .. tostring(value)
    if s:match("\n" .. pat .. "=[^\r\n]*") or s:match("^" .. pat .. "=[^\r\n]*") then
        s = s:gsub("([\r\n]" .. pat .. ")=[^\r\n]*", "%1=" .. tostring(value), 1)
        s = s:gsub("^(" .. pat .. ")=[^\r\n]*", "%1=" .. tostring(value), 1)
    else
        if s:sub(-1) ~= "\n" then s = s .. "\n" end
        s = s .. line .. "\n"
    end
    local f = io.open(ini_path(), "w")
    if not f then return false end
    f:write(s)
    f:close()
    return true
end

--[[ ---------------------------------------------------------------------------
  The log box.
--------------------------------------------------------------------------- ]]
local log_lines = {}
local function logf(col, fmt, ...)
    local ok, s = pcall(string.format, fmt, ...)
    if not ok then s = fmt end
    log_lines[#log_lines + 1] = { col = col, s = s }
    if #log_lines > 200 then table.remove(log_lines, 1) end
end

--[[ ---------------------------------------------------------------------------
  Integrity.

  Checks the files this mod actually loads, in the places it loads them from.
  A missing shader blob or a stale layer DLL is the difference between "the mod
  is off" and "the mod is broken", and those look identical from the outside.
--------------------------------------------------------------------------- ]]
local function file_exists(p)
    local f = io.open(p, "rb")
    if f then f:close() return true end
    return false
end

local integrity_ok = nil    -- nil = not run, true/false = result

-- Set when the attach answer was not available at check time, cleared by the
-- re-check in the builder once the layer reports.
local attach_pending = false

local function verify_integrity()
    local root = SYSTEM_DIRECTORY or ""
    local base = root .. "MotionVectors\\"
    local checks = {
        { "Vulkan layer",     base .. "build\\vklayer\\VkLayer_mv.dll" },
        { "Layer manifest",   base .. "build\\vklayer\\VkLayer_mv.json" },
        { "Plugin",           root .. "Resources\\plugins\\MotionVectors\\64\\win.xpl" },
        { "Live controls",    ini_path() },
    }
    local bad = 0
    logf(C_AMBER, "Verifying files...")
    for _, c in ipairs(checks) do
        if file_exists(c[2]) then
            logf(C_GREEN, "  OK      %s", c[1])
        else
            logf(C_RED,   "  MISSING %s", c[1])
            bad = bad + 1
        end
    end
    -- The layer being present on disk says nothing about it being LOADED. That
    -- is a separate failure with a separate fix, so it is a separate line.
    -- ---- ATTACHMENT IS NOT KNOWN YET AT SCRIPT LOAD.
    --
    -- FlyWithLua runs this when the flight starts, which is BEFORE the layer
    -- has reported itself: the layer sets layerAttached on its first frame
    -- through the shared block, and the plugin only publishes the dataref
    -- after that. So a check here reads 0 and writes "NOT attached" into the
    -- log - a line that then sits there, stale and wrong, for the rest of the
    -- session while the layer is demonstrably running.
    --
    -- It cost real time twice tonight: the message was read as a genuine
    -- handshake failure and chased, when the plugin's own log said
    -- layerAttached=1 the whole time.
    --
    -- So say "waiting" rather than "PROBLEM", do not count it against the
    -- verdict, and let the deferred re-check below settle it once the answer
    -- exists.
    if get("taaimpl/layer_attached", 0) == 1 then
        logf(C_GREEN, "  OK      Vulkan layer is attached to this process")
    elseif not have["taaimpl/layer_attached"] then
        logf(C_RED,   "  PROBLEM Plugin dataref missing - the plugin is not loaded")
        bad = bad + 1
    else
        logf(C_AMBER, "  ...     Vulkan layer has not reported yet - rechecking")
        attach_pending = true
    end
    integrity_ok = (bad == 0)
    logf(integrity_ok and C_GREEN or C_RED,
         integrity_ok and "Integrity check passed." or
                          string.format("Integrity check found %d problem(s).", bad))
end

--[[ ---------------------------------------------------------------------------
  Version compatibility, and the countdown.

  X-Plane reports its version as an integer: 12.4.4 arrives as 124400. Anything
  newer than the constant above is untested rather than known-broken, so it
  prompts rather than refuses - and if nobody answers, the safe default is to
  take the mod out of the picture instead of leaving an untested combination
  running silently.
--------------------------------------------------------------------------- ]]
local function xp_version_string()
    local v = get("sim/version/xplane_internal_version", 0)
    if v == 0 then return "unknown" end
    local major = math.floor(v / 10000)
    local minor = math.floor((v % 10000) / 100)
    local patch = v % 100
    return string.format("%d.%d.%d", major, minor, patch)
end

local function version_is_newer(a, b)
    local am, an, ap = a:match("(%d+)%.(%d+)%.(%d+)")
    local bm, bn, bp = b:match("(%d+)%.(%d+)%.(%d+)")
    if not am or not bm then return false end
    am, an, ap = tonumber(am), tonumber(an), tonumber(ap)
    bm, bn, bp = tonumber(bm), tonumber(bn), tonumber(bp)
    if am ~= bm then return am > bm end
    if an ~= bn then return an > bn end
    return ap > bp
end

local prompt_active   = false
local prompt_deadline = 0
local prompt_resolved = false

local function begin_prompt()
    prompt_active   = true
    prompt_resolved = false
    prompt_deadline = os.clock() + 10.0
    logf(C_RED, "You are on X-Plane version %s, the latest compatible version is",
         xp_version_string())
    logf(C_RED, "X-Plane version %s! Would you like to continue?", MAX_TESTED_XP)
end

local function expire_prompt()
    prompt_active   = false
    prompt_resolved = true
    ini_set("taa.enable", "0")
    logf(C_AMBER, "No answer - the mod has been switched off for this session.")
    logf(C_DIM,   "Press On to enable it anyway.")
end

--[[ ---------------------------------------------------------------------------
  Startup.
--------------------------------------------------------------------------- ]]
local started = false
local function first_frame()
    started = true
    logf(C_AMBER, "TAA for X-Plane 12  -  version %s", MOD_VERSION)
    verify_integrity()
    local xp = xp_version_string()
    if xp ~= "unknown" and version_is_newer(xp, MAX_TESTED_XP) then
        begin_prompt()
    else
        logf(C_GREEN, "X-Plane %s is the tested version.", xp)
    end
end

--[[ ---------------------------------------------------------------------------
  Backends. Only TAA exists; the rest are drawn greyed so the roadmap is
  visible rather than implied.
--------------------------------------------------------------------------- ]]
local BACKENDS = { "TAA", "DLSS", "FSR", "XeSS", "DLAA", "FG", "MFG" }

local show_advanced = false

local win_bg_pushed = 0

local wnd = nil
local close_requested = false

-- Window geometry, tracked here because FlyWithLua can set it but not report
-- it. XPLM screen coordinates: origin bottom-left, so top > bottom.
local win_l, win_t = 120, 0        -- filled in on open()
local WIN_W, WIN_H = 720, 360
local dragging   = false
local title_held = false
local drag_mx, drag_my = 0, 0

local function apply_geometry()
    if wnd then
        float_wnd_set_geometry(wnd, win_l, win_t, win_l + WIN_W, win_t - WIN_H)
    end
end

local function handle_drag(held)
    if held then
        if not dragging then
            dragging = true
            drag_mx, drag_my = MOUSE_X, MOUSE_Y
        else
            local dx = MOUSE_X - drag_mx
            local dy = MOUSE_Y - drag_my
            if dx ~= 0 or dy ~= 0 then
                win_l, win_t = win_l + dx, win_t + dy
                drag_mx, drag_my = MOUSE_X, MOUSE_Y
                apply_geometry()
            end
        end
    else
        dragging = false
    end
end

-- A transparent grab strip, for empty space only.
--
-- The previous version covered the WHOLE panel and was submitted first, which
-- made it ImGui's active item on mouse-down - so every button and slider under
-- it stopped responding. This binding exposes no SetItemAllowOverlap, so the
-- only safe drag areas are ones nothing else occupies.
local function drag_strip(id, w, h)
    local n = pushcol("Button",        0x00000000)
            + pushcol("ButtonHovered", 0x18FFFFFF)
            + pushcol("ButtonActive",  0x28FFFFFF)
    imgui.Button(id, w, h)
    local held = imgui.IsItemActive()
    popcol(n)
    return held
end

-- The title bar: a flat full-width button that reports being held, plus a
-- close box. A Button is used rather than a label because IsItemActive only
-- answers for an interactive item.
local function title_bar()
    local n = 0
    n = n + pushcol("Button",        0x00000000)
    n = n + pushcol("ButtonHovered", 0x30FFFFFF)
    n = n + pushcol("ButtonActive",  0x30FFFFFF)
    n = n + pushcol("Text",          C_TEXT)
    imgui.Button("TAA For X-Plane 12", WIN_W - 96, 24)
    local held = imgui.IsItemActive()
    popcol(n)

    title_held = held

    same(WIN_W - 88)
    if styled_button("X", 44, 24, C_RED, C_GREY_BG, true) then
        close_requested = true
    end
    imgui.Separator()
end

local function build(w, x, y)
    -- Balance last frame's push before doing anything else.
    if win_bg_pushed > 0 then popcol(win_bg_pushed); win_bg_pushed = 0 end
    if not started then first_frame() end

    -- ---- SETTLE THE ATTACH QUESTION ONCE THE ANSWER EXISTS.
    --
    -- The layer reports itself a frame or two into the flight, after
    -- verify_integrity() has already run. Rather than leave a stale line in
    -- the log, watch for the transition and say so when it happens.
    if attach_pending and get("taaimpl/layer_attached", 0) == 1 then
        attach_pending = false
        logf(C_GREEN, "  OK      Vulkan layer is attached to this process")
        if integrity_ok ~= false then
            integrity_ok = true
            logf(C_GREEN, "Integrity check passed.")
        end
    end
    if prompt_active and os.clock() > prompt_deadline then expire_prompt() end

    -- The panel colour has to be PAINTED, not pushed: FlyWithLua has already
    -- begun the ImGui window by the time this builder runs, so PushStyleColor
    -- on WindowBg arrives too late. A full-size child with ChildBg is what
    -- gives the whole plugin one background, and its alpha is what lets the
    -- scene through the way the reference dialog does.
    local nWin = pushcol("ChildBg", 0x00000000)   -- the WINDOW carries the tint;
                                              -- tinting here too stacked two
                                              -- dark layers into a black band
    imgui.BeginChild("panel", -1, -1, false)
    title_bar()
    -- Drag strips live in genuinely empty space, so they can never sit on top
    -- of a control. The title bar is the main grab area; these make the rest of
    -- the panel feel draggable without stealing clicks.
    local strip_held = drag_strip("##dragmid", WIN_W - 260, 18)
    handle_drag(title_held or strip_held)

    -- Our own title, because the window is undecorated now: X-Plane's frame
    -- cannot be recoloured, so the only way to make the border match the panel
    -- is not to have one.

    ------------------------------------------------------- version + actions
    text(C_TEXT, "Version:")
    local yTop = imgui.GetCursorPosY()

    local nFld = pushcol("FrameBg", C_FIELD) + pushcol("Border", C_FIELD_ED)
    imgui.PushItemWidth(440)
    imgui.InputText("", string.format("Mod %s        X-Plane %s",
                                      MOD_VERSION, xp_version_string()), 128)
    imgui.PopItemWidth()
    popcol(nFld)

    -- Right-hand column, stacked, as the reference has Paste over Clear.
    local BX, BW = 470, 216
    imgui.SetCursorPosY(yTop)
    imgui.SetCursorPosX(BX)
    if styled_button("Verify Integrity", BW, 22, C_GREEN, C_GREEN_BG, true) then
        verify_integrity()
    end
    imgui.SetCursorPosX(BX)
    if styled_button("Reset Settings", BW, 22, C_GREEN, C_GREEN_BG, true) then
        for k, v in pairs({
            ["taa.enable"]       = "1",    ["taa.mode"]         = "3",
            ["taa.alpha"]        = "0.10", ["taa.alpha_moving"] = "0.35",
            ["taa.jitter_scale"] = "1.0",  ["taa.unjitter"]     = "0",
            ["taa.novec_cov"]    = "-1.0", ["taa.clear_mode"]   = "1",
            ["taa.mv_pass"]      = "-1",
        }) do ini_set(k, v) end
        logf(C_AMBER, "Settings reset to defaults.")
    end
    imgui.SetCursorPosX(BX)
    local cont_label = prompt_active
        and string.format("Continue  %.0fs", math.max(0, prompt_deadline - os.clock()))
        or  "Continue"
    if styled_button(cont_label, BW, 22, C_GREY_ED, C_GREY_BG, prompt_active) then
        prompt_active = false
        logf(C_GREEN, "Continuing on an untested X-Plane version.")
    end

    ------------------------------------------------------------------- log
    local nChild = pushcol("ChildBg", C_LOG)
    imgui.BeginChild("log", -1, 132, true)
    for _, l in ipairs(log_lines) do text(l.col, l.s) end
    imgui.EndChild()
    popcol(nChild)

    --------------------------------------------------------------- bottom row
    -- The empty band to the left of the bottom buttons doubles as a grab area.
    local strip2 = drag_strip("##dragbot", BX - 24, 46)
    if strip2 then handle_drag(true) end
    imgui.SetCursorPosY(imgui.GetCursorPosY() - 46)
    imgui.SetCursorPosX(BX)
    styled_button("Developed by Vihaan2012", BW, 22, C_GREY_ED, C_GREY_BG, false)
    imgui.SetCursorPosX(BX)
    local on = (ini_get("taa.enable", "1") == "1")
    if styled_button(on and "On" or "Off", BW, 22,
                     on and C_GREEN or C_GREY_ED,
                     on and C_GREEN_BG or C_GREY_BG, true) then
        ini_set("taa.enable", on and "0" or "1")
        logf(C_AMBER, on and "Mod switched off." or "Mod switched on.")
    end

    ---------------------------------------------------- everything below fold
    imgui.Separator()
    if imgui.Button(show_advanced and "Hide advanced settings"
                                   or "Show advanced settings", 220, 22) then
        show_advanced = not show_advanced
    end

    -- Backends live down here, not at the top. Only one of them exists, so a
    -- row of greyed buttons was taking the most valuable space on the panel to
    -- advertise things that do nothing.
    imgui.TextUnformatted("")
    for i, b in ipairs(BACKENDS) do
        local enabled = (b == "TAA")
        if i > 1 then same() end
        styled_button(b, 76, 22, enabled and C_GREEN or C_DIM,
                      enabled and C_GREEN_BG or C_GREY_BG, enabled)
    end
    text(C_DIM, "DLSS, FSR, XeSS, DLAA, FG and MFG are not implemented yet.")

    if show_advanced then
        imgui.TextUnformatted("")
        text(C_RED, "Do not change these unless you know what you are doing.")
        text(C_DIM, "Several of them can make the picture worse in ways that are")
        text(C_DIM, "not obvious until you are moving.")
        imgui.TextUnformatted("")

        local function num_setting(key, label, lo, hi)
            local cur = tonumber(ini_get(key, "0")) or 0
            imgui.PushItemWidth(200)
            local changed, v = imgui.SliderFloat(label, cur, lo, hi, "%.3f")
            imgui.PopItemWidth()
            if changed then ini_set(key, string.format("%.3f", v)) end
        end
        local function int_setting(key, label, lo, hi, default)
            local cur = tonumber(ini_get(key, tostring(default))) or default
            imgui.PushItemWidth(200)
            local ch, v = imgui.SliderFloat(label, cur, lo, hi, "%.0f")
            imgui.PopItemWidth()
            if ch then ini_set(key, string.format("%d", math.floor(v + 0.5))) end
        end

        num_setting("taa.alpha",        "alpha",        0.01, 1.0)
        num_setting("taa.alpha_moving", "alpha_moving", 0.01, 1.0)
        num_setting("taa.jitter_scale", "jitter_scale", 0.0,  1.0)
        num_setting("taa.vel_scale",    "vel_scale",    0.0,  2.0)
        int_setting("taa.mv_pass",      "mv_pass",     -2,   13, -1)
        int_setting("taa.clear_mode",   "clear_mode",   0,    2,  1)
        text(C_DIM, "mv_pass: -2 census winner, -1 all passes, 0..13 pin one.")
        text(C_DIM, "clear_mode: 0 never, 1 in-pass, 2 after the resolve.")
    end

    ------------------------------------------------------------- diagnostics
    if get("taaimpl/layer_attached", 0) ~= 1 then
        heading("STATUS")
        text(C_RED, "  VULKAN LAYER NOT ATTACHED")
        text(C_DIM, "  Start X-Plane from the Motion Vectors launcher. The layer")
        text(C_DIM, "  is enabled for that process only, so no other Vulkan")
        text(C_DIM, "  application ever loads it.")
    else
        heading("INJECTION")
        local rejected = get("taaimpl/pipelines_rejected", 0)
        row("PIPELINES CARRYING VELOCITY",
            string.format("%d", get("taaimpl/pipelines_patched", 0)))
        row("REJECTED BY THE DRIVER", string.format("%d", rejected),
            rejected == 0 and C_GREEN or C_RED)

        heading("RENDER")
        row("RESOLUTION", string.format("%d x %d", get("taaimpl/viewport_w", 0),
                                                  get("taaimpl/viewport_h", 0)))
        row("SCALE", string.format("%.2f x", get("taaimpl/render_scale", 1.0)))
        row("DEPTH", get("taaimpl/reverse_z", 0) == 1 and "reverse-Z" or "standard")
        row("JITTER", string.format("%d phases", get("taaimpl/jitter_phases", 0)))
        row("MOVING OBJECTS", string.format("%d", get("taaimpl/moving_objects", 0)))

        heading("MEMORY")
        row("PROCESS VRAM", string.format("%d MB", get("taaimpl/vram_used_mb", 0)))
        row("DRIVER BUDGET", string.format("%d / %d MB",
            get("taaimpl/vram_budget_mb", 0), get("taaimpl/vram_total_mb", 0)), C_DIM)

        heading("TUNING")
        local changed, v = imgui.SliderFloat("LOD bias", get("taaimpl/lod_bias", 0.0),
                                             -2.0, 0.0, "%.2f")
        if changed and have["taaimpl/lod_bias"] then MV_lod_bias = v end
    end

    imgui.EndChild()
    popcol(nWin)

    -- Applies to the NEXT Begin(), which is the only way to reach a window
    -- FlyWithLua has already opened. Popped at the top of the next frame.
    win_bg_pushed = pushcol("WindowBg", C_PANEL)
end

function mv_open()
    if wnd then
        float_wnd_destroy(wnd)
        wnd = nil
        return
    end
    -- decoration 0: no X-Plane frame. Its colour cannot be changed, so the
    -- only way to make the border match the panel is not to have one.
    wnd = float_wnd_create(WIN_W, WIN_H, 0, true)
    float_wnd_set_imgui_builder(wnd, "build")
    float_wnd_set_onclose(wnd, "onclose")
    -- Place it once, in our own coordinates, so dragging has a known origin.
    win_l = 120
    win_t = (SCREEN_HEIGHT or 1080) - 120
    apply_geometry()
end

function onclose(w)
    wnd = nil
end

-- Destroying the window from inside its own ImGui builder is not safe, so the
-- close box only sets a flag and the deferred callback acts on it.
function mv_tick()
    if close_requested then
        close_requested = false
        if wnd then float_wnd_destroy(wnd); wnd = nil end
    end
end
do_often("mv_tick()")

_G.build = build

-- Whether the panel is currently up. Defined BEFORE add_macro: FlyWithLua
-- evaluates the macro's deactivate string at registration time, so a function
-- declared afterwards does not exist yet and the whole script is quarantined
-- for the error.
function mv_open_state() return wnd ~= nil end

-- Both strings are guarded as well. A macro string that throws does not just
-- fail - it gets the file moved to Scripts (Quarantine), which looks from the
-- outside like the script vanishing.
add_macro("TAA for X-Plane 12: panel",
          "if mv_open then mv_open() end",
          "if mv_open_state and mv_open_state() and mv_open then mv_open() end",
          "deactivate")
create_command("FlyWithLua/MotionVectors/toggle_panel",
               "TAA for X-Plane 12: toggle panel",
               "if mv_open then mv_open() end", "", "")

mv_open()
