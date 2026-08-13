--[[----------------------------------------------------------------------------
  MOTION VECTORS - status panel

  A front end for the MotionVectors plugin, which derives per-pixel motion
  vectors for X-Plane 12.

  There is almost nothing to choose here, and that is deliberate. The project
  produces a velocity field; this reports on it. Every value on screen reads a
  dataref, and the one control that writes is a tuning value the plugin clamps
  itself - so this panel holds no state of its own and cannot drift from what
  the plugin believes.

  Requires FlyWithLua NG+ (ImGui).
------------------------------------------------------------------------------]]

if not SUPPORTS_FLOATING_WINDOWS then
    logMsg("Motion Vectors: FlyWithLua NG+ with floating window support is required.")
    return
end

--[[ ---------------------------------------------------------------------------
  Datarefs.

  Wrapped in pcall so the panel survives the plugin being absent. A script that
  throws on load takes the whole thing down, and the message the user actually
  needs - "the plugin is not installed" - ends up buried under a traceback about
  a nil dataref. The predecessor's panel was quarantined by FlyWithLua for
  exactly this reason.
--------------------------------------------------------------------------- ]]
local have = {}

-- ---- ASK WHETHER IT EXISTS BEFORE BINDING IT.
--
-- pcall is NOT enough. FlyWithLua's dataref() reports a missing name through
-- its own error path, above Lua, so the failure is not something pcall can
-- catch - it quarantines the whole script and takes the Lua engine with it.
-- That is exactly what happened: taaimpl/render_scale is documented in the
-- plugin and never actually registered, and one missing name killed every
-- script the user had loaded.
--
-- XPLMFindDataRef just returns nil for a name that does not exist, which is the
-- question actually being asked here. pcall stays as a second line for the
-- binding itself.
local function try_dataref(name, kind)
    local var = name:gsub("/", "_"):gsub("^taaimpl_", "MV_")
    if XPLMFindDataRef == nil then have[name] = false return false end
    local ref = XPLMFindDataRef(name)
    if ref == nil or ref == 0 then have[name] = false return false end
    local ok = pcall(dataref, var, name, kind)
    have[name] = ok
    return ok
end

try_dataref("taaimpl/enabled",            "writable")
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
try_dataref("taaimpl/mv_residual_px",     "readonly")
try_dataref("taaimpl/mv_residual_p95_px", "readonly")
try_dataref("taaimpl/mv_samples",         "readonly")
try_dataref("taaimpl/velocity_mb",        "readonly")
try_dataref("taaimpl/pipelines_patched",  "readonly")
try_dataref("taaimpl/pipelines_rejected", "readonly")
try_dataref("taaimpl/taa_enabled",        "writable")
try_dataref("taaimpl/taa_blend",          "writable")
try_dataref("taaimpl/taa_dispatches",     "readonly")

local function get(name, default)
    if not have[name] then return default end
    local var = name:gsub("/", "_"):gsub("^taaimpl_", "MV_")
    local v = _G[var]
    if v == nil then return default end
    return v
end

--[[ ---------------------------------------------------------------------------
  Colour and layout.

  A mix of the two earlier panels: the boxed cards and column alignment of the
  first, the restraint of the second. The card row exists because the residual
  is the one number that decides whether any of this works, and a number in a
  box under its own label is read at a glance where the same number in a list
  is not.

  Amber for headings, tan for labels, white for values, and green/red used ONLY
  where something is genuinely good or genuinely wrong. Colour that means
  nothing is worse than no colour: if half the panel is green, green stops being
  a signal, and the residual loses the only visual weight it has.
--------------------------------------------------------------------------- ]]
local AMBER = 0xFF3BA7FF   -- ImGui takes ABGR, not RGBA
local LABEL = 0xFFB0793B
local VALUE = 0xFFF0F0F0
local GOOD  = 0xFF5BD65B
local BAD   = 0xFF4B4BFF
local DIM   = 0xFF808080

local function text(col, s)
    imgui.PushStyleColor(imgui.constant.Col.Text, col)
    imgui.TextUnformatted(s)
    imgui.PopStyleColor()
end

-- ---- EVERY OPTIONAL IMGUI CALL GOES THROUGH HERE.
--
-- FlyWithLua's ImGui binding does not expose the whole API and which parts it
-- exposes varies between builds. A missing function is a hard Lua error, and a
-- hard error in a draw callback does not merely blank the panel - FlyWithLua
-- quarantines the script and takes every other script the user has loaded down
-- with it. That already happened here once, over a single missing dataref.
--
-- So anything not certain to exist is called through this, and the panel draws
-- a little plainer on an older binding instead of killing the engine.
local function opt(fn, ...)
    if type(fn) ~= "function" then return false end
    local ok = pcall(fn, ...)
    return ok
end

-- A label/value pair with the value at a fixed column, so a stack of them lines
-- up regardless of label length.
local function row(label, value, col, indent)
    text(LABEL, "  " .. label)
    imgui.SameLine(indent or 210)
    text(col or VALUE, value)
end

local function heading(s)
    imgui.TextUnformatted("")
    text(AMBER, s)
    imgui.Separator()
end

--[[ ---------------------------------------------------------------------------
  The verdict.

  The residual is the depth-free epipolar distance: how far the field says a
  pixel was from where the geometry says it could have been.

  The thresholds come from the verified suite, which runs 0.000 to 0.005 px
  across rotation, pitch, translation, head movement and powered flight. A tenth
  of a pixel is already an order of magnitude worse than anything measured, so
  that is where good ends. Above a pixel the field is not usable by a consumer.
--------------------------------------------------------------------------- ]]
local function verdict(res)
    if res <  0.0  then return DIM,   "NO DATA" end
    if res <= 0.10 then return GOOD,  "SUB-PIXEL" end
    if res <= 1.00 then return AMBER, "DEGRADED" end
    return BAD, "BROKEN"
end

-- A boxed statistic: title, then the value, then a note.
local function card(id, w, title, value, vcol, note, ncol)
    if not opt(imgui.BeginChild, id, w, 64, true) then
        -- No child windows in this binding: fall back to a plain line rather
        -- than losing the number entirely.
        text(LABEL, "  " .. title)
        imgui.SameLine(210)
        text(vcol or VALUE, value .. (note and ("   " .. note) or ""))
        return
    end
    text(LABEL, title)
    text(vcol or VALUE, "  " .. value)
    if note then text(ncol or DIM, "  " .. note) end
    opt(imgui.EndChild)
end

local function build(w, x, y)
    if get("taaimpl/layer_attached", 0) ~= 1 then
        text(BAD, "VULKAN LAYER NOT ATTACHED")
        imgui.Separator()
        imgui.TextUnformatted("")
        -- Say what to do about it, not just that it happened. The layer is
        -- explicit: it loads only when the loader is told to, which the launcher
        -- does and a plain Steam start does not.
        text(VALUE, "Start X-Plane from the Motion Vectors launcher.")
        imgui.TextUnformatted("")
        text(DIM, "The layer is enabled for that process only, so no other")
        text(DIM, "Vulkan application ever loads it.")
        return
    end

    local res      = get("taaimpl/mv_residual_px", -1.0)
    local p95      = get("taaimpl/mv_residual_p95_px", -1.0)
    local samples  = get("taaimpl/mv_samples", 0)
    local rejected = get("taaimpl/pipelines_rejected", 0)
    local vcol, vtext = verdict(res)

    text(AMBER, "MOTION VECTOR ACCURACY")
    imgui.Separator()

    -- ---- THE CARD ROW.
    --
    -- Three numbers answering three different questions: is the field right, is
    -- it right EVERYWHERE, and is any geometry missing from it. The last is a
    -- card rather than a buried line because 14,835 rejected pipelines once left
    -- the field a function of screen position that passed every test then being
    -- run.
    local cw = 176
    card("c_res", cw, " MEDIAN RESIDUAL",
         res < 0.0 and "--" or string.format("%.3f px", res), vcol, vtext, vcol)
    imgui.SameLine(0)
    card("c_p95", cw, " 95TH PERCENTILE",
         p95 < 0.0 and "--" or string.format("%.3f px", p95), VALUE,
         string.format("%d k samples", math.floor(samples / 1000)))
    imgui.SameLine(0)
    card("c_rej", cw, " PIPELINES REJECTED",
         string.format("%d", rejected), rejected == 0 and GOOD or BAD,
         rejected == 0 and "none - complete" or "holes in the field",
         rejected == 0 and DIM or BAD)

    text(DIM, "  Distance between where the field says a pixel was and where the")
    text(DIM, "  geometry says it could have been. Needs no depth, so it holds")
    text(DIM, "  under rotation and translation alike.")

    --[[ -----------------------------------------------------------------------
      TAA.

      Directly under the residual, because the residual is what makes the switch
      safe to use: a temporal resolve reprojects the previous frame through the
      velocity field, and if the field is wrong the result smears. The number
      immediately above says whether it is.
    ------------------------------------------------------------------------ ]]
    heading("TEMPORAL ANTI-ALIASING")
    local taaOn    = get("taaimpl/taa_enabled", 0)
    local dispatch = get("taaimpl/taa_dispatches", 0)

    if have["taaimpl/taa_enabled"] then
        local ok, changed, v = pcall(imgui.Checkbox, "  Enable TAA", taaOn == 1)
        if ok and changed then MV_taa_enabled = v and 1 or 0 end
    end

    if taaOn == 1 then
        -- Running, or merely switched on? Different claims. A resolve that
        -- silently never runs looks exactly like one that runs and does nothing,
        -- and that happened here - a descriptor pool ran dry after eight frames
        -- and the pass quietly stopped. "Enabled" is not evidence.
        if dispatch > 0 then
            row("RESOLVE", string.format("running, %d frames", dispatch), GOOD)
        else
            row("RESOLVE", "SWITCHED ON, NOT RUNNING", BAD)
        end
        if have["taaimpl/taa_blend"] then
            local ok, ch, b = pcall(imgui.SliderFloat, "  Current frame weight",
                                    get("taaimpl/taa_blend", 0.1), 0.01, 1.0, "%.2f")
            if ok and ch then MV_taa_blend = b end
        end
        text(DIM, "  Lower keeps more history: smoother, slower to respond.")
    else
        text(DIM, "  Off. The velocity field is produced either way - this is")
        text(DIM, "  the first thing that consumes it.")
    end

    -- ---- TWO COLUMNS.
    --
    -- Render state and memory are both short lists nobody reads top to bottom,
    -- so stacking them wastes the height the card row needs.
    heading("RENDER AND MEMORY")
    local twoCol = opt(imgui.Columns, 2, "rm", false)
    row("RESOLUTION", string.format("%d x %d", get("taaimpl/viewport_w", 0),
                                               get("taaimpl/viewport_h", 0)), VALUE, 150)
    row("SCALE",  string.format("%.2f x", get("taaimpl/render_scale", 1.0)), VALUE, 150)
    row("DEPTH",  get("taaimpl/reverse_z", 0) == 1 and "reverse-Z" or "standard", VALUE, 150)
    row("JITTER", string.format("%d phases", get("taaimpl/jitter_phases", 0)), VALUE, 150)
    if twoCol then opt(imgui.NextColumn) end
    row("VELOCITY TARGET", string.format("%d MB", get("taaimpl/velocity_mb", 0)), VALUE, 170)
    row("PROCESS VRAM",    string.format("%d MB", get("taaimpl/vram_used_mb", 0)), VALUE, 170)
    row("DRIVER BUDGET",   string.format("%d MB", get("taaimpl/vram_budget_mb", 0)), DIM, 170)
    row("MOVING OBJECTS",  string.format("%d", get("taaimpl/moving_objects", 0)), DIM, 170)
    if twoCol then opt(imgui.Columns, 1, "rm", false) end

    heading("TUNING")
    local ok, changed, v = pcall(imgui.SliderFloat, "  LOD bias",
                                 get("taaimpl/lod_bias", 0.0), -2.0, 0.0, "%.2f")
    if ok and changed and have["taaimpl/lod_bias"] then MV_lod_bias = v end
    text(DIM, "  Every value here is a dataref under taaimpl/.")
end

local wnd = nil

-- ---- THE DRAW CALLBACK NEVER THROWS.
--
-- FlyWithLua quarantines a script whose callback errors, and quarantining takes
-- the Lua engine with it - every other script the user has loaded, for one
-- mistake in this one. A panel that draws an error line is recoverable; a sim
-- with no scripts is not.
function build_guarded(w, x, y)
    local ok, err = pcall(build, w, x, y)
    if not ok then
        text(BAD, "panel error (the plugin is unaffected):")
        text(DIM, tostring(err))
    end
end

local function open()
    if wnd then
        float_wnd_destroy(wnd)
        wnd = nil
        return
    end
    wnd = float_wnd_create(600, 560, 1, true)
    float_wnd_set_title(wnd, "Motion Vectors")
    float_wnd_set_imgui_builder(wnd, "build_guarded")
    float_wnd_set_onclose(wnd, "onclose")
end

function onclose(w)
    wnd = nil
end

add_macro("Motion Vectors: panel", "open()", "if wnd then open() end", "deactivate")
create_command("FlyWithLua/MotionVectors/toggle_panel",
               "Motion Vectors: toggle panel", "open()", "", "")
open()
