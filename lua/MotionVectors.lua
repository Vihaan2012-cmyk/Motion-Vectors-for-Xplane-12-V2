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

local function get(name, default)
    if not have[name] then return default end
    local var = name:gsub("/", "_"):gsub("^taaimpl_", "MV_")
    local v = _G[var]
    if v == nil then return default end
    return v
end

--[[ ---------------------------------------------------------------------------
  Colour.

  Amber for headings, cyan-grey for labels, white for values, and green/red used
  ONLY where something is genuinely good or genuinely wrong. Colour that means
  nothing is worse than no colour: if every panel is half green, green stops
  being a signal, and the one number that matters here - the residual - loses
  the only visual weight it has.
--------------------------------------------------------------------------- ]]
local AMBER  = 0xFF3BA7FF   -- ImGui takes ABGR
local LABEL  = 0xFFB0793B
local VALUE  = 0xFFE8E8E8
local GOOD   = 0xFF5BD65B
local BAD    = 0xFF4B4BFF
local DIM    = 0xFF707070

local function text(col, s)
    imgui.PushStyleColor(imgui.constant.Col.Text, col)
    imgui.TextUnformatted(s)
    imgui.PopStyleColor()
end

local function same(x) imgui.SameLine(x) end

-- A label/value row on one line, with the value at a fixed column so a column
-- of them lines up regardless of label length.
local function row(label, value, col, indent)
    text(LABEL, string.format("  %s", label))
    same(indent or 190)
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
  pixel was from where the geometry says it could have been. It is the one
  number that decides whether this project works, so it gets the top of the
  panel and it gets the only strong colour.

  The thresholds come from the verified suite, which runs 0.000 to 0.004 px
  across rotation, pitch, translation and head movement. A tenth of a pixel is
  already an order of magnitude worse than anything measured, so that is where
  "good" ends. Above a pixel the field is not usable by a consumer at all.
--------------------------------------------------------------------------- ]]
local function verdict(res)
    if res < 0.0    then return DIM,  "NO MEASUREMENT YET" end
    if res <= 0.10  then return GOOD, "SUB-PIXEL - VERIFIED" end
    if res <= 1.00  then return AMBER, "DEGRADED" end
    return BAD, "BROKEN"
end

local function build(w, x, y)
    if get("taaimpl/layer_attached", 0) ~= 1 then
        text(BAD, "VULKAN LAYER NOT ATTACHED")
        imgui.Separator()
        imgui.TextUnformatted("")
        -- Say what to do about it, not just that it happened. The layer is
        -- explicit: it loads only when the loader is told to, which the
        -- launcher does and a plain Steam start does not.
        text(VALUE, "Start X-Plane from the Motion Vectors launcher.")
        imgui.TextUnformatted("")
        text(DIM, "The layer is enabled for that process only, so no")
        text(DIM, "other Vulkan application ever loads it.")
        return
    end

    -- The MOTION VECTOR ACCURACY section is gone: its numbers only live while
    -- the 63 MB readback dump runs, and that dump is a stutter machine kept
    -- off in normal flying - a permanently-frozen headline row taught nothing.
    text(AMBER, "MOTION VECTORS")
    imgui.Separator()

    heading("INJECTION")
    local patched  = get("taaimpl/pipelines_patched", 0)
    local rejected = get("taaimpl/pipelines_rejected", 0)
    row("PIPELINES CARRYING VELOCITY", string.format("%d", patched))
    -- Rejections are the failure that hides. 14,835 of them once left the field
    -- a function of screen position that passed every test being run at the
    -- time, so this is red the moment it is not zero.
    row("REJECTED BY THE DRIVER", string.format("%d", rejected),
        rejected == 0 and GOOD or BAD)
    if rejected > 0 then
        text(BAD, "  Rejected pipelines are holes in the field.")
    end

    heading("RENDER")
    row("RESOLUTION", string.format("%d x %d", get("taaimpl/viewport_w", 0),
                                                get("taaimpl/viewport_h", 0)))
    row("SCALE", string.format("%.2f x", get("taaimpl/render_scale", 1.0)))
    row("DEPTH", get("taaimpl/reverse_z", 0) == 1 and "reverse-Z" or "standard")
    row("JITTER", string.format("%d phases", get("taaimpl/jitter_phases", 0)))
    row("MOVING OBJECTS", string.format("%d", get("taaimpl/moving_objects", 0)))

    heading("MEMORY")
    local used, budget, total =
        get("taaimpl/vram_used_mb", 0),
        get("taaimpl/vram_budget_mb", 0),
        get("taaimpl/vram_total_mb", 0)
    row("VELOCITY TARGET", string.format("%d MB", get("taaimpl/velocity_mb", 0)))
    row("PROCESS VRAM", string.format("%d MB", used))
    row("DRIVER BUDGET", string.format("%d / %d MB", budget, total), DIM)

    heading("TUNING")
    local changed, v = imgui.SliderFloat("LOD bias", get("taaimpl/lod_bias", 0.0),
                                         -2.0, 0.0, "%.2f")
    if changed and have["taaimpl/lod_bias"] then MV_lod_bias = v end
    text(DIM, "  Everything on this panel is a dataref under taaimpl/.")
end

local wnd = nil

local function open()
    if wnd then
        float_wnd_destroy(wnd)
        wnd = nil
        return
    end
    wnd = float_wnd_create(560, 620, 1, true)
    float_wnd_set_title(wnd, "Motion Vectors")
    float_wnd_set_imgui_builder(wnd, "build")
    float_wnd_set_onclose(wnd, "onclose")
end

function onclose(w)
    wnd = nil
end

-- Exposed so the builder callback can find it by name.
_G.build = build

add_macro("Motion Vectors: panel", "open()", "if wnd then open() end", "deactivate")
create_command("FlyWithLua/MotionVectors/toggle_panel",
               "Motion Vectors: toggle panel", "open()", "", "")
open()
