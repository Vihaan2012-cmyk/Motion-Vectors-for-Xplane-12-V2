--[[----------------------------------------------------------------------------
  Motion Vectors - status panel

  A front end for the MotionVectors plugin, which derives per-pixel motion
  vectors for X-Plane 12. There is nothing to choose here: the project produces
  vectors and this reports on them. Every widget reads a dataref; the few that
  write are tuning values the plugin clamps itself, so this never holds state of
  its own and cannot drift from what the plugin believes.

  Requires FlyWithLua NG+ (ImGui).
------------------------------------------------------------------------------]]

if not SUPPORTS_FLOATING_WINDOWS then
    logMsg("Motion Vectors: FlyWithLua NG+ with floating window support is required.")
    return
end

-- Wrapped in pcall so the panel survives the plugin being absent. A script that
-- throws on load takes the whole thing down, and then the message the user
-- actually needs - "the plugin is not installed" - is buried under a traceback
-- about a nil dataref. The predecessor's panel ended up quarantined by
-- FlyWithLua for exactly this reason.
local have = {}
local function try_dataref(name, kind)
    local var = name:gsub("/", "_"):gsub("^taaimpl_", "MV_")
    local ok = pcall(dataref, var, name, kind)
    have[name] = ok
    return ok
end

try_dataref("taaimpl/enabled",        "writable")
try_dataref("taaimpl/layer_attached", "readonly")
try_dataref("taaimpl/render_scale",   "readonly")
try_dataref("taaimpl/viewport_w",     "readonly")
try_dataref("taaimpl/viewport_h",     "readonly")
try_dataref("taaimpl/jitter_phases",  "readonly")
try_dataref("taaimpl/moving_objects", "readonly")
try_dataref("taaimpl/reverse_z",      "readonly")
try_dataref("taaimpl/lod_bias",       "writable")
try_dataref("taaimpl/vram_used_mb",   "readonly")
try_dataref("taaimpl/vram_budget_mb", "readonly")
try_dataref("taaimpl/vram_total_mb",  "readonly")

local wnd = nil

local function get(name, default)
    if not have[name] then return default end
    local var = name:gsub("/", "_"):gsub("^taaimpl_", "MV_")
    local v = _G[var]
    if v == nil then return default end
    return v
end

local function build(w, x, y)
    local attached = get("taaimpl/layer_attached", 0)

    if attached ~= 1 then
        imgui.TextUnformatted("Vulkan layer NOT attached")
        imgui.TextUnformatted("")
        -- Say what to do about it, not just that it happened. The layer is
        -- explicit: it loads only when the loader is told to load it, which the
        -- launcher does and a plain Steam start does not.
        imgui.TextUnformatted("Start X-Plane from the Motion Vectors shortcut.")
        imgui.TextUnformatted("The layer is enabled for that process only, so")
        imgui.TextUnformatted("no other Vulkan application loads it.")
        return
    end

    imgui.TextUnformatted("Vulkan layer attached")
    imgui.TextUnformatted("")

    local vw = get("taaimpl/viewport_w", 0)
    local vh = get("taaimpl/viewport_h", 0)
    local scale = get("taaimpl/render_scale", 1.0)
    imgui.TextUnformatted(string.format("Render      %dx%d  (%.2fx)", vw, vh, scale))
    imgui.TextUnformatted(string.format("Depth       %s",
        get("taaimpl/reverse_z", 0) == 1 and "reverse-Z" or "standard"))
    imgui.TextUnformatted(string.format("Jitter      %d phases (off until the vectors measure 1)",
        get("taaimpl/jitter_phases", 0)))
    imgui.TextUnformatted(string.format("Moving objs %d", get("taaimpl/moving_objects", 0)))

    imgui.TextUnformatted("")
    imgui.TextUnformatted(string.format("VRAM        %d / %d MB  of %d",
        get("taaimpl/vram_used_mb", 0),
        get("taaimpl/vram_budget_mb", 0),
        get("taaimpl/vram_total_mb", 0)))

    imgui.TextUnformatted("")
    local changed, v = imgui.SliderFloat("LOD bias", get("taaimpl/lod_bias", 0.0),
                                         -2.0, 0.0, "%.2f")
    if changed and have["taaimpl/lod_bias"] then MV_lod_bias = v end

    imgui.TextUnformatted("")
    imgui.TextUnformatted("Everything here is a dataref under taaimpl/.")
end

local function open()
    if wnd then
        float_wnd_destroy(wnd)
        wnd = nil
        return
    end
    wnd = float_wnd_create(420, 260, 1, true)
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
