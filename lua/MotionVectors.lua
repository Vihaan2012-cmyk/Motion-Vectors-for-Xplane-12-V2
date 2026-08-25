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
try_dataref("taaimpl/enabled",            "readonly")
try_dataref("taaimpl/mv_residual_px",     "readonly")
try_dataref("taaimpl/mv_residual_p95_px", "readonly")
try_dataref("taaimpl/mv_samples",         "readonly")
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
local MOD_VERSION      = "1.1.1"

-- ---- BUG REPORT DISCORD WEBHOOK.
--
-- Paste a Discord webhook URL between the quotes to enable one-click upload of
-- the bug report (description + MotionVectors_Debug.txt + a screenshot). Leave
-- it empty and "Report a bug" still writes the debug dump and takes the shot
-- locally in the X-Plane folder - it just does not upload.
local BUG_WEBHOOK = "https://discord.com/api/webhooks/1541762723807891587/jPBYZvgpodrv4vPWLRjFOR3gCj44lL3ysNzGaB6eSYa3YARNV7D1lXYE79NTn8AVw8tb"
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
    -- TAA_LIVE_FILE FIRST, exactly as the layer resolves it.
    --
    -- live::path() checks this variable before falling back to %TEMP%, and the
    -- launcher sets it. Checking only %TEMP% here meant that under the dev
    -- launcher the panel edited one file while the layer read another, and
    -- every setting written from the panel silently did nothing.
    local e = os.getenv("TAA_LIVE_FILE")
    if e and e ~= "" then return e end
    local t = os.getenv("TEMP") or os.getenv("TMP") or "."
    return t .. "\\taa_live.ini"
end

-- The file may not exist yet: the layer writes a template on first run, so a
-- panel opened before the layer has ever attached has nothing to edit. Create
-- it rather than refusing, so the panel works on a fresh install.
local function ini_ensure()
    local f = io.open(ini_path(), "r")
    if f then f:close(); return true end
    f = io.open(ini_path(), "w")
    if not f then return false end
    f:write("# Motion Vectors live control file.\n" ..
            "# Written by the panel. The layer re-reads it every few frames.\n")
    f:close()
    return true
end

local function ini_read()
    local f = io.open(ini_path(), "r")
    if not f then return nil end
    local s = f:read("*a")
    f:close()
    return s
end

-- A PARSED CACHE, because the panel now shows 88 settings.
--
-- ini_get() re-opened and re-read the whole file per call. At six settings that
-- was merely wasteful; at eighty-eight it is eighty-eight file opens per frame,
-- inside the ImGui builder, on the main thread. Parsed once every refresh
-- instead, and invalidated the moment we write.
local ini_map, ini_map_at = {}, -1e9

local function ini_refresh(force)
    local now = os.clock()
    if not force and (now - ini_map_at) < 0.5 then return end
    ini_map_at = now
    local m = {}
    local txt = ini_read()
    if txt then
        for line in txt:gmatch("[^\r\n]+") do
            if not line:match("^%s*[#;]") then
                local k, v = line:match("^%s*([%w_.]+)%s*=%s*(.-)%s*$")
                if k then m[k] = v end
            end
        end
    end
    ini_map = m
end

-- nil when the key is absent, which is what "still at the compiled default"
-- looks like. Distinguishing that from an explicit value is the whole point:
-- the panel must not claim a user set something they did not.
local function ini_raw(key)
    return ini_map[key]
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
    ini_refresh(true)
    return true
end

-- Remove a key entirely, which is how a setting goes back to the value
-- compiled into the layer. Writing the default back as a literal would look
-- identical in the file and behave differently the day the default changes.
local function ini_clear(key)
    local txt = ini_read()
    if not txt then return false end
    local pat = key:gsub("%.", "%%.")
    txt = txt:gsub("[\r\n]" .. pat .. "%s*=[^\r\n]*", "")
    txt = txt:gsub("^" .. pat .. "%s*=[^\r\n]*[\r\n]?", "")
    local f = io.open(ini_path(), "w")
    if not f then return false end
    f:write(txt)
    f:close()
    ini_refresh(true)
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
    -- ---- TWO LAYOUTS EXIST, AND ONLY ONE OF THEM WAS CHECKED.
    --
    -- This looked only in MotionVectors\build\vklayer\, which is where a
    -- DEVELOPMENT tree keeps the layer. The release zip installs it to
    -- MotionVectors\ instead, so every installed user was told
    --
    --     MISSING Vulkan layer
    --     MISSING Layer manifest
    --     OK      Vulkan layer is attached to this process
    --
    -- which contradicts itself - it cannot be missing and attached at once -
    -- and reported a broken install to people whose install was fine.
    --
    -- Both layouts are legitimate, so both are accepted.
    local function first_existing(paths)
        for _, q in ipairs(paths) do
            if file_exists(q) then return q end
        end
        return nil
    end

    -- THREE layouts now. The product split moved development output to
    -- build\\<product>\\vklayer, and dropping the old path without
    -- adding the new one made the panel report MISSING for a layer that was
    -- loaded and running - the same self-contradiction the comment above
    -- records, reintroduced from the other direction.
    local dll  = { base .. "VkLayer_mv.dll",
                   base .. "build\\MotionVectors\\vklayer\\VkLayer_mv.dll",
                   base .. "build\\vklayer\\VkLayer_mv.dll" }
    local json = { base .. "VkLayer_mv.json",
                   base .. "build\\MotionVectors\\vklayer\\VkLayer_mv.json",
                   base .. "build\\vklayer\\VkLayer_mv.json" }

    local checks = {
        { "Vulkan layer",     first_existing(dll)  or dll[1]  },
        { "Layer manifest",   first_existing(json) or json[1] },
        { "Plugin",           root .. "Resources\\plugins\\MotionVectors\\64\\win.xpl" },
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
    -- ---- THE LIVE CONTROL FILE IS NOT AN INSTALLATION FILE.
    --
    -- It used to sit in the list above and be counted as a failure, so every
    -- clean install reported "Integrity check found 1 problem(s)" the first
    -- time it ran - on an installation with nothing whatsoever wrong with it.
    --
    -- Its absence is the NORMAL state before anyone has changed a setting. The
    -- layer says so itself, in its own words, in the log a few lines earlier:
    -- "no config at ... - using defaults". A file the layer is documented to
    -- do without is not part of whether the mod is installed.
    --
    -- What IS worth reporting is being unable to create it, because then the
    -- panel cannot write a setting and every control in it would silently do
    -- nothing - which is a real fault, and one that looks like the panel being
    -- broken rather than like a permissions problem.
    if file_exists(ini_path()) then
        logf(C_GREEN, "  OK      Live controls")
    elseif ini_ensure() then
        logf(C_GREEN, "  OK      Live controls (created - defaults until changed)")
    else
        logf(C_RED,   "  PROBLEM Cannot create %s - the panel will not be able "
                      .. "to save anything", ini_path())
        bad = bad + 1
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

-- BEGIN GENERATED SETTINGS -- produced by gen_lua.py from the layer
-- source; every live:: key in src/vklayer is listed here. Do not edit
-- by hand: a hand-kept copy is what left writeTemplate() at 6 of 88.
local SETTINGS = {
  { key = "crash.enable", label = "enable", kind = "bool", def = false, lo = nil, hi = nil,
    group = "CRASH", help = "" },
  { key = "rd.capture", label = "capture", kind = "int", def = 0, lo = nil, hi = nil,
    group = "DEBUG", help = "" },
  { key = "report", label = "report", kind = "int", def = 0, lo = nil, hi = nil,
    group = "DEBUG", help = "One-shot: fires once and clears its own key, so leaving it set in the file by accident costs one report rather than one per frame forever." },
  { key = "report.every", label = "every", kind = "int", def = 0, lo = nil, hi = nil,
    group = "DEBUG", help = "" },
  { key = "taa.alpha", label = "alpha", kind = "float", def = 0.05, lo = 0.01, hi = 1.0,
    group = "TAA", help = "" },
  { key = "taa.alpha_moving", label = "alpha_moving", kind = "float", def = 0.35, lo = 0.01, hi = 1.0,
    group = "TAA", help = "Blend weight while the camera moves." },
  { key = "taa.alpha_moving_px", label = "alpha_moving_px", kind = "float", def = 3.0, lo = 0.0, hi = 16.0,
    group = "TAA", help = "Speed, in px/frame, at which alpha_moving is fully applied." },
  { key = "taa.clear_after_resolve", label = "clear_after_resolve", kind = "bool", def = false, lo = nil, hi = nil,
    group = "TAA", help = "" },
  { key = "taa.enable", label = "enable", kind = "bool", def = false, lo = nil, hi = nil,
    group = "TAA", help = "---- EVERY KNOB IS LIVE." },
  { key = "taa.force_reset", label = "force_reset", kind = "bool", def = false, lo = nil, hi = nil,
    group = "TAA", help = "" },
  { key = "taa.freeze_history", label = "freeze_history", kind = "bool", def = false, lo = nil, hi = nil,
    group = "TAA", help = "" },
  { key = "taa.gain", label = "gain", kind = "float", def = 4.0, lo = 0.0, hi = 16.0,
    group = "TAA", help = "" },
  { key = "taa.hist_catmull", label = "hist_catmull", kind = "bool", def = true, lo = nil, hi = nil,
    group = "TAA", help = "Sharp history resampling." },
  { key = "taa.jitter_scale", label = "jitter_scale", kind = "float", def = 0.0, lo = 0.0, hi = 2.0,
    group = "TAA", help = "Jitter defaults OFF even with the resolve on: the unjitter cancellation has never been verified on screen, and the first flight that ran it showed exa" },
  { key = "taa.max_resolves", label = "max_resolves", kind = "int", def = 1, lo = 1, hi = 8,
    group = "TAA", help = "" },
  { key = "taa.mode", label = "mode", kind = "int", def = 0, lo = 0, hi = 3,
    group = "TAA", help = "0 passthrough, 1 reproject, 2 full (variance clip), 3 clean" },
  { key = "taa.moved_dead", label = "moved_dead", kind = "float", def = 0.0, lo = 0.0, hi = 4.0,
    group = "TAA", help = "Deadband on the clamp correction, in units of the noise floor." },
  { key = "taa.moved_eps", label = "moved_eps", kind = "float", def = 1e-4, lo = nil, hi = nil,
    group = "TAA", help = "Not camDelta: that is translation only, and a camera rotating in place moves every pixel while translating zero." },
  { key = "taa.mv_pass", label = "mv_pass", kind = "int", def = -2, lo = -2, hi = 13,
    group = "TAA", help = "-2 census winner, -1 all passes, 0..13 pin one" },
  { key = "taa.nearfield_m", label = "nearfield_m", kind = "float", def = nil, lo = nil, hi = nil,
    group = "TAA", help = "" },
  { key = "taa.nearfield_view", label = "nearfield_view", kind = "int", def = -1, lo = -1, hi = 2000,
    group = "TAA", help = "---- NEAR-FIELD DISTANCE, in metres, into the spare .z of the jitter vec4." },
  { key = "taa.no_accum", label = "no_accum", kind = "bool", def = false, lo = nil, hi = nil,
    group = "TAA", help = "" },
  { key = "taa.no_motion", label = "no_motion", kind = "bool", def = false, lo = nil, hi = nil,
    group = "TAA", help = "" },
  { key = "taa.novec_alpha", label = "novec_alpha", kind = "float", def = 0.05, lo = 0.0, hi = 1.0,
    group = "TAA", help = "0.5 was chosen to stop the ground crawling, but it also refuses to accumulate on every pixel the sentinel calls unwritten - which is most of an extern" },
  { key = "taa.novec_by_vel", label = "novec_by_vel", kind = "bool", def = false, lo = nil, hi = nil,
    group = "TAA", help = "" },
  { key = "taa.novec_cov", label = "novec_cov", kind = "float", def = -1.0, lo = nil, hi = nil,
    group = "TAA", help = "---- THE UNWRITTEN-PIXEL REJECTION IS OFF BY DEFAULT." },
  { key = "taa.objflags", label = "objflags", kind = "bool", def = true, lo = nil, hi = nil,
    group = "TAA", help = "The gbuffer_vel weight override - on by default because everything it addresses (prop halo, airframe streaks, cockpit shake) is worse than the aliasin" },
  { key = "taa.quad_needs_depth", label = "quad_needs_depth", kind = "bool", def = true, lo = nil, hi = nil,
    group = "TAA", help = "" },
  { key = "taa.quad_needs_pull", label = "quad_needs_pull", kind = "bool", def = true, lo = nil, hi = nil,
    group = "TAA", help = "" },
  { key = "taa.reactive", label = "reactive", kind = "bool", def = false, lo = nil, hi = nil,
    group = "TAA", help = "The C14 reactive mask - on by default for the same reason as the flag override: flicker parked in history is worse than aliasing on the flickering con" },
  { key = "taa.smul_x", label = "smul_x", kind = "float", def = 0.5, lo = -2.0, hi = 2.0,
    group = "TAA", help = "" },
  { key = "taa.smul_y", label = "smul_y", kind = "float", def = -0.5, lo = -2.0, hi = 2.0,
    group = "TAA", help = "" },
  { key = "taa.sticky_colour", label = "sticky_colour", kind = "int", def = -1, lo = -1, hi = 1,
    group = "TAA", help = "" },
  { key = "taa.unjitter", label = "unjitter", kind = "bool", def = true, lo = nil, hi = nil,
    group = "TAA", help = "The unjitter alignment - isolation knob for the aligned sampling, so its contribution can be removed live without touching the jitter itself." },
  { key = "taa.varclip", label = "varclip", kind = "float", def = 8.0, lo = 0.5, hi = 16.0,
    group = "TAA", help = "8.0, not 1.25." },
  { key = "taa.vel_max", label = "vel_max", kind = "float", def = 1.0, lo = nil, hi = nil,
    group = "TAA", help = "" },
  { key = "taa.vel_scale", label = "vel_scale", kind = "float", def = 1.0, lo = 0.0, hi = 4.0,
    group = "TAA", help = "---- REMOVE ONE INPUT AT A TIME." },
  { key = "taa.vel_ypos", label = "vel_ypos", kind = "bool", def = false, lo = nil, hi = nil,
    group = "TAA", help = "-1.0 is the shipping belief (negative-height viewport => d(uv_y) = -vel_y)." },
  { key = "taa.viz", label = "viz", kind = "int", def = 0, lo = 0, hi = 10,
    group = "TAA", help = "0 off, 1 motion, 2 magnitude, 3 invalid, 4 history, 5 weight, 6 clamp, 8 written, 10 view depth" },
  { key = "taa.viz_scale", label = "viz_scale", kind = "float", def = 1.0, lo = 0.0, hi = 8.0,
    group = "TAA", help = "" },
  { key = "vram.adaptive", label = "adaptive", kind = "bool", def = true, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.age_frames", label = "age_frames", kind = "int", def = 1800, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.bench", label = "bench", kind = "int", def = 0, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.budget_alpha", label = "budget_alpha", kind = "float", def = 0.02, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.deflate_frames", label = "deflate_frames", kind = "int", def = 600, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.deflate_mb", label = "deflate_mb", kind = "int", def = 512, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.enable", label = "enable", kind = "bool", def = true, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.explain", label = "explain", kind = "int", def = 1, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.governor", label = "governor", kind = "bool", def = true, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.hold_max_mb", label = "hold_max_mb", kind = "int", def = 512, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.lookahead", label = "lookahead", kind = "int", def = 300, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.migrate", label = "migrate", kind = "int", def = 1, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.migrate_every", label = "migrate_every", kind = "int", def = 300, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.priority", label = "priority", kind = "bool", def = true, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.recycle", label = "recycle", kind = "bool", def = true, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.recycle_hold_frames", label = "recycle_hold_frames", kind = "int", def = 180, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.recycle_max_mb", label = "recycle_max_mb", kind = "int", def = 256, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.recycle_pressure_mb", label = "recycle_pressure_mb", kind = "int", def = 64, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.report", label = "report", kind = "int", def = 0, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.reserve_c", label = "reserve_c", kind = "int", def = 768, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.reserve_g", label = "reserve_g", kind = "int", def = 128, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.reserve_o", label = "reserve_o", kind = "int", def = 384, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.reserve_r", label = "reserve_r", kind = "int", def = 512, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.reserve_y", label = "reserve_y", kind = "int", def = 256, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.retain_max_mb", label = "retain_max_mb", kind = "int", def = 256, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.rot_floor_deg", label = "rot_floor_deg", kind = "float", def = 1.0, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.shape", label = "shape", kind = "bool", def = true, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.speed_reserve", label = "speed_reserve", kind = "float", def = 0.01, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.teleport_frames", label = "teleport_frames", kind = "int", def = 900, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.teleport_m", label = "teleport_m", kind = "float", def = 2000.0, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.tex_drop_above", label = "tex_drop_above", kind = "int", def = 0, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.tex_streamed_to", label = "tex_streamed_to", kind = "int", def = 0, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.trace_every", label = "trace_every", kind = "int", def = 600, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.upload_c", label = "upload_c", kind = "int", def = 8, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.upload_cache", label = "upload_cache", kind = "bool", def = true, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.upload_g", label = "upload_g", kind = "int", def = 0, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.upload_max_hold", label = "upload_max_hold", kind = "int", def = 2, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.upload_o", label = "upload_o", kind = "int", def = 64, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.upload_r", label = "upload_r", kind = "int", def = 24, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.upload_y", label = "upload_y", kind = "int", def = 0, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.w_category", label = "w_category", kind = "float", def = 0.40, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.w_frequency", label = "w_frequency", kind = "float", def = 0.15, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.w_recency", label = "w_recency", kind = "float", def = 0.25, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.w_recreate", label = "w_recreate", kind = "float", def = 0.10, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.w_size", label = "w_size", kind = "float", def = 0.10, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.w_spatial", label = "w_spatial", kind = "float", def = 0.20, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.warmup_frames", label = "warmup_frames", kind = "int", def = 900, lo = nil, hi = nil,
    group = "VRAM", help = "" },
  { key = "vram.warmup_mb", label = "warmup_mb", kind = "int", def = 512, lo = nil, hi = nil,
    group = "VRAM", help = "" },
}
-- END GENERATED SETTINGS


--[[ ---------------------------------------------------------------------------
  The settings browser.

  Every live:: key in the layer, edited here and written straight into the
  control file the layer polls. There is no dataref in this path and no restart:
  the layer re-reads the file on a timestamp change every fifteen frames, so a
  slider moved here takes effect in well under a second.

  Two rules the rest of this file also follows.

  A key that is ABSENT from the file is not the same as a key set to its default
  value. Absent means "whatever the layer was compiled with", and that is the
  value that ships. So a row shows "default" in dim text until it is explicitly
  overridden, and Reset deletes the line rather than writing the default back.

  A range is a claim about what is useful. Only the keys someone has actually
  reasoned about carry lo/hi and get a slider; the rest get a number box, which
  is honest about not knowing rather than putting a confident-looking slider on
  a span nobody has thought about.
--------------------------------------------------------------------------- ]]
local SET_GROUPS = { "TAA", "VRAM", "CRASH", "DEBUG" }
local set_group  = "TAA"
local set_edit   = {}      -- key -> in-progress text, so typing is not fought

-- FlyWithLua builds differ in which imgui entry points exist, and calling a
-- missing one quarantines the script. Same defence as pushcol().
local function imhas(fn) return imgui and type(imgui[fn]) == "function" end

local function set_default_str(sd)
    if sd.def == nil then return "?" end
    if sd.kind == "bool" then return sd.def and "on" or "off" end
    if sd.kind == "int"  then return string.format("%d", sd.def) end
    return string.format("%.4g", sd.def)
end

local function setting_row(sd)
    local rawv       = ini_raw(sd.key)
    local overridden = (rawv ~= nil)

    -- Label, coloured by whether the user owns this value or the layer does.
    text(overridden and C_TEXT or C_DIM, "  " .. sd.label)
    same(190)

    if sd.kind == "bool" then
        local cur
        if overridden then cur = not (rawv == "0" or rawv == "off" or rawv == "false")
        else                cur = (sd.def == true) end
        if styled_button((cur and "On" or "Off") .. "##" .. sd.key, 54, 18,
                         cur and C_GREEN or C_GREY_ED,
                         cur and C_GREEN_BG or C_GREY_BG, true) then
            ini_ensure()
            ini_set(sd.key, cur and "0" or "1")
            logf(C_AMBER, "%s = %s", sd.key, cur and "0" or "1")
        end
    else
        local cur = tonumber(rawv) or tonumber(sd.def) or 0
        if sd.lo and sd.hi and imhas("SliderFloat") then
            imgui.PushItemWidth(210)
            local fmt = (sd.kind == "int") and "%.0f" or "%.3f"
            local ch, v = imgui.SliderFloat("##" .. sd.key, cur, sd.lo, sd.hi, fmt)
            imgui.PopItemWidth()
            if ch then
                ini_ensure()
                local out = (sd.kind == "int")
                            and string.format("%d", math.floor(v + 0.5))
                            or  string.format("%.4f", v)
                ini_set(sd.key, out)
            end
        elseif imhas("InputText") then
            -- Held in set_edit while typing: writing the file on every
            -- keystroke would make the layer reload mid-word and would fight
            -- the caret.
            local buf = set_edit[sd.key]
            if buf == nil then
                buf = overridden and rawv or set_default_str(sd)
            end
            imgui.PushItemWidth(150)
            local ch, v = imgui.InputText("##" .. sd.key, buf, 32)
            imgui.PopItemWidth()
            if ch then set_edit[sd.key] = v end
            same(350)
            if styled_button("Set##" .. sd.key, 44, 18, C_GREEN, C_GREEN_BG, true) then
                local val = set_edit[sd.key] or buf
                if tonumber(val) then
                    ini_ensure()
                    ini_set(sd.key, val)
                    set_edit[sd.key] = nil
                    logf(C_AMBER, "%s = %s", sd.key, val)
                else
                    logf(C_RED, "%s: %q is not a number", sd.key, tostring(val))
                end
            end
        else
            text(C_DIM, overridden and rawv or set_default_str(sd))
        end
    end

    -- Reset: delete the line. Only offered when there IS a line to delete.
    if overridden then
        same(470)
        if styled_button("Reset##" .. sd.key, 52, 18, C_GREY_ED, C_GREY_BG, true) then
            ini_clear(sd.key)
            set_edit[sd.key] = nil
            logf(C_AMBER, "%s back to default (%s)", sd.key, set_default_str(sd))
        end
    else
        same(470)
        text(C_DIM, "default " .. set_default_str(sd))
    end

    if sd.help ~= "" then text(C_DIM, "      " .. sd.help) end
end

function settings_browser()
    ini_refresh(false)

    -- Group tabs. Counted from the table, so a new key in a new group appears
    -- without this function being told about it.
    local counts = {}
    for _, sd in ipairs(SETTINGS) do
        counts[sd.group] = (counts[sd.group] or 0) + 1
    end
    for i, g in ipairs(SET_GROUPS) do
        if i > 1 then same() end
        local on = (set_group == g)
        if styled_button(string.format("%s (%d)", g, counts[g] or 0), 104, 20,
                         on and C_GREEN or C_GREY_ED,
                         on and C_GREEN_BG or C_GREY_BG, true) then
            set_group = g
        end
    end

    -- How many in this group the user has actually overridden, so "am I running
    -- stock?" is answerable without scrolling the list.
    local n_over = 0
    for _, sd in ipairs(SETTINGS) do
        if sd.group == set_group and ini_raw(sd.key) ~= nil then n_over = n_over + 1 end
    end
    text(n_over > 0 and C_AMBER or C_DIM,
         string.format("  %d of %d overridden in this file", n_over,
                       counts[set_group] or 0))
    text(C_DIM, "  " .. ini_path())
    imgui.Separator()

    local child = imhas("BeginChild")
    if child then imgui.BeginChild("mv_settings", 690, 260, true) end
    for _, sd in ipairs(SETTINGS) do
        if sd.group == set_group then setting_row(sd) end
    end
    if child then imgui.EndChild() end

    imgui.Separator()
    if styled_button("Reset this group to defaults", 240, 20, C_RED, C_RED_BG, true) then
        local n = 0
        for _, sd in ipairs(SETTINGS) do
            if sd.group == set_group and ini_raw(sd.key) ~= nil then
                ini_clear(sd.key); n = n + 1
            end
        end
        set_edit = {}
        logf(C_AMBER, "%s: cleared %d override(s).", set_group, n)
    end
end

local show_advanced = false


local wnd = nil
local close_requested = false
-- Set by build() once it has run a frame WITHOUT the trailing WindowBg push,
-- so the ImGui colour stack is balanced before the window is destroyed. See
-- the close sequence at the bottom of build().
local safe_to_close = false
local close_ticks   = 0

-- Window geometry, tracked here because FlyWithLua can set it but not report
-- it. XPLM screen coordinates: origin bottom-left, so top > bottom.
local win_l, win_t = 120, 0        -- filled in on open()
local WIN_W, WIN_H = 720, 360
-- The settings browser needs room the status panel does not. Kept as a
-- separate constant so the closed panel is exactly the size it always was.
local WIN_H_ADV = 720
local dragging   = false
local title_held = false
local drag_mx, drag_my = 0, 0

local function apply_geometry()
    if wnd then
        -- The browser needs the taller body; the status panel keeps the size
        -- it has always had. Resizing on the toggle rather than opening large
        -- means the common case is unchanged.
        local h = show_advanced and WIN_H_ADV or WIN_H
        if bug_open then h = h + 150 end   -- room for the bug description box
        float_wnd_set_geometry(wnd, win_l, win_t, win_l + WIN_W, win_t - h)
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

local win_bg_pushed = 0

-- ============================================================================
--  BUG REPORTER
--
--  "Report a bug" asks the user what went wrong, writes MotionVectors_Debug.txt
--  (versions, launched-by, render/injection state, VRAM, the live settings),
--  takes one X-Plane screenshot, and uploads the description + both files to a
--  Discord webhook. The upload is a detached curl in a one-shot .bat so the sim
--  never freezes on it. All local-only if BUG_WEBHOOK is left empty.
-- ============================================================================
bug_open            = false    -- module global: apply_geometry() (defined above) reads it
local bug_text      = ""
local bug_status    = ""
local bug_upload_at = nil      -- os.clock() moment to run the upload; nil = idle
local bug_before    = {}       -- screenshots on disk before the shot was taken
local bug_debug     = nil      -- path of the debug dump awaiting upload

local function bug_shot_dir()
    return (SYSTEM_DIRECTORY or "") .. "Output\\screenshots\\"
end

-- The set of screenshot names on disk now, so the shot taken for THIS report
-- can be picked out afterwards by set difference (Lua has no reliable mtime).
local function bug_ss_set()
    local set = {}
    if directory_to_table then
        local ok, t = pcall(directory_to_table, bug_shot_dir())
        if ok and t then
            for _, n in ipairs(t) do
                if type(n) == "string" and n:lower():match("%.png$") then set[n] = true end
            end
        end
    end
    return set
end

local function bug_write_debug(desc)
    local path = (SYSTEM_DIRECTORY or "") .. "MotionVectors_Debug.txt"
    local f = io.open(path, "w")
    if not f then return nil end
    local att = get("taaimpl/layer_attached", 0)
    f:write("MotionVectors bug report\n========================\n")
    f:write("mod version : " .. tostring(MOD_VERSION) .. "\n")
    f:write("X-Plane ver : " .. tostring(get("sim/version/xplane_internal_version", "?")) .. "\n")
    f:write("launched by : " .. ((att == 1)
        and "MotionVectors launcher (Vulkan layer attached)"
        or  "NOT via the launcher - Vulkan layer is NOT attached") .. "\n")
    f:write("\n--- user description ---\n" .. (desc ~= "" and desc or "(none provided)") .. "\n")
    f:write("\n--- render / injection ---\n")
    f:write(string.format("resolution         = %d x %d\n", get("taaimpl/viewport_w",0), get("taaimpl/viewport_h",0)))
    f:write(string.format("render_scale       = %.2f\n", get("taaimpl/render_scale",1.0)))
    f:write(string.format("reverse_z          = %s\n", tostring(get("taaimpl/reverse_z",0))))
    f:write(string.format("jitter_phases      = %d\n", get("taaimpl/jitter_phases",0)))
    f:write(string.format("moving_objects     = %d\n", get("taaimpl/moving_objects",0)))
    f:write(string.format("pipelines_patched  = %d\n", get("taaimpl/pipelines_patched",0)))
    f:write(string.format("pipelines_rejected = %d\n", get("taaimpl/pipelines_rejected",0)))
    f:write(string.format("lod_bias           = %.2f\n", get("taaimpl/lod_bias",0)))
    f:write("\n--- vram (MB) ---\n")
    f:write(string.format("used=%d  budget=%d  total=%d  velocity=%d\n",
        get("taaimpl/vram_used_mb",0), get("taaimpl/vram_budget_mb",0),
        get("taaimpl/vram_total_mb",0), get("taaimpl/velocity_mb",0)))
    f:write("\n--- resolve / velocity ---\n")
    f:write(string.format("layer_attached     = %s\n", tostring(get("taaimpl/layer_attached",0))))
    f:write(string.format("taa_enabled        = %s\n", tostring(get("taaimpl/enabled",0))))
    f:write(string.format("mv_residual_px     = %.4f  (p95 %.4f)\n",
        get("taaimpl/mv_residual_px",0), get("taaimpl/mv_residual_p95_px",0)))
    f:write(string.format("mv_samples         = %d\n", get("taaimpl/mv_samples",0)))
    f:close()
    return path
end

local function json_escape(s)
    return (s or ""):gsub("\\", "\\\\"):gsub('"', '\\"'):gsub("\r", ""):gsub("\n", "\\n")
end

local function bug_do_upload(debugPath, shotPath, desc)
    if BUG_WEBHOOK == nil or BUG_WEBHOOK == "" then
        bug_status = "Saved locally - set BUG_WEBHOOK to upload."
        return
    end
    local root = SYSTEM_DIRECTORY or ""
    local payload = root .. "MotionVectors_payload.json"
    local pf = io.open(payload, "w")
    if pf then
        -- A proper Discord rich embed: the description carries the user's text,
        -- the state goes in inline fields, and the screenshot is shown inside
        -- the embed via attachment:// (curl renames it to screenshot.png below).
        local function fld(name, value, inline)
            return string.format('{"name":"%s","value":"%s","inline":%s}',
                                  name, value, inline and "true" or "false")
        end
        local layer = (get("taaimpl/layer_attached",0) == 1) and "attached" or "NOT attached"
        local embed = '{"embeds":[{'
            .. '"title":"MotionVectors bug report",'
            .. '"description":"' .. json_escape(desc ~= "" and desc or "(no description)") .. '",'
            .. '"color":15158332,'
            .. '"timestamp":"' .. os.date("!%Y-%m-%dT%H:%M:%SZ") .. '",'
            .. '"fields":['
            ..   fld("Mod", tostring(MOD_VERSION), true) .. ','
            ..   fld("X-Plane", tostring(get("sim/version/xplane_internal_version","?")), true) .. ','
            ..   fld("Vulkan layer", layer, true) .. ','
            ..   fld("TAA", (get("taaimpl/enabled",0) == 1) and "on" or "off", true) .. ','
            ..   fld("Resolution", string.format("%dx%d", get("taaimpl/viewport_w",0), get("taaimpl/viewport_h",0)), true) .. ','
            ..   fld("Render scale", string.format("%.2fx", get("taaimpl/render_scale",1.0)), true) .. ','
            ..   fld("Pipelines", string.format("%d patched / %d rej", get("taaimpl/pipelines_patched",0), get("taaimpl/pipelines_rejected",0)), true) .. ','
            ..   fld("MV residual", string.format("%.3f px", get("taaimpl/mv_residual_px",0)), true) .. ','
            ..   fld("Velocity buf", string.format("%d MB", get("taaimpl/velocity_mb",0)), true)
            .. ']'
            .. (shotPath and ',"image":{"url":"attachment://screenshot.png"}' or '')
            .. ',"footer":{"text":"MotionVectors bug reporter"}'
            .. '}]}'
        pf:write(embed)
        pf:close()
    end
    -- Quoting lives inside the .bat, away from cmd's own start-parsing.
    local cmd = 'curl.exe -s -o nul -m 30 '
        .. '-F "payload_json=<' .. payload .. ';type=application/json" '
        .. '-F "files[0]=@' .. debugPath .. ';type=text/plain" '
    if shotPath then cmd = cmd .. '-F "files[1]=@' .. shotPath .. ';filename=screenshot.png;type=image/png" ' end
    cmd = cmd .. '"' .. BUG_WEBHOOK .. '"'
    -- Run curl FULLY HIDDEN: the .bat silences every stream (no Discord JSON in a
    -- console), and a one-line VBS shim launches it with window style 0 so no
    -- console window ever pops. Detached (Run ..., 0, False) so the sim never
    -- waits on the upload. Falls back to a plain detached start if VBS is absent.
    local batp = root .. "MotionVectors_send.bat"
    local bf = io.open(batp, "w")
    if bf then bf:write("@echo off\r\n" .. cmd .. " >nul 2>&1\r\n"); bf:close() end
    local vbsp = root .. "MotionVectors_send.vbs"
    local vf = io.open(vbsp, "w")
    if vf then
        vf:write('CreateObject("WScript.Shell").Run """' .. batp .. '""", 0, False\r\n')
        vf:close()
        os.execute('start "" /b wscript //nologo "' .. vbsp .. '"')
    else
        os.execute('start "" /b "' .. batp .. '"')
    end
    bug_status = "Report sent."
end

-- Runs from mv_tick (do_often) so it completes even with the panel closed.
function bug_tick()
    if not bug_upload_at then return end
    if os.clock() < bug_upload_at then return end
    bug_upload_at = nil
    local shot = nil
    for name in pairs(bug_ss_set()) do
        if not bug_before[name] then shot = bug_shot_dir() .. name; break end
    end
    bug_do_upload(bug_debug, shot, bug_text)
end

local function bug_start()
    bug_debug = bug_write_debug(bug_text)
    if not bug_debug then bug_status = "Could not write the debug file."; return end
    bug_before = bug_ss_set()
    if XPLMFindCommand and XPLMCommandOnce then
        local c = XPLMFindCommand("sim/operation/screenshot")
        if c then XPLMCommandOnce(c) end
    end
    bug_status = "Collecting screenshot..."
    bug_upload_at = os.clock() + 1.2   -- give X-Plane a moment to write the PNG
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
    -- ---- THE WINDOW CARRIES THE TINT. THE CHILD STAYS TRANSPARENT.
    --
    -- Moving the colour here to silence an ImGui style-stack assert cost the
    -- panel its transparency AND its drag: FlyWithLua's own window kept its
    -- default opaque background, the child painted over it, and the drag
    -- strips stopped behaving. "The panel looks the same" was wrong.
    --
    -- The assert is cosmetic. A panel you cannot see through or move is not.
    -- So the cross-frame push is back, and the ImGui message with it, until
    -- there is a fix that does not cost the look - most likely painting the
    -- background with a draw-list rectangle, which touches no style stack at
    -- all.
    -- ---- BACKGROUND BY DRAW LIST, NOT BY A CROSS-FRAME STYLE PUSH.
    --
    -- The panel tint used to be a WindowBg colour pushed at the END of build()
    -- to reach the NEXT Begin(), then popped at the start of the next build().
    -- That push outlived a single frame, so destroying the window between the
    -- push and the pop orphaned it on ImGui's global colour stack, and
    -- reopening popped a stack that no longer held it - PopStyleColor on an
    -- empty vector, which took X-Plane down. Painting the same translucent
    -- rectangle straight onto the window draw list happens inside THIS frame
    -- and touches no style stack, so nothing is left dangling to crash on.
    -- This is exactly the fix the old comment above said was needed.
    if imgui.DrawList_AddRectFilled then
        imgui.DrawList_AddRectFilled(0, 0,
            imgui.GetWindowWidth(), imgui.GetWindowHeight(), C_PANEL)
    end

    local nWin = pushcol("ChildBg", 0x00000000)
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
        apply_geometry()
    end

    ------------------------------------------------------------ report a bug
    same(232)
    if imgui.Button(bug_open and "Cancel##bug" or "Report a bug", 160, 22) then
        bug_open = not bug_open
        if bug_open then bug_status = "" end
        apply_geometry()   -- grow/shrink the window for the description box
    end
    if bug_open then
        text(C_TEXT, "What went wrong? (a sentence is plenty)")
        if imhas("InputText") then
            imgui.PushItemWidth(WIN_W - 40)
            local ch, v = imgui.InputText("##bugtext", bug_text, 512)
            imgui.PopItemWidth()
            if ch then bug_text = v end
        end
        if styled_button("Send report", 160, 22, C_GREEN, C_GREEN_BG, true) then
            bug_start()
        end
        if bug_status ~= "" then same(180); text(C_AMBER, bug_status) end
        text(C_DIM, "Writes MotionVectors_Debug.txt + a screenshot to your X-Plane"
                 .. " folder" .. (BUG_WEBHOOK ~= "" and " and uploads them to Discord."
                                                    or " (set BUG_WEBHOOK to upload)."))
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

        settings_browser()
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

    -- No cross-frame style push any more: the background is painted by the
    -- draw-list rectangle at the top of build(), so the colour stack is already
    -- balanced right here and the window can be torn down on ANY frame without
    -- orphaning anything. win_bg_pushed stays 0 (the start-of-build balance is
    -- now a permanent no-op) and the close no longer has to be staged - but the
    -- signal is kept so mv_tick() still performs the teardown it owns.
    win_bg_pushed = 0
    if close_requested then safe_to_close = true end
end

function mv_open()
    if wnd then
        -- Request, do not destroy. Tearing the window down here skips the
        -- balancing frame and leaves the colour stack dangling, which is the
        -- crash described at the bottom of build().
        close_requested = true
        return
    end
    safe_to_close = false
    close_ticks   = 0
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
    -- Closed from outside our own sequence, so the balancing frame never ran
    -- and a push may still be on the stack. Forgetting it leaks one entry;
    -- remembering it makes the next build() pop a stack that may no longer
    -- hold it, which is the crash. A cosmetic leak beats a crash.
    win_bg_pushed = 0
    close_requested = false
    safe_to_close   = false
end

-- Destroying the window from inside its own ImGui builder is not safe, so the
-- close box only sets a flag and the deferred callback acts on it.
function mv_tick()
    if close_requested then
        -- Wait for build() to run its balancing frame. If it never does - the
        -- window is not being drawn at all - close anyway rather than leave a
        -- panel that will not shut, and drop the count instead of popping a
        -- stack we can no longer reason about.
        close_ticks = close_ticks + 1
        if safe_to_close or close_ticks > 4 then
            if not safe_to_close then win_bg_pushed = 0 end
            close_requested = false
            safe_to_close   = false
            close_ticks     = 0
            if wnd then float_wnd_destroy(wnd); wnd = nil end
        end
    end
    bug_tick()
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
