// Which mod this build IS.
//
// One source tree produces two products:
//
//   Motion Vectors           per-pixel motion vectors, TAA, upscaling
//   Realistic Crash Physics  airframe fragmentation on impact
//
// They are built from the same layer because crash destruction is not a
// separate technology - it is the SAME SPIR-V injection, the same pipeline and
// descriptor interception, the same push-constant plumbing and the same live
// config that motion vectors needed first. Forking the tree would mean fixing
// the injector twice, and this project has already lost whole evenings to one
// copy of a thing disagreeing with another.
//
// So the layer is shared and the FEATURES are compiled in or out.
//
// ---- EVERY NAME HERE EXISTS BECAUSE BOTH MODS MAY BE INSTALLED AT ONCE.
//
// Two plugins that both call themselves TAAImplementation, both open
// "Local\TAAImpl_Matrices", both register a layer named VK_LAYER_mv and both
// answer to datarefs under taaimpl/ do not merely conflict - they silently
// corrupt each other, because the second one to start finds the first one's
// shared block already there and maps it as its own.
//
// A user running motion vectors AND crash physics is the expected case, not an
// exotic one. Nothing below may be shared between the two.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// The build system sets exactly one of these. Defaulting to motion vectors
// keeps every existing command line working unchanged.
#ifndef MV_PRODUCT_CRASH
#define MV_PRODUCT_CRASH 0
#endif

#if MV_PRODUCT_CRASH

// ---- REALISTIC CRASH PHYSICS
#define MV_PRODUCT_ID       "RealisticCrashPhysics"
#define MV_PLUGIN_NAME      "Realistic Crash Physics"
#define MV_PLUGIN_SIG       "com.nitin.realisticcrashphysics"
#define MV_PLUGIN_DESC      "Airframe fragmentation on impact, from the aircraft's own geometry."
#define MV_LAYER_NAME       "VK_LAYER_rcp"
#define MV_LAYER_DLL        "VkLayer_rcp.dll"
#define MV_LAYER_JSON       "VkLayer_rcp.json"
#define MV_SHARE_NAME       "Local\\RCPhys_Matrices"
#define MV_CONTROL_NAME     "Local\\RCPhys_Control"
#define MV_DATAREF_PREFIX   "crashphys/"
#define MV_LIVE_FILE        "rcp_live.ini"
#define MV_LIVE_ENV         "RCP_LIVE_FILE"
#define MV_TRACE_FILE       "rcp_layer.txt"
#define MV_PANEL_LUA        "RealisticCrashPhysics.lua"

// What is compiled in.
#define MV_FEATURE_TAA      0
#define MV_FEATURE_CRASH    1

#else

// ---- MOTION VECTORS
#define MV_PRODUCT_ID       "MotionVectors"
#define MV_PLUGIN_NAME      "TAAImplementation"
#define MV_PLUGIN_SIG       "com.nitin.taaimplementation"
#define MV_PLUGIN_DESC      "Per-pixel motion vectors, temporal anti-aliasing and upscaling."
#define MV_LAYER_NAME       "VK_LAYER_mv"
#define MV_LAYER_DLL        "VkLayer_mv.dll"
#define MV_LAYER_JSON       "VkLayer_mv.json"
#define MV_SHARE_NAME       "Local\\TAAImpl_Matrices"
#define MV_CONTROL_NAME     "Local\\TAAImpl_Control"
#define MV_DATAREF_PREFIX   "taaimpl/"
#define MV_LIVE_FILE        "taa_live.ini"
#define MV_LIVE_ENV         "TAA_LIVE_FILE"
#define MV_TRACE_FILE       "taa_layer.txt"
#define MV_PANEL_LUA        "MotionVectors.lua"

// ---- WHY CRASH IS STILL COMPILED INTO MOTION VECTORS FOR NOW.
//
// It is off at runtime (crash.enable defaults to 0) and has been shipping
// dormant since 0.0.19. Setting this to 0 removes it from the binary
// altogether, which is the goal - but doing that in the same change as the
// product split would mean two large edits landing together, and the crash
// work is mid-flight: Task 10 is written and unverified.
//
// Left at 1 deliberately, so the split can be proved on its own before
// anything is deleted. Flip it when the crash build stands up.
#define MV_FEATURE_TAA      1
#define MV_FEATURE_CRASH    1

#endif

// The names above must never coincide. Checked here rather than trusted,
// because the failure is silent corruption rather than a build error: two mods
// mapping one another's shared block look like one mod behaving strangely.
#if MV_PRODUCT_CRASH
#  if MV_FEATURE_TAA
#    error "the crash product must not compile TAA in - it would duplicate the whole motion-vector path in a second layer"
#  endif
#endif
