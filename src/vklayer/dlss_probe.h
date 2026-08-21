// What DLSS needs from the Vulkan device, asked of NGX itself.
//
// WHY A LAYER CAN DO THIS AND AN APPLICATION CANNOT
//
// DLSS requires instance and device extensions to be enabled when the instance
// and device are CREATED. An application that did not ask for them at startup
// cannot add them later, which is why "DLSS support" is normally a decision
// taken before anything is drawn.
//
// A layer sits in front of vkCreateInstance and vkCreateDevice. It sees both
// create-infos before they reach the driver and may append to them. That is the
// whole reason this is worth attempting here: X-Plane never asked for DLSS, and
// it does not have to.
//
// WHY THE ANSWER IS ASKED FOR RATHER THAN HARDCODED
//
// NGX publishes two queries for exactly this:
//
//   GetFeatureInstanceExtensionRequirements  needs no instance, so it can run
//                                            before vkCreateInstance
//   GetFeatureDeviceExtensionRequirements    needs the instance and physical
//                                            device, so it runs before
//                                            vkCreateDevice
//
// The XeSS probe records why this matters: Streamline's requirement list was
// being printed and then ignored, and Reflex failed with -229 for want of three
// extensions it had already named. A hardcoded list is a list that goes stale.
//
// ---- WHY THIS IS OFF BY DEFAULT, AND WHAT KILLED THE LAST ONE.
//
// The XeSS probe CRASHED X-Plane inside vkCreateDevice, and the two candidate
// causes were never separated:
//
//   LoadLibrary of a 77 MB DLL while the Vulkan loader holds its lock, or
//   re-entering the loader from inside one of its own calls.
//
// nvngx_dlss.dll is 56 MB, so the same hazard applies and the same discipline
// is needed:
//
//   * the library is loaded BEFORE descending into the loader, never from
//     inside a nested call;
//   * every step traces before it runs, so a crash names the step it died in
//     rather than leaving two suspects;
//   * the whole thing is behind dlss.probe, default off, so a user who never
//     asks for it cannot be taken down by it.
//
// NOTHING HERE RUNS DLSS. It asks what DLSS would need and reports the answer.
// Whether that answer is usable is a separate question, kept separate on
// purpose - the availability report has three distinct verdicts and this fills
// in exactly one of them.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>
#include <vector>

namespace dlssprobe {

// What the probe found, in the terms the availability report needs.
struct Requirements {
    bool loaded   = false;   // the NGX library resolved
    bool queried  = false;   // the requirement queries answered
    // Extension names NGX asked for. Pointers are owned by NGX and outlive the
    // query, but they are copied here so the caller never depends on that.
    std::vector<std::string> instanceExts;
    std::vector<std::string> deviceExts;
    // The feature chain NGX wants appended to VkDeviceCreateInfo::pNext.
    // Borrowed, not owned: NGX keeps it, and copying a pNext chain would mean
    // copying structures whose types are not known here.
    const void *featureChain = nullptr;
    unsigned major = 0, minor = 0, patch = 0;
    // Why it did not work, when it did not. Never left empty on failure: a
    // blank reason is what made the upscaler report unfalsifiable before.
    std::string why;
};

// ---- THE PROBE.
//
// Split in two because the two queries happen at different moments and only one
// of them can be answered without an instance.
//
// instanceStage() runs before vkCreateInstance and answers "which instance
// extensions". deviceStage() runs before vkCreateDevice, needs what
// instanceStage produced, and answers "which device extensions and features".
//
// Neither creates anything. Both are safe to call and discard.
Requirements &state();

// Loads the NGX library and asks for instance extensions. Call from the TOP of
// the vkCreateInstance hook, before the call descends - see the note above on
// why the library must not be loaded from inside a nested loader call.
bool instanceStage(const char *sdkPath);

// Asks for device extensions and the feature chain. Needs a live instance and
// physical device, so it belongs at the top of the vkCreateDevice hook.
bool deviceStage(void *instance, void *physicalDevice);

} // namespace dlssprobe
