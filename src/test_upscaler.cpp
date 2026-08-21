// Offline tests for upscaler backend selection.
//
// The whole point of upscaler_policy.h being Vulkan-free is that this runs at
// a command line. Every check here is a question that would otherwise be
// answered by launching X-Plane on hardware we do not have.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdio>
#include <cstring>
#include "vklayer/upscaler_policy.h"

static int g_fails = 0;
static int g_checks = 0;

static void check(bool cond, const char *what)
{
    ++g_checks;
    if (!cond) { ++g_fails; printf("  FAIL  %s\n", what); }
    else       {            printf("  ok    %s\n", what); }
}

// The three machines that matter, built as QUERIED facts rather than names.
static upscaler::DeviceCaps rdna4()
{
    upscaler::DeviceCaps c;
    c.vendorId = upscaler::VENDOR_ID_AMD; c.coopMatrix = true; c.valid = true;
    return c;
}
static upscaler::DeviceCaps amdOld()
{
    upscaler::DeviceCaps c;
    c.vendorId = upscaler::VENDOR_ID_AMD; c.coopMatrix = false; c.valid = true;
    return c;
}
static upscaler::DeviceCaps nvidia()
{
    upscaler::DeviceCaps c;
    c.vendorId = upscaler::VENDOR_ID_NVIDIA; c.coopMatrix = true; c.valid = true;
    return c;
}
static upscaler::DeviceCaps intel()
{
    upscaler::DeviceCaps c;
    c.vendorId = upscaler::VENDOR_ID_INTEL; c.valid = true;
    return c;
}

int main()
{
    printf("upscaler selection tests\n");

    // ---------------------------------------------------- hardware verdict
    printf("\nhardware verdict is about silicon, not about our code\n");
    check(upscaler::hardwareCanRun(TAA_UPSCALER_FSR4, rdna4()),
          "FSR 4 hardware-runs on AMD with cooperative matrix");
    check(!upscaler::hardwareCanRun(TAA_UPSCALER_FSR4, amdOld()),
          "FSR 4 does not hardware-run on AMD without cooperative matrix");
    check(!upscaler::hardwareCanRun(TAA_UPSCALER_FSR4, nvidia()),
          "FSR 4 does not hardware-run on NVIDIA even with matrix cores");
    check(upscaler::hardwareCanRun(TAA_UPSCALER_DLSS, nvidia()),
          "DLSS hardware-runs on NVIDIA");
    check(!upscaler::hardwareCanRun(TAA_UPSCALER_DLSS, rdna4()),
          "DLSS does not hardware-run on AMD");

    // FSR 2's whole selling point is that it is not vendor-locked. If this
    // ever starts failing, someone has added a restriction AMD does not.
    check(upscaler::hardwareCanRun(TAA_UPSCALER_FSR2, nvidia()) &&
          upscaler::hardwareCanRun(TAA_UPSCALER_FSR2, intel()) &&
          upscaler::hardwareCanRun(TAA_UPSCALER_FSR2, amdOld()),
          "FSR 2 hardware-runs on all three vendors");

    // ---------------------------------------------------- the two reasons
    printf("\nthe three reasons stay distinguishable\n");
    // This is the check that protects the design decision. There are three
    // different ways a backend can be unavailable and they have three
    // different fixes:
    //
    //   no-gpu      wrong silicon          - buy a different card
    //   no-support  no code in this layer  - a developer has work to do
    //   no-library  no SDK for this build  - install the runtime
    //
    // Collapsing any two of them sends someone to solve the wrong problem.
    // On an RDNA 4 card FSR 4 is possible and merely unimplemented; on an
    // older AMD card it is impossible however much code we write.
    check(upscaler::availability(TAA_UPSCALER_FSR4, rdna4()) == TAA_AVAIL_NO_SUPPORT,
          "FSR 4 on RDNA 4 reports no-support, not no-gpu");
    check(upscaler::availability(TAA_UPSCALER_FSR4, amdOld()) == TAA_AVAIL_NO_GPU,
          "FSR 4 on older AMD reports no-gpu");
    check(upscaler::availability(TAA_UPSCALER_DLSS, nvidia()) == TAA_AVAIL_NO_SUPPORT,
          "DLSS on NVIDIA reports no-support while unimplemented");
    check(upscaler::availability(TAA_UPSCALER_DLSS, amdOld()) == TAA_AVAIL_NO_GPU,
          "DLSS on AMD reports no-gpu");

    // The hardware verdict outranks everything. A backend we have not written
    // is still impossible on the wrong card, and saying "no-support" there
    // would imply that writing it would help.
    check(upscaler::availability(TAA_UPSCALER_FSR4, nvidia()) == TAA_AVAIL_NO_GPU,
          "hardware outranks the implementation verdict");

    // ---------------------------------------------------- ours always works
    printf("\nours answers without a device\n");
    upscaler::DeviceCaps none;   // valid == false: nothing has answered yet
    check(upscaler::availability(TAA_UPSCALER_TAA, none) == TAA_AVAIL_OK,
          "TAA is available before any device has been seen");
    check(upscaler::availability(TAA_UPSCALER_OFF, none) == TAA_AVAIL_OK,
          "Off is always available");
    // A zeroed struct must not be able to advertise a vendor backend.
    check(upscaler::availability(TAA_UPSCALER_FSR4, none) == TAA_AVAIL_UNKNOWN,
          "FSR 4 is unknown - not OK - before a device answers");
    check(upscaler::availability(TAA_UPSCALER_DLSS, none) == TAA_AVAIL_UNKNOWN,
          "DLSS is unknown - not OK - before a device answers");

    // ---------------------------------------------------- resolution
    printf("\nresolution falls back to TAA and says why\n");
    const char *why = nullptr;
    check(upscaler::resolve(TAA_UPSCALER_TAA, rdna4(), &why) == TAA_UPSCALER_TAA,
          "a TAA request resolves to TAA");
    check(why != nullptr && why[0] == 0,
          "why is empty on success, never left dangling");

    why = nullptr;
    check(upscaler::resolve(TAA_UPSCALER_DLSS, rdna4(), &why) == TAA_UPSCALER_TAA,
          "a DLSS request on AMD falls back to TAA");
    check(why != nullptr && strstr(why, "GPU") != nullptr,
          "and the reason names the GPU");

    why = nullptr;
    check(upscaler::resolve(TAA_UPSCALER_FSR4, rdna4(), &why) == TAA_UPSCALER_TAA,
          "an FSR 4 request on RDNA 4 still falls back - not implemented");
    check(why != nullptr && strstr(why, "GPU") == nullptr,
          "and the reason does NOT blame the GPU, which is capable");

    // Fallback is TAA, not OFF. Dropping to OFF looks exactly like the layer
    // having detached, which has already caused a wrong diagnosis once.
    why = nullptr;
    check(upscaler::resolve(TAA_UPSCALER_FSR2, intel(), &why) == TAA_UPSCALER_TAA,
          "fallback is TAA rather than OFF");

    why = nullptr;
    check(upscaler::resolve(99, rdna4(), &why) == TAA_UPSCALER_TAA,
          "an out-of-range request resolves to TAA");
    check(why != nullptr && why[0] != 0,
          "and an out-of-range request still explains itself");

    // ---------------------------------------------------- the whole array
    printf("\nthe array the plugin reads\n");
    int32_t avail[TAA_UPSCALER_COUNT];
    memset(avail, 0xFF, sizeof(avail));
    upscaler::availabilityAll(rdna4(), avail);
    bool allSet = true;
    for (int i = 0; i < TAA_UPSCALER_COUNT; ++i)
        if (avail[i] < 0 || avail[i] > TAA_AVAIL_NO_SUPPORT) allSet = false;
    check(allSet, "every slot is written with a legal verdict");
    check(avail[TAA_UPSCALER_TAA] == TAA_AVAIL_OK,
          "TAA reads OK through the array path too");

    // ---------------------------------------------------- naming
    printf("\nnames exist for everything the enum can hold\n");
    bool named = true;
    for (int i = 0; i < TAA_UPSCALER_COUNT; ++i)
        if (strcmp(upscaler::name(i), "?") == 0) named = false;
    check(named, "no backend in the enum is missing a display name");


    // ------------------------------------------------ XeSS is cross-vendor
    printf("\nXeSS is not gated on Intel\n");
    check(upscaler::hardwareCanRun(TAA_UPSCALER_XESS, nvidia()) &&
          upscaler::hardwareCanRun(TAA_UPSCALER_XESS, rdna4()) &&
          upscaler::hardwareCanRun(TAA_UPSCALER_XESS, intel()),
          "XeSS hardware-runs on all three vendors");
    // Why this matters: on the development machine, a 4060, XeSS is the only
    // vendor upscaler whose HARDWARE verdict is yes. Gating it on Intel would
    // have refused the one backend that can actually run here.
    check(upscaler::availability(TAA_UPSCALER_XESS, nvidia()) != TAA_AVAIL_NO_GPU,
          "XeSS is never refused for hardware reasons on NVIDIA");

    // -------------------------------- having the SDK is not having the code
    printf("\nan SDK on disk is not an implementation\n");
    // The distinction that briefly broke. SDK detection found three SDKs, a
    // single flag went to 1, and every vendor backend would have reported
    // ready while no record() existed for any of them - the panel would have
    // offered XeSS and the user would have got silence.
    check(upscaler::backendImplemented(TAA_UPSCALER_TAA),
          "TAA is implemented");
    check(!upscaler::backendImplemented(TAA_UPSCALER_XESS),
          "no XeSS backend is implemented yet, and the policy says so");
    check(!upscaler::backendImplemented(TAA_UPSCALER_FSR2),
          "no FSR 2 backend is implemented yet");
    check(!upscaler::backendImplemented(TAA_UPSCALER_DLSS),
          "no DLSS backend is implemented yet");

    // Every backend must resolve to something that actually runs, whatever
    // the machine. A request can be refused; it can never leave nothing.
    {
        bool allRunnable = true, allExplained = true;
        const upscaler::DeviceCaps machines[3] = { nvidia(), rdna4(), intel() };
        for (int m = 0; m < 3; ++m) {
            for (int u = 0; u < TAA_UPSCALER_COUNT; ++u) {
                const char *w = nullptr;
                const int r = upscaler::resolve(u, machines[m], &w);
                if (r != TAA_UPSCALER_OFF && r != TAA_UPSCALER_TAA && r != u)
                    allRunnable = false;
                if (w == nullptr) allExplained = false;
            }
        }
        check(allRunnable, "every backend on every machine resolves to a runnable choice");
        check(allExplained, "and never leaves the reason pointer unset");
    }

    printf("\n%d checks, %d failed\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
