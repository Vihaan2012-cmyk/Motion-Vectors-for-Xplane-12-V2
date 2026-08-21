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
    printf("\nno-gpu and no-library stay distinguishable\n");
    // This is the check that protects the design decision. On an RDNA 4 card
    // FSR 4 is possible and merely uncompiled; on an older AMD card it is
    // impossible. Collapsing both to "unavailable" sends the user to buy a
    // card they already have.
    check(upscaler::availability(TAA_UPSCALER_FSR4, rdna4()) == TAA_AVAIL_NO_LIBRARY,
          "FSR 4 on RDNA 4 reports no-library, not no-gpu");
    check(upscaler::availability(TAA_UPSCALER_FSR4, amdOld()) == TAA_AVAIL_NO_GPU,
          "FSR 4 on older AMD reports no-gpu");
    check(upscaler::availability(TAA_UPSCALER_DLSS, nvidia()) == TAA_AVAIL_NO_LIBRARY,
          "DLSS on NVIDIA reports no-library");
    check(upscaler::availability(TAA_UPSCALER_DLSS, amdOld()) == TAA_AVAIL_NO_GPU,
          "DLSS on AMD reports no-gpu");

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
          "an FSR 4 request on RDNA 4 still falls back - no runtime compiled");
    check(why != nullptr && strstr(why, "runtime") != nullptr,
          "and the reason names the runtime, not the GPU");

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

    printf("\n%d checks, %d failed\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
