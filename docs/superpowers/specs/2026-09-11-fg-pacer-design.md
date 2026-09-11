# FG pacer: closed-loop display feedback ⏱️

**Goal:** even frame spacing for 2x now, plus the timing instrumentation 3x needs. Sub-project 1 of multi-frame generation. 3x itself = sub-project 2, own spec.

**Decision (user, 2026-09-11):** pacer first 🟢. Approach A = closed-loop feedback pacer on `VK_KHR_present_id` + `VK_KHR_present_wait`. No `VK_EXT_present_timing` as a wait primitive. No third frame until this ships.

**Display in play:** LG Ultragear 2560x1440 @ 120 Hz on the RTX 4060. VRR on/off = unknown ❓ → M1 answers it with numbers, not a menu screenshot.

---

## 1. What FFX does today 🔍

All in `third_party/FidelityFX-SDK/sdk/src/backends/vk/FrameInterpolationSwapchain/`.

- `interpolationThread` (`FrameInterpolationSwapchainVK.cpp` ~1236-1290): every time a pair finishes interpolating it measures the gap since the last pair (`deltaQpc`), feeds a `SimpleMovingAverage`, and sets **one** delta for both frames of the pair:
  `delta = avg*0.5 - variance*varianceFactor(0.1) - safetyMargin(0.1 ms)`
- presenter threads (`copyAndPresent_presenterThread` ~941, `composeAndPresent_presenterThread` ~1086, which one runs depends on the composition mode FFX picks - we pass `NOT_FORCED`): for G then R → `waitForPerformanceCount(previousPresentQpc + delta)` → present → `previousPresentQpc = now`.
- `waitForPerformanceCount` (`_Helpers.cpp:29`) = pure QPC spin 🔥.
- `presentToSwapChain` (~766): `VkPresentInfoKHR.pNext = nullptr`. No present ids. Nobody ever learns **when the frame hit the glass**.

What's wrong with it:

- **open loop** 🔁 the anchor is when we *called* present, not when it *showed*. Driver queue, compositor, VRR window all add their own delay and the pacer never sees it.
- **mean + variance** one hitch drags the schedule for a whole averaging window. A median forgets a hitch next frame.
- **same delta for G and R** the real frame is placed relative to G's present, so G's error becomes R's error.
- **pure spin** ~T/2 of one core burnt per pair for nothing.
- **12.4.4 reality (measured 2026-09-11):** the proxy present mode is now `IMMEDIATE` (0) where 12.4.3 gave `MAILBOX` (1). Nothing downstream paces for us any more; whatever spacing we present with is the spacing the panel gets.

Driver check (vulkaninfo 2026-09-11): `VK_KHR_present_id` rev 1, `VK_KHR_present_wait` rev 1 both offered. ✅

---

## 2. Milestones 🪜

| | what | pass when |
|---|---|---|
| **M1** instrumentation | ids on every present, flip thread, ring, `PACE:` trace line. **No scheduling change.** Baseline. | line appears every 300 pairs · `FPS:` still 2.00x · wait timeouts 0 on IMMEDIATE · answers "is the Ultragear VRR" |
| **M2** feedback scheduler + hybrid wait | measured schedule replaces the guessed delta; timer+spin wait | generated placement error **p95 < 1.0 ms** on VRR vs the M1 baseline · late presents not up · eye says smoother · A/B by live key |
| **M3** refresh governor (optional, keyed) | real-fps cap for fixed-refresh panels | with `lock=30` on a fixed 120 Hz panel: vblank-quantised ≥ 95 %, 2-1-2-1 gone. **Skipped** if M1 shows VRR on the Ultragear |

M1 ships with `taa.fg_pace=0` (stats only). M2 flips the default to 1 after acceptance.

---

## 3. Threads & data 🧵

```
game thread ── presentInterpolated() ──► scheduledInterpolations
interpolationThread: pair ready → T = median8(pair-ready intervals) → entry.periodQpc, entry.readyQpc → pacerEvent
presenter thread:    for G then R:  target = fgPaceTarget(...) → pacerWait(target) → presentToSwapChain(+VkPresentIdKHR)
                                    ring[id] = {id, type, idealQpc, targetQpc, presentQpc}
flip thread (NEW):   vkWaitForPresentKHR(realSwapchain, id, 100 ms) → ring[id].flipQpc = QPC now
```

**Ring** 🔁 64 slots, index `id & 63`. Each field has exactly one writer (presenter writes the schedule fields, flip thread writes `flipQpc`) → no lock. Readers check `slot.id == wanted` before trusting a slot.

```cpp
struct MvPaceSlot {
    uint64_t id;          // present id, 1-based per real swapchain
    uint8_t  type;        // 0 = generated, 1 = real, 2 = passthrough (generation off)
    int64_t  idealQpc;    // anchor + T/2 (G) or anchor + T (R), BEFORE correction
    int64_t  targetQpc;   // what we actually waited for
    int64_t  presentQpc;  // QPC after vkQueuePresentKHR returned
    int64_t  flipQpc;     // 0 = not known yet
};
```

**Present ids** 🎫 `presentToSwapChain` chains `VkPresentIdKHR{ swapchainCount=1, pPresentIds=&id }` when ids are available. The id is taken **inside** `swapchainCriticalSection`, because the passthrough presents (~2313, ~2582, game thread) and the presenter thread share that function and the spec wants ids strictly increasing per swapchain.

**Flip thread** 🧵 one per swapchain object, created in `init()` right after the real swapchain, killed in `destroySwapchain()` before the real swapchain goes. Loop: next id = last waited + 1; if nothing presented yet, wait on an event (set by `presentToSwapChain`); `vkWaitForPresentKHR(device, realSwapchain, id, 100 ms)`; on success stamp `flipQpc`, publish `lastFlipReal` / `lastFlipGen` atomics.

**Scheduler** 🧮 pure functions in `src/vklayer/fg_pacer.h`, host-testable, no Vulkan:

```cpp
struct FgPaceIn {
    int64_t now;
    int64_t anchorFlipReal;      // flip of the previous REAL frame, 0 = not known yet
    int64_t anchorPresentReal;   // its present time (always known)
    int64_t presentToFlipEst;    // median8 of (flip - present) over real frames, 0 until measured
    int64_t periodT;             // median8 of pair-ready intervals
    int64_t corr;                // integral correction, already clamped
    int64_t targetG;             // for R: the G target of this pair
    bool    isReal;
};
int64_t fgPaceTarget(const FgPaceIn& in);
```

Rules:

- `anchor = anchorFlipReal ? anchorFlipReal : anchorPresentReal + presentToFlipEst`
- **G:** `target = anchor + T/2 + corr`
- **R:** `target = max(anchor + T, targetG + T/8)` — never on the same refresh as its G
- both: `target = max(target, now)` (content is ready = now) and `target = min(target, now + T)` (stale anchor → present now, don't hang)
- `corr = clamp(-gain * sum(e[last 4]), -T/4, +T/4)`, `e_i = flipG_i - idealG_i` (positive = landed late). `gain` = `taa.fg_pace_gain`, default 0.5. Only slots with a known flip count.
- `T` = median of the last 8 pair-ready intervals, the same samples FFX feeds its average today. Reset on the same 100 ms gap FFX uses (`deltaQpcResetThreashold`) and on `resetTimer`. Fewer than 3 samples → stock delta.
- **stock mode** (`fg_pace=0`): `target = previousPresentQpc + delta` exactly as today. Ids, ring and stats still run — that is the M1 baseline.

Why T comes from arrivals and not from flips: if T were measured from our own flips and R were placed at anchor+T, the loop would feed itself and could never speed up. Arrivals are the game's cadence, independent of what we do with presents. Flips only ever *correct placement*.

**Hybrid wait** 😴 `pacerWait(targetQpc, spinUs)` in `_Helpers.cpp`: high-resolution waitable timer (`CreateWaitableTimerExW` + `CREATE_WAITABLE_TIMER_HIGH_RESOLUTION`, one handle per thread, static thread-local) until `target - spinUs`, then QPC spin to `target`. `spinUs=0` → timer only. `spinUs ≥ T` → pure spin = today. Default 500 µs, which covers the timer's worst case on Windows 11.

---

## 4. Extensions & features 🔌

- `layer.cpp` `kWanted[]` (~17366) += `VK_KHR_present_id`, `VK_KHR_present_wait`, only when `taa.fg_pace_ext=1` (launch-time; `live::onoff` runs `loadNow()` on first use so the ini is readable at device creation).
- **Names alone do nothing** ⚠️ two static feature structs chained on `ci2.pNext` exactly like `unusedFeat`: `VkPhysicalDevicePresentIdFeaturesKHR{presentId=VK_TRUE}`, `VkPhysicalDevicePresentWaitFeaturesKHR{presentWait=VK_TRUE}`. Trace `DEVICE: present ids requested (present_id + present_wait)`.
- The low-latency exclusion is untouched: it keys on `wantLowLatency` (env `TAA_SL_LOW_LATENCY`), not on whether `present_id` happens to be enabled, so adding `present_id` does **not** wake `VK_NV_low_latency2`.
- `vkWaitForPresentKHR` is a device-extension entry point the loader does not export. The swapchain calls it **by name** like its other 56 direct calls: `build.ps1` adds `-DvkWaitForPresentKHR=mvFfxWaitForPresentKHR`, `src/vklayer/ffx_fg_shim.cpp` adds the forwarder resolved through `mvFfxDeviceProc(name)` → continues DOWN the chain, never back into our own hooks.
- The layer decides `idsAvailable` at proxy creation = both extensions made it into the device's enabled list. It rides in the config (§5). Absent, or `fg_pace_ext=0` → **everything dormant**: no pNext, no thread, one trace line `PACE: present ids unavailable (present_id=%d present_wait=%d) - stock FFX pacing, no flip feedback`.

---

## 5. Config plumbing 🎛️

New configure key in `sdk/include/FidelityFX/host/backends/vk/ffx_vk.h`:

```cpp
FFX_FI_SWAPCHAIN_CONFIGURE_KEY_MV_PACER = 100,   // valuePtr -> FfxMvPacerConfig

typedef struct FfxMvPacerConfig {
    uint32_t size;          // sizeof(FfxMvPacerConfig), version guard
    uint32_t mode;          // 0 stock FFX schedule, 1 feedback
    uint32_t idsAvailable;  // layer verdict, §4
    uint32_t spinUs;        // taa.fg_pace_spin_us
    float    gain;          // taa.fg_pace_gain
    float    lockFps;       // taa.fg_pace_lock (M3), 0 = off
    uint32_t refreshHz;     // taa.fg_pace_hz, quantisation stat only
} FfxMvPacerConfig;
```

- Handled in `ffxConfigureFrameInterpolationSwapchainVK` next to `FRAMEPACINGTUNING`; stored under `criticalSectionUpdateConfig`; each pacing entry carries a copy so a live change lands at a pair boundary, never mid-pair.
- Pushed by the layer from the present hook, right after the `FfxFrameGenerationConfig` block (`layer.cpp` ~9728): build the struct from the live keys every present, `memcmp` against the last pushed one, push on change and once right after `FG: ACTIVE`. Trace on push: `PACE CONFIG: mode=%u ids=%u spin=%uus gain=%.2f lock=%.0f hz=%u`.

---

## 6. Keys 🔑

Same three inis as always: `config/taa_live.ini`, `%TEMP%\taa_live.ini`, `dist/stage-v1.2.0-dev/MotionVectors/taa_live.ini`. Ini only, no panel rows.

```ini
# ---- frame pacing (spec docs/superpowers/specs/2026-09-11-fg-pacer-design.md)
taa.fg_pace=0            # 1 = feedback scheduler (flip-anchored, median T), 0 = stock FFX schedule. Live A/B.
taa.fg_pace_ext=1        # launch-time: request VK_KHR_present_id + present_wait (needed for any flip feedback)
taa.fg_pace_spin_us=500  # final spin after the timer wait; 0 = timer only, huge = pure spin like stock
taa.fg_pace_gain=0.5     # integral gain on generated-frame placement error, clamped to +-T/4
taa.fg_pace_hz=120       # panel refresh, for the vblank-quantised stat only
taa.fg_pace_lock=0       # M3: cap real fps (30 = 60 presented on a fixed 120 Hz panel). 0 = off
```

---

## 7. Trace 📈

One line per 300 pairs (600 presents) from the swapchain through `mvFgTrace` (declared `extern "C"` in the swapchain, defined in `layer.cpp`):

```
PACE: mode=feedback ids=on present=IMMEDIATE T=27.9ms (p95 jitter 1.4) | gen placement err p50=0.21ms p95=0.62ms (4.4% of T/2) | real flip interval p05/p50/p95=26.1/27.9/30.4ms | late presents 3 | wait timeouts 0 | flip unknown at schedule 0 | replaced 0 | vblank-quantised 4% (8.33ms) | spin 0.48ms/pair
```

- **late present** = content ready more than 1 ms after its target (the game was slower than the schedule) — not a pacer error, a game hitch.
- **flip unknown at schedule** = the fallback anchor (`anchorPresentReal + presentToFlipEst`) was used.
- **replaced** = a slot whose `flipQpc` equals its successor's (mailbox discarded it). Excluded from placement stats.
- **vblank-quantised** = share of consecutive flip intervals within ±0.3 ms of a multiple of `1000/hz` ms. ~100 % = fixed refresh, low = VRR. **This is the VRR answer.** ✅
- **spin** = CPU time spent in the spin phase, per pair.

Also once per proxy: `PACE: ids available=%s (present_id=%d present_wait=%d) mode=%s present mode=%s`. Stock mode prints the same line with `mode=stock` so M1 and M2 read identically.

---

## 8. Reset & errors 🧯

- **Proxy rebuild** (resize, FSR toggle, monitor move): new swapchain object → new ring, ids from 1, T/corr empty, new flip thread. The old one is joined in `destroySwapchain()` with a 250 ms cap; the wait timeout is 100 ms so the join always completes.
- `vkWaitForPresentKHR` → `VK_SUCCESS` stamp · `VK_TIMEOUT` count + retry the same id up to 3× then skip · `VK_ERROR_OUT_OF_DATE_KHR` / `SURFACE_LOST` / `DEVICE_LOST` → thread exits, scheduler runs on present-time anchors, one line `PACE: flip thread stopped (%d) - anchors are present times now`. The presenter **never** blocks on the flip thread.
- Presenter outruns the flip thread by > 64 ids → the flip thread jumps to the latest id, skipped ids counted.
- Scheduler sanity: `T <= 0` or fewer than 3 samples → stock delta; target clamped to `[now, now + T]`.
- Mailbox comes back some day → the "replaced" rule above; nothing else changes.
- M3 `lockFps > 0`: the layer's present hook holds X-Plane's present so real frames are ≥ `1/lockFps` apart (timer + spin, same `pacerWait`). Adds up to one frame of latency. Off by default, never auto-enabled.

---

## 9. Tests 🧪

- **Host** `src/test_fg_pacer.cpp`, built like `test_fg_flags` (`g++ -O2 -std=c++17 -I src -o build/test_fg_pacer.exe`): median8 (odd/even count, reset, < 3 samples → stock), `fgPaceTarget` G and R (corr clamp ±T/4, R ≥ G + T/8, `[now, now+T]` clamp, fallback anchor when flip unknown, stock passthrough), ring slot write/read by id with wrap, vblank-quantised stat on a fixed-refresh series (100 %) and a VRR series (~0 %).
- **In sim** M1: one flight with `fg_pace=0`, keep the `PACE:` lines as the baseline. M2: same flight, `fg_pace=1`, flip the key live for the A/B. Numbers in §2 decide; the eye is the second opinion.
- **Regression** `FPS:` still 2.00x · `FG SNAP AUDIT` unchanged · late presents not up · no new `taa_layer` errors · quit + relaunch clean (thread join).

---

## 10. Files 📁

- `third_party/FidelityFX-SDK/sdk/src/backends/vk/FrameInterpolationSwapchain/FrameInterpolationSwapchainVK.cpp` / `.h` — ids in `presentToSwapChain`, flip thread, ring, `PacingData.periodQpc/readyQpc`, target computation in **both** presenter threads (shared helper), configure key, `PACE:` stats.
- `.../FrameInterpolationSwapchainVK_Helpers.cpp` / `.h` — `pacerWait` (timer + spin). `waitForPerformanceCount` stays for stock mode.
- `third_party/FidelityFX-SDK/sdk/include/FidelityFX/host/backends/vk/ffx_vk.h` — key + `FfxMvPacerConfig`.
- `src/vklayer/fg_pacer.h` **new** — median, ring, `fgPaceTarget`, stats math. Pure C++. The swapchain includes it: `build.ps1` `$ffxInc += "-I$root\src\vklayer"`.
- `src/vklayer/ffx_fg_shim.cpp` + `build.ps1` define list — `mvFfxWaitForPresentKHR`.
- `src/vklayer/layer.cpp` — `kWanted` + feature structs + `idsAvailable`; config push in the present hook; M3 hold.
- `src/test_fg_pacer.cpp` **new**.
- the three inis, this spec, the plan.

Build note: `build.ps1` recompiles an FFX object only when its `.o` is older than the `.cpp`; the header-only `fg_pacer.h` change does not trigger that, so the plan deletes `build/ffx_obj/FrameInterpolationSwapchainVK*.o` explicitly.

---

## 11. Out of scope 🚫

3x itself (sub-project 2) · `VK_EXT_present_timing` as a wait primitive · DLSS-G / Reflex / `VK_NV_low_latency2` · anything 12.4.4 · panel rows for the keys · touching the HUD-less or trust/snap paths.
