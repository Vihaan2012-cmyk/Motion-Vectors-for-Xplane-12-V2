#pragma once
#include <cmath>

// Crash detection, done here rather than taken from X-Plane.
//
// WHY NOT sim/flightmodel2/misc/has_crashed
//
// Four reasons, and only the first is about thresholds:
//
//   1. It is ONE BIT. The fragment simulation needs an impact VECTOR and a
//      MAGNITUDE to seed velocities with, so those have to be read from the G
//      datarefs regardless - at which point the bit adds nothing.
//   2. X-Plane ACTS on it. The sim shows a dialog and repositions the
//      aeroplane, which fights override_planepath and would tidy the wreckage
//      away. sim/operation/prefs/reset_on_crash = 0 stops that, and then this
//      layer owns the aftermath completely.
//   3. Damage is a CONTINUUM. A firm landing, a gear collapse and a terrain
//      impact at 200 kt want different responses; one bit cannot say which.
//   4. It misses things. A wingtip strike or a gear collapse on rollout is
//      visible in the G traces and the contact flags, and not in that bit.
//
// WHY A PEAK OVER A WINDOW, NOT AN INSTANT COMPARISON
//
// g_nrml is noisy, and this project has already been bitten by treating a
// noisy signal as a binary state: the near-field select keyed on cameraMoved,
// which is TRUE on a parked aeroplane because the airframe trembles on its
// gear with the engines running, and the resulting flicker pumped the image.
// A single-frame comparison against 6 g would fire on taxiing over a seam.
//
// So the trigger takes the PEAK over a short window after ground contact, and
// latches once. Re-arming requires being properly airborne again, so one
// impact produces one event no matter how the airframe bounces afterwards.

namespace destruct {

enum Severity {
    SEV_NONE   = 0,   // ordinary landing - nothing happens
    SEV_HARD   = 1,   // firm arrival: gear compression, panel flex, no debris
    SEV_STRUCT = 2,   // structural failure: fragments break loose
    SEV_BREAKUP = 3   // full breakup
};

// Thresholds. Deliberately explicit rather than derived, because the right
// values are a LOOK and will be tuned by eye; every one of these is exposed as
// a crash.* live key so tuning does not need a rebuild.
struct Thresholds {
    float hardFpm;      // vertical speed at contact, negative is downward
    float hardG;
    float structFpm;
    float structG;
    float breakupG;
    int   windowFrames; // how long after contact the peak is taken over
    int   rearmFrames;  // airborne frames required before another event
};

inline Thresholds defaultThresholds()
{
    Thresholds t;
    // A normal airliner touchdown is 100-300 fpm and about 1.2-1.4 g.
    // 600 fpm / 2.0 g is firm but survivable - the aeroplane is fine, the
    // passengers noticed.
    t.hardFpm     = -600.0f;
    t.hardG       =  2.0f;
    // 1000 fpm is a heavy landing that damages the airframe. 6 g exceeds what
    // transport-category structure is certified for.
    t.structFpm   = -1000.0f;
    t.structG     =  6.0f;
    // 15 g is not survivable as structure. Everything comes apart.
    t.breakupG    = 15.0f;
    // A real impact is a spike a few frames wide; 20 frames covers it at any
    // sane rate without letting a later bounce contribute to the same event.
    t.windowFrames = 20;
    // A full second airborne before the trigger can fire again, so a bounce
    // down the runway is one crash rather than four.
    t.rearmFrames  = 60;
    return t;
}

// Everything the detector reads, sampled once per flight loop.
struct Sample {
    float vhFpm;        // sim/flightmodel/position/vh_ind_fpm
    float gNrml;        // sim/flightmodel/forces/g_nrml
    float gAxil;        // sim/flightmodel/forces/g_axil
    float gSide;        // sim/flightmodel/forces/g_side
    int   onGround;     // sim/flightmodel/failures/onground_any
};

// What the trigger produces when it fires.
struct Event {
    int      fired;
    Severity severity;
    float    peakG;
    float    contactFpm;
    float    impulse[3]; // body-axis deceleration at the peak, for seeding
};

struct Detector {
    Thresholds th;
    int   armed;          // 1 once properly airborne, so a cold start on the
                          // ground cannot fire on the first frame
    int   airborneCount;
    int   inWindow;       // frames remaining in the post-contact window
    int   latched;        // an event has fired and not been re-armed
    float peakG;
    float contactFpm;
    float peakAxis[3];
};

inline Detector makeDetector(const Thresholds &t)
{
    Detector d;
    d.th = t;
    d.armed = 0;
    d.airborneCount = 0;
    d.inWindow = 0;
    d.latched = 0;
    d.peakG = 0.0f;
    d.contactFpm = 0.0f;
    d.peakAxis[0] = d.peakAxis[1] = d.peakAxis[2] = 0.0f;
    return d;
}

// Total G magnitude. g_nrml reads about 1 in level flight because it includes
// the reaction to gravity, so the resting state is 1 rather than 0 and a
// threshold of 2 means "one g of acceleration beyond holding the aeroplane up".
inline float totalG(const Sample &s)
{
    return std::sqrt(s.gNrml * s.gNrml + s.gAxil * s.gAxil + s.gSide * s.gSide);
}

inline Severity classify(const Thresholds &t, float peakG, float contactFpm)
{
    if (peakG >= t.breakupG)                                  return SEV_BREAKUP;
    if (peakG >= t.structG || contactFpm <= t.structFpm)       return SEV_STRUCT;
    if (peakG >= t.hardG   || contactFpm <= t.hardFpm)         return SEV_HARD;
    return SEV_NONE;
}

// One step. Returns an Event with fired=0 on every frame except the one that
// closes a contact window with a severity above SEV_NONE.
inline Event step(Detector &d, const Sample &s)
{
    Event e;
    e.fired = 0;
    e.severity = SEV_NONE;
    e.peakG = 0.0f;
    e.contactFpm = 0.0f;
    e.impulse[0] = e.impulse[1] = e.impulse[2] = 0.0f;

    if (!s.onGround) {
        if (++d.airborneCount >= d.th.rearmFrames) {
            d.armed = 1;
            d.latched = 0;      // properly airborne again: allow another event
        }
        // Leaving the ground abandons any window in progress. A bounce is part
        // of the SAME arrival, and letting the window survive it would blend
        // two impacts into one reading.
        d.inWindow = 0;
        return e;
    }

    // On the ground from here.
    const int justLanded = (d.airborneCount > 0);
    d.airborneCount = 0;

    if (justLanded && d.armed && !d.latched) {
        d.inWindow = d.th.windowFrames;
        d.contactFpm = s.vhFpm;      // the descent rate AT contact, not later
        d.peakG = 0.0f;
        d.peakAxis[0] = d.peakAxis[1] = d.peakAxis[2] = 0.0f;
    }

    if (d.inWindow > 0) {
        float g = totalG(s);
        if (g > d.peakG) {
            d.peakG = g;
            d.peakAxis[0] = s.gAxil;
            d.peakAxis[1] = s.gNrml;
            d.peakAxis[2] = s.gSide;
        }
        if (--d.inWindow == 0) {
            Severity sev = classify(d.th, d.peakG, d.contactFpm);
            if (sev != SEV_NONE) {
                e.fired = 1;
                e.severity = sev;
                e.peakG = d.peakG;
                e.contactFpm = d.contactFpm;
                for (int i = 0; i < 3; ++i) e.impulse[i] = d.peakAxis[i];
                d.latched = 1;   // one impact, one event
            }
        }
    }
    return e;
}

}  // namespace destruct
