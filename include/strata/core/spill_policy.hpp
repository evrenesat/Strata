// The parked capsule's capture-time policy, as arithmetic: how long a disk capture is predicted to take, whether that
// fits the budget before a queued VIP has waited too long, and how often the engine may tell the scheduler about
// progress.  Pure functions and one small state class: no clock, no I/O, no engine state, so the engine's capture
// path and its tests share exactly the same rules.
//
// These are conservative policy defaults, not measurements of any particular disk.
#pragma once

#include <cstdint>
#include <limits>

namespace strata::core {

// Publishing a disk capture has to finish inside this budget, starting when the pause was accepted (so the pause
// drain and a refused RAM attempt both eat into it).  It bounds cooperative work; it cannot interrupt a syscall.
inline constexpr double kSpillCaptureBudgetSeconds = 60.0;
// Predicted capture time: a fixed setup plus 1.5x the payload at the expected rate, so a slow disk is predicted
// pessimistically rather than optimistically.
inline constexpr double kSpillEstimateBaseSeconds = 5.0;
inline constexpr double kSpillEstimateSlack = 1.5;
// Before this process has completed a capture on this directory, assume this rate.
inline constexpr double kSpillDefaultRateBytesPerSecond = 128.0 * (1 << 20);

// External progress is coalesced: 256 MiB of newly completed work, or ten seconds since the last event, whichever
// comes first - plus an explicit event at the start and at each major phase transition.  Never one event per block,
// and never a timer-only event while nothing is completing.
inline constexpr uint64_t kSpillProgressBytes = 256ull << 20;
inline constexpr double kSpillProgressSeconds = 10.0;

// The last successful capture's rate on this process and directory.  A sample counts only when it describes real
// work: a finite, strictly positive duration and a nonzero payload.  The effective rate never exceeds the default,
// so a suspiciously fast sample cannot make the estimate optimistic.
struct SpillCaptureRate {
    double seconds = 0;
    uint64_t bytes = 0;

    bool have() const { return seconds > 0 && bytes != 0; }
    double bytes_per_second() const {
        if (!have()) return kSpillDefaultRateBytesPerSecond;
        const double measured = static_cast<double>(bytes) / seconds;
        if (!(measured > 0)) return kSpillDefaultRateBytesPerSecond;   // also rejects NaN
        return measured < kSpillDefaultRateBytesPerSecond ? measured : kSpillDefaultRateBytesPerSecond;
    }
    // Keeps only a usable sample; anything else leaves the previous one in place.
    void sample(double seconds_, uint64_t bytes_) {
        if (!(seconds_ > 0) || bytes_ == 0) return;
        if (seconds_ > 1e9 || !(seconds_ < std::numeric_limits<double>::infinity())) return;
        seconds = seconds_;
        bytes = bytes_;
    }
};

// The predicted time to publish a capture of `payload_bytes`, in seconds.
inline double spill_capture_estimate_seconds(uint64_t payload_bytes, const SpillCaptureRate& rate) {
    const double per_second = rate.bytes_per_second();
    const double transfer = per_second > 0 ? static_cast<double>(payload_bytes) / per_second : 0;
    return kSpillEstimateBaseSeconds + kSpillEstimateSlack * transfer;
}

// True when the prediction fits what remains of the budget.  A non-finite or non-positive remaining budget admits
// nothing: the caller must never publish an already-expired capture.
inline bool spill_capture_admits(double estimate_seconds, double remaining_budget_seconds) {
    if (!(estimate_seconds >= 0) || !(remaining_budget_seconds > 0)) return false;
    if (!(remaining_budget_seconds < std::numeric_limits<double>::infinity())) return true;
    return estimate_seconds <= remaining_budget_seconds;
}

// The external-progress gate.  `due` answers whether an event is owed now: enough newly completed bytes, or enough
// elapsed time *with* new completed work.  A fresh gate owes its first (transfer-start) event at once, and `force()`
// does the same at a major phase transition.
class SpillProgressGate {
public:
    explicit SpillProgressGate(double now = 0) : last_at_(now) {}

    bool due(double now, uint64_t completed_bytes) const {
        if (pending_) return true;
        if (completed_bytes <= last_bytes_) return false;                 // no new work: no event, however long
        if (completed_bytes - last_bytes_ >= kSpillProgressBytes) return true;
        return now - last_at_ >= kSpillProgressSeconds;
    }
    void emitted(double now, uint64_t completed_bytes) {
        last_at_ = now;
        last_bytes_ = completed_bytes;
        pending_ = false;
    }
    // The first event and each major phase transition ask for one regardless of the thresholds.
    void force() { pending_ = true; }
    uint64_t last_emitted_bytes() const { return last_bytes_; }

private:
    double last_at_ = 0;
    uint64_t last_bytes_ = 0;
    bool pending_ = true;   // the initial transfer event
};

}  // namespace strata::core
