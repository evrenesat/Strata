// Host-only tests of the capture-time policy: the prediction, the budget verdict and the progress gate.
#include "strata/core/spill_policy.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
constexpr double mib = 1 << 20;
}  // namespace

int main() {
    // ---- the rate sample -------------------------------------------------------------
    {
        SpillCaptureRate rate;
        check(!rate.have(), "no sample before the first successful capture");
        check(rate.bytes_per_second() == kSpillDefaultRateBytesPerSecond, "an unknown rate is the conservative default");
        rate.sample(0, 1 << 20);                 // no duration
        check(!rate.have(), "a zero-duration sample is ignored");
        rate.sample(-1, 1 << 20);
        check(!rate.have(), "a negative-duration sample is ignored");
        rate.sample(1, 0);
        check(!rate.have(), "a zero-byte sample is ignored");
        rate.sample(std::numeric_limits<double>::quiet_NaN(), 1 << 20);
        check(!rate.have(), "a NaN sample is ignored");
        rate.sample(std::numeric_limits<double>::infinity(), 1 << 20);
        check(!rate.have(), "an infinite sample is ignored");
        rate.sample(2.0, 64ull << 20);           // a real capture: 32 MiB/s
        check(rate.have() && rate.bytes_per_second() == 32 * mib, "a measured rate is used");
        rate.sample(0.5, 512ull << 20);          // an implausibly fast sample
        check(rate.bytes_per_second() == kSpillDefaultRateBytesPerSecond,
              "a sample faster than the default is clamped to it");
    }
    // ---- the prediction --------------------------------------------------------------
    {
        SpillCaptureRate unknown;
        const double estimate = spill_capture_estimate_seconds(256ull << 20, unknown);
        check(std::fabs(estimate - (kSpillEstimateBaseSeconds + kSpillEstimateSlack * 2.0)) < 1e-9,
              "256 MiB at the default rate predicts setup plus 1.5x the transfer");
        SpillCaptureRate slow;
        slow.sample(4.0, 32ull << 20);           // 8 MiB/s
        const double slow_estimate = spill_capture_estimate_seconds(64ull << 20, slow);
        check(slow_estimate > estimate, "a slower rate predicts a longer capture");
        check(spill_capture_estimate_seconds(0, unknown) == kSpillEstimateBaseSeconds,
              "an empty payload still costs the setup");
    }
    // ---- the budget verdict ----------------------------------------------------------
    {
        check(spill_capture_admits(59.9, 60.0), "a prediction inside the budget is admitted");
        check(!spill_capture_admits(60.1, 60.0), "a prediction past the budget is refused before any write");
        check(spill_capture_admits(1.0, std::numeric_limits<double>::infinity()),
              "an unbounded budget admits any finite prediction");
        check(!spill_capture_admits(1.0, 0.0), "an exhausted budget admits nothing");
        check(!spill_capture_admits(1.0, -1.0), "a negative budget admits nothing");
        check(!spill_capture_admits(std::numeric_limits<double>::quiet_NaN(), 60.0),
              "an unusable prediction admits nothing");
    }
    // ---- the progress gate -----------------------------------------------------------
    {
        SpillProgressGate gate(0.0);
        check(gate.due(0.0, 0), "the first event is owed immediately");
        gate.force();
        check(gate.due(0.0, 0), "a phase transition forces an event");
        gate.emitted(0.0, 0);
        check(!gate.due(0.5, 0), "nothing completed: no event");
        check(!gate.due(9.9, 0), "elapsed time alone never emits");
        check(gate.due(0.5, kSpillProgressBytes), "256 MiB of new work emits");
        gate.emitted(0.5, kSpillProgressBytes);
        check(!gate.due(1.0, kSpillProgressBytes + 16 * (1ull << 20)),
              "a block under the byte threshold does not emit early");
        check(gate.due(10.5, kSpillProgressBytes + 16 * (1ull << 20)),
              "ten seconds with new work emits");
        gate.emitted(10.5, kSpillProgressBytes + 16 * (1ull << 20));
        check(!gate.due(30.0, kSpillProgressBytes + 16 * (1ull << 20)),
              "a stalled capture emits nothing however long it stalls");
        check(gate.due(30.0, kSpillProgressBytes + 16 * (1ull << 20) + 1),
              "a single completed byte after the stall emits on the time rule");
        gate.emitted(30.0, kSpillProgressBytes + 16 * (1ull << 20) + 1);
        check(!gate.due(31.0, kSpillProgressBytes + 16 * (1ull << 20)),
              "an older cursor never rewinds the gate");
    }
    std::printf("spill_policy_test: %d checks passed\n", checks);
}
