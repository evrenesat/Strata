// Host-only tests of the spill memory budget and disk preflight arithmetic.
#include "strata/core/spill_budget.hpp"

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
constexpr uint64_t mib = 1 << 20;
}  // namespace

int main() {
    // The phase allowances are the plan's numbers, and they stay small next to the payload they replace.
    check(spill_capture_additional_bytes() == 17 * mib, "capture adds one 16 MiB staging block and the proof allowance");
    check(spill_restore_additional_bytes() == 41 * mib,
          "restore adds two 16 MiB buffers, the reader allowance and the proof allowance");
    check(kSpillCapsuleLimitBytes > kSpillFreeFloorBytes, "the capsule budget is above its floor");
    check(spill_capture_additional_bytes() + kSpillFreeFloorBytes < kSpillCapsuleLimitBytes,
          "capture scratch plus the floor still fits the capsule budget");

    // The capture peak: bounded dirty cache, and a small payload is bounded by the payload.
    {
        const auto small = spill_capture_peak(4 * mib);
        check(small.anonymous == 17 * mib && small.dirty == 4 * mib, "a small capture dirties only its own bytes");
        const auto big = spill_capture_peak(4ull << 30);
        check(big.dirty == kSpillWritebackBytes, "a large capture never leaves more than one window dirty");
        const auto empty = spill_capture_peak(0);
        check(empty.dirty == 0 && empty.anonymous == 17 * mib, "an empty capture dirties nothing");
    }
    // The disk preflight: payload plus reserve, and no capture without a known free-space reading.
    {
        const uint64_t payload = 4ull << 30;
        check(spill_disk_preflight(true, payload + 4096 * mib, payload, 4096 * mib), "exact room is enough");
        check(!spill_disk_preflight(true, payload + 4096 * mib - 1, payload, 4096 * mib), "one byte short is refused");
        check(spill_disk_preflight(true, payload, payload, 0), "a zero reserve still requires the payload");
        check(!spill_disk_preflight(true, payload - 1, payload, 0), "a zero reserve does not remove the payload check");
        check(!spill_disk_preflight(false, 1ull << 62, payload, 0), "unknown free space never admits a capture");
        check(!spill_disk_preflight(true, std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max(), 1),
              "an overflowing requirement is refused rather than wrapped");
    }
    std::printf("spill_budget_test: %d checks passed\n", checks);
}
