#include "strata/core/kv_transfer.hpp"

#include <utility>

namespace strata::core {

KvTransferResult kv_transfer_run(const KvTransfer& transfer) {
    KvTransferResult result;
    if (!transfer.drive || !transfer.write) {
        result.outcome = KvTransferOutcome::refused;
        result.error = "K/V transfer has no source or no destination";
        return result;
    }
    std::string error;
    bool sink_refused = false;
    // Every block goes through here in order: verify, then write, then count.  Once a write is attempted the live
    // state may already have changed, so the boundary is recorded before the call, not after it.
    const KvBlockSink checked = [&](size_t part, uint64_t offset, const uint8_t* data, size_t n, std::string& why) {
        if (transfer.verify && !transfer.verify(part, offset, data, n, why)) {
            sink_refused = true;
            return false;
        }
        result.mutated = true;   // set before the write: a write that fails partway is already fatal
        if (!transfer.write(part, offset, data, n, why)) {
            sink_refused = true;
            return false;
        }
        result.verified_bytes += n;
        return true;
    };
    const KvDrive drove = transfer.drive(checked, error);
    result.error = error;
    if (drove == KvDrive::ok && !sink_refused) {
        result.ok = true;
        result.outcome = KvTransferOutcome::refused;   // not meaningful on success; callers read ok
        return result;
    }
    if (result.mutated) {
        result.outcome = KvTransferOutcome::fatal;
        if (result.error.empty()) result.error = "K/V restore failed after live state was written";
        return result;
    }
    result.outcome = sink_refused ? KvTransferOutcome::lost : (drove == KvDrive::lost ? KvTransferOutcome::lost
                                                                                    : KvTransferOutcome::refused);
    if (result.error.empty())
        result.error = result.outcome == KvTransferOutcome::lost ? "K/V payload unusable"
                                                                 : "K/V transfer refused before writing";
    return result;
}

}  // namespace strata::core
