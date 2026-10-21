#include "strata/core/kv_transfer.hpp"

#include <utility>

namespace strata::core {

namespace {
// The one place the ending is classified, for both entry points.
KvTransferResult classify(KvDrive drove, bool sink_refused, bool mutated, uint64_t verified, std::string error) {
    const bool ok = drove == KvDrive::ok;
    KvTransferResult result;
    result.mutated = mutated;
    result.verified_bytes = verified;
    result.error = std::move(error);
    if (ok && !sink_refused) {
        result.ok = true;
        result.outcome = KvTransferOutcome::refused;   // not meaningful on success; callers read ok
        return result;
    }
    if (mutated) {
        result.outcome = KvTransferOutcome::fatal;
        if (result.error.empty()) result.error = "K/V restore failed after live state was written";
        return result;
    }
    // A sink refusal and a source that reports loss both mean the payload was unusable; only a plain driver refusal
    // is the recoverable "nothing was transferred" case.
    result.outcome = (sink_refused || drove == KvDrive::lost) ? KvTransferOutcome::lost : KvTransferOutcome::refused;
    if (result.error.empty())
        result.error = result.outcome == KvTransferOutcome::lost ? "K/V payload unusable"
                                                                 : "K/V transfer refused before writing";
    return result;
}
}  // namespace

KvTransferResult kv_layer_transfer_run(const KvLayerDriveFn& drive, const KvLayerSink& write) {
    if (!drive || !write)
        return classify(KvDrive::refused, false, false, 0, "K/V transfer has no source or no destination");
    bool sink_refused = false;
    bool mutated = false;
    uint64_t verified = 0;
    const KvLayerSink checked = [&](size_t layer, bool draft, size_t part, uint64_t offset, const uint8_t* data,
                                    size_t n, std::string& why) {
        mutated = true;   // before the write: a write that fails partway is already fatal
        if (!write(layer, draft, part, offset, data, n, why)) {
            sink_refused = true;
            return false;
        }
        verified += n;
        return true;
    };
    std::string error;
    const KvDrive drove = drive(checked, error);
    return classify(drove, sink_refused, mutated, verified, error);
}

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
    return classify(drove, sink_refused, result.mutated, result.verified_bytes, error);
}

}  // namespace strata::core
