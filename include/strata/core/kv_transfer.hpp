// The restore transaction's rules, as host-only code: one ordered pass over a K/V payload that verifies each block
// before it reaches the live destination, starts one transaction-wide mutation boundary at the first destination
// write, and classifies the failure by what had already happened to the engine.
//
// This is the part that must not be re-derived at each call site.  The RAM path pushes its segmented buffers, the
// disk capsule pushes verified spill blocks, and the engine supplies the one device-touching callback (the write into
// the live pools); this file decides ordering, counting and the outcome.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace strata::core {

// How a restore (or a proof pass) ended.  The distinction is about the LIVE state, not the wording of an error:
//   refused - nothing was written and nothing needs unwinding: an admission or driver refusal before the first
//             write, so the capsule may be retried or cancelled exactly as before
//   lost    - the payload is unusable (short read, digest mismatch, truncation) before any write: the capsule is
//             dead, the engine is intact, and the request ends with one typed terminal
//   fatal   - a failure at or after the first destination write: the live state is unknown and the engine must not
//             keep decoding
enum class KvTransferOutcome { refused, lost, fatal };

struct KvTransferResult {
    bool ok = false;
    KvTransferOutcome outcome = KvTransferOutcome::refused;
    uint64_t verified_bytes = 0;   // bytes verified AND written before any failure
    bool mutated = false;          // at least one destination write was attempted
    std::string error;
};

// A verified block on its way to the destination: exactly `n` bytes of `part` at `offset`, already checked.
using KvBlockSink = std::function<bool(size_t part, uint64_t offset, const uint8_t* data, size_t n, std::string& error)>;

// What the payload source reports when it stops: `ok` done, `refused` nothing was transferred for an admission
// reason (memory, a thread that could not start), `lost` the payload itself is unusable (I/O, size, digest).
enum class KvDrive { ok, refused, lost };
using KvDriveFn = std::function<KvDrive(const KvBlockSink& sink, std::string& error)>;

struct KvTransfer {
    // Pushes the payload into `sink` in manifest order and returns how it ended.
    KvDriveFn drive;
    // Optional per-block integrity check before the destination write.  A disk read that already verified its blocks
    // leaves this empty; the RAM path and a proof readback can use it.
    std::function<bool(size_t part, uint64_t offset, const uint8_t* data, size_t n, std::string& error)> verify;
    // Writes one verified block into the live destination.  A failure here is fatal: the write may have touched part
    // of the destination before failing.
    KvBlockSink write;
};

// Runs one transfer.  Ordering: verify, then write, then count.  The mutation boundary is set BEFORE the first write
// call, never after it, so a write that fails halfway is already fatal rather than "recoverable".
KvTransferResult kv_transfer_run(const KvTransfer& transfer);

}  // namespace strata::core
