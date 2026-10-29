// What a parked-capsule spill costs in memory and on disk, as checks a caller can reason about before it allocates.
// Pure constants and checked arithmetic: the engine's capture and restore paths use these numbers instead of
// re-deriving them, and the acceptance fixture bounds its measurements with the same ones.
#pragma once

#include <cstdint>
#include <limits>

namespace strata::core {

// The capture path holds ONE reusable 16 MiB staging block (the writer assembles whole blocks there) and lets at
// most one 64 MiB writeback window sit dirty before it is written back and dropped from the page cache.
inline constexpr uint64_t kSpillCaptureStagingBytes = 16ull << 20;
inline constexpr uint64_t kSpillWritebackBytes = 64ull << 20;

// The restore path reads through TWO 16 MiB buffers with one reader thread, and the reader's own runtime plus the
// clean read cache it is allowed to leave behind are budgeted separately.
inline constexpr uint64_t kSpillRestoreBuffersBytes = 2 * (16ull << 20);
inline constexpr uint64_t kSpillReaderAllowanceBytes = 8ull << 20;
inline constexpr uint64_t kSpillReadCacheAllowanceBytes = 32ull << 20;

// The proof/readback allowance the engine already reserves while a session is captured.
inline constexpr uint64_t kSpillProofAllowanceBytes = 1ull << 20;

// Additional memory each phase may ask for beyond what the engine already holds.  These bound the *new* allocation,
// not the engine's resident working set: a spilled capsule avoids the capsule's own K/V copy, not the pools it reads.
inline constexpr uint64_t spill_capture_additional_bytes() {
    return kSpillCaptureStagingBytes + kSpillProofAllowanceBytes;
}
inline constexpr uint64_t spill_restore_additional_bytes() {
    return kSpillRestoreBuffersBytes + kSpillReaderAllowanceBytes + kSpillProofAllowanceBytes;
}

// The capsule budget and floor the RAM path already enforces, restated here so a disk capture is admitted against the
// same limits rather than a parallel set.
inline constexpr uint64_t kSpillCapsuleLimitBytes = 4608ull * 1024 * 1024;
inline constexpr uint64_t kSpillFreeFloorBytes = 2560ull * 1024 * 1024;

// The peak a capture of `payload_bytes` adds: the staging block plus, while a window is outstanding, that window and
// its writeback.  Two separate figures because they are different kinds of memory (anonymous vs dirty cache).
struct SpillCapturePeak {
    uint64_t anonymous = 0;
    uint64_t dirty = 0;
};
inline constexpr SpillCapturePeak spill_capture_peak(uint64_t payload_bytes) {
    SpillCapturePeak peak{};
    peak.anonymous = kSpillCaptureStagingBytes + kSpillProofAllowanceBytes;
    peak.dirty = payload_bytes < kSpillWritebackBytes ? payload_bytes : kSpillWritebackBytes;
    return peak;
}

// Whether the disk has room for this capture: the payload plus the configured extra reserve (which a value of 0
// removes, leaving the payload requirement).  A preflight, not a reservation: other writers can take the space later,
// which is why the write path must still handle ENOSPC and EDQUOT.
inline bool spill_disk_preflight(bool free_known, uint64_t free_bytes, uint64_t payload_bytes,
                                 uint64_t reserve_bytes) {
    if (!free_known) return false;                       // unknown free space never admits a capture
    if (payload_bytes > std::numeric_limits<uint64_t>::max() - reserve_bytes) return false;
    return free_bytes >= payload_bytes + reserve_bytes;
}

}  // namespace strata::core
