#pragma once

#include <cstdint>
#include <istream>
#include <optional>
#include <string>

namespace strata::core {

// Host physical memory, not swap/commit or a container/job memory reservation.
// Unknown telemetry is deliberately distinct from a measured zero.
std::optional<uint64_t> conversation_available_memory();
std::optional<uint64_t> conversation_mem_available(std::istream& meminfo);

// One cgroup memory file reading (memory.max, memory.current, memory.limit_in_bytes,
// memory.usage_in_bytes) as a byte count.  "max" (v2) and the v1 no-limit sentinel read as
// UINT64_MAX (an unbounded allowance); a missing, unreadable or malformed file is unknown.
std::optional<uint64_t> conversation_cgroup_memory_value(std::istream& file);

// Conservative reclaimable file-cache credit for one cgroup level, from its memory.stat.
//
// `usage` is the level's memory.current (v2) or memory.usage_in_bytes (v1).  With U = usage,
// I = inactive_file, F = file, D = dirty and W = writeback bytes, the credit is
//     min(U, max(0, min(I, F) - (D + W)))
// so the level admits `limit - (U - credit)` bytes.  Only clean, reclaimable file cache can be
// credited: active cache, anonymous/pinned/shared memory and arbitrary `file` bytes are never
// counted.  This estimates what the kernel could reclaim; it is not a reservation and not
// proof that every credited page can be reclaimed.
//
// Required keys are `inactive_file`, `file`, `file_dirty`, `file_writeback` (v2) or the
// hierarchical `total_inactive_file`, `total_cache`, `total_dirty`, `total_writeback` totals
// (v1).  A missing, unreadable, duplicated or malformed required entry, or unusable counters,
// yields no credit at all (`nullopt`), which keeps the older fully conservative bound.
std::optional<uint64_t> conversation_cgroup_memory_credit(std::istream& stat, uint64_t usage,
                                                          bool v1_stats);

// The process's cgroup memory allowance.  v2 first (`0::<path>` in the proc file, then
// <root>/<path>/memory.max minus memory.current), else the v1 memory controller line
// (<root>/memory<path>/memory.limit_in_bytes minus memory.usage_in_bytes).  Each bounded level
// subtracts its reclaimable file-cache credit from usage first, and the smallest corrected
// allowance in the hierarchy wins.  Pure and injectable (paths, not the real /proc) so tests
// can use temp files.
struct ConversationCgroupMemory {
    bool present = false;               // a cgroup memory controller binds this process
    std::optional<uint64_t> available;  // bytes left in the allowance; unknown = unreadable
    std::optional<uint64_t> usage;      // raw usage of the level that set the allowance
    std::optional<uint64_t> credit;     // inactive-file credit subtracted at that level
};
ConversationCgroupMemory conversation_cgroup_memory(const std::string& proc_cgroup_path,
                                                    const std::string& root);

// The admitted budget: the smaller of the host view and the cgroup allowance.  Unknown
// telemetry (no host reading, or a discovered cgroup whose files cannot be read) fails closed.
std::optional<uint64_t> conversation_memory_bound(std::optional<uint64_t> host,
                                                  const ConversationCgroupMemory& cgroup);

inline bool conversation_memory_admit(std::optional<uint64_t> available,
                                      uint64_t allocation, uint64_t floor) {
    return available && *available >= floor && allocation <= *available - floor;
}

} // namespace strata::core
