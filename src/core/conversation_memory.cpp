#include "strata/core/conversation_memory.hpp"

#include <algorithm>
#include <charconv>
#include <cerrno>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strata::core {
namespace {

// The v1 "no limit" value for memory.limit_in_bytes (0x7FFFFFFFFFFFF000); a v2 "max" or a
// value at or above the sentinel reads as an unbounded allowance (UINT64_MAX) end to end.
constexpr uint64_t cgroup_no_limit = 0x7FFFFFFFFFFFF000ULL;

// A limit file (memory.max / memory.limit_in_bytes): a missing file means the controller (or
// its limit) is not there, so no bound applies; any other failure or malformed content is
// unknown and the caller fails closed.
std::optional<uint64_t> read_cgroup_limit(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        if (errno == ENOENT) return std::numeric_limits<uint64_t>::max();
        return {};
    }
    return conversation_cgroup_memory_value(file);
}

// A usage file (memory.current / memory.usage_in_bytes): missing or unreadable is unknown.
std::optional<uint64_t> read_cgroup_usage(const std::string& path) {
    std::ifstream file(path);
    if (!file) return {};
    return conversation_cgroup_memory_value(file);
}

std::string cgroup_relative_path(const std::string& path) {
    if (path.empty() || path == "/") return "";
    return path;                                  // always starts with '/' when non-empty
}

uint64_t saturating_add(uint64_t a, uint64_t b) {
    const auto max = std::numeric_limits<uint64_t>::max();
    return b > max - a ? max : a + b;
}

} // namespace

// The credit is a reclamation estimate, so every unusable input withdraws it completely: a
// missing key, a repeated key, a token that is not strict unsigned decimal, or an I/O failure
// all return no value and the caller keeps the older, fully conservative bound.
std::optional<uint64_t> conversation_cgroup_memory_credit(std::istream& stat, uint64_t usage,
                                                          bool v1_stats) {
    const char* const names[4] = {
        v1_stats ? "total_inactive_file" : "inactive_file",
        v1_stats ? "total_cache" : "file",
        v1_stats ? "total_dirty" : "file_dirty",
        v1_stats ? "total_writeback" : "file_writeback",
    };
    if (!stat) return {};
    std::optional<uint64_t> found[4];
    std::string line;
    while (std::getline(stat, line)) {
        std::istringstream fields(line);
        std::string key, token, extra;
        if (!(fields >> key)) continue;                    // blank line
        const auto slot = [&]() -> int {
            for (int i = 0; i < 4; ++i) if (key == names[i]) return i;
            return -1;
        }();
        if (!(fields >> token) || (fields >> extra)) {     // no value, or trailing fields
            if (slot >= 0) return {};
            continue;
        }
        if (slot < 0) continue;                            // an unrelated stat key
        if (found[slot]) return {};                        // a repeated required key
        uint64_t value = 0;
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
        if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size()) return {};
        found[slot] = value;
    }
    if (stat.bad() || (stat.fail() && !stat.eof())) return {};   // a partial or failed read
    for (const auto& value : found) if (!value) return {};       // a missing required key
    const uint64_t reclaimable = std::min(*found[0], *found[1]);
    const uint64_t unwritten = saturating_add(*found[2], *found[3]);
    const uint64_t clean = reclaimable > unwritten ? reclaimable - unwritten : 0;
    return std::min(usage, clean);                            // never credit more than usage
}

std::optional<uint64_t> conversation_mem_available(std::istream& meminfo) {
    std::optional<uint64_t> result;
    std::string line;
    while (std::getline(meminfo, line)) {
        std::istringstream fields(line);
        std::string key, value, unit, extra;
        if (!(fields >> key) || key != "MemAvailable:") continue;
        if (result || !(fields >> value >> unit) || unit != "kB" || (fields >> extra)) return {};
        uint64_t kb = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), kb);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
            kb > std::numeric_limits<uint64_t>::max() / 1024) return {};
        result = kb * 1024;
    }
    if (meminfo.bad() || (meminfo.fail() && !meminfo.eof())) return {};
    return result;
}

std::optional<uint64_t> conversation_cgroup_memory_value(std::istream& file) {
    if (!file) return {};
    std::string text;
    std::getline(file, text);
    if (file.bad() || (file.fail() && !file.eof())) return {};
    // "max" (v2) or the v1 sentinel: the cgroup holds no byte limit for this process.
    if (text == "max") return std::numeric_limits<uint64_t>::max();
    uint64_t bytes = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), bytes);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) return {};
    if (bytes >= cgroup_no_limit) return std::numeric_limits<uint64_t>::max();
    return bytes;
}

ConversationCgroupMemory conversation_cgroup_memory(const std::string& proc_cgroup_path,
                                                    const std::string& root) {
    // /proc/self/cgroup's v2 line is `0::<path>`; a v1 memory line is `<n>:memory:<path>`
    // (the controller list is comma-separated).  v2 wins when both are present.
    std::ifstream self_cgroup(proc_cgroup_path);
    if (!self_cgroup) return {};                  // no cgroup view: the host view alone applies
    std::optional<std::string> v2_path, v1_path;
    std::string line;
    while (std::getline(self_cgroup, line)) {
        if (!v2_path && line.rfind("0::", 0) == 0) v2_path = line.substr(3);
        if (!v1_path) {
            const auto first = line.find(':');
            const auto second = first == std::string::npos ? std::string::npos
                                                           : line.find(':', first + 1);
            if (second != std::string::npos) {
                const std::string controllers = line.substr(first + 1, second - first - 1);
                if (controllers == "memory" ||
                    controllers.find(",memory") != std::string::npos ||
                    controllers.find("memory,") != std::string::npos ||
                    controllers.find(",memory,") != std::string::npos)
                    v1_path = line.substr(second + 1);
            }
        }
    }
    if (self_cgroup.bad()) return {};
    // One bounded level: the limit file, the usage file and the level's memory.stat.  A missing
    // or unreadable stat file grants no credit, which keeps the pre-credit conservative bound.
    const auto allowance = [&](const std::string& base, const std::string& limit_suffix,
                               const std::string& usage_suffix,
                               bool v1_stats) -> ConversationCgroupMemory {
        const auto limit = read_cgroup_limit(base + limit_suffix);
        if (!limit) return {true, {}, {}, {}};    // discovered but unreadable: fail closed
        if (*limit == std::numeric_limits<uint64_t>::max()) return {true, *limit, {}, {}};  // no bound
        const auto used = read_cgroup_usage(base + usage_suffix);
        if (!used) return {true, {}, {}, {}};     // a real limit whose usage cannot be read
        uint64_t credit = 0;
        if (std::ifstream stat(base + "/memory.stat"); stat)
            credit = conversation_cgroup_memory_credit(stat, *used, v1_stats).value_or(0);
        const auto effective_used = *used > credit ? *used - credit : 0;
        return {true, *limit > effective_used ? *limit - effective_used : 0, *used, credit};
    };
    // Limits are hierarchical: an ancestor's memory.max binds the process even when the
    // leaf is "max", so walk every level from the leaf to the hierarchy root and keep the
    // smallest allowance seen.  A missing limit file at a level means that level is unbound.
    const auto walk = [&](const std::string& prefix, const std::string& start,
                          const std::string& limit_suffix, const std::string& usage_suffix,
                          bool v1_stats) -> ConversationCgroupMemory {
        std::string path = cgroup_relative_path(start);
        ConversationCgroupMemory bound;           // the smallest level allowance seen
        while (true) {
            const auto level = allowance(prefix + path, limit_suffix, usage_suffix, v1_stats);
            if (!level.available) return {true, {}, {}, {}};   // a level that cannot be read
            if (!bound.available || *level.available < *bound.available) bound = level;
            if (path.empty() || path == "/") break;
            const auto slash = path.rfind('/');
            path = slash == std::string::npos ? std::string() : path.substr(0, slash);
        }
        return bound;
    };
    // Which hierarchy binds this process:
    //   - the v2 layout, when there is no v1 memory line (a pure v2 host) or when the unified root really controls
    //     memory, which its own `cgroup.controllers` file states;
    //   - otherwise the v1 memory limit, because on a hybrid host the unified tree has no memory controller and its
    //     files do not exist - reading that path finds nothing, which reads as "unbounded" and silently fails open.
    // So a v2 line whose listing cannot be read falls back to v1 when a v1 memory line exists: the safe direction,
    // and there is no third case, since a v2 line with no v1 line is handled by the first condition.
    const auto v2_controls_memory = [&]() {
        std::ifstream controllers(root + "/cgroup.controllers");
        if (!controllers) return false;
        std::string listing;
        while (std::getline(controllers, listing)) {
            std::istringstream tokens(listing);
            std::string token;
            while (tokens >> token)
                if (token == "memory") return true;
        }
        return false;
    };
    if (v2_path && (!v1_path || v2_controls_memory()))
        return walk(root, *v2_path, "/memory.max", "/memory.current", false);
    if (v1_path) return walk(root + "/memory", *v1_path, "/memory.limit_in_bytes",
                             "/memory.usage_in_bytes", true);
    return {};                                    // no memory controller: the host view alone applies
}

std::optional<uint64_t> conversation_memory_bound(std::optional<uint64_t> host,
                                                  const ConversationCgroupMemory& cgroup) {
    if (!host || (cgroup.present && !cgroup.available)) return {};
    if (!cgroup.present) return host;
    return std::min(*host, *cgroup.available);
}

std::optional<uint64_t> conversation_available_memory() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof status;
    if (GlobalMemoryStatusEx(&status)) return status.ullAvailPhys;
    return {};
#elif defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo) return {};
    const auto host = conversation_mem_available(meminfo);
    return conversation_memory_bound(host,
        conversation_cgroup_memory("/proc/self/cgroup", "/sys/fs/cgroup"));
#else
    return {};
#endif
}

} // namespace strata::core
