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

} // namespace

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
    const auto allowance = [&](const std::string& limit_file, const std::string& usage_file)
        -> ConversationCgroupMemory {
        const auto limit = read_cgroup_limit(limit_file);
        if (!limit) return {true, {}};            // discovered but unreadable: fail closed
        if (*limit == std::numeric_limits<uint64_t>::max()) return {true, *limit};  // no bound
        const auto used = read_cgroup_usage(usage_file);
        if (!used) return {true, {}};             // a real limit whose usage cannot be read
        return {true, *limit > *used ? *limit - *used : 0};
    };
    // Limits are hierarchical: an ancestor's memory.max binds the process even when the
    // leaf is "max", so walk every level from the leaf to the hierarchy root and keep the
    // smallest allowance seen.  A missing limit file at a level means that level is unbound.
    const auto walk = [&](const std::string& base, const std::string& limit_suffix,
                          const std::string& usage_suffix) -> ConversationCgroupMemory {
        std::string path = cgroup_relative_path(base);
        std::optional<uint64_t> bound;            // the smallest level allowance seen
        while (true) {
            const auto level = allowance(root + path + limit_suffix, root + path + usage_suffix);
            if (!level.available) return {true, {}};      // a discovered level that cannot be read
            if (!bound || *level.available < *bound) bound = level.available;
            if (path.empty() || path == "/") break;
            const auto slash = path.rfind('/');
            path = slash == std::string::npos ? std::string() : path.substr(0, slash);
        }
        return {true, bound};
    };
    if (v2_path) return walk(*v2_path, "/memory.max", "/memory.current");
    if (v1_path) return walk("/memory" + cgroup_relative_path(*v1_path),
                             "/memory.limit_in_bytes", "/memory.usage_in_bytes");
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
