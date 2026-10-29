#include "strata/core/conversation_memory.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strata::core {

namespace fs = std::filesystem;

namespace {

std::atomic<uint64_t> g_memory_limit_bytes{0};

// "<key> <n> kB", exactly once, nothing else on the line; the value in bytes.  Anything else is unknown.
std::optional<uint64_t> meminfo_kb_bytes(std::istream& meminfo, const char* wanted) {
    std::optional<uint64_t> result;
    std::string line;
    while (std::getline(meminfo, line)) {
        std::istringstream fields(line);
        std::string key, value, unit, extra;
        if (!(fields >> key) || key != wanted) continue;
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

// One cgroup control file: absent (no such file), a number, "max", or unusable (unreadable, empty, not a number).
struct Value {
    enum Kind { absent, bad, max, number } kind = absent;
    uint64_t n = 0;
};

Value read_value(const fs::path& path) {
    std::error_code ec;
    const bool there = fs::exists(path, ec);
    if (ec) return {Value::bad, 0};
    if (!there) return {};
    std::ifstream file(path);
    if (!file) return {Value::bad, 0};
    char buf[64];
    file.read(buf, sizeof buf);
    if (file.bad()) return {Value::bad, 0};
    size_t len = (size_t) file.gcount();
    if (len == sizeof buf) return {Value::bad, 0};
    while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r' || buf[len - 1] == ' ' || buf[len - 1] == '\t')) --len;
    if (len == 0) return {Value::bad, 0};
    if (len == 3 && std::string(buf, 3) == "max") return {Value::max, 0};
    Value v{Value::number, 0};
    const auto parsed = std::from_chars(buf, buf + len, v.n);
    if (parsed.ec != std::errc{} || parsed.ptr != buf + len) return {Value::bad, 0};
    return v;
}

// The one reclaimable kind that is credited: clean inactive file cache, from the same group's memory.stat
// (v2 inactive_file, v1 total_inactive_file), less its dirty and writeback pages when those are listed, and never
// more than the group's charged usage.  active_file (it holds mlocked pages) and shmem (pinned complements, the KV
// pool) are not credited.  A memory.stat that is missing or unparsable gives no credit - not an unknown sample.
uint64_t reclaimable_cache(const fs::path& dir, uint64_t current, bool v1) {
    std::ifstream stat(dir / "memory.stat");
    if (!stat) return 0;
    const char* inactive_key = v1 ? "total_inactive_file" : "inactive_file";
    const char* dirty_key = v1 ? "total_dirty" : "file_dirty";
    const char* writeback_key = v1 ? "total_writeback" : "file_writeback";
    std::optional<uint64_t> inactive;
    uint64_t dirty = 0, writeback = 0;
    std::string line;
    while (std::getline(stat, line)) {
        std::istringstream fields(line);
        std::string key, value, extra;
        if (!(fields >> key >> value) || (fields >> extra)) continue;
        uint64_t* target = key == dirty_key ? &dirty : key == writeback_key ? &writeback : nullptr;
        uint64_t n = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), n);
        const bool ok = parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size();
        if (key == inactive_key) { if (!ok || inactive) return 0; inactive = n; }
        else if (target != nullptr) { if (!ok) return 0; *target = n; }
    }
    if (!inactive) return 0;
    uint64_t credit = std::min(*inactive, current);
    credit = dirty >= credit ? 0 : credit - dirty;
    credit = writeback >= credit ? 0 : credit - writeback;
    return credit;
}

// What the cgroup files say: every finite limit with the usage charged against it (and the cache credit of that
// group), and the usage and credit of the topmost visible group (the root of this namespace - what a container's own
// limit is measured over).
struct CgroupScan {
    struct Limit { uint64_t limit, current, credit; };
    std::vector<Limit> limits;
    std::optional<uint64_t> root_current;
    uint64_t root_credit = 0;
    bool any_files = false;
};

// `dirs` runs from the top visible group down to the process's own.  False when a file that is there cannot be used.
bool scan_levels(const std::vector<fs::path>& dirs, const char* const* limit_names, size_t n_limits,
                 const char* current_name, bool v1, CgroupScan& out) {
    for (size_t i = 0; i < dirs.size(); ++i) {
        const Value current = read_value(dirs[i] / current_name);
        std::vector<Value> limits;
        for (size_t k = 0; k < n_limits; ++k) limits.push_back(read_value(dirs[i] / limit_names[k]));
        bool unusable = current.kind == Value::bad || current.kind == Value::max;
        bool finite_limit = false;
        for (const Value& v : limits) {
            unusable = unusable || v.kind == Value::bad;
            finite_limit = finite_limit || v.kind == Value::number;
        }
        if (unusable) return false;
        if (current.kind == Value::absent) {
            if (finite_limit) return false;   // a limit with no usage beside it: the room cannot be told
            continue;                          // no memory controller at this level (e.g. the host's root group)
        }
        out.any_files = true;
        const uint64_t credit = reclaimable_cache(dirs[i], current.n, v1);
        if (i == 0) { out.root_current = current.n; out.root_credit = credit; }
        for (const Value& v : limits) {
            if (v.kind != Value::number) continue;
            if (v1 && v.n >= (1ull << 62)) continue;   // v1's "unlimited" is a number near 2^63
            out.limits.push_back({v.n, current.n, credit});
        }
    }
    return true;
}

// "/a/b/c" -> root, root/a, root/a/b, root/a/b/c, cut at the first one that is not a directory here: a container
// without its own cgroup namespace sees its group's host path in /proc/self/cgroup, but only its own subtree under
// the mount, so the deepest one that exists (at worst the mount itself) is the nearest group we can read.
bool group_dirs(const fs::path& root, const std::string& group, std::vector<fs::path>& dirs) {
    dirs.clear();
    std::vector<std::string> names;
    std::istringstream parts(group);
    for (std::string part; std::getline(parts, part, '/');) {
        if (part.empty()) continue;
        if (part == "." || part == "..") return false;   // never walk out of the mount, nor trust a path that tries
        names.push_back(part);
    }
    std::error_code ec;
    if (!fs::is_directory(root, ec)) return true;   // nothing mounted: no cgroup terms
    dirs.push_back(root);
    fs::path at = root;
    for (const std::string& part : names) {
        at /= part;
        if (!fs::is_directory(at, ec)) break;
        dirs.push_back(at);
    }
    return true;
}

bool scan_cgroups(const MemoryProbePaths& paths, CgroupScan& out) {
    std::optional<std::string> v2_group, v1_group;
    std::ifstream groups(paths.self_cgroup);
    if (!groups) {
        v2_group = "/";   // no /proc/self/cgroup: the mount's own top group is still readable
    } else {
        std::string line;
        while (std::getline(groups, line)) {
            const size_t c1 = line.find(':'), c2 = c1 == std::string::npos ? c1 : line.find(':', c1 + 1);
            if (c2 == std::string::npos) continue;
            const std::string controllers = line.substr(c1 + 1, c2 - c1 - 1), group = line.substr(c2 + 1);
            if (line.compare(0, c1, "0") == 0 && controllers.empty()) { v2_group = group; continue; }
            std::istringstream names(controllers);
            for (std::string name; std::getline(names, name, ',');)
                if (name == "memory") v1_group = group;
        }
    }
    std::vector<fs::path> dirs;
    if (v2_group) {
        if (!group_dirs(paths.cgroup_root, *v2_group, dirs)) return false;
        static const char* const names[] = {"memory.max", "memory.high"};
        if (!scan_levels(dirs, names, 2, "memory.current", false, out)) return false;
    }
    if (!out.any_files && v1_group) {   // cgroup v1: the memory controller's own tree
        if (!group_dirs(fs::path(paths.cgroup_root) / "memory", *v1_group, dirs)) return false;
        static const char* const names[] = {"memory.limit_in_bytes"};
        if (!scan_levels(dirs, names, 1, "memory.usage_in_bytes", true, out)) return false;
    }
    return true;
}

std::optional<HostMemoryReading> combine(std::optional<uint64_t> available, std::optional<uint64_t> total,
                                         const CgroupScan& cg, uint64_t explicit_limit) {
    if (!available) return {};
    HostMemoryReading r;
    r.mem_available = *available;
    r.available = *available;
    r.source = MemorySource::meminfo;
    r.limit = total.value_or(0);
    r.current = total && *total >= *available ? *total - *available : 0;
    for (const CgroupScan::Limit& l : cg.limits) {
        const uint64_t used = l.current - l.credit;   // credit <= current
        const uint64_t room = l.limit > used ? l.limit - used : 0;
        if (room < r.available) {
            r.available = room;
            r.source = MemorySource::cgroup;
            r.limit = l.limit;
            r.current = l.current;
            r.credit = l.credit;
        }
    }
    if (explicit_limit != 0) {
        uint64_t used = 0, credit = 0;
        if (cg.root_current) { used = *cg.root_current; credit = cg.root_credit; }
        else if (total && *total >= *available) used = *total - *available;
        else return {};
        const uint64_t net = used - credit;
        const uint64_t room = explicit_limit > net ? explicit_limit - net : 0;
        if (room <= r.available) {   // a tie goes to the operator's number: it is the one that names the real cap
            r.available = room;
            r.source = MemorySource::flag;
            r.limit = explicit_limit;
            r.current = used;
            r.credit = credit;
        }
    }
    return r;
}

} // namespace

const char* memory_source_name(MemorySource source) {
    switch (source) {
    case MemorySource::flag: return "flag";
    case MemorySource::cgroup: return "cgroup";
    default: return "meminfo";
    }
}

std::optional<uint64_t> conversation_mem_available(std::istream& meminfo) {
    return meminfo_kb_bytes(meminfo, "MemAvailable:");
}

void set_memory_limit_bytes(uint64_t bytes) { g_memory_limit_bytes.store(bytes, std::memory_order_relaxed); }
uint64_t memory_limit_bytes() { return g_memory_limit_bytes.load(std::memory_order_relaxed); }

std::optional<HostMemoryReading> sample_host_memory(const MemoryProbePaths& paths, uint64_t explicit_limit) {
    std::ifstream avail_file(paths.meminfo);
    if (!avail_file) return {};
    const std::optional<uint64_t> available = conversation_mem_available(avail_file);
    if (!available) return {};
    std::ifstream total_file(paths.meminfo);
    std::optional<uint64_t> total;
    if (total_file) total = meminfo_kb_bytes(total_file, "MemTotal:");
    CgroupScan cg;
    if (!scan_cgroups(paths, cg)) {
        // Without an operator cap the cgroup files are only an extra: the guard falls back to MemAvailable alone (what
        // 0.1.41 used), once-warned.  With a cap the usage cannot be told, so the sample stays unknown.
        if (explicit_limit != 0) return {};
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            std::fprintf(stderr, "[strata] memory guard: the cgroup memory files could not be read or parsed, using "
                                 "MemAvailable; set --memory-limit-mib to the container's limit\n");
        cg = CgroupScan{};
    }
    return combine(available, total, cg, explicit_limit);
}

std::optional<HostMemoryReading> sample_host_memory() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof status;
    // #1607: the commit limit (RAM + page file) refuses an allocation while physical RAM is still free
    if (!GlobalMemoryStatusEx(&status)) return {};
    return combine(std::min<uint64_t>(status.ullAvailPhys, status.ullAvailPageFile), status.ullTotalPhys, CgroupScan{}, memory_limit_bytes());
#elif defined(__linux__)
    return sample_host_memory(MemoryProbePaths{}, memory_limit_bytes());
#else
    return {};
#endif
}

std::optional<uint64_t> available_host_bytes() {
    const auto reading = sample_host_memory();
    if (!reading) return {};
    return reading->available;
}

// Parked-capsule/priority spill admission: the strict fail-closed cgroup view (fork feature,
// kept beside the upstream sampler: an unreadable discovered cgroup fails admission closed
// instead of falling back to MemAvailable).
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
