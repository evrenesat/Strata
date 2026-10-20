#include "strata/core/conversation_memory.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace strata::core;
namespace fs = std::filesystem;
namespace {

int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

// The injectable cgroup layout the tests write: a fake /proc/self/cgroup plus a fake
// /sys/fs/cgroup root, both plain files under one temporary directory.
struct CgroupTree {
    fs::path root;
    explicit CgroupTree(const char* name) : root(fs::temp_directory_path() /
                                                (std::string(name) + "-" + std::to_string(::getpid()))) {
        std::error_code ignored;
        fs::remove_all(root, ignored);
        fs::create_directories(root);
    }
    ~CgroupTree() { std::error_code ignored; fs::remove_all(root, ignored); }
    fs::path proc() const { return root / "self.cgroup"; }
    void write(const fs::path& path, const std::string& text) const {
        fs::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << text;
    }
    ConversationCgroupMemory read() const {
        return conversation_cgroup_memory(proc().string(), root.string());
    }
};

} // namespace

int main() {
    for (const char* text : {"", "MemFree: 100 kB\n", "MemAvailable: -1 kB\n",
                            "MemAvailable: +1 kB\n", "MemAvailable: 1 MB\n",
                            "MemAvailable: 1\n", "MemAvailable: 1x kB\n",
                            "MemAvailable: 18446744073709551615 kB\n",
                            "MemAvailable: 18446744073709551616 kB\n",
                            "MemAvailable: 1 kB trailing\n",
                            "MemAvailable: 1 kB\nMemAvailable: 2 kB\n"}) {
        std::istringstream input(text);
        check(!conversation_mem_available(input), "missing/malformed telemetry is unknown");
    }
    std::istringstream normal("MemTotal: 999999 kB\nMemAvailable:    12345 kB\nSwapFree: 777 kB\n");
    check(conversation_mem_available(normal) == 12345ULL * 1024, "only MemAvailable is counted");
    std::istringstream zero("MemAvailable: 0 kB");
    check(conversation_mem_available(zero) == 0, "zero available memory is known");
    std::istringstream broken("MemAvailable: 123 kB\n");
    broken.setstate(std::ios::badbit);
    check(!conversation_mem_available(broken), "I/O failure declines admission");
    check(!conversation_memory_admit({}, 0, 0), "unknown fails closed even with zero floor");
    check(conversation_memory_admit(100, 40, 60), "exact allocation plus floor fits");
    check(!conversation_memory_admit(99, 40, 60), "one byte below required memory rejected");
    check(!conversation_memory_admit(59, 0, 60), "floor subtraction cannot underflow");
    check(conversation_memory_admit(60, 0, 60), "post-capture floor check");
    check(!conversation_memory_admit(100, std::numeric_limits<uint64_t>::max(), 1), "allocation arithmetic cannot overflow");
    check(conversation_memory_admit(std::numeric_limits<uint64_t>::max(),
                                   std::numeric_limits<uint64_t>::max(), 0), "maximal exact bound");
    check(!conversation_memory_admit(std::numeric_limits<uint64_t>::max(),
                                    std::numeric_limits<uint64_t>::max(), 1), "maximal sum overflow rejected");
    // Test the real provider without assuming any particular amount of free RAM.
    const auto available = conversation_available_memory();
    check(!available || conversation_memory_admit(available, 0, 0), "provider returns bytes or unknown");
#if defined(_WIN32)
    // Unknown must fail closed in production, but must not let a broken Windows provider pass this test.
    // Compare with total physical memory, not a second available reading: other processes can allocate
    // between calls. No pressure allocation or fixed free-memory assumption is needed.
    MEMORYSTATUSEX physical{};
    physical.dwLength = sizeof physical;
    check(GlobalMemoryStatusEx(&physical) != 0, "Windows physical-memory API is available");
    check(available.has_value() && *available <= physical.ullTotalPhys,
          "Windows provider returns a known, physically bounded sample");
#endif
    std::printf("conversation_memory_test: %d checks passed\n", checks);

    // ---- cgroup memory admission (P2-d): the budget is min(host, cgroup allowance) ----
    // One cgroup memory file reading: "max"/sentinel are unbounded, malformed is unknown.
    {
        std::istringstream max_text("max\n");
        check(conversation_cgroup_memory_value(max_text) == std::numeric_limits<uint64_t>::max(),
              "cgroup 'max' reads as unbounded");
        std::istringstream sentinel("9223372036854771712\n");
        check(conversation_cgroup_memory_value(sentinel) == std::numeric_limits<uint64_t>::max(),
              "the v1 no-limit sentinel reads as unbounded");
        std::istringstream plain("1073741824\n");
        check(conversation_cgroup_memory_value(plain) == 1073741824ULL, "a byte limit parses");
        std::istringstream zero("0\n");
        check(conversation_cgroup_memory_value(zero) == 0, "a zero limit is known");
        for (const char* bad : {"", "max extra\n", "abc\n", "-3\n", "+3\n", "1x\n", "12 kb\n",
                                "18446744073709551616\n"}) {
            std::istringstream input(bad);
            check(!conversation_cgroup_memory_value(input), "malformed cgroup telemetry is unknown");
            (void)bad;
        }
    }
    // Host-only: no memory controller line (or no cgroup file at all) never changes the host view.
    {
        CgroupTree tree("strata-cgmem-host");
        tree.write(tree.proc(), "12:hugetlb:/\n11:blkio:/\n");
        const auto none = tree.read();
        check(!none.present, "no memory controller line: no cgroup bound applies");
        const auto host = std::optional<uint64_t>(12345 * 1024);
        check(conversation_memory_bound(host, none) == host, "host-only keeps the host view");
        const auto missing = conversation_cgroup_memory((tree.root / "absent").string(), tree.root.string());
        check(!missing.present, "a missing cgroup file is not a cgroup bound");
    }
    // v2: a tighter cgroup allowance is the admitted budget.
    {
        CgroupTree tree("strata-cgmem-v2");
        tree.write(tree.proc(), "0::/job\n");
        tree.write(tree.root / "job" / "memory.max", "1000\n");
        tree.write(tree.root / "job" / "memory.current", "400\n");
        const auto bound = tree.read();
        check(bound.present && bound.available == 600, "v2 allowance is max minus current");
        check(conversation_memory_bound(12345 * 1024, bound) == 600, "a tighter cgroup wins");
        tree.write(tree.root / "job" / "memory.max", "999999999999\n");
        check(conversation_memory_bound(600, tree.read()) == 600, "a looser cgroup never raises the host view");
        tree.write(tree.root / "job" / "memory.max", "max\n");
        const auto unbounded = tree.read();
        check(unbounded.present && conversation_memory_bound(600, unbounded) == 600,
              "cgroup 'max' means the host view alone applies");
        tree.write(tree.root / "job" / "memory.max", "1000\n");
        tree.write(tree.root / "job" / "memory.current", "1400\n");
        check(conversation_memory_bound(600, tree.read()) == 0, "an over-drawn cgroup admits nothing");
        // Fail closed: a discovered cgroup whose telemetry cannot be read or parsed.
        tree.write(tree.root / "job" / "memory.max", "soon\n");
        check(!conversation_memory_bound(600, tree.read()), "a malformed limit fails closed");
        tree.write(tree.root / "job" / "memory.max", "1000\n");
        std::error_code ignored;
        fs::remove(tree.root / "job" / "memory.current", ignored);
        check(!conversation_memory_bound(600, tree.read()), "an unreadable usage fails closed");
    }
    // v1 fallback: the memory controller hierarchy under <root>/memory.
    {
        CgroupTree tree("strata-cgmem-v1");
        tree.write(tree.proc(), "9:memory:/jobA\n");
        tree.write(tree.root / "memory" / "jobA" / "memory.limit_in_bytes", "5000\n");
        tree.write(tree.root / "memory" / "jobA" / "memory.usage_in_bytes", "1200\n");
        check(tree.read().available == 3800, "v1 limit minus usage is the allowance");
        tree.write(tree.root / "memory" / "jobA" / "memory.limit_in_bytes", "9223372036854771712\n");
        check(conversation_memory_bound(12345 * 1024, tree.read()) == 12345 * 1024,
              "the v1 no-limit sentinel leaves the host view alone");
    }
    // The real provider still returns bytes or unknown, whichever the host truth is.
    {
        const auto available = conversation_available_memory();
        check(!available || conversation_memory_admit(available, 0, 0),
              "provider returns bytes or unknown");
    }
    std::printf("conversation_memory_test: %d checks passed\n", checks);
}
