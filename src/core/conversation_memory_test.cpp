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
    // Hybrid v1/v2 host: BOTH lines exist, the unified root does not control memory, and the real limit is the v1
    // one.  Reading the v2 path there finds no file, which must not read as "unbounded".
    {
        CgroupTree tree("strata-cgmem-hybrid");
        tree.write(tree.proc(), "0::/a\n7:memory:/a\n");
        tree.write(tree.root / "a" / "memory.max", "max\n");          // the unified tree's (irrelevant) view
        tree.write(tree.root / "a" / "memory.current", "0\n");
        tree.write(tree.root / "memory" / "a" / "memory.limit_in_bytes", "5000\n");
        tree.write(tree.root / "memory" / "a" / "memory.usage_in_bytes", "1200\n");
        const auto bound = tree.read();
        check(bound.present && bound.available == 3800, "a hybrid host uses its v1 memory limit, not the unified tree");
        check(conversation_memory_bound(1ULL << 40, bound) == 3800, "the v1 bound is what admission sees");
        // The same host, once the unified root really does control memory: the v2 view wins.
        tree.write(tree.root / "cgroup.controllers", "cpu io memory pids\n");
        tree.write(tree.root / "a" / "memory.max", "1000\n");
        tree.write(tree.root / "a" / "memory.current", "400\n");
        check(tree.read().available == 600, "a v2 root that controls memory is preferred");
        // ...but a root listing only other controllers is not a memory-capable v2 root.
        tree.write(tree.root / "cgroup.controllers", "cpu io pids\n");
        check(tree.read().available == 3800, "a unified root without memory in cgroup.controllers is not the bound");
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
    // Hierarchical v2: an "unlimited" leaf is still bounded by a finite ancestor limit.
    {
        CgroupTree tree("strata-cgmem-nest");
        tree.write(tree.proc(), "0::/user.slice/app.slice/strata.service\n");
        tree.write(tree.root / "user.slice/app.slice/strata.service/memory.max", "max\n");
        tree.write(tree.root / "user.slice/app.slice/strata.service/memory.current", "100\n");
        tree.write(tree.root / "user.slice/app.slice/memory.max", "max\n");
        tree.write(tree.root / "user.slice/app.slice/memory.current", "0\n");
        tree.write(tree.root / "user.slice/memory.max", "8000\n");
        tree.write(tree.root / "user.slice/memory.current", "1000\n");
        check(tree.read().available == 7000, "the finite ancestor binds an unlimited leaf");
        tree.write(tree.root / "user.slice/app.slice/memory.max", "3000\n");
        tree.write(tree.root / "user.slice/app.slice/memory.current", "100\n");
        check(tree.read().available == 2900, "the tightest level in the hierarchy wins");
        tree.write(tree.root / "user.slice/memory.current", "bogus\n");
        check(!conversation_memory_bound(600, tree.read()), "an unreadable ancestor fails closed");
    }
    // Hierarchical v1: the memory hierarchy under <root>/memory walks the same way.
    {
        CgroupTree tree("strata-cgmem-nest-v1");
        tree.write(tree.proc(), "9:memory:/job/leaf\n");
        tree.write(tree.root / "memory/job/leaf/memory.limit_in_bytes", "max\n");
        tree.write(tree.root / "memory/job/leaf/memory.usage_in_bytes", "0\n");
        tree.write(tree.root / "memory/job/memory.limit_in_bytes", "9000\n");
        tree.write(tree.root / "memory/job/memory.usage_in_bytes", "500\n");
        check(tree.read().available == 8500, "the v1 parent binds an unlimited leaf");
    }
    // ---- reclaimable-file credit: mapped inactive cache is not live pressure ----
    // The strict parser first: only clean, inactive cache is creditable, and only up to usage.
    {
        std::istringstream stat("anon 100\nfile 800\nactive_file 100\ninactive_file 700\n"
                                "file_dirty 10\nfile_writeback 5\n");
        check(conversation_cgroup_memory_credit(stat, 900, false) == 685,
              "credit is min(inactive,file) minus dirty/writeback, clamped by usage");
        std::istringstream clamp("inactive_file 5000\nfile 4000\nfile_dirty 0\nfile_writeback 0\n");
        check(conversation_cgroup_memory_credit(clamp, 4000, false) == 4000,
              "credit never exceeds usage");
        std::istringstream over("inactive_file 100\nfile 100\n"
                                "file_dirty 9223372036854775807\n"
                                "file_writeback 9223372036854775807\n");
        check(conversation_cgroup_memory_credit(over, 1000, false) == 0,
              "dirty/writeback addition saturates instead of overflowing");
        for (const char* bad : {
                 "inactive_file 100\nfile 200\nfile_dirty 10\n",                       // missing key
                 "inactive_file 100\nfile 200\nfile_dirty 10\nfile_writeback 5\n"
                 "file_dirty 6\n",                                                    // duplicate key
                 "inactive_file -1\nfile 200\nfile_dirty 10\nfile_writeback 5\n",      // signed
                 "inactive_file +1\nfile 200\nfile_dirty 10\nfile_writeback 5\n",      // signed
                 "inactive_file 100 extra\nfile 200\nfile_dirty 10\nfile_writeback 5\n",
                 "inactive_file 1x\nfile 200\nfile_dirty 10\nfile_writeback 5\n",
                 "inactive_file 100\nfile 200\nfile_dirty 10\nfile_writeback 5\n"
                 "file_writeback duplicate\n",
                 "inactive_file 100\nfile 200\nfile_dirty 18446744073709551616\n"
                 "file_writeback 5\n",                                                // out of range
             }) {
            std::istringstream input(bad);
            check(!conversation_cgroup_memory_credit(input, 1000, false),
                  "unusable stat telemetry grants no credit");
        }
        std::istringstream unrelated_only("anon 500\nslab 20\n");
        check(!conversation_cgroup_memory_credit(unrelated_only, 1000, false),
              "unrelated keys alone are not a credit");
        std::istringstream v1("total_inactive_file 400\ntotal_cache 600\n"
                              "total_dirty 0\ntotal_writeback 0\n");
        check(conversation_cgroup_memory_credit(v1, 1000, true) == 400,
              "v1 hierarchical totals produce credit");
    }
    // v2 level: the mapped inactive cache fills the raw limit, yet resident work still fits.
    {
        CgroupTree tree("strata-cgmem-credit-v2");
        tree.write(tree.proc(), "0::/job\n");
        tree.write(tree.root / "job" / "memory.max", "10000\n");
        tree.write(tree.root / "job" / "memory.current", "9000\n");
        tree.write(tree.root / "job" / "memory.stat",
                   "anon 1000\nfile 8000\nactive_file 1000\ninactive_file 7000\n"
                   "file_dirty 0\nfile_writeback 0\n");
        const auto bound = tree.read();
        check(bound.available == 8000, "inactive file cache is not live pressure");
        check(bound.usage == 9000 && bound.credit == 7000,
              "raw usage and applied credit stay reportable");
        check(conversation_memory_bound(1ULL << 40, bound) == 8000, "credit binds the budget");
        check(conversation_memory_admit(bound.available, 4000, 2560),
              "a modest resident allocation fits under reclaimable cache");
        // Anonymous use, not cache, is what must still fit.
        tree.write(tree.root / "job" / "memory.current", "9000\n");
        tree.write(tree.root / "job" / "memory.stat",
                   "anon 8500\nfile 500\nactive_file 0\ninactive_file 500\n"
                   "file_dirty 0\nfile_writeback 0\n");
        check(tree.read().available == 1500, "anonymous pressure is not credited");
        check(!conversation_memory_admit(tree.read().available, 3000, 0),
              "an anonymous allocation beyond the corrected allowance is refused");
        // Dirty and writeback bytes cannot be reclaimed, so they remove credit.
        tree.write(tree.root / "job" / "memory.stat",
                   "anon 1000\nfile 8000\nactive_file 1000\ninactive_file 7000\n"
                   "file_dirty 6000\nfile_writeback 500\n");
        check(tree.read().available == 1500, "dirty and writeback remove credit");
        // A stat file that is absent, empty or unusable keeps the old conservative bound.
        std::error_code ignored;
        fs::remove(tree.root / "job" / "memory.stat", ignored);
        check(tree.read().available == 1000, "absent stat telemetry grants no credit");
        tree.write(tree.root / "job" / "memory.stat", "");
        check(tree.read().available == 1000, "empty stat telemetry grants no credit");
        tree.write(tree.root / "job" / "memory.stat", "inactive_file 7000\nfile nope\n");
        check(tree.read().available == 1000, "malformed stat telemetry grants no credit");
    }
    // v1 level uses its hierarchical totals.
    {
        CgroupTree tree("strata-cgmem-credit-v1");
        tree.write(tree.proc(), "9:memory:/jobA\n");
        tree.write(tree.root / "memory" / "jobA" / "memory.limit_in_bytes", "10000\n");
        tree.write(tree.root / "memory" / "jobA" / "memory.usage_in_bytes", "9000\n");
        tree.write(tree.root / "memory" / "jobA" / "memory.stat",
                   "total_inactive_file 6000\ntotal_cache 8000\ntotal_dirty 100\n"
                   "total_writeback 0\n");
        check(tree.read().available == 6900, "v1 hierarchical totals produce credit");
    }
    // Hierarchy: credit never lifts an ancestor bound, and an unlimited leaf stays bounded.
    {
        CgroupTree tree("strata-cgmem-credit-nest");
        tree.write(tree.proc(), "0::/slice/leaf\n");
        tree.write(tree.root / "slice/leaf" / "memory.max", "max\n");
        tree.write(tree.root / "slice/leaf" / "memory.current", "0\n");
        tree.write(tree.root / "slice/leaf" / "memory.stat",
                   "inactive_file 50000\nfile 50000\nfile_dirty 0\nfile_writeback 0\n");
        tree.write(tree.root / "slice" / "memory.max", "8000\n");
        tree.write(tree.root / "slice" / "memory.current", "1000\n");
        tree.write(tree.root / "slice" / "memory.stat",
                   "inactive_file 200\nfile 300\nfile_dirty 0\nfile_writeback 0\n");
        const auto bound = tree.read();
        check(bound.available == 7200, "an unlimited leaf stays bounded by its ancestor");
        check(bound.credit == 200 && bound.usage == 1000,
              "the binding level reports its own usage and credit");
        tree.write(tree.root / "slice" / "memory.current", "1000\n");
        tree.write(tree.root / "slice" / "memory.stat", "inactive_file 0\nfile 0\n");
        check(tree.read().available == 7000, "an ancestor without credit keeps its bound");
    }
    // The real provider still returns bytes or unknown, whichever the host truth is.
    {
        const auto available = conversation_available_memory();
        check(!available || conversation_memory_admit(available, 0, 0),
              "provider returns bytes or unknown");
    }
    std::printf("conversation_memory_test: %d checks passed\n", checks);
}
