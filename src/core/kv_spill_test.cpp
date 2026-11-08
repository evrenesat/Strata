// Host-only tests of the parked-capsule disk fallback (kv_spill): the pinned directory, the anonymous file, the
// manifest arithmetic, the streaming writer and the bounded two-buffer reader.  No engine, no GPU, no service.
#include "strata/core/kv_spill.hpp"

#include "strata/core/conversation_file.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {

int checks = 0;
void check_detail(bool ok, const char* label, const std::string& detail) {
    if (ok) {
        ++checks;
        return;
    }
    std::fprintf(stderr, "FAIL: %s (%s)\n", label, detail.c_str());
    std::exit(1);
}
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        std::exit(1);
    }
}

// A temp directory the tests own: mode 0700 on a supported filesystem, removed at the end either way.
struct TestDir {
    fs::path path;
    explicit TestDir(const char* name) {
        path = fs::temp_directory_path() / (std::string(name) + "-" + std::to_string(::getpid()));
        std::error_code ec;
        fs::remove_all(path, ec);
        fs::create_directory(path, ec);
        ::chmod(path.c_str(), 0700);
    }
    ~TestDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string str() const { return path.string(); }
    size_t entries() const {
        size_t count = 0;
        std::error_code ec;
        for (auto it = fs::directory_iterator(path, ec); !ec && it != fs::directory_iterator();
             it.increment(ec))
            ++count;
        return count;
    }
};

size_t open_descriptors() {
    size_t count = 0;
    std::error_code ec;
    for (auto it = fs::directory_iterator("/proc/self/fd", ec); !ec && it != fs::directory_iterator();
         it.increment(ec))
        ++count;
    return count;
}

// Deterministic content: every byte is a function of its logical file offset, so a 100 MiB roundtrip needs no 100 MiB
// of expected data and a shifted or repeated block cannot pass by accident.
uint8_t pattern_byte(uint64_t at) { return static_cast<uint8_t>(((at * 131u) + 17u) >> 2); }

SpillManifest make_manifest(uint64_t request_id, const std::vector<std::array<uint64_t, 5>>& main_sizes,
                            const std::optional<std::array<uint64_t, 5>>& draft_sizes = std::nullopt) {
    SpillManifest manifest;
    manifest.request_id = request_id;
    for (size_t i = 0; i < main_sizes.size(); ++i) {
        SpillLayer layer;
        layer.geometry.layer = static_cast<int64_t>(i);
        layer.geometry.include_index = true;
        layer.geometry.cells = 131072;
        layer.geometry.heads = 2;
        layer.geometry.head_dim = 256;
        layer.sizes = main_sizes[i];
        manifest.main.push_back(layer);
    }
    if (draft_sizes) {
        SpillLayer draft;
        draft.geometry.layer = 48;
        draft.geometry.include_index = false;
        draft.sizes = *draft_sizes;
        manifest.draft = draft;
    }
    return manifest;
}

std::vector<uint64_t> ordered_parts(const SpillManifest& manifest) {
    std::vector<uint64_t> parts;
    for (const SpillLayer& layer : manifest.main)
        for (uint64_t size : layer.sizes) parts.push_back(size);
    if (manifest.draft)
        for (uint64_t size : manifest.draft->sizes) parts.push_back(size);
    return parts;
}

// Writes every declared part with a varying chunk size (crossing block and part boundaries), finishes the capsule and
// reads it back through the pipeline, checking every byte against the pattern.
uint64_t roundtrip(const TestDir& dir, SpillManifest manifest, uint64_t* verified_out = nullptr,
                   uint64_t before_counter = 0) {
    std::string error;
    check(spill_manifest_layout(manifest, error), "the capsule's parts lay out");
    SpillDirectory spot = SpillDirectory::open(dir.str(), error);
    check(spot.valid(), "the spill directory opens");
    SpillWriter writer = SpillWriter::create(spot, manifest, error);
    check(writer.valid(), "the capsule file opens");

    const std::vector<uint64_t> parts = ordered_parts(manifest);
    static const size_t chunks[] = {1, 7, 4093, 65536, 1u << 20, 333};
    size_t chunk_index = 0;
    std::vector<uint8_t> buffer(1u << 20);
    uint64_t at = 0;
    for (uint64_t size : parts) {
        uint64_t left = size;
        while (left != 0) {
            const size_t want = static_cast<size_t>(std::min<uint64_t>(left, chunks[chunk_index++ % 6]));
            for (size_t i = 0; i < want; ++i) buffer[i] = pattern_byte(at + i);
            check(writer.append(buffer.data(), want, error), "append within the part");
            at += want;
            left -= want;
        }
    }
    check(writer.part_remaining() == 0, "every part is full");
    check(writer.written() == at, "the writer accounts for every byte");
    check(writer.finish(error), "the capsule finishes");
    manifest = writer.manifest();   // the writer's copy carries the block digests

    SpillFile file = writer.release();
    check(file.valid(), "a finished capsule owns its descriptor");
    SpillReader reader = SpillReader::open(std::move(file), manifest, error);
    check(reader.valid(), "the capsule opens for reading");
    uint64_t verified = 0;
    const SpillSink sink = [&](const SpillBlockRef& block, std::string& why) {
        const SpillLayer& layer = block.draft ? *manifest.draft : manifest.main[block.layer];
        const uint64_t base = layer.extents[block.part].offset + block.offset;
        for (size_t i = 0; i < block.size; ++i) {
            if (block.data[i] != pattern_byte(base + i)) {
                why = "restored bytes do not match";
                return false;
            }
        }
        verified += block.size;
        return true;
    };
    const SpillReadResult result = reader.run(sink);
    check_detail(result.ok, "the capsule reads back", result.error);
    check(result.verified_bytes == at && verified == at, "every byte is verified and consumed");
    if (verified_out) *verified_out = verified;
    if (before_counter) {
        const uint64_t after = spill_write_bytes_total();
        check(after >= before_counter + at, "the cumulative write counter counts every byte");
    }
    return at;
}

}  // namespace

int main() {
    // ---------------------------------------------------------------- manifest arithmetic
    {
        std::string error;
        SpillManifest empty;
        check(!spill_manifest_layout(empty, error), "a capsule needs a main layer");

        SpillLayer negative;
        negative.geometry.heads = -1;
        negative.sizes = {10, 0, 0, 0, 0};
        SpillManifest bad;
        bad.main.push_back(negative);
        check(!spill_manifest_layout(bad, error), "negative geometry is refused");

        // The total must be refused by arithmetic, before any digest list is allocated.
        SpillManifest huge = make_manifest(1, {{1ull << 62, 1ull << 62, 0, 0, 0}});
        check(!spill_manifest_layout(huge, error), "an unrepresentable payload is refused");

        SpillManifest good = make_manifest(7, {{100, 0, 50, 0, 0}}, std::array<uint64_t, 5>{30, 10, 0, 0, 0});
        check(spill_manifest_layout(good, error), "a valid capsule lays out");
        check(good.payload_bytes == 190, "the payload is the sum of the parts");
        check(good.main[0].extents[0].blocks.size() == 1, "a partial part has one block digest");
        check(good.main[0].extents[1].blocks.empty(), "a zero-length part has no digests");
        check(good.main[0].extents[2].offset == 100, "extents follow the declared parts contiguously");
        check(good.draft->extents[0].offset == 150, "the draft entry follows the main layers");
        check(spill_manifest_check(good, error), "a laid-out manifest checks out");

        SpillManifest drifted = good;
        drifted.main[0].extents[2].offset += 1;
        check(!spill_manifest_check(drifted, error), "a non-contiguous manifest is refused");

        SpillManifest short_blocks = good;
        short_blocks.main[0].extents[0].blocks.pop_back();
        check(!spill_manifest_check(short_blocks, error), "a wrong block count is refused");

        SpillManifest wrong_total = good;
        wrong_total.payload_bytes += 1;
        check(!spill_manifest_check(wrong_total, error), "a payload that disagrees with its extents is refused");

        SpillManifest draft_index = make_manifest(1, {{10, 0, 0, 0, 0}}, std::array<uint64_t, 5>{10, 0, 0, 0, 0});
        draft_index.draft->geometry.include_index = true;
        check(!spill_manifest_layout(draft_index, error), "a draft entry may not claim the index row");

        SpillManifest no_main;
        SpillLayer draft_only;
        no_main.draft = draft_only;
        check(!spill_manifest_layout(no_main, error), "a draft entry alone is not a capsule");
    }

    // ---------------------------------------------------------------- the pinned directory
    {
        TestDir dir("strata-kvspill-dir");
        std::string error;
        SpillDirectory good = SpillDirectory::open(dir.str(), error);
        check(good.valid(), "an owned 0700 directory on ext4 opens");
        check(good.free_bytes() && *good.free_bytes() > 0, "free space is reportable");
        check(good.anonymous_supported(error), "the filesystem accepts anonymous capsule files");

        const SpillDirectory moved = std::move(good);
        check(moved.valid() && !good.valid(), "the pinned directory moves as one owner");

        SpillDirectory loose = SpillDirectory::open(dir.str(), error);
        check(loose.valid(), "reopening the directory works");
        ::chmod(dir.path.c_str(), 0755);
        check(!SpillDirectory::open(dir.str(), error).valid(), "a directory without mode 0700 is refused");
        ::chmod(dir.path.c_str(), 0700);

        const SpillFault failing = [](const char* step) {
            return std::strcmp(step, "dir_open") == 0 ? EACCES : 0;
        };
        check(!SpillDirectory::open(dir.str(), error, failing).valid(), "a failing directory open is reported");

        check(!SpillDirectory::open("relative/spill", error).valid(), "a relative path is refused");
        check(!SpillDirectory::open(dir.str() + "/../spill", error).valid(), "a '..' component is refused");
        check(!SpillDirectory::open(dir.str() + "/absent", error).valid(), "a missing directory is refused");

        const fs::path link = dir.path / "link";
        std::error_code ec;
        fs::create_directory_symlink(dir.path, link, ec);
        check(!SpillDirectory::open(link.string(), error).valid(), "a symbolic link is refused");

        const fs::path file = dir.path / "plain";
        { std::ofstream(file) << "x"; }
        check(!SpillDirectory::open(file.string(), error).valid(), "a regular file is refused");

        // An unsupported filesystem type is refused (tmpfs here; the check is the type, not the medium).
        const fs::path shm = "/dev/shm/strata-kvspill-unsupported";
        std::error_code shm_ec;
        fs::remove_all(shm, shm_ec);
        fs::create_directory(shm, shm_ec);
        if (!shm_ec) {
            ::chmod(shm.c_str(), 0700);
            const std::string why = error;
            check(!SpillDirectory::open(shm.string(), error).valid(), "an unsupported filesystem is refused");
            check(error.find("filesystem") != std::string::npos, "the refusal names the filesystem");
            (void)why;
            fs::remove_all(shm, shm_ec);
        }

        // Wrong ownership: another uid must not be accepted (this test needs root to change it).
        if (::geteuid() == 0) {
            const fs::path other = dir.path / "other";
            fs::create_directory(other, ec);
            ::chmod(other.c_str(), 0700);
            if (::chown(other.c_str(), 65534, 65534) == 0) {
                check(!SpillDirectory::open(other.string(), error).valid(), "a directory owned by another uid is refused");
            }
            fs::remove(other, ec);
        }
    }

    // ---------------------------------------------------------------- the anonymous file
    {
        TestDir dir("strata-kvspill-file");
        std::string error;
        SpillDirectory spot = SpillDirectory::open(dir.str(), error);
        check(spot.valid(), "the spill directory opens");
        SpillFile file = SpillFile::create(spot.fd(), error);
        check(file.valid(), "an anonymous capsule file is created");
        check(dir.entries() == 0, "the anonymous file has no name in the directory");
        check((::fcntl(file.fd(), F_GETFD) & FD_CLOEXEC) != 0, "the capsule descriptor is close-on-exec");

        // Exec: a same-numbered descriptor must not survive into a child program.
        {
            const pid_t pid = ::fork();
            check(pid >= 0, "fork works");
            if (pid == 0) {
                const std::string probe = "test -e /proc/self/fd/" + std::to_string(file.fd());
                ::execl("/bin/sh", "sh", "-c", probe.c_str(), static_cast<char*>(nullptr));
                ::_exit(127);
            }
            int status = 0;
            ::waitpid(pid, &status, 0);
            check(WIFEXITED(status) && WEXITSTATUS(status) == 1, "the capsule descriptor is not inherited by exec");
        }

        SpillFile second = std::move(file);
        check(second.valid() && !file.valid(), "the file moves as one owner");
        SpillFile third;
        third = std::move(second);
        check(third.valid() && !second.valid(), "move assignment transfers the descriptor");
        const int fd = third.fd();
        check(::fcntl(fd, F_GETFD) >= 0, "the transferred descriptor is open");
        third.reset();
        check(!third.valid() && ::fcntl(fd, F_GETFD) < 0, "reset closes the descriptor");

        const SpillFault failing = [](const char* step) {
            return std::strcmp(step, "open") == 0 ? EACCES : 0;
        };
        check(!SpillFile::create(spot.fd(), error, failing).valid(), "a failing open is reported");
    }

    // ---------------------------------------------------------------- roundtrips
    {
        TestDir dir("strata-kvspill-roundtrip");
        const uint64_t before = spill_write_bytes_total();
        // 0 bytes, 1 byte, one byte short of a block, exactly a block, one byte over, past the 64 MiB writeback window,
        // and a 100 MiB capsule.
        roundtrip(dir, make_manifest(1, {{0, 0, 0, 0, 0}}));
        roundtrip(dir, make_manifest(2, {{1, 0, 0, 0, 0}}));
        roundtrip(dir, make_manifest(3, {{kSpillBlockBytes - 1, 0, 0, 0, 0}}));
        roundtrip(dir, make_manifest(4, {{kSpillBlockBytes, 0, 0, 0, 0}}));
        roundtrip(dir, make_manifest(5, {{kSpillBlockBytes + 1, 0, 0, 0, 0}}));
        roundtrip(dir, make_manifest(6, {{kSpillBlockBytes * 4 + 1, 0, 0, 0, 0}}));
        roundtrip(dir, make_manifest(7, {{kSpillBlockBytes * 6 + 4 * 1024 * 1024, 0, 0, 0, 0}}));
        // Parts that are empty or partial, in the main entry and in the draft entry.
        roundtrip(dir, make_manifest(8, {{100, 0, 50, 0, 0}, {0, 7, 0, 0, 0}},
                                     std::array<uint64_t, 5>{30, 0, 0, 0, 0}));
        // The largest case is also the cumulative-counter check.
        roundtrip(dir, make_manifest(9, {{static_cast<uint64_t>(100) << 20, 0, 0, 0, 0}}), nullptr, before);
        check(dir.entries() == 0, "no capsule left a name behind");
    }

    // ---------------------------------------------------------------- writer refusals
    {
        TestDir dir("strata-kvspill-writer");
        std::string error;
        SpillDirectory spot = SpillDirectory::open(dir.str(), error);
        check(spot.valid(), "the spill directory opens");

        SpillManifest manifest = make_manifest(11, {{kSpillBlockBytes * 5, 0, 0, 0, 0}});
        check(spill_manifest_layout(manifest, error), "the capsule lays out");
        // One block: a capsule writes whole blocks, so a refusal is reached when a block completes.
        std::vector<uint8_t> buffer(static_cast<size_t>(kSpillBlockBytes), 0x5A);

        // ENOSPC and EDQUOT arrive as typed failures, not as silent truncation.
        for (const int code : {ENOSPC, EDQUOT, EIO}) {
            const SpillFault failing = [code](const char* step) {
                return std::strcmp(step, "write") == 0 ? code : 0;
            };
            SpillWriter writer = SpillWriter::create(spot, manifest, error, failing);
            check(writer.valid(), "the capsule file opens");
            check(!writer.append(buffer.data(), buffer.size(), error), "a failing write refuses the capture");
            check(!error.empty(), "the refusal carries a reason");
        }
        // Zero progress is a failure rather than an endless loop.
        {
            const SpillFault stuck = [](const char* step) {
                return std::strcmp(step, "write") == 0 ? -1 : 0;
            };
            SpillWriter writer = SpillWriter::create(spot, manifest, error, stuck);
            check(!writer.append(buffer.data(), buffer.size(), error), "a zero-byte write is a failure");
            check(error.find("no progress") != std::string::npos, "the zero-progress failure says so");
        }
        // EINTR is retried, not reported: the capture succeeds after one interruption.
        {
            std::atomic<int> interruptions{1};
            const SpillFault once = [&interruptions](const char* step) {
                if (std::strcmp(step, "write") != 0) return 0;
                return interruptions.exchange(0) > 0 ? EINTR : 0;
            };
            SpillWriter writer = SpillWriter::create(spot, manifest, error, once);
            check(writer.append(buffer.data(), buffer.size(), error), "an interrupted write is retried");
        }
        // The writeback window and the advice are checked, and they fail the capture when they fail.
        for (const char* step : {"writeback", "advice"}) {
            const std::string wanted(step);
            const SpillFault failing = [wanted](const char* name) {
                return wanted == name ? EIO : 0;
            };
            SpillWriter writer = SpillWriter::create(spot, manifest, error, failing);
            check(writer.valid(), "the capsule file opens");
            bool failed = false;
            uint64_t written = 0;
            // Cross the 64 MiB writeback window so the failing step is reached.
            while (written < kSpillWritebackWindow + kSpillBlockBytes) {
                if (!writer.append(buffer.data(), buffer.size(), error)) {
                    failed = true;
                    break;
                }
                written += buffer.size();
            }
            check(failed, "a failing writeback or advice refuses the capture");
        }
        // The capture path fills the writer's own staging buffer; a refused source leaves the capsule as it was.
        {
            SpillWriter writer = SpillWriter::create(spot, make_manifest(13, {{200, 0, 0, 0, 0}}), error);
            check(writer.valid(), "the source-test capsule opens");
            const SpillWriter::FillBlock refusing = [](uint8_t*, size_t, std::string& why) {
                why = "source failed";
                return false;
            };
            check(!writer.append_filled(64, refusing, error), "a refused source fails the append");
            check(error == "source failed", "the source's own reason is kept");
            check(writer.written() == 0, "a refused source writes nothing");
            const SpillWriter::FillBlock filling = [](uint8_t* dst, size_t n, std::string&) {
                std::memset(dst, 0x42, n);
                return true;
            };
            check(writer.append_filled(200, filling, error), "a filled block is accepted");
            check(writer.finish(error), "the filled capsule finishes");
        }
        // Declared sizes are enforced, and an unfinished capsule cannot be published.
        {
            SpillWriter writer = SpillWriter::create(spot, make_manifest(12, {{100, 0, 0, 0, 0}}), error);
            check(writer.valid(), "the small capsule opens");
            check(writer.part_remaining() == 100, "the part reports its remaining room");
            check(!writer.append(buffer.data(), 101, error), "an append past the part is refused");
            check(writer.append(buffer.data(), 60, error), "an append inside the part succeeds");
            check(!writer.finish(error), "an incomplete capsule does not finish");
            check(writer.append(buffer.data(), 40, error), "the rest of the part can still follow");
            check(writer.finish(error), "the complete capsule finishes");
            check(!writer.finish(error), "a capsule does not finish twice");
            check(!writer.append(buffer.data(), 1, error), "a finished capsule takes no more data");
        }
    }

    // ---------------------------------------------------------------- reader refusals and integrity
    {
        TestDir dir("strata-kvspill-reader");
        std::string error;
        SpillDirectory spot = SpillDirectory::open(dir.str(), error);
        check(spot.valid(), "the spill directory opens");

        const auto write_capsule = [&](SpillManifest& manifest, SpillWriter& out) {
            check(spill_manifest_layout(manifest, error), "the capsule lays out");
            out = SpillWriter::create(spot, manifest, error);
            check(out.valid(), "the capsule file opens");
            std::vector<uint8_t> buffer(1u << 20);
            for (uint64_t at = 0; at < manifest.payload_bytes; at += buffer.size()) {
                const size_t take = static_cast<size_t>(std::min<uint64_t>(buffer.size(), manifest.payload_bytes - at));
                for (size_t i = 0; i < take; ++i) buffer[i] = pattern_byte(at + i);
                check(out.append(buffer.data(), take, error), "the capsule fills");
            }
            check(out.finish(error), "the capsule finishes");
            manifest = out.manifest();   // the writer's copy carries the block digests
        };

        // A corrupted late block is refused, and exactly the earlier blocks reach the sink.
        {
            SpillManifest manifest = make_manifest(21, {{kSpillBlockBytes * 3, 0, 0, 0, 0}});
            SpillWriter writer;
            write_capsule(manifest, writer);
            SpillFile file = writer.release();
            const char corrupt = 0x7F;
            check(::pwrite(file.fd(), &corrupt, 1, static_cast<off_t>(kSpillBlockBytes + 5)) == 1,
                  "one byte of a later block is overwritten");
            SpillReader reader = SpillReader::open(std::move(file), manifest, error);
            check(reader.valid(), "the corrupted capsule opens");
            uint64_t sunk = 0;
            const SpillSink sink = [&](const SpillBlockRef& block, std::string&) {
                sunk += block.size;
                return true;
            };
            const SpillReadResult result = reader.run(sink);
            check(!result.ok, "a corrupted block fails the read");
            check(result.error.find("digest") != std::string::npos, "the failure is a digest mismatch");
            check(sunk == kSpillBlockBytes && result.verified_bytes == kSpillBlockBytes,
                  "only the verified block reached the sink");
        }
        // A pass that fails after a successful open still leaves the descriptor with the reader, so a caller can
        // recover the capsule instead of losing it on an admission or storage refusal.
        {
            SpillManifest manifest = make_manifest(29, {{kSpillBlockBytes * 2, 0, 0, 0, 0}});
            SpillWriter writer;
            write_capsule(manifest, writer);
            SpillFile file = writer.release();
            std::atomic<int> fails{0};
            const SpillFault failing = [&fails](const char* step) {
                if (std::strcmp(step, "read") != 0) return 0;
                return fails.fetch_add(1) == 0 ? EIO : 0;
            };
            SpillReader reader = SpillReader::open(std::move(file), manifest, error, failing);
            check(reader.valid(), "the capsule opens with a failing first read");
            const auto result = reader.run({});
            check(!result.ok && !result.admission, "a read failure is a payload failure, not admission");
            SpillFile back = reader.release_file();
            check(back.valid(), "the descriptor survives a failed pass");
        }
        // A corrupted first block reaches no sink at all: nothing was mutated.
        {
            SpillManifest manifest = make_manifest(22, {{kSpillBlockBytes * 2, 0, 0, 0, 0}});
            SpillWriter writer;
            write_capsule(manifest, writer);
            SpillFile file = writer.release();
            const char corrupt = 0x11;
            check(::pwrite(file.fd(), &corrupt, 1, 0) == 1, "the first block is overwritten");
            SpillReader reader = SpillReader::open(std::move(file), manifest, error);
            check(reader.valid(), "the corrupted capsule opens");
            uint64_t sunk = 0;
            const SpillSink sink = [&](const SpillBlockRef& block, std::string&) {
                sunk += block.size;
                return true;
            };
            const SpillReadResult result = reader.run(sink);
            check(!result.ok && sunk == 0 && result.verified_bytes == 0,
                  "a corrupt first block reaches no sink");
        }
        // A read failure is reported at its own position, after the blocks before it.
        {
            SpillManifest manifest = make_manifest(23, {{kSpillBlockBytes * 3, 0, 0, 0, 0}});
            SpillWriter writer;
            write_capsule(manifest, writer);
            SpillFile file = writer.release();
            std::atomic<int> calls{0};
            const SpillFault failing = [&calls](const char* step) {
                if (std::strcmp(step, "read") != 0) return 0;
                return calls.fetch_add(1) == 1 ? EIO : 0;   // fail the second block
            };
            SpillReader reader = SpillReader::open(std::move(file), manifest, error, failing);
            check(reader.valid(), "the capsule opens with an injected read fault");
            uint64_t sunk = 0;
            const SpillSink sink = [&](const SpillBlockRef& block, std::string&) {
                sunk += block.size;
                return true;
            };
            const SpillReadResult result = reader.run(sink);
            check(!result.ok, "the injected read failure fails the read");
            check(sunk == kSpillBlockBytes && result.verified_bytes == kSpillBlockBytes,
                  "the blocks before the failure are the ones consumed");
        }
        // A refused sink stops the walk with its own reason.
        {
            SpillManifest manifest = make_manifest(24, {{kSpillBlockBytes * 2, 0, 0, 0, 0}});
            SpillWriter writer;
            write_capsule(manifest, writer);
            SpillReader reader = SpillReader::open(writer.release(), manifest, error);
            check(reader.valid(), "the capsule opens");
            const SpillSink refusing = [](const SpillBlockRef&, std::string& why) {
                why = "sink refused";
                return false;
            };
            const SpillReadResult result = reader.run(refusing);
            check(!result.ok && result.error == "sink refused" && result.verified_bytes == 0,
                  "a sink refusal stops before any block is counted as verified");
        }
        // A refused open still owns the descriptor: the capsule survives for a retry or a cancellation.
        {
            SpillManifest manifest = make_manifest(28, {{kSpillBlockBytes, 0, 0, 0, 0}});
            SpillWriter writer;
            write_capsule(manifest, writer);
            SpillFile file = writer.release();
            check(::ftruncate(file.fd(), 0) == 0, "the capsule is emptied");
            SpillReader reader = SpillReader::open(std::move(file), manifest, error);
            check(!reader.valid(), "an emptied capsule does not open");
            SpillFile back = reader.release_file();
            check(back.valid(), "the refused open hands the descriptor back");
            const auto result = reader.run({});
            check(!result.ok && result.admission, "a reader that never opened refuses as an admission, not a loss");
        }
        // A truncated capsule is refused before the read starts.
        {
            SpillManifest manifest = make_manifest(25, {{kSpillBlockBytes * 2, 0, 0, 0, 0}});
            SpillWriter writer;
            write_capsule(manifest, writer);
            SpillFile file = writer.release();
            check(::ftruncate(file.fd(), static_cast<off_t>(kSpillBlockBytes)) == 0, "the capsule is truncated");
            check(!SpillReader::open(std::move(file), manifest, error).valid(), "a truncated capsule is refused");
        }
        // The pipeline overlaps: the second block is read while the first is being consumed, and no third block is
        // read while both buffers are in use.
        {
            SpillManifest manifest = make_manifest(26, {{kSpillBlockBytes * 3, 0, 0, 0, 0}});
            SpillWriter writer;
            write_capsule(manifest, writer);
            std::atomic<int> reads{0};
            const SpillFault counting = [&reads](const char* step) {
                if (std::strcmp(step, "read") == 0) reads.fetch_add(1);
                return 0;
            };
            SpillReader reader = SpillReader::open(writer.release(), manifest, error, counting);
            check(reader.valid(), "the capsule opens");
            const SpillSink sink = [&reads](const SpillBlockRef& block, std::string&) {
                if (block.offset == 0) {
                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
                    while (reads.load() < 2 && std::chrono::steady_clock::now() < deadline)
                        std::this_thread::yield();
                    check(reads.load() == 2, "the next block is read while the first is consumed");
                    std::this_thread::yield();
                    check(reads.load() == 2, "no third block is read while both buffers are in use");
                }
                return true;
            };
            const SpillReadResult result = reader.run(sink);
            check(result.ok, "the overlapped read completes");
            check(reads.load() == 3, "every block is read exactly once");
        }
    }

    // ---------------------------------------------------------------- a killed child leaves nothing behind
    {
        TestDir dir("strata-kvspill-kill");
        std::string error;
        SpillDirectory spot = SpillDirectory::open(dir.str(), error);
        check(spot.valid(), "the spill directory opens");
        const size_t descriptors_before = open_descriptors();
        const pid_t pid = ::fork();
        check(pid >= 0, "fork works");
        if (pid == 0) {
            // The child writes a capsule and exits without any cleanup: closing the process is the whole contract.
            SpillManifest manifest = make_manifest(31, {{kSpillBlockBytes * 2, 0, 0, 0, 0}});
            std::string why;
            if (!spill_manifest_layout(manifest, why)) ::_exit(1);
            SpillWriter writer = SpillWriter::create(spot, manifest, why);
            if (!writer.valid()) ::_exit(2);
            std::vector<uint8_t> buffer(1u << 20, 0x33);
            while (writer.part_remaining() != 0) {
                const size_t take = std::min<size_t>(writer.part_remaining(), buffer.size());
                if (!writer.append(buffer.data(), take, why)) ::_exit(3);
            }
            ::_exit(0);
        }
        int status = 0;
        ::waitpid(pid, &status, 0);
        check(WIFEXITED(status) && WEXITSTATUS(status) == 0, "the child wrote its capsule and exited");
        check(dir.entries() == 0, "a dead capsule owner leaves no name behind");
        check(open_descriptors() == descriptors_before, "the child's descriptor did not outlive it");
    }

    std::printf("kv_spill_test: %d checks passed\n", checks);
}
