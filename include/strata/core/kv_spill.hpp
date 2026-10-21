// Optional disk fallback for the one parked priority request: the main and MTP K/V of a suspended coding capsule,
// streamed into ONE anonymous temporary file and read back verified, while the driver, the recurrent state, the
// image data, the MTP working buffers and the checkpoints stay in RAM.
//
// This is a transport, not a format: the file holds raw part bytes in manifest order and nothing else.  It cannot
// outlive the process that owns it (an anonymous O_TMPFILE inode is reclaimed when the last descriptor closes), so
// there is no recovery story, no stale payload to sweep and no crash-consistency contract.  The per-block digests
// detect accidental corruption - an incomplete write, a torn page, a device error - and are NOT an authentication
// or tamper check, and closing the file is not secure erasure.
//
// Pure host code: no CUDA or HIP.  The engine supplies the bytes (from device memory or pinned host pools) and
// consumes them through a visitor; this layer never allocates a whole K/V copy and never touches live engine state.
//
// Everything here reports concrete I/O or integrity failures.  It does not decide whether the running engine is
// still safe: a failure before the first restored block can leave live state untouched, while the same failure after
// that block may be fatal.  That classification belongs to the caller that owns the live buffers.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

// Digest granularity, capture staging size and read-buffer size.  Every block is hashed whole, including the final
// partial block of a part; a zero-length part has no blocks and moves no bytes.
inline constexpr uint64_t kSpillBlockBytes = 16ull << 20;

// The K/V parts of one layer, in file order: k, v, k_scale, v_scale, pooled.
inline constexpr size_t kSpillParts = 5;

// Writeback window: after this much appended data the file is written back and dropped from the page cache, so the
// captured bytes do not accumulate as dirty cache behind the engine's own accounting.
inline constexpr uint64_t kSpillWritebackWindow = 64ull << 20;

// A test-only fault hook; never an HTTP parameter or a configuration value.  Return 0 to go on, an errno value to make
// that step fail, or -1 to make a transfer report zero progress (which the caller must treat as a failure).  Steps:
// "dir_open", "open", "write", "writeback", "advice", "read".
using SpillFault = std::function<int(const char* step)>;

const char* spill_error_name(int err);

// What identifies the bytes of one layer as the engine's K/V layout saw them.  The geometry record is stored once per
// layer entry and is compared on restore together with the part sizes.
struct SpillGeometry {
    int64_t layer = 0;              // QSA layer index (the draft layer's index for the MTP entry)
    int format = 0;                 // K/V pool format of the source
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    bool include_index = false;     // main layers carry the index row; the MTP draft extent does not
};

// One part of a layer: where it starts in the file, how long it is and the digest of each 16 MiB block.
struct SpillExtent {
    uint64_t offset = 0;
    uint64_t size = 0;
    std::vector<uint64_t> blocks;
};

struct SpillLayer {
    SpillGeometry geometry;
    std::array<uint64_t, kSpillParts> sizes{};
    std::array<SpillExtent, kSpillParts> extents{};
};

// The whole capsule's layout.  `main` holds one entry per owned QSA layer, then the optional draft entry.  Offsets
// are contiguous from 0 in manifest order, so a reader walks the file exactly once.
struct SpillManifest {
    uint64_t request_id = 0;        // the hash seed: one capsule's digests do not transfer to another's
    uint64_t payload_bytes = 0;     // logical bytes the file holds
    std::vector<SpillLayer> main;
    std::optional<SpillLayer> draft;
};

// Lays out `manifest.main` and `manifest.draft` contiguously in part order and sizes their block lists.  Every size
// is checked for negative, out-of-range and overflowing values; the caller must not have filled the extents yet.
// Returns false with a reason for an empty layer list, a size that cannot be represented or a total too large.
bool spill_manifest_layout(SpillManifest& manifest, std::string& error);

// Independent revalidation of a laid-out manifest: part counts, exact block counts, non-overlap, contiguity from 0
// and the final file size.  Run before any restore mutates live state; it trusts nothing but the numbers.
bool spill_manifest_check(const SpillManifest& manifest, std::string& error);

// Visit every block the writer stored, in manifest order, one whole block at a time.
struct SpillBlockRef {
    size_t layer = 0;               // index into `main` (or the draft entry when `draft` is true)
    bool draft = false;
    size_t part = 0;                // 0..4 = k, v, k_scale, v_scale, pooled
    uint64_t offset = 0;            // offset inside the part
    const uint8_t* data = nullptr;  // a complete block: `size` bytes, valid only during the call
    size_t size = 0;
};
// Returns false to stop the walk with `error`.
using SpillSink = std::function<bool(const SpillBlockRef& block, std::string& error)>;

// One anonymous capsule file, move-only, close-on-exec, the only owner of its inode.  There is no name: a moving
// owner therefore moves the file itself, and a reset or a process exit is what reclaims the blocks.
class SpillFile {
public:
    SpillFile() = default;
    SpillFile(SpillFile&& other) noexcept;
    SpillFile& operator=(SpillFile&& other) noexcept;
    SpillFile(const SpillFile&) = delete;
    SpillFile& operator=(const SpillFile&) = delete;
    ~SpillFile();

    // Opens the anonymous inode in `dir` (O_TMPFILE|O_RDWR|O_CLOEXEC, mode 0600, plus O_EXCL so nothing can ever link
    // it).  A filesystem, kernel or resource refusal leaves no descriptor and no name behind.
    static SpillFile create(int dir_fd, std::string& error, const SpillFault& fault = {});
    bool valid() const { return fd_ >= 0; }
    int fd() const { return fd_; }
    // The inode's current on-disk size, or nullopt when it cannot be read.
    std::optional<uint64_t> size() const;
    void reset();

private:
    int fd_ = -1;
};

// The pinned spill directory: an existing absolute directory this process owns, mode 0700, on ext4, XFS or Btrfs.
// Every component below the filesystem root is opened without following a link, so the descriptor cannot be
// redirected to another directory afterwards.  The filesystem check is about the type, not the physical medium: the
// operator still has to verify that the intended local SSD backs it.
class SpillDirectory {
public:
    SpillDirectory() = default;
    SpillDirectory(SpillDirectory&& other) noexcept;
    SpillDirectory& operator=(SpillDirectory&& other) noexcept;
    SpillDirectory(const SpillDirectory&) = delete;
    SpillDirectory& operator=(const SpillDirectory&) = delete;
    ~SpillDirectory();

    // Validates and pins `path`.  `error` says which check failed; nothing is created, changed or chmod-ed.
    static SpillDirectory open(const std::string& path, std::string& error, const SpillFault& fault = {});
    // The same for a filesystem-magic override: only tests use it (an unsupported type must fail in production).
    bool valid() const { return fd_ >= 0; }
    int fd() const { return fd_; }
    // Free bytes for unprivileged writers on this filesystem (f_bavail * f_frsize), or nullopt when unknown.
    std::optional<uint64_t> free_bytes() const;
    // True when this filesystem can hold an anonymous capsule file: probed by opening and closing one, so a
    // filesystem that refuses O_TMPFILE is refused at startup rather than at the first park.
    bool anonymous_supported(std::string& error) const;
    void reset();

private:
    int fd_ = -1;
};

// A capsule file being filled: sequential, one part after another, in manifest order.  Each successful write counts
// towards a process-lifetime byte total, including partial writes of an attempt that is later refused.
class SpillWriter {
public:
    SpillWriter() = default;
    SpillWriter(SpillWriter&&) noexcept;
    SpillWriter& operator=(SpillWriter&&) noexcept;
    SpillWriter(const SpillWriter&) = delete;
    SpillWriter& operator=(const SpillWriter&) = delete;
    ~SpillWriter();

    static SpillWriter create(const SpillDirectory& dir, SpillManifest manifest, std::string& error,
                              const SpillFault& fault = {});
    bool valid() const { return file_.valid(); }
    const SpillManifest& manifest() const { return manifest_; }
    SpillManifest& manifest() { return manifest_; }
    // Bytes the current part can still take; 0 once every part is full.
    size_t part_remaining() const;
    uint64_t written() const { return written_; }

    // Appends the next bytes of the current part.  `n` must not cross the part's declared size; blocks that complete
    // are hashed and, every `kSpillBlockBytes` block boundary, written back and dropped from the page cache.
    bool append(const void* data, size_t n, std::string& error);
    // The same, from a source that copies into the writer's own staging buffer: `fill(dst, n, why)` fills exactly `n`
    // bytes.  This is the capture path - one 16 MiB staging allocation for the whole capsule, no second copy.
    using FillBlock = std::function<bool(uint8_t* dst, size_t n, std::string& error)>;
    bool append_filled(size_t n, const FillBlock& fill, std::string& error);
    // Flushes the final partial block and the last window, verifies the inode's size against the manifest and freezes
    // it (block lists complete, payload bytes fixed).  Only a finished writer may be read.
    bool finish(std::string& error);
    // Detaches the capsule file from the writer (for the reader, or for a test that corrupts one block on purpose).
    SpillFile release();
    void reset();

private:
    // One part to fill, in file order.  Zero-length parts are not listed: they move no bytes and carry no digests.
    struct PartCursor {
        uint64_t offset = 0;
        uint64_t size = 0;
        size_t layer = 0;
        bool draft = false;
        size_t part = 0;
    };
    bool flush_block(std::string& error);
    bool write_back(std::string& error);

    SpillFile file_;
    SpillManifest manifest_;
    std::vector<PartCursor> parts_;
    size_t cursor_ = 0;             // the part being filled
    std::vector<uint8_t> block_;    // the block being assembled, written once complete
    uint64_t written_ = 0;          // logical bytes appended
    uint64_t unsynced_ = 0;         // bytes since the last writeback window
    bool finished_ = false;
    SpillFault fault_;
};

// The result of one verified read pass.  `verified_bytes` is the number of bytes hashed and handed to the sink before
// any failure: zero means no live state was touched yet, a non-zero value after an error means it was.
struct SpillReadResult {
    bool ok = false;
    uint64_t verified_bytes = 0;
    // True when the pass never started for an admission reason (buffers could not be allocated, the reader thread could
    // not start): the capsule is intact and may be retried.  False on any storage or integrity failure.
    bool admission = false;
    std::string error;
};

// Reads a finished capsule back: two 16 MiB buffers and exactly one reader thread, so block N+1 is being read while
// the owner verifies and consumes block N.  The owner hashes a complete block before handing any of it to the sink;
// a hash, size or I/O failure stops the walk and is reported in order.  The thread is joined before the file is
// closed or the buffers are freed on every path.
class SpillReader {
public:
    SpillReader() = default;
    SpillReader(SpillReader&&) noexcept;
    SpillReader& operator=(SpillReader&&) noexcept;
    SpillReader(const SpillReader&) = delete;
    SpillReader& operator=(const SpillReader&) = delete;
    ~SpillReader();

    // Validates the manifest's arithmetic, the inode's size and that this process can start the reader thread before
    // anything is read.  `sink` is called with every complete block; an empty sink verifies only.
    // Takes ownership of `file` immediately: a failed open still HOLDS it, so the caller can take it back with
    // release_file() and keep the capsule for a retry or a cancellation instead of losing its data.
    static SpillReader open(SpillFile file, const SpillManifest& manifest, std::string& error,
                            const SpillFault& fault = {});
    bool valid() const { return opened_; }        // true only when the capsule really opened
    SpillFile release_file() { return std::move(file_); }
    SpillReadResult run(const SpillSink& sink);

private:
    SpillFile file_;
    SpillManifest manifest_;
    SpillFault fault_;
    bool opened_ = false;
};

// The process-lifetime logical bytes every spill writer has written, including partial writes of refused attempts.
// Monotonic until the engine restarts; it is logical write traffic, not physical device writes.
uint64_t spill_write_bytes_total();

} // namespace strata::core
