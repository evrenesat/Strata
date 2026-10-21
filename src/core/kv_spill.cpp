#include "strata/core/kv_spill.hpp"

#include "strata/core/conversation_file.hpp"  // SessionHasher: the same block digest the file layer uses

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/vfs.h>
#include <unistd.h>
#endif

namespace strata::core {
namespace {

// The filesystem types this MVP can hold an anonymous capsule on.  The magic describes the type, not the physical
// medium: an operator still has to verify that the intended local SSD backs it.
constexpr long kFilesystemExt4 = 0xEF53;
constexpr long kFilesystemXfs = 0x58465342;
constexpr long kFilesystemBtrfs = 0x9123683E;

#if defined(__linux__)
constexpr int kOpenDirectory = O_RDONLY | O_DIRECTORY | O_CLOEXEC;
#endif

std::atomic<uint64_t> g_spill_write_bytes{0};

uint64_t page_bytes() {
#if defined(__linux__)
    const long page = ::sysconf(_SC_PAGESIZE);
    return page > 0 ? static_cast<uint64_t>(page) : 4096;
#else
    return 4096;
#endif
}

uint64_t round_down_page(uint64_t value) { return value - value % page_bytes(); }

uint64_t round_up_page(uint64_t value) {
    const uint64_t page = page_bytes();
    const uint64_t rem = value % page;
    if (rem == 0) return value;
    const uint64_t add = page - rem;
    const uint64_t max = std::numeric_limits<uint64_t>::max();
    return add > max - value ? max : value + add;
}

bool add_checked(uint64_t& total, uint64_t value) {
    const uint64_t max = std::numeric_limits<uint64_t>::max();
    if (value > max - total) return false;
    total += value;
    return true;
}

// The digest of one block: the request identity and the block's place in the capsule come first, then the bytes.  Two
// capsules never share digests, a moved block is detected, and a digest is not an authentication tag.  The file lives
// and dies with this process, so the byte order of the header is a local detail rather than a format.
uint64_t spill_block_digest(uint64_t request_id, bool draft, size_t layer, size_t part, uint64_t block_index,
                            uint64_t block_bytes, const uint8_t* data) {
    const std::array<uint64_t, 5> header{{
        draft ? 1u : 0u, static_cast<uint64_t>(layer), static_cast<uint64_t>(part), block_index, block_bytes}};
    SessionHasher hasher(request_id);
    hasher.update(header.data(), sizeof header);
    hasher.update(data, static_cast<size_t>(block_bytes));
    return hasher.digest();
}

// One block of the ordered walk: where it lives in the file and what it must hash to.
struct ReadBlock {
    uint64_t file_offset = 0;
    uint64_t size = 0;
    uint64_t block_index = 0;   // index of the block inside its part
    uint64_t digest = 0;
    size_t layer = 0;
    bool draft = false;
    size_t part = 0;
    uint64_t part_offset = 0;
};

std::vector<ReadBlock> spill_blocks(const SpillManifest& manifest) {
    std::vector<ReadBlock> blocks;
    const auto walk = [&](const SpillLayer& layer, size_t index, bool draft) {
        for (size_t part = 0; part < kSpillParts; ++part) {
            const SpillExtent& extent = layer.extents[part];
            uint64_t at = 0;
            for (size_t block = 0; at < extent.size; ++block) {
                const uint64_t size = std::min(kSpillBlockBytes, extent.size - at);
                ReadBlock entry;
                entry.file_offset = extent.offset + at;
                entry.size = size;
                entry.block_index = block;
                entry.digest = block < extent.blocks.size() ? extent.blocks[block] : 0;
                entry.layer = index;
                entry.draft = draft;
                entry.part = part;
                entry.part_offset = at;
                blocks.push_back(entry);
                at += size;
            }
        }
    };
    for (size_t i = 0; i < manifest.main.size(); ++i) walk(manifest.main[i], i, false);
    if (manifest.draft) walk(*manifest.draft, 0, true);
    return blocks;
}

const char* errno_name(int err) {
    switch (err) {
        case 0: return "none";
        case ENOENT: return "missing";
        case EACCES: return "permission";
        case EPERM: return "not-permitted";
        case EINVAL: return "invalid";
        case ELOOP: return "symlink";
        case ENOTDIR: return "not-a-directory";
        case ENOSPC: return "no-space";
#ifdef EDQUOT
        case EDQUOT: return "quota";
#endif
        case EFBIG: return "file-too-large";
        case ENOMEM: return "out-of-memory";
        case EIO: return "io";
        case EINTR: return "interrupted";
        case EOPNOTSUPP: return "unsupported";
        case EMFILE: return "too-many-files";
        case ENFILE: return "file-table-full";
        case EEXIST: return "exists";
        default: return "error";
    }
}

std::string describe(const char* step, int err) {
    std::string text(step);
    text += " failed: ";
    text += errno_name(err);
    return text;
}

// A transfer that never silently loses bytes: short writes and reads are continued, EINTR is retried, a zero-byte
// result is a failure rather than an infinite loop, and every written byte counts towards the process-lifetime total.
bool transfer_all(int fd, uint8_t* buffer, size_t bytes, uint64_t at, bool writing, std::string& error,
                  const SpillFault& fault) {
#if defined(__linux__)
    size_t done = 0;
    while (done < bytes) {
        if (fault) {
            const int injected = fault(writing ? "write" : "read");
            if (injected == EINTR) continue;
            if (injected == -1) {
                error = writing ? "spill write made no progress" : "spill read made no progress";
                return false;
            }
            if (injected > 0) {
                error = describe(writing ? "spill write" : "spill read", injected);
                return false;
            }
        }
        const ssize_t moved =
            writing ? ::pwrite(fd, buffer + done, bytes - done, static_cast<off_t>(at + done))
                    : ::pread(fd, buffer + done, bytes - done, static_cast<off_t>(at + done));
        if (moved < 0) {
            if (errno == EINTR) continue;
            error = describe(writing ? "spill write" : "spill read", errno);
            return false;
        }
        if (moved == 0) {
            error = writing ? "spill write made no progress" : "spill read reached the end early";
            return false;
        }
        done += static_cast<size_t>(moved);
        if (writing) g_spill_write_bytes.fetch_add(static_cast<uint64_t>(moved), std::memory_order_relaxed);
    }
    return true;
#else
    (void)fd;
    (void)buffer;
    (void)bytes;
    (void)at;
    (void)writing;
    (void)fault;
    error = "spill storage is only implemented on Linux";
    return false;
#endif
}

}  // namespace

const char* spill_error_name(int err) { return errno_name(err); }

uint64_t spill_write_bytes_total() { return g_spill_write_bytes.load(std::memory_order_relaxed); }

// ---------------------------------------------------------------- manifest arithmetic

bool spill_manifest_layout(SpillManifest& manifest, std::string& error) {
    if (manifest.main.empty()) {
        error = "a spill capsule needs at least one main K/V layer";
        return false;
    }
    // The total is proven representable before any digest list is allocated: an out-of-range manifest must be refused
    // by arithmetic, not by failing to allocate terabytes of digests.
    uint64_t total = 0;
    const auto account = [&](const SpillLayer& layer) {
        for (size_t part = 0; part < kSpillParts; ++part) {
            if (!add_checked(total, layer.sizes[part]) ||
                total > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
                error = "spill payload is too large for this platform";
                return false;
            }
        }
        return true;
    };
    for (const SpillLayer& layer : manifest.main)
        if (!account(layer)) return false;
    if (manifest.draft && !account(*manifest.draft)) return false;
    uint64_t offset = 0;
    const auto lay = [&](SpillLayer& layer) {
        for (size_t part = 0; part < kSpillParts; ++part) {
            const uint64_t size = layer.sizes[part];
            SpillExtent& extent = layer.extents[part];
            extent.offset = offset;
            extent.size = size;
            const uint64_t count = size / kSpillBlockBytes + (size % kSpillBlockBytes != 0 ? 1 : 0);
            if (count > static_cast<uint64_t>(extent.blocks.max_size())) {
                error = "spill part needs more block digests than can be stored";
                return false;
            }
            try {
                extent.blocks.assign(static_cast<size_t>(count), 0);
            } catch (const std::bad_alloc&) {
                error = "spill manifest digest list could not be allocated";
                return false;
            }
            if (!add_checked(offset, size)) {
                error = "spill payload size overflows";
                return false;
            }
            if (offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
                error = "spill payload is too large for this platform";
                return false;
            }
        }
        return true;
    };
    for (SpillLayer& layer : manifest.main) {
        if (layer.geometry.layer < 0 || layer.geometry.cells < 0 || layer.geometry.heads < 0 ||
            layer.geometry.head_dim < 0 || layer.geometry.page_size < 0 || layer.geometry.pooled_rows < 0 ||
            layer.geometry.idx_dim < 0) {
            error = "spill layer geometry is negative";
            return false;
        }
        if (!lay(layer)) return false;
    }
    if (manifest.draft) {
        SpillLayer& draft = *manifest.draft;
        if (draft.geometry.include_index) {
            error = "a draft spill entry must not claim the index row";
            return false;
        }
        if (!lay(draft)) return false;
    }
    manifest.payload_bytes = offset;
    return true;
}

bool spill_manifest_check(const SpillManifest& manifest, std::string& error) {
    if (manifest.main.empty()) {
        error = "manifest has no main K/V layer";
        return false;
    }
    if (manifest.draft && manifest.draft->geometry.include_index) {
        error = "draft manifest entry claims the index row";
        return false;
    }
    uint64_t expected = 0;
    const auto check_layer = [&](const SpillLayer& layer) {
        for (size_t part = 0; part < kSpillParts; ++part) {
            const SpillExtent& extent = layer.extents[part];
            if (extent.offset != expected) {
                error = "manifest extents are not contiguous from the start of the file";
                return false;
            }
            if (!add_checked(expected, extent.size)) {
                error = "manifest payload size overflows";
                return false;
            }
            const uint64_t count = extent.size / kSpillBlockBytes + (extent.size % kSpillBlockBytes != 0 ? 1 : 0);
            if (extent.blocks.size() != count) {
                error = "manifest block count does not match the extent size";
                return false;
            }
        }
        return true;
    };
    for (const SpillLayer& layer : manifest.main)
        if (!check_layer(layer)) return false;
    if (manifest.draft && !check_layer(*manifest.draft)) return false;
    if (expected != manifest.payload_bytes) {
        error = "manifest payload size does not match its extents";
        return false;
    }
    if (expected > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
        error = "manifest payload exceeds this platform's file offsets";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------- the anonymous file

SpillFile::~SpillFile() { reset(); }

SpillFile::SpillFile(SpillFile&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }

SpillFile& SpillFile::operator=(SpillFile&& other) noexcept {
    if (this != &other) {
        reset();
        fd_ = other.fd_;
        other.fd_ = -1;
    }
    return *this;
}

void SpillFile::reset() {
#if defined(__linux__)
    if (fd_ >= 0) ::close(fd_);
#endif
    fd_ = -1;
}

SpillFile SpillFile::create(int dir_fd, std::string& error, const SpillFault& fault) {
    SpillFile file;
#if defined(__linux__)
    if (dir_fd < 0) {
        error = "spill directory is not open";
        return file;
    }
    if (fault) {
        const int injected = fault("open");
        if (injected == -1) {
            error = "spill open reported no progress";
            return file;
        }
        if (injected > 0 && injected != EINTR) {
            error = describe("spill open", injected);
            return file;
        }
    }
    // O_EXCL makes the inode unlinkable for anyone else, including a later linkat() from this process.
    const int fd = ::openat(dir_fd, ".", O_TMPFILE | O_RDWR | O_CLOEXEC | O_EXCL, 0600);
    if (fd < 0) {
        error = describe("spill open", errno);
        return file;
    }
    file.fd_ = fd;
#else
    (void)dir_fd;
    (void)fault;
    error = "spill storage is only implemented on Linux";
#endif
    return file;
}

std::optional<uint64_t> SpillFile::size() const {
#if defined(__linux__)
    if (fd_ < 0) return {};
    struct stat st {};
    if (::fstat(fd_, &st) != 0) return {};
    if (st.st_size < 0) return {};
    return static_cast<uint64_t>(st.st_size);
#else
    return {};
#endif
}

// ---------------------------------------------------------------- the pinned directory

SpillDirectory::~SpillDirectory() { reset(); }

SpillDirectory::SpillDirectory(SpillDirectory&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }

SpillDirectory& SpillDirectory::operator=(SpillDirectory&& other) noexcept {
    if (this != &other) {
        reset();
        fd_ = other.fd_;
        other.fd_ = -1;
    }
    return *this;
}

void SpillDirectory::reset() {
#if defined(__linux__)
    if (fd_ >= 0) ::close(fd_);
#endif
    fd_ = -1;
}

SpillDirectory SpillDirectory::open(const std::string& path, std::string& error, const SpillFault& fault) {
    SpillDirectory dir;
#if defined(__linux__)
    if (path.empty() || path.front() != '/') {
        error = "spill directory must be an absolute path";
        return dir;
    }
    if (fault) {
        const int injected = fault("dir_open");
        if (injected != 0 && injected != EINTR) {
            error = describe("spill directory", injected == -1 ? EIO : injected);
            return dir;
        }
    }
    int current = ::open("/", kOpenDirectory);
    if (current < 0) {
        error = describe("spill directory root", errno);
        return dir;
    }
    // Every component is opened without following a link, so a replaced path cannot redirect this descriptor later.
    std::string rest = path;
    while (!rest.empty() && rest.front() == '/') rest.erase(rest.begin());
    while (!rest.empty()) {
        const size_t slash = rest.find('/');
        const std::string component = rest.substr(0, slash);
        rest = slash == std::string::npos ? std::string() : rest.substr(slash + 1);
        if (component.empty() || component == ".") continue;
        if (component == "..") {
            ::close(current);
            error = "spill directory must not contain '..'";
            return dir;
        }
        const int next = ::openat(current, component.c_str(), kOpenDirectory | O_NOFOLLOW);
        if (next < 0) {
            const int err = errno;
            ::close(current);
            error = err == ELOOP ? "spill directory component is a symbolic link"
                                 : describe("spill directory component", err);
            return dir;
        }
        ::close(current);
        current = next;
        if (!rest.empty()) continue;      // intermediate levels only have to be real directories
        struct stat st {};
        if (::fstat(current, &st) != 0) {
            const int err = errno;
            ::close(current);
            error = describe("spill directory", err);
            return dir;
        }
        if (!S_ISDIR(st.st_mode)) {
            ::close(current);
            error = "spill directory is not a directory";
            return dir;
        }
        if (st.st_uid != ::geteuid()) {
            ::close(current);
            error = "spill directory is not owned by this process";
            return dir;
        }
        if ((st.st_mode & 07777) != 0700) {
            ::close(current);
            error = "spill directory must have mode 0700";
            return dir;
        }
        struct statfs fs {};
        if (::fstatfs(current, &fs) != 0) {
            const int err = errno;
            ::close(current);
            error = describe("spill filesystem", err);
            return dir;
        }
        if (fs.f_type != kFilesystemExt4 && fs.f_type != kFilesystemXfs && fs.f_type != kFilesystemBtrfs) {
            ::close(current);
            error = "spill directory is not on a supported filesystem (ext4, XFS or Btrfs)";
            return dir;
        }
    }
    dir.fd_ = current;
#else
    (void)path;
    (void)fault;
    error = "spill storage is only implemented on Linux";
#endif
    return dir;
}

std::optional<uint64_t> SpillDirectory::free_bytes() const {
#if defined(__linux__)
    if (fd_ < 0) return {};
    struct statvfs fs {};
    if (::fstatvfs(fd_, &fs) != 0) return {};
    const uint64_t block = fs.f_frsize != 0 ? fs.f_frsize : fs.f_bsize;
    if (block != 0 && fs.f_bavail > std::numeric_limits<uint64_t>::max() / block) return {};
    return static_cast<uint64_t>(fs.f_bavail) * block;
#else
    return {};
#endif
}

bool SpillDirectory::anonymous_supported(std::string& error) const {
    if (fd_ < 0) {
        error = "spill directory is not open";
        return false;
    }
    SpillFile probe = SpillFile::create(fd_, error);
    return probe.valid();
}

// ---------------------------------------------------------------- writing

SpillWriter::~SpillWriter() = default;
SpillWriter::SpillWriter(SpillWriter&&) noexcept = default;
SpillWriter& SpillWriter::operator=(SpillWriter&&) noexcept = default;

SpillWriter SpillWriter::create(const SpillDirectory& dir, SpillManifest manifest, std::string& error,
                                const SpillFault& fault) {
    SpillWriter writer;
    if (!spill_manifest_layout(manifest, error)) return writer;
    SpillFile file = SpillFile::create(dir.fd(), error, fault);
    if (!file.valid()) return writer;
    writer.parts_.clear();
    const auto collect = [&writer](const SpillLayer& layer, size_t index, bool draft) {
        for (size_t part = 0; part < kSpillParts; ++part) {
            const SpillExtent& extent = layer.extents[part];
            if (extent.size == 0) continue;   // a zero-length part moves no bytes and has no digests
            SpillWriter::PartCursor cursor;
            cursor.offset = extent.offset;
            cursor.size = extent.size;
            cursor.layer = index;
            cursor.draft = draft;
            cursor.part = part;
            writer.parts_.push_back(cursor);
        }
    };
    for (size_t i = 0; i < manifest.main.size(); ++i) collect(manifest.main[i], i, false);
    if (manifest.draft) collect(*manifest.draft, 0, true);
    try {
        writer.block_.reserve(static_cast<size_t>(kSpillBlockBytes));
    } catch (const std::bad_alloc&) {
        error = "spill block buffer allocation failed";
        return SpillWriter{};
    }
    writer.file_ = std::move(file);
    writer.manifest_ = std::move(manifest);
    writer.fault_ = fault;
    return writer;
}

SpillFile SpillWriter::release() { return std::move(file_); }

void SpillWriter::reset() {
    file_.reset();
    manifest_ = SpillManifest{};
    parts_.clear();
    cursor_ = 0;
    block_.clear();
    written_ = 0;
    unsynced_ = 0;
    finished_ = false;
}

size_t SpillWriter::part_remaining() const {
    if (!file_.valid() || finished_ || cursor_ >= parts_.size()) return 0;
    const PartCursor& cursor = parts_[cursor_];
    const uint64_t inside = written_ - cursor.offset;
    if (inside >= cursor.size) return 0;
    return static_cast<size_t>(cursor.size - inside);
}

bool SpillWriter::write_back(std::string& error) {
    if (unsynced_ == 0) return true;
    if (!file_.valid()) {
        error = "spill file is not open";
        return false;
    }
#if defined(__linux__)
    const uint64_t begin = round_down_page(written_ - unsynced_);
    const uint64_t end = round_up_page(written_);
    const uint64_t length = end > begin ? end - begin : 0;
    if (length != 0) {
        if (fault_) {
            const int injected = fault_("writeback");
            if (injected != 0 && injected != EINTR) {
                error = describe("spill writeback", injected == -1 ? EIO : injected);
                return false;
            }
        }
        if (::sync_file_range(file_.fd(), static_cast<off_t>(begin), static_cast<off_t>(length),
                              SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE |
                                  SYNC_FILE_RANGE_WAIT_AFTER) != 0) {
            error = describe("spill writeback", errno);
            return false;
        }
        if (fault_) {
            const int injected = fault_("advice");
            if (injected != 0 && injected != EINTR) {
                error = describe("spill advice", injected == -1 ? EIO : injected);
                return false;
            }
        }
        // Checked by return value: posix_fadvise reports the error itself, not through errno.
        const int advised = ::posix_fadvise(file_.fd(), static_cast<off_t>(begin), static_cast<off_t>(length),
                                            POSIX_FADV_DONTNEED);
        if (advised != 0) {
            error = describe("spill advice", advised);
            return false;
        }
    }
    unsynced_ = 0;
    return true;
#else
    unsynced_ = 0;
    return true;
#endif
}

bool SpillWriter::flush_block(std::string& error) {
    const PartCursor cursor = parts_[cursor_];
    const uint64_t inside = written_ - cursor.offset;
    const size_t block_index = static_cast<size_t>(inside / kSpillBlockBytes);
    const uint64_t digest = spill_block_digest(manifest_.request_id, cursor.draft, cursor.layer, cursor.part,
                                               block_index, block_.size(), block_.data());
    if (!transfer_all(file_.fd(), block_.data(), block_.size(), written_, true, error, fault_)) return false;
    SpillExtent& extent = cursor.draft ? manifest_.draft->extents[cursor.part]
                                       : manifest_.main[cursor.layer].extents[cursor.part];
    if (block_index >= extent.blocks.size()) {
        error = "spill block index is outside the manifest";
        return false;
    }
    extent.blocks[block_index] = digest;
    written_ += block_.size();
    unsynced_ += block_.size();
    block_.clear();
    if (unsynced_ >= kSpillWritebackWindow && !write_back(error)) return false;
    if (written_ - cursor.offset == cursor.size) ++cursor_;   // this part is full: the next one follows
    return true;
}

bool SpillWriter::append(const void* data, size_t n, std::string& error) {
    const uint8_t* source = static_cast<const uint8_t*>(data);
    return append_filled(n, [&source](uint8_t* dst, size_t take, std::string&) {
        std::memcpy(dst, source, take);
        source += take;
        return true;
    }, error);
}

bool SpillWriter::append_filled(size_t n, const FillBlock& fill, std::string& error) {
    if (!file_.valid()) {
        error = "spill file is not open";
        return false;
    }
    if (finished_) {
        error = "spill capsule is already finished";
        return false;
    }
    if (n == 0) return true;
    if (n > part_remaining()) {
        error = "spill append would cross the current part's declared size";
        return false;
    }
    size_t left = n;
    while (left != 0) {
        const PartCursor& cursor = parts_[cursor_];
        const uint64_t inside = written_ - cursor.offset;
        const uint64_t part_left = cursor.size - inside;
        const uint64_t block_left = kSpillBlockBytes - block_.size();
        const size_t take = static_cast<size_t>(
            std::min<uint64_t>({static_cast<uint64_t>(left), part_left, block_left}));
        const size_t start = block_.size();
        block_.resize(start + take);
        if (!fill(block_.data() + start, take, error)) {
            block_.resize(start);   // a refused source leaves the capsule as it was
            if (error.empty()) error = "spill source refused the block";
            return false;
        }
        left -= take;
        const bool complete_block = block_.size() == kSpillBlockBytes;
        const bool complete_part = written_ + block_.size() - cursor.offset == cursor.size;
        if ((complete_block || complete_part) && !flush_block(error)) return false;
    }
    return true;
}

bool SpillWriter::finish(std::string& error) {
    if (!file_.valid()) {
        error = "spill file is not open";
        return false;
    }
    if (finished_) {
        error = "spill capsule is already finished";
        return false;
    }
    if (!block_.empty()) {
        error = "spill capsule has an incomplete block";
        return false;
    }
    if (written_ != manifest_.payload_bytes) {
        error = "spill capsule is incomplete";
        return false;
    }
    if (!write_back(error)) return false;   // the final window is flushed before the capsule is published
    const auto size = file_.size();
    if (!size) {
        error = "spill file size cannot be read";
        return false;
    }
    if (*size != manifest_.payload_bytes) {
        error = "spill file size does not match the manifest";
        return false;
    }
    finished_ = true;
    return true;
}

// ---------------------------------------------------------------- reading

namespace {

enum class SlotState { free, filling, ready, error };

struct ReadSlot {
    std::vector<uint8_t> data;
    SlotState state = SlotState::free;
    size_t filled = 0;
};

// What the reader thread and the owner share: one mutex, one signal per direction, and the error of the block the
// reader could not produce.
struct ReaderShared {
    std::mutex mutex;
    std::condition_variable owner_ready;    // the reader produced or refused the block the owner waits for
    std::condition_variable reader_ready;   // the owner released a buffer
    bool stop = false;
    bool done = false;
    std::string error;
};

}  // namespace

SpillReader::~SpillReader() = default;
SpillReader::SpillReader(SpillReader&&) noexcept = default;
SpillReader& SpillReader::operator=(SpillReader&&) noexcept = default;

SpillReader SpillReader::open(SpillFile file, const SpillManifest& manifest, std::string& error,
                              const SpillFault& fault) {
    SpillReader reader;
    if (!spill_manifest_check(manifest, error)) return reader;
    if (!file.valid()) {
        error = "spill file is not open";
        return reader;
    }
    const auto size = file.size();
    if (!size || *size != manifest.payload_bytes) {
        error = "spill file size does not match the manifest";
        return reader;
    }
#if defined(__linux__)
    // Sequential, one pass, never re-read: tell the kernel not to build a read-ahead window we will not use.
    (void)::posix_fadvise(file.fd(), 0, 0, POSIX_FADV_RANDOM);
#endif
    reader.file_ = std::move(file);
    reader.manifest_ = manifest;
    reader.fault_ = fault;
    return reader;
}

SpillReadResult SpillReader::run(const SpillSink& sink) {
    SpillReadResult result;
    if (!file_.valid()) {
        result.error = "spill file is not open";
        return result;
    }
    const std::vector<ReadBlock> blocks = spill_blocks(manifest_);
    if (blocks.empty()) {
        result.ok = true;
        return result;
    }
    // Two 16 MiB buffers and exactly one reader thread: while the owner hashes and consumes block N, the reader
    // fills block N+1.  Nothing is read ahead of that: two buffers is the whole pipeline.
    std::array<ReadSlot, 2> slots;
    try {
        for (ReadSlot& slot : slots) slot.data.resize(static_cast<size_t>(kSpillBlockBytes));
    } catch (const std::bad_alloc&) {
        result.error = "spill read buffers could not be allocated";
        return result;      // before any read and before any live mutation
    }
    ReaderShared shared;
    const int fd = file_.fd();
    std::thread reader_thread;
    try {
        reader_thread = std::thread([&shared, &slots, &blocks, fd, this]() {
            for (size_t i = 0; i < blocks.size(); ++i) {
                ReadSlot& slot = slots[i % 2];
                {
                    std::unique_lock<std::mutex> lock(shared.mutex);
                    shared.reader_ready.wait(lock, [&]() { return shared.stop || slot.state == SlotState::free; });
                    if (shared.stop) return;
                    slot.state = SlotState::filling;   // claimed under the lock: the owner cannot see it yet
                }
                std::string error;
                const bool read_ok = transfer_all(fd, slot.data.data(), static_cast<size_t>(blocks[i].size),
                                                  blocks[i].file_offset, false, error, fault_);
                {
                    std::lock_guard<std::mutex> lock(shared.mutex);
                    if (shared.stop) return;
                    if (read_ok) {
                        slot.filled = static_cast<size_t>(blocks[i].size);
                        slot.state = SlotState::ready;
                    } else {
                        slot.state = SlotState::error;
                        shared.error = error.empty() ? "spill read failed" : error;
                    }
                    shared.owner_ready.notify_one();
                }
            }
            std::lock_guard<std::mutex> lock(shared.mutex);
            shared.done = true;
            shared.owner_ready.notify_one();
        });
    } catch (const std::system_error&) {
        result.error = "spill reader thread could not be started";
        return result;      // nothing was read and no live state was touched
    }

    const auto stop_and_join = [&shared, &reader_thread]() {
        {
            std::lock_guard<std::mutex> lock(shared.mutex);
            shared.stop = true;
        }
        shared.owner_ready.notify_all();
        shared.reader_ready.notify_all();
        if (reader_thread.joinable()) reader_thread.join();   // joined before the file closes or buffers free
    };

    for (size_t i = 0; i < blocks.size(); ++i) {
        ReadSlot& slot = slots[i % 2];
        {
            std::unique_lock<std::mutex> lock(shared.mutex);
            shared.owner_ready.wait(lock, [&]() {
                return slot.state == SlotState::ready || slot.state == SlotState::error;
            });
        }
        if (slot.state == SlotState::error) {
            std::string why;
            {
                std::lock_guard<std::mutex> lock(shared.mutex);
                why = shared.error;
            }
            result.error = why.empty() ? "spill read failed" : why;
            stop_and_join();
            return result;
        }
        // The whole block is verified before any of it reaches the sink.
        const uint64_t digest = spill_block_digest(manifest_.request_id, blocks[i].draft, blocks[i].layer,
                                                   blocks[i].part, blocks[i].block_index, blocks[i].size,
                                                   slot.data.data());
        if (digest != blocks[i].digest) {
            result.error = "spill block digest mismatch";
            stop_and_join();
            return result;
        }
        if (sink) {
            SpillBlockRef ref;
            ref.layer = blocks[i].layer;
            ref.draft = blocks[i].draft;
            ref.part = blocks[i].part;
            ref.offset = blocks[i].part_offset;
            ref.data = slot.data.data();
            ref.size = static_cast<size_t>(blocks[i].size);
            std::string sink_error;
            if (!sink(ref, sink_error)) {
                result.error = sink_error.empty() ? "spill sink refused the block" : sink_error;
                stop_and_join();
                return result;
            }
        }
        result.verified_bytes += blocks[i].size;
        {
            std::lock_guard<std::mutex> lock(shared.mutex);
            slot.state = SlotState::free;
        }
        shared.reader_ready.notify_one();
    }
    stop_and_join();
    result.ok = true;
    return result;
}

}  // namespace strata::core
