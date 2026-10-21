// Host-only tests of the restore transaction's rules: ordering, the mutation boundary and the outcome classification.
#include "strata/core/kv_transfer.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
constexpr size_t kBlock = 4;   // small blocks keep the fake payload readable

struct Fake {
    std::vector<int> writes;    // (part, offset) of each attempted write, in order
    int verified = 0;
};

// Drives `blocks` blocks of part 0, each `kBlock` bytes, then reports `end`.
KvDriveFn push(int blocks, KvDrive end, std::vector<int>* pushed = nullptr) {
    return [blocks, end, pushed](const KvBlockSink& sink, std::string& error) {
        std::vector<uint8_t> data(kBlock, 0x5A);
        for (int i = 0; i < blocks; ++i) {
            if (!sink(0, uint64_t(i) * kBlock, data.data(), data.size(), error)) return KvDrive::ok;   // a refusal ends the pass
            if (pushed) pushed->push_back(i);
        }
        if (end != KvDrive::ok) error = end == KvDrive::lost ? "spill read failed" : "admission refused";
        return end;
    };
}

KvBlockSink writer(Fake& fake, int fail_at = -1, int verify_fail_as_write = 0) {
    return [&fake, fail_at, verify_fail_as_write](size_t part, uint64_t offset, const uint8_t*, size_t n,
                                                  std::string& error) {
        fake.writes.push_back(int(part * 1000 + offset));
        fake.verified += int(n);
        if (fail_at >= 0 && int(fake.writes.size()) - 1 == fail_at) {
            error = verify_fail_as_write ? "device copy failed" : "sink refused";
            return false;
        }
        return true;
    };
}
}  // namespace

int main() {
    // ---- the happy path ------------------------------------------------------------------
    {
        Fake fake;
        KvTransfer t;
        t.drive = push(3, KvDrive::ok);
        t.write = writer(fake);
        const auto r = kv_transfer_run(t);
        check(r.ok && r.mutated && r.verified_bytes == 3 * kBlock, "every verified block is written and counted");
        check(fake.writes.size() == 3, "the sink saw each block once, in order");
        check(fake.writes[0] == 0 && fake.writes[1] == int(kBlock) && fake.writes[2] == int(2 * kBlock),
              "blocks reach the destination in manifest order");
    }
    {
        Fake fake;
        KvTransfer t;
        t.drive = push(0, KvDrive::ok);          // an empty payload (all parts zero-length)
        t.write = writer(fake);
        const auto r = kv_transfer_run(t);
        check(r.ok && !r.mutated && r.verified_bytes == 0 && fake.writes.empty(),
              "an empty payload succeeds without touching the destination");
    }
    // ---- a source refusal before any write ----------------------------------------------
    {
        Fake fake;
        KvTransfer t;
        t.drive = push(0, KvDrive::refused);
        t.write = writer(fake);
        const auto r = kv_transfer_run(t);
        check(!r.ok && !r.mutated && r.outcome == KvTransferOutcome::refused && fake.writes.empty(),
              "a driver refusal before any write is refused, not lost");
    }
    {
        Fake fake;
        KvTransfer t;
        t.drive = push(0, KvDrive::lost);
        t.write = writer(fake);
        const auto r = kv_transfer_run(t);
        check(!r.ok && !r.mutated && r.outcome == KvTransferOutcome::lost,
              "payload loss before any write is lost, not refused");
    }
    // ---- verification before the destination --------------------------------------------
    {
        Fake fake;
        KvTransfer t;
        t.drive = push(2, KvDrive::ok);
        t.verify = [](size_t, uint64_t offset, const uint8_t*, size_t, std::string& why) {
            if (offset == 0) { why = "block digest mismatch"; return false; }
            return true;
        };
        t.write = writer(fake);
        const auto r = kv_transfer_run(t);
        check(!r.ok && r.outcome == KvTransferOutcome::lost && !r.mutated && fake.writes.empty(),
              "a first-block verification failure never reaches the destination");
        check(r.error == "block digest mismatch", "the verification reason is kept");
    }
    {
        Fake fake;
        KvTransfer t;
        t.drive = push(3, KvDrive::ok);
        t.verify = [](size_t, uint64_t offset, const uint8_t*, size_t, std::string& why) {
            if (offset == kBlock) { why = "block digest mismatch"; return false; }
            return true;
        };
        t.write = writer(fake);
        const auto r = kv_transfer_run(t);
        // The plan's rule: corruption in a later extent is discovered after live state changed, so it is fatal rather
        // than a lost capsule - the engine cannot unwind a half-restored layer.
        check(!r.ok && r.outcome == KvTransferOutcome::fatal && r.verified_bytes == kBlock,
              "a verification failure after an earlier live write is fatal, not a lost capsule");
        check(fake.writes.size() == 1, "the refused block never reached the destination");
    }
    // ---- the mutation boundary -----------------------------------------------------------
    {
        Fake fake;
        KvTransfer t;
        t.drive = push(3, KvDrive::ok);
        t.write = writer(fake, 0);               // the very first write fails partway
        const auto r = kv_transfer_run(t);
        check(!r.ok && r.outcome == KvTransferOutcome::fatal && r.mutated,
              "a first write that fails partway is already fatal");
        check(r.verified_bytes == 0, "a failed write is not counted as verified");
    }
    {
        Fake fake;
        KvTransfer t;
        t.drive = push(3, KvDrive::ok);
        t.write = writer(fake, 1);
        const auto r = kv_transfer_run(t);
        check(!r.ok && r.outcome == KvTransferOutcome::fatal && r.verified_bytes == kBlock,
              "a later write failure is fatal and keeps the earlier count");
    }
    {
        // A driver that reports a clean refusal *after* delivering blocks is still fatal: the destination changed.
        Fake fake;
        std::vector<int> pushed;
        KvTransfer t;
        t.drive = push(2, KvDrive::refused, &pushed);
        t.write = writer(fake);
        const auto r = kv_transfer_run(t);
        check(!r.ok && r.outcome == KvTransferOutcome::fatal && r.verified_bytes == 2 * kBlock,
              "a refusal after live writes has already lost the ability to unwind");
    }
    {
        // And so is one that reports payload loss late.
        Fake fake;
        KvTransfer t;
        t.drive = push(1, KvDrive::lost);
        t.write = writer(fake);
        const auto r = kv_transfer_run(t);
        check(!r.ok && r.outcome == KvTransferOutcome::fatal, "late payload loss is fatal, not a lost capsule");
    }
    // ---- degenerate transfers -------------------------------------------------------------
    {
        KvTransfer t;
        const auto r = kv_transfer_run(t);
        check(!r.ok && r.outcome == KvTransferOutcome::refused && !r.error.empty(),
              "a transfer without a source or destination is refused, never accepted");
    }
    std::printf("kv_transfer_test: %d checks passed\n", checks);
}
