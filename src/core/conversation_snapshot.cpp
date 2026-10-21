#include "strata/core/conversation_snapshot.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "conversation_checked.hpp"
#include <cuda_runtime.h>

#include <array>
#include <cstring>
#include <limits>
#include <memory>

namespace strata::core {
namespace {

// The layout and target verdicts live in conversation_kv_layout.cpp so the RAM path, the disk capsule and the tests
// share one definition; these are the names the call sites in this file read.
using Layout = ConversationKvLayout;

std::array<void*, 5> pools(const QsaState& st, bool resident = false) {
    const bool host = st.kv_mode != 0 && !resident;
    if (st.kv_hybrid)
        return {host ? st.host.k_q : st.k_q, host ? st.host.v_q4 : st.v_q4, host ? st.host.k_scale : st.k_scale,
                nullptr, st.idx_pooled};
    if (st.kv_q4)
        return {host ? st.host.k_q4 : st.k_q4, host ? st.host.v_q4 : st.v_q4, nullptr, nullptr, st.idx_pooled};
    if (st.kv_int8)
        return {host ? st.host.k_q : st.k_q, host ? st.host.v_q : st.v_q,
                host ? st.host.k_scale : st.k_scale, host ? st.host.v_scale : st.v_scale, st.idx_pooled};
    return {host ? st.host.k_pool : st.k_pool, host ? st.host.v_pool : st.v_pool, nullptr, nullptr, st.idx_pooled};
}

bool transfer(void* dst, const void* src, size_t n, std::string& error) {
    if (!n) return true;
    if (!src || !dst) { error = "conversation snapshot: missing state buffer"; return false; }
    // Default handles both device allocations and device-mapped host pool aliases.
    const cudaError_t e = cudaMemcpy(dst, src, n, cudaMemcpyDefault);
    if (e == cudaSuccess) return true;
    error = std::string("conversation snapshot copy: ") + cudaGetErrorString(e);
    return false;
}
} // namespace

bool conversation_kv_capture_bytes(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                                   int64_t upto, bool index, size_t& bytes, std::string& error) {
    Layout l{};
    if (!conversation_kv_layout(st, g, upto, index, l, error) ||
        !conversation_kv_layout_targets(l, st, upto, error)) return false;
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<const ConversationBuffer*,5> buffers = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    bytes = 0;
    for (size_t i = 0; i < buffers.size(); ++i) {
        const size_t n = buffers[i]->allocation_peak(sizes[i]);
        if (n == SIZE_MAX || !conversation_detail::add(bytes, n)) {
            error = "conversation snapshot: segmented K/V allocation overflow";
            return false;
        }
    }
    return true;
}

size_t conversation_kv_bytes(const QsaState& st, const ModelGeometry& g, int64_t upto, bool index) {
    size_t bytes = 0;
    std::string error;
    return conversation_kv_capture_bytes({}, st, g, upto, index, bytes, error) ? bytes : 0;
}

bool conversation_kv_save(ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                          int64_t upto, bool index, std::string& error,
                          int64_t unchanged_tokens, size_t* reused_bytes) {
    Layout l{};
    if (!conversation_kv_layout(st, g, upto, index, l, error)) return false;
    if (!conversation_kv_layout_targets(l, st, upto, error)) return false;
    if (unchanged_tokens < 0 || unchanged_tokens > upto || unchanged_tokens > image.cells) {
        error = "conversation snapshot: invalid unchanged prefix";
        return false;
    }
    const int64_t whole_cells = (unchanged_tokens / l.page_size) * l.page_size;
    image.format = l.format;
    image.cells = l.cells;
    image.heads = g.n_head_kv;
    image.head_dim = g.head_dim;
    image.page_size = l.page_size;
    image.pooled_rows = l.pooled_rows;
    image.idx_dim = g.idx_key_dim;
    const auto src = pools(st);
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<ConversationBuffer*,5> dst = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    for (size_t i = 0; i < dst.size(); ++i) {
        // Recopy the partial page and the indexer's moving spare row. Completed
        // pages/rows strictly before the first rewritten token remain identical.
        const size_t keep = i == 4 ? (index ? size_t(unchanged_tokens / strata::kernels::qsa_real_shapes().idx_block) * g.idx_key_dim * 4 : 0)
                                  : l.cells ? (sizes[i] / size_t(l.cells)) * size_t(whole_cells) : 0;
        if (keep > dst[i]->size()) { error = "conversation snapshot: missing reusable prefix"; return false; }
        dst[i]->resize(sizes[i]);
        if (!dst[i]->visit(keep, sizes[i] - keep, [&](uint8_t* p, size_t n, size_t at) {
                return transfer(p, src[i] ? static_cast<const uint8_t*>(src[i]) + at : nullptr, n, error);
            })) return false;
        if (reused_bytes) *reused_bytes += keep;
    }
    return true;
}

bool conversation_kv_validate(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                              int64_t upto, bool index, std::string& error) {
    Layout l{};
    if (!conversation_kv_layout(st, g, upto, index, l, error)) return false;
    if (!conversation_kv_layout_targets(l, st, upto, error)) return false;
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<const ConversationBuffer*,5> src = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    if (image.format != l.format || image.cells != l.cells || image.heads != g.n_head_kv ||
        image.head_dim != g.head_dim || image.page_size != l.page_size || image.pooled_rows != l.pooled_rows ||
        image.idx_dim != g.idx_key_dim) {
        error = "conversation snapshot: incompatible K/V geometry";
        return false;
    }
    for (size_t i = 0; i < src.size(); ++i)
        if (src[i]->size() != sizes[i]) { error = "conversation snapshot: invalid K/V payload size"; return false; }
    const auto dst = pools(st);
    for (size_t i = 0; i < src.size(); ++i)
        if (sizes[i] && !dst[i]) { error = "conversation snapshot: missing target state buffer"; return false; }
    return true;
}

bool conversation_kv_restore(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                             int64_t upto, bool index, std::string& error) {
    if (!conversation_kv_validate(image, st, g, upto, index, error)) return false;
    const std::array<const ConversationBuffer*,5> src = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    const auto dst = pools(st);
    for (size_t i = 0; i < src.size(); ++i)
        if (!src[i]->visit(0, src[i]->size(), [&](const uint8_t* p, size_t n, size_t at) {
                return transfer(static_cast<uint8_t*>(dst[i]) + at, p, n, error);
            })) return false;
    return conversation_kv_finish_restore(st, g, upto, error);
}

bool conversation_kv_finish_restore(const QsaState& st, const ModelGeometry& g, int64_t upto, std::string& error) {
    // VRAM slots still contain the outgoing conversation. Resolve must refill
    // them from the restored authoritative pools before any attention reads.
    if (st.kv_mode == 1) strata::kernels::kv_stream_reset(st.map, nullptr);
    if (st.kv_mode == 2 && upto > 0) {
        auto shapes = strata::kernels::qsa_real_shapes();
        shapes.n_head_kv = g.n_head_kv; shapes.head_dim = g.head_dim;
        const int64_t end = (upto + shapes.page_size - 1) / shapes.page_size;
        strata::kernels::kv_ring_restore(qsa_attn_pools(st), st.host, qsa_kv_format(st),
                                        std::max<int64_t>(0, end - st.n_slots), end, st.n_slots, shapes, nullptr);
    }
    const auto status = cudaGetLastError();
    if (status == cudaSuccess) return true;
    error = std::string("conversation snapshot residency restore: ") + cudaGetErrorString(status);
    return false;
}

bool conversation_kv_visit(const ConversationKv& image, const KvBlockSink& sink, std::string& error) {
    if (!sink) { error = "conversation snapshot: no K/V sink"; return false; }
    const std::array<const ConversationBuffer*,5> buffers = {&image.k,&image.v,&image.k_scale,&image.v_scale,
                                                             &image.pooled};
    for (size_t part = 0; part < buffers.size(); ++part) {
        if (buffers[part]->empty()) continue;
        if (!buffers[part]->visit(0, buffers[part]->size(), [&](const uint8_t* p, size_t n, size_t at) {
                return sink(part, at, p, n, error);
            })) return false;
    }
    return true;
}

KvBlockSink conversation_kv_writer(const QsaState& st, const ConversationKvLayout& layout, std::string& error) {
    const auto sizes = layout.part_sizes();
    const auto destinations = pools(st);
    for (size_t part = 0; part < sizes.size(); ++part)
        if (sizes[part] && !destinations[part]) {
            error = "conversation snapshot: missing target state buffer";
            return {};
        }
    return [sizes, destinations](size_t part, uint64_t offset, const uint8_t* data, size_t n,
                                 std::string& why) -> bool {
        if (part >= sizes.size() || offset > sizes[part] || n > sizes[part] - offset) {
            why = "conversation snapshot: K/V block out of range";
            return false;
        }
        return transfer(static_cast<uint8_t*>(destinations[part]) + offset, data, n, why);
    };
}

ConversationKvRestoreResult conversation_kv_restore_source(const QsaState& st, const ModelGeometry& g, int64_t upto,
                                                           bool include_index, const KvDriveFn& drive) {
    ConversationKvRestoreResult result;
    Layout l{};
    if (!conversation_kv_layout(st, g, upto, include_index, l, result.error) ||
        !conversation_kv_layout_targets(l, st, upto, result.error)) return result;   // invalid: nothing written
    KvBlockSink writer = conversation_kv_writer(st, l, result.error);
    if (!writer) return result;
    KvTransfer transfer;
    transfer.drive = drive;
    transfer.write = writer;
    const KvTransferResult ran = kv_transfer_run(transfer);
    result.detail = ran.outcome;
    result.verified_bytes = ran.verified_bytes;
    result.error = ran.error;
    if (!ran.ok) {
        result.outcome = ran.outcome == KvTransferOutcome::fatal ? ConversationRestore::transfer_failed
                                                                 : ConversationRestore::invalid;
        return result;
    }
    if (!conversation_kv_finish_restore(st, g, upto, result.error)) {
        result.outcome = ConversationRestore::transfer_failed;   // the pools are written; residency did not follow
        return result;
    }
    result.outcome = ConversationRestore::restored;
    return result;
}

ConversationKvRestoreResult conversation_kv_restore_layers(const ModelGeometry& g,
                                                           const std::vector<ConversationKvTarget>& targets,
                                                           const KvLayerDriveFn& drive,
                                                           const std::function<bool(std::string&)>& before_draft) {
    ConversationKvRestoreResult result;
    if (targets.empty()) { result.error = "conversation snapshot: no K/V layers to restore"; return result; }
    if (!drive) { result.error = "conversation snapshot: no K/V source"; return result; }
    // Every layout validated and every destination prepared BEFORE the first write, across all layers: that is what
    // keeps the capsule's single mutation boundary meaningful.
    size_t main_layers = 0;
    for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i].draft && i + 1 != targets.size()) {
            result.error = "conversation snapshot: the draft entry must be the last target";
            return result;
        }
        if (!targets[i].draft) ++main_layers;
    }
    std::vector<Layout> layouts(targets.size());
    std::vector<KvBlockSink> writers(targets.size());
    for (size_t i = 0; i < targets.size(); ++i) {
        const auto& target = targets[i];
        if (!target.state) { result.error = "conversation snapshot: missing K/V target state"; return result; }
        if (!conversation_kv_layout(*target.state, g, target.upto, !target.draft, layouts[i], result.error) ||
            !conversation_kv_layout_targets(layouts[i], *target.state, target.upto, result.error)) return result;
        writers[i] = conversation_kv_writer(*target.state, layouts[i], result.error);
        if (!writers[i]) return result;
    }
    const bool has_draft = targets.back().draft;
    bool draft_hook_ran = false;
    const KvLayerSink sink = [&](size_t layer, bool draft, size_t part, uint64_t offset, const uint8_t* data,
                                 size_t n, std::string& why) -> bool {
        if (draft) {
            if (!has_draft) { why = "conversation snapshot: K/V block for a draft layer this session does not own"; return false; }
            // Every main block is in place now, so the main layers' recurrent state can be restored before the draft's
            // own K/V starts: one hook, exactly once, at that boundary.
            if (!draft_hook_ran) {
                draft_hook_ran = true;
                if (before_draft && !before_draft(why)) return false;
            }
            return writers.back()(part, offset, data, n, why);
        }
        if (layer >= main_layers) { why = "conversation snapshot: K/V block for a layer this session does not own"; return false; }
        return writers[layer](part, offset, data, n, why);
    };
    const KvTransferResult ran = kv_layer_transfer_run(drive, sink);
    result.detail = ran.outcome;
    result.verified_bytes = ran.verified_bytes;
    result.error = ran.error;
    if (!ran.ok) {
        result.outcome = ran.outcome == KvTransferOutcome::fatal ? ConversationRestore::transfer_failed
                                                                 : ConversationRestore::invalid;
        return result;
    }
    for (const auto& target : targets)
        if (!conversation_kv_finish_restore(*target.state, g, target.upto, result.error)) {
            result.outcome = ConversationRestore::transfer_failed;   // written, but residency did not follow
            return result;
        }
    result.outcome = ConversationRestore::restored;
    return result;
}

ConversationKvRestoreResult conversation_kv_compare_layers(const ModelGeometry& g,
                                                           const std::vector<ConversationKvTarget>& targets,
                                                           const KvLayerDriveFn& drive, uint64_t& fingerprint) {
    ConversationKvRestoreResult result;
    fingerprint = 1469598103934665603ull;
    if (targets.empty()) { result.error = "conversation snapshot: no K/V layers to compare"; return result; }
    if (!drive) { result.error = "conversation snapshot: no K/V source"; return result; }
    size_t main_layers = 0;
    std::vector<Layout> layouts(targets.size());
    for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i].draft && i + 1 != targets.size()) {
            result.error = "conversation snapshot: the draft entry must be the last target";
            return result;
        }
        if (!targets[i].draft) ++main_layers;
        if (!targets[i].state) { result.error = "conversation snapshot: missing K/V target state"; return result; }
        if (!conversation_kv_layout(*targets[i].state, g, targets[i].upto, !targets[i].draft, layouts[i],
                                    result.error)) return result;
    }
    const bool has_draft = targets.back().draft;
    uint64_t hash = fingerprint;
    std::array<uint8_t, 65536> scratch;
    // A compare-only walk: nothing is written, so this needs no mutation boundary - it either matches or it does not.
    const auto compare_block = [&](size_t index, size_t part, uint64_t offset, const uint8_t* data, size_t n,
                                   std::string& why) -> bool {
        const auto sizes = layouts[index].part_sizes();
        if (part >= sizes.size() || offset > sizes[part] || n > sizes[part] - offset) {
            why = "conversation snapshot: K/V block out of range";
            return false;
        }
        const auto pools = ::strata::core::pools(*targets[index].state);   // the live destination
        const auto* device = static_cast<const uint8_t*>(pools[part]) + offset;
        for (size_t at = 0; at < n;) {
            const size_t take = std::min(scratch.size(), n - at);
            if (!transfer(scratch.data(), device + at, take, why)) return false;
            if (std::memcmp(scratch.data(), data + at, take) != 0) {
                why = "conversation snapshot: restored K/V bytes differ";
                return false;
            }
            for (size_t i = 0; i < take; ++i) { hash ^= scratch[i]; hash *= 1099511628211ull; }
            at += take;
        }
        return true;
    };
    const KvLayerSink sink = [&](size_t layer, bool draft, size_t part, uint64_t offset, const uint8_t* data,
                                 size_t n, std::string& why) -> bool {
        if (draft) {
            if (!has_draft) { why = "conversation snapshot: K/V block for a draft layer this session does not own"; return false; }
            return compare_block(targets.size() - 1, part, offset, data, n, why);
        }
        if (layer >= main_layers) { why = "conversation snapshot: K/V block for a layer this session does not own"; return false; }
        return compare_block(layer, part, offset, data, n, why);
    };
    const KvTransferResult ran = kv_layer_transfer_run(drive, sink);
    result.detail = ran.outcome;
    result.verified_bytes = ran.verified_bytes;
    result.error = ran.error;
    fingerprint = hash;
    result.outcome = ran.ok ? ConversationRestore::restored
                            : (ran.outcome == KvTransferOutcome::fatal ? ConversationRestore::transfer_failed
                                                                      : ConversationRestore::invalid);
    return result;
}

bool conversation_kv_part_sizes(const QsaState& st, const ModelGeometry& g, int64_t upto, bool index,
                                std::array<uint64_t, 5>& sizes, std::string& error) {
    Layout l{};
    if (!conversation_kv_layout(st, g, upto, index, l, error)) return false;
    sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    return true;
}

bool conversation_kv_source(SessionKvSource& out, const QsaState& st, const ModelGeometry& g,
                            int64_t upto, bool index, std::string& error) {
    Layout l{};
    if (!conversation_kv_layout(st, g, upto, index, l, error) ||
        !conversation_kv_layout_targets(l, st, upto, error)) return false;
    SessionKvSource s;
    s.format = l.format; s.cells = l.cells; s.heads = g.n_head_kv; s.head_dim = g.head_dim;
    s.page_size = l.page_size; s.pooled_rows = l.pooled_rows; s.idx_dim = g.idx_key_dim;
    s.sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const auto src = pools(st);
    const auto sizes = s.sizes;
    auto why = std::make_shared<std::string>();
    s.read = [src, sizes, why](size_t part, size_t offset, void* dst, size_t n) {
        if (part >= 5 || offset > sizes[part] || n > sizes[part] - offset) { *why = "K/V read out of range"; return false; }
        return transfer(dst, src[part] ? static_cast<const uint8_t*>(src[part]) + offset : nullptr, n, *why);
    };
    s.error = [why] { return *why; };
    out = std::move(s);
    return true;
}

bool conversation_kv_verify(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                            int64_t upto, bool index, uint64_t& fingerprint, std::string& error,bool restored_residency) {
    if (!conversation_kv_validate(image, st, g, upto, index, error)) return false;
    std::array<uint8_t, 65536> buffer;
    uint64_t hash = 1469598103934665603ull;
    auto compare = [&](const void* device, const uint8_t* expected, size_t bytes, bool include_hash) {
        for (size_t offset = 0; offset < bytes;) {
            const size_t n = std::min(buffer.size(), bytes - offset);
            if (!transfer(buffer.data(), static_cast<const uint8_t*>(device) + offset, n, error)) return false;
            if (std::memcmp(buffer.data(), expected + offset, n) != 0) {
                error = "conversation snapshot: restored K/V bytes differ";
                return false;
            }
            if (include_hash) for (size_t i = 0; i < n; ++i) { hash ^= buffer[i]; hash *= 1099511628211ull; }
            offset += n;
        }
        return true;
    };
    const std::array<const ConversationBuffer*, 5> saved = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    const auto authoritative = pools(st);
    for (size_t i = 0; i < saved.size(); ++i)
        if (!saved[i]->visit(0, saved[i]->size(), [&](const uint8_t* p, size_t n, size_t at) {
                return compare(static_cast<const uint8_t*>(authoritative[i]) + at, p, n, true);
            })) return false;
    if (restored_residency && st.kv_mode == 2 && image.cells > 0) {
        const auto resident = pools(st, true);
        const int64_t end = image.cells / image.page_size;
        const int64_t begin = std::max<int64_t>(0, end - st.n_slots);
        for (size_t i = 0; i < 4; ++i) {
            if (saved[i]->empty()) continue;
            const size_t page_bytes = saved[i]->size() / size_t(end);
            for (int64_t page = begin; page < end; ++page)
                if (!saved[i]->visit(size_t(page) * page_bytes, page_bytes,
                        [&](const uint8_t* p, size_t n, size_t at) {
                            const size_t offset = size_t(page % st.n_slots) * page_bytes + at - size_t(page) * page_bytes;
                            return compare(static_cast<const uint8_t*>(resident[i]) + offset, p, n, false);
                        })) return false;
        }
    }
    fingerprint = hash;
    return true;
}

} // namespace strata::core
