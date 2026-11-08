#include "strata/core/conversation_kv_layout.hpp"

#include "conversation_checked.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/qsa.hpp"

#include <limits>

namespace strata::core {
namespace {

bool fail(std::string& error, const char* why) {
    error = why;
    return false;
}

bool valid_extent(const QsaState& st, int64_t upto, std::string& error) {
    const int64_t page = strata::kernels::qsa_real_shapes().page_size;
    if (upto < 0 || upto > st.max_cells || upto > std::numeric_limits<int64_t>::max() - (page - 1)) {
        error = "conversation snapshot: invalid K/V extent";
        return false;
    }
    return true;
}

}  // namespace

bool conversation_kv_layout(const QsaState& st, const ModelGeometry& g, int64_t upto, bool include_index,
                            ConversationKvLayout& out, std::string& error) {
    if (!valid_extent(st, upto, error)) return false;
    if (st.kv_hybrid && (st.kv_mode == 2 || st.kv_q4 || st.kv_int8)) {  // a streamed one reads its host copy
        return fail(error, "conversation snapshot: hybrid K8V4 cannot be a ring and needs distinct format flags");
    }
    const auto s = strata::kernels::qsa_real_shapes();
    const bool int8_keys = st.kv_int8 || st.kv_hybrid;
    if (g.n_head_kv <= 0 || g.head_dim <= 0 || g.head_dim > INT32_MAX || g.idx_key_dim <= 0 ||
        (st.kv_q4 && g.head_dim % 32) || (int8_keys && !st.kv_q4 && g.head_dim % 64)) {
        return fail(error, "conversation snapshot: invalid K/V geometry");
    }
    const int64_t cells = ((upto + s.page_size - 1) / s.page_size) * s.page_size;
    const size_t per = st.kv_q4 ? (size_t) strata::kernels::kv_q4_bytes_per_head((int) g.head_dim)
                                : (size_t) g.head_dim * (int8_keys ? 1 : 2);
    // Include the moving spare row, not only completed blocks. The checkpoint restore reconstructs that row when
    // rewinding to an earlier prefix.
    const int64_t pooled = include_index && upto > 0 ? upto / s.idx_block + 1 : 0;
    // Format 3 identifies snapshot K8V4 only; never pass it to the block movers.  Rotated INT8 K/V (#293,
    // STRATA_KV_ROT=1) is another format too (+16): its bytes mean nothing to a state that does not rotate, so a
    // snapshot or prompt checkpoint taken with the rotation never restores into one without it, nor the reverse.
    // (Q4_0 is always rotated: nothing to tell apart; without the rotation the format is the one it always was.)
    const int rotated = st.kv_rot && !st.kv_q4 && !st.kv_hybrid ? 16 : 0;
    ConversationKvLayout layout{};
    layout.format = (st.kv_hybrid ? 3 : qsa_kv_format(st)) + rotated;
    layout.cells = cells;
    layout.pooled_rows = pooled;
    layout.page_size = s.page_size;
    using conversation_detail::product;
    if (!product(layout.data, {(uint64_t) cells, (uint64_t) g.n_head_kv, per}) ||
        !product(layout.scales, {(uint64_t) cells, (uint64_t) g.n_head_kv,
                                 int8_keys && !st.kv_q4 ? (uint64_t) (g.head_dim / 64) * 2 : 0}) ||
        !product(layout.pooled, {(uint64_t) pooled, (uint64_t) g.idx_key_dim, sizeof(float)})) {
        return fail(error, "conversation snapshot: K/V byte count overflow");
    }
    layout.value_data = layout.data;
    layout.value_scales = layout.scales;
    if (st.kv_hybrid) {
        if (!product(layout.value_data, {(uint64_t) cells, (uint64_t) g.n_head_kv,
                                         (uint64_t) strata::kernels::kv_q4_bytes_per_head((int) g.head_dim)})) {
            return fail(error, "conversation snapshot: hybrid V byte count overflow");
        }
        layout.value_scales = 0;
    }
    out = layout;
    return true;
}

bool conversation_kv_layout_targets(const ConversationKvLayout& layout, const QsaState& st, int64_t upto,
                                    std::string& error) {
    if (upto < 0 || upto > st.max_cells || st.n_pages < 0 || layout.cells / layout.page_size > st.n_pages ||
        st.kv_mode < 0 || st.kv_mode > 2 || st.n_slots <= 0 ||
        (st.kv_mode == 0 && st.n_slots < st.n_pages) ||
        layout.pooled_rows > st.idx_pooled_rows || (st.kv_mode != 0 && !st.host.present())) {
        return fail(error, "conversation snapshot: invalid K/V extent or missing authoritative host pool");
    }
    if (st.kv_mode == 1) {
        const auto& m = st.map;
        if (m.n_blocks != st.n_pages || m.n_slots != st.n_slots || !m.page_table || !m.slot_block ||
            !m.slot_stamp || !m.slot_ref || !m.ctl || !m.miss_block || !m.miss_slot) {
            return fail(error, "conversation snapshot: invalid streaming map");
        }
    }
    if (st.kv_mode == 2 && (!st.page_table ||
        (st.kv_q4 ? !st.k_q4 || !st.v_q4 : st.kv_int8 ? !st.k_q || !st.v_q || !st.k_scale || !st.v_scale
                                                      : !st.k_pool || !st.v_pool))) {
        return fail(error, "conversation snapshot: invalid draft ring");
    }
    return true;
}

}  // namespace strata::core
