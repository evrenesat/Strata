// Host-only tests of the shared K/V layout verdicts: what a snapshot of N tokens holds, and whether a state can
// restore into it.  No device, no CUDA: the same functions the snapshot, the capsule and the disk path call.
#include "strata/core/conversation_kv_layout.hpp"

#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/qsa.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
const auto shapes = strata::kernels::qsa_real_shapes();
constexpr int64_t kHead = 128;

ModelGeometry geometry(int64_t kv_heads = 2) {
    ModelGeometry g{};
    g.n_head_kv = kv_heads;
    g.head_dim = kHead;
    g.idx_key_dim = 128;
    return g;
}

// A full-VRAM state (mode 0) sized for `cells`, which is the baseline every case starts from.
QsaState full_state(int64_t cells) {
    QsaState st{};
    st.max_cells = cells;
    st.n_pages = cells / shapes.page_size;
    st.n_slots = st.n_pages;
    st.kv_mode = 0;
    st.idx_pooled_rows = cells / shapes.idx_block + 2;
    return st;
}
}  // namespace

int main() {
    // ---- the layout: geometry, padding and part sizes ---------------------------------
    {
        std::string e;
        const auto g = geometry();
        auto st = full_state(4096);
        ConversationKvLayout l{};
        const int64_t upto = 1001;   // deliberately not a whole number of pages
        const int64_t padded = ((upto + shapes.page_size - 1) / shapes.page_size) * shapes.page_size;
        check(conversation_kv_layout(st, g, upto, true, l, e), "a plain fp16 layout is computed");
        check(l.cells == padded && l.cells > upto, "cells pad up to whole pages");
        check(l.page_size == shapes.page_size, "the layout records the page size");
        check(l.pooled_rows == upto / shapes.idx_block + 1, "the indexer's moving spare row is included");
        check(l.data == uint64_t(l.cells) * 2 * kHead * 2, "fp16 K holds two bytes per element");
        check(l.value_data == l.data && l.scales == 0, "plain fp16 has no scale part");
        const auto parts = l.part_sizes();
        check(parts[4] == uint64_t(l.pooled_rows) * g.idx_key_dim * 4, "pooled keys are fp32 rows");

        ConversationKvLayout no_index{};
        check(conversation_kv_layout(st, g, upto, false, no_index, e), "a layout without the index row is computed");
        check(no_index.pooled_rows == 0 && no_index.pooled == 0, "no index row means no pooled bytes");

        st.kv_int8 = true;
        ConversationKvLayout int8{};
        check(conversation_kv_layout(st, g, upto, true, int8, e), "an INT8 layout is computed");
        check(int8.data == uint64_t(int8.cells) * 2 * kHead && int8.scales > 0,
              "INT8 K is one byte per element plus scales");
        check(int8.format != l.format, "INT8 and fp16 are different formats");

        st = full_state(4096); st.kv_rot = true; st.kv_int8 = true;
        ConversationKvLayout rotated{};
        check(conversation_kv_layout(st, g, upto, true, rotated, e), "a rotated INT8 layout is computed");
        check(rotated.format == int8.format + 16, "rotation is its own format");
        st = full_state(4096); st.kv_rot = true;
        ConversationKvLayout rotated16{};
        check(conversation_kv_layout(st, g, upto, true, rotated16, e) && rotated16.format == l.format + 16,
              "any non-Q4, non-hybrid format records the rotation in its format");

        st = full_state(4096); st.kv_q4 = true;
        ConversationKvLayout q4{};
        check(conversation_kv_layout(st, g, upto, true, q4, e), "a Q4_0 layout is computed");
        check(q4.data == uint64_t(q4.cells) * 2 * strata::kernels::kv_q4_bytes_per_head(int(kHead)),
              "Q4_0 K uses the packed per-head size");
        check(q4.format != l.format, "Q4_0 is its own format");

        st = full_state(4096); st.kv_hybrid = true;
        ConversationKvLayout hybrid{};
        check(conversation_kv_layout(st, g, upto, true, hybrid, e), "a hybrid K8V4 layout is computed");
        check(hybrid.format == 3, "hybrid K8V4 is format 3");
        check(hybrid.value_data != hybrid.data && hybrid.value_scales == 0,
              "hybrid V is Q4_0 and carries no V scales");
    }
    // ---- the layout refusals ----------------------------------------------------------
    {
        std::string e;
        const auto g = geometry();
        ConversationKvLayout l{};
        auto st = full_state(4096);
        check(!conversation_kv_layout(st, g, -1, true, l, e), "a negative extent is refused");
        check(!conversation_kv_layout(st, g, 4097, true, l, e), "an extent past the arena is refused");
        auto hybrid_ring = full_state(4096);
        hybrid_ring.kv_hybrid = true; hybrid_ring.kv_mode = 2;
        check(!conversation_kv_layout(hybrid_ring, g, 100, true, l, e), "hybrid K8V4 cannot be a ring");
        auto hybrid_q4 = full_state(4096);
        hybrid_q4.kv_hybrid = true; hybrid_q4.kv_q4 = true;
        check(!conversation_kv_layout(hybrid_q4, g, 100, true, l, e), "hybrid K8V4 needs distinct format flags");
        auto bad_heads = geometry(0);
        check(!conversation_kv_layout(full_state(4096), bad_heads, 100, true, l, e), "no KV heads is refused");
        auto odd_dim = geometry(); odd_dim.head_dim = 96;
        auto int8_state = full_state(4096); int8_state.kv_int8 = true;
        check(!conversation_kv_layout(int8_state, odd_dim, 100, true, l, e),
              "INT8 needs a head dimension divisible by 64");
        auto q4_odd = full_state(4096); q4_odd.kv_q4 = true;
        auto not_32 = geometry(); not_32.head_dim = 100;
        check(!conversation_kv_layout(q4_odd, not_32, 100, true, l, e),
              "Q4_0 needs a head dimension divisible by 32");
        auto no_index_dim = geometry(); no_index_dim.idx_key_dim = 0;
        check(!conversation_kv_layout(full_state(4096), no_index_dim, 100, true, l, e),
              "an indexer without an index dimension is refused");
    }
    // ---- the target verdicts ----------------------------------------------------------
    {
        std::string e;
        const auto g = geometry();
        auto st = full_state(4096);
        ConversationKvLayout l{};
        const int64_t upto = 1000;
        check(conversation_kv_layout(st, g, upto, true, l, e), "the baseline layout is computed");
        check(conversation_kv_layout_targets(l, st, upto, e), "a full-VRAM state can hold it");

        auto short_pages = full_state(4096);
        short_pages.n_pages = 1;                       // fewer pages than the layout needs
        short_pages.n_slots = 1;
        check(!conversation_kv_layout_targets(l, short_pages, upto, e),
              "a state with too few pages cannot hold the layout");

        auto unmatched = full_state(4096);
        unmatched.n_slots = 0;
        check(!conversation_kv_layout_targets(l, unmatched, upto, e), "a state with no slots is refused");

        auto over_extent = full_state(4096);
        check(!conversation_kv_layout_targets(l, over_extent, 8192, e), "an extent past the state is refused");

        auto small_index = full_state(4096);
        small_index.idx_pooled_rows = 1;
        check(!conversation_kv_layout_targets(l, small_index, upto, e), "too few pooled rows is refused");

        auto streamed = full_state(4096);
        streamed.kv_mode = 1;
        streamed.n_slots = 2;
        check(!conversation_kv_layout_targets(l, streamed, upto, e),
              "a streamed state without its authoritative host pool is refused");
        streamed.host.k_pool = reinterpret_cast<uint16_t*>(0x1000);
        check(!conversation_kv_layout_targets(l, streamed, upto, e),
              "a streamed state without its map is refused");
        streamed.map.n_blocks = streamed.n_pages;
        streamed.map.n_slots = streamed.n_slots;
        streamed.map.page_table = reinterpret_cast<int32_t*>(0x2000);
        streamed.map.slot_block = reinterpret_cast<int32_t*>(0x2000);
        streamed.map.slot_stamp = reinterpret_cast<int32_t*>(0x2000);
        streamed.map.slot_ref = reinterpret_cast<int32_t*>(0x2000);
        streamed.map.ctl = reinterpret_cast<int32_t*>(0x2000);
        streamed.map.miss_block = reinterpret_cast<int32_t*>(0x2000);
        streamed.map.miss_slot = reinterpret_cast<int32_t*>(0x2000);
        check(conversation_kv_layout_targets(l, streamed, upto, e), "a streamed state with host pools and a map fits");
        streamed.map.n_slots = 99;
        check(!conversation_kv_layout_targets(l, streamed, upto, e), "a map of another geometry is refused");

        auto ring = full_state(4096);
        ring.kv_mode = 2;
        ring.n_slots = 2;
        ring.host.k_pool = reinterpret_cast<uint16_t*>(0x1000);
        check(!conversation_kv_layout_targets(l, ring, upto, e), "a ring without its page table or pools is refused");
        ring.page_table = reinterpret_cast<int32_t*>(0x3000);
        ring.k_pool = reinterpret_cast<uint16_t*>(0x3000);
        ring.v_pool = reinterpret_cast<uint16_t*>(0x3000);
        check(conversation_kv_layout_targets(l, ring, upto, e), "a ring with its table and pools fits");
        ring.kv_int8 = true;   // INT8 ring needs all four pointers
        check(!conversation_kv_layout_targets(l, ring, upto, e), "an incomplete INT8 ring is refused");
        ring.k_q = reinterpret_cast<int8_t*>(0x3000);
        ring.v_q = reinterpret_cast<int8_t*>(0x3000);
        ring.k_scale = reinterpret_cast<uint16_t*>(0x3000);
        ring.v_scale = reinterpret_cast<uint16_t*>(0x3000);
        check(conversation_kv_layout_targets(l, ring, upto, e), "a complete INT8 ring fits");
    }
    std::printf("conversation_kv_layout_test: %d checks passed\n", checks);
}
