// What a conversation's K/V snapshot of `upto` tokens holds, and whether a given state can restore into it.  Pure
// arithmetic over the state's own flags and the model geometry: no device call, no state read, no allocation.  The
// snapshot, the disk capsule and their tests all use these verdicts, so the RAM and file paths cannot drift apart.
#pragma once

#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::core {

// One K/V layer's stored geometry and the byte count of each part, in file order: k, v, k_scale, v_scale, pooled.
// `format` includes the rotated INT8 variant (+16) and the hybrid marker (3); `cells` is padded up to whole pages
// and includes the moving spare row the indexer needs; `pooled_rows` is the pooled-key row count.
struct ConversationKvLayout {
    int format = 0;
    int64_t cells = 0, pooled_rows = 0, page_size = 0;
    size_t data = 0, scales = 0, pooled = 0, value_data = 0, value_scales = 0;

    std::array<uint64_t, 5> part_sizes() const { return {data, value_data, scales, value_scales, pooled}; }
};

// The layout a snapshot of `upto` tokens would have on this state.  Refused (with a reason) for a negative or
// over-range extent, an impossible hybrid-plus-ring combination, unusable geometry, or a byte count that would
// overflow.
bool conversation_kv_layout(const QsaState& st, const ModelGeometry& g, int64_t upto, bool include_index,
                            ConversationKvLayout& out, std::string& error);

// Whether this state can hold that layout and be written into: the page/slot/page-table identities for the mode, the
// authoritative host pool for a streamed mode, the streaming map, and the draft ring.  No device call, no write.
bool conversation_kv_layout_targets(const ConversationKvLayout& layout, const QsaState& st, int64_t upto,
                                    std::string& error);

}  // namespace strata::core
