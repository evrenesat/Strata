#pragma once

// Owned, CPU-readable request state. Never serialize graph, stream or weight pointers.
#include "strata/core/conversation_cache.hpp"
#include "strata/kernels/sampler.hpp"
#include <array>
#include <string>

namespace strata::core {

struct MtpRequestState {
    static constexpr uint32_t schema_version = 1;
    uint32_t version = schema_version;
    // max_t, hc*n_embd, n_head, window, cells, cap, vocab, draft-vocab,
    // hnorm, q4, q4-head, coupled allocation, q allocation.
    std::array<int64_t, 13> geometry{};
    int64_t prompt_len = 0, kv_extent = 0;
    int max_drafts = 0;
    bool coupled = false;
    kernels::SamplerParams sampling{}, device_sampling{};
    ConversationKv kv; // includes speculative lookahead beyond main's committed extent
    std::vector<int32_t> host_tokens, host_steps, host_positions, host_row, host_out;
    std::vector<int32_t> device_tokens, device_steps, device_positions, device_row, device_out;
    std::vector<int32_t> history, ring, q, top2;
    std::vector<float> host_prob, device_prob, input_R, residual_R, window_R, debug_logits;
    double ms_draft = 0, ms_prefill = 0;
    int64_t rounds = 0;
    MtpRequestState() = default;
    MtpRequestState(const MtpRequestState&) = delete;
    MtpRequestState& operator=(const MtpRequestState&) = delete;
    MtpRequestState(MtpRequestState&&) noexcept = default;
    MtpRequestState& operator=(MtpRequestState&&) noexcept = default;
    size_t bytes() const {
        size_t n = sizeof(*this) + kv.bytes();
        for (const auto* v : {&host_tokens,&host_steps,&host_positions,&host_row,&host_out,
                             &device_tokens,&device_steps,&device_positions,&device_row,&device_out,
                             &history,&ring,&q,&top2}) n += v->capacity()*sizeof(int32_t);
        for (const auto* v : {&host_prob,&device_prob,&input_R,&residual_R,&window_R,&debug_logits})
            n += v->capacity()*sizeof(float);
        return n;
    }
    bool validate_layout(const std::array<int64_t,13>& expected, int64_t expected_prompt,
                         int64_t expected_extent, std::string& error) const;
};

} // namespace strata::core
