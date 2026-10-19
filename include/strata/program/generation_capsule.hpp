#pragma once

#include "strata/core/conversation_memory.hpp"
#include "strata/core/mtp_request.hpp"
#include "strata/spec/suffix_drafter.hpp"
#include "strata/spec/draft_policy.hpp"
#include <optional>
#include <limits>
#include <string>
#include <chrono>

namespace strata::program {

enum class GenerationPhase { prefill, decode, parked, terminal };
struct GenerationProgress {
    int64_t total = 0, from = 0, reached = 0, next_checkpoint = 0;
    bool tail_saved = false;
    std::chrono::steady_clock::time_point began{};
};
struct GenerationAccounting {
    double wait=0,pool=0,host=0,plan=0,actq=0,jobs=0,run=0;
    int64_t misses=0,entries=0,hits=0,pcie=0,ram=0,files=0,lookups=0,offload=0;
    uint64_t file_bytes=0;
};
GenerationAccounting operator+(GenerationAccounting,const GenerationAccounting&);
GenerationAccounting operator-(GenerationAccounting,const GenerationAccounting&);

struct PrefillSegment {
    int64_t begin = 0, end = 0, offset = 0, chunk = 0;
    std::vector<int64_t> cuts;
    // The callback runs before Prefill publishes ple_prev; only a successful
    // complete run return can set returned. Never infer it from on_chunk.
    bool returned = false, prefetch_drained = false, loans_returned = false;
    std::array<int32_t,2> published_ple{};
    struct Loan { int64_t chunk = 0, first = 0; std::vector<std::pair<int32_t,int32_t>> slots; };
    std::vector<Loan> geometry; // recreate the exact loan/chunk layout, not a remaining-length heuristic
};

struct GenerationDriver {
    static constexpr uint32_t schema_version = 1;
    uint32_t version = schema_version;
    GenerationPhase phase = GenerationPhase::prefill;
    uint64_t request_id = 0, model_identity = 0, config_identity = 0;
    int64_t original_prompt_len = 0, max_context = 0, max_new = 0, produced = 0;
    int64_t position = 0, read_from = 0, reused = 0, reread_to = -1;
    int32_t next_token = 0;
    int window_size = 0, mtp_window_size = 0, history_size = 0;
    bool first_window = true, cancelled = false, cvec = true, checkpoints_enabled = true, uses_mtp = false;
    int64_t pin = 0, short_read = 0, tail_checkpoint_len = -1;
    uint64_t checkpoint_clock = 0;
    int32_t ple_token = -1;
    GenerationProgress progress;
    GenerationAccounting accounting_baseline, accounting_at_pause, foreground_excluded;
    std::chrono::steady_clock::time_point began{}, decode_began{}, parked_at{};
    double paused_prefill_ms=0, paused_decode_ms=0;
    double pcie_fraction = 0, spec_min_p = 0;
    kernels::SamplerParams sampling{}; // seed is resolved once, Philox counter is absolute
    std::string finish = "length";
    std::vector<int64_t> prompt;
    std::vector<int32_t> consumed, drafts, window, output, suffix_buffer, chain_buffer, chain_tail, history_stage;
    std::vector<float> draft_probabilities;
    // At the supported boundary every committed output has been emitted. Retain
    // the cursor explicitly so later partial emission cannot silently be lost.
    std::vector<int32_t> output_remainder;
    size_t output_cursor = 0;
    std::vector<core::ConversationImageKey> images;
    std::vector<float> image_rows;
    std::vector<int64_t> image_row_offsets; // -1: text row; offsets into owned image_rows
    std::vector<int32_t> mrope;
    bool mrope_identity = true;
    PrefillSegment prefill;
    std::optional<spec::SuffixDrafter> suffix;
    std::optional<spec::DraftPolicy> policy;
    int64_t rounds = 0, draft_offered = 0, draft_accepted = 0;
    int64_t suffix_windows = 0, suffix_drafts = 0, suffix_accepted = 0;
    int64_t chain_windows = 0, chain_drafts = 0, chain_accepted = 0;
    std::array<int64_t,8> top2_rejected{}, top2_hit{}, depth_tested{}, depth_accepted{};
    double verify_ms = 0, commit_ms = 0, draft_ms = 0;
    int64_t decode_windows = 0, decode_rows = 0;
    GenerationDriver() = default;
    GenerationDriver(const GenerationDriver&) = delete;
    GenerationDriver& operator=(const GenerationDriver&) = delete;
    GenerationDriver(GenerationDriver&&) noexcept = default;
    GenerationDriver& operator=(GenerationDriver&&) noexcept = default;
    size_t bytes() const;
    bool validate(std::string& error) const;
    // Invoke only after commit, whole-window emit, draft and adaptive work drain.
    bool next_window(int32_t token, int committed_rows, std::string& error);
    bool prefill_return(int64_t end, std::array<int32_t,2> published,
                        bool drained, bool loans_returned, std::string& error);
};

// While a request is active, lookup objects also belong to its typed driver.
// Reuse the ordinary suffix allocation and process-learned policy on the legacy
// path. A parked driver keeps its own policy branch while another request runs.
class GenerationLookupLease {
    GenerationDriver& driver_;
    spec::SuffixDrafter& suffix_;
    spec::DraftPolicy& policy_;
    int64_t& rounds_;
public:
    GenerationLookupLease(GenerationDriver&, spec::SuffixDrafter&, spec::DraftPolicy&,
                          int64_t& rounds, int min_match, size_t capacity);
    ~GenerationLookupLease();
    GenerationLookupLease(const GenerationLookupLease&) = delete;
};

struct GenerationCapsule {
    static constexpr uint32_t schema_version = 1;
    uint32_t version = schema_version;
    GenerationPhase saved_phase = GenerationPhase::terminal;
    GenerationDriver driver;
    core::SavedConversation main; // owns checkpoints by move; no optional LRU owner
    std::optional<core::MtpRequestState> mtp;
    size_t admitted_bytes = 0;
    bool occupied = false;
    GenerationCapsule() = default;
    GenerationCapsule(const GenerationCapsule&) = delete;
    GenerationCapsule& operator=(const GenerationCapsule&) = delete;
    GenerationCapsule(GenerationCapsule&& other) noexcept;
    GenerationCapsule& operator=(GenerationCapsule&& other) noexcept;
    size_t bytes() const;
};

inline constexpr uint64_t capsule_limit = 4608ull*1024*1024;
inline constexpr uint64_t capsule_free_floor = 2560ull*1024*1024;
bool generation_memory_admit(std::optional<uint64_t> available, uint64_t allocation,
                             uint64_t retained, std::string& error);
bool generation_memory_postcheck(std::optional<uint64_t> available, uint64_t retained,
                                 std::string& error);
// CPU ownership transaction after native readback: all checks precede moves.
bool generation_publish(GenerationCapsule&,GenerationDriver&,
    std::vector<core::ConversationCheckpoint>&,core::SavedConversation&&,
    std::optional<core::MtpRequestState>&&,size_t,std::optional<uint64_t>,std::string&);
void generation_take(GenerationCapsule&,GenerationDriver&,
                     std::vector<core::ConversationCheckpoint>&) noexcept;

} // namespace strata::program
