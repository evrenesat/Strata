#pragma once

#include "strata/program/generation_capsule.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/kv_spill.hpp"
#include "strata/core/spill_budget.hpp"
#include "strata/core/spill_policy.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/verify.hpp"
#include "strata/prefill/prefill.hpp"
#include <functional>
#include "strata/program/generation_proof.hpp"

namespace strata::program {

// References to the one stable working session, never part of a capsule.
// Later protocol code supplies real ownership counts, not advisory progress.
struct GenerationRuntime {
    core::SessionState& session;
    const core::ModelGeometry& geometry;
    core::MtpDrafter* mtp = nullptr;
    uint64_t model_identity = 0, config_identity = 0;
    int batch_slots = 0, layer_splits = 0, pipeline_windows = 0;
    size_t expert_loans = 0;
    bool prefetch_live = false, commit_live = false, adaptation_live = false;
    bool poisoned = false;
    bool* engine_poisoned = nullptr;
    int32_t* mrope_device = nullptr;
    size_t mrope_capacity = 0;
    core::Verifier* verifier = nullptr;
    prefill::Prefill* prefill = nullptr;
    std::vector<const float*>* image_row_pointers = nullptr;
    int32_t* history_device = nullptr;
    size_t history_capacity = 0;
    int* pcie_num = nullptr;
    uint64_t* checkpoint_clock = nullptr;
    int64_t* tail_checkpoint_len = nullptr;
    GenerationProgress* progress = nullptr;
    std::vector<int64_t>* current_prompt = nullptr;
    std::vector<core::ConversationImageKey>* request_images = nullptr;
    std::vector<int32_t>* mrope_host = nullptr;
    bool* mrope_identity = nullptr;
    std::function<bool(const PrefillSegment&,std::string&)> validate_prefill_geometry;
    std::function<bool(const PrefillSegment&,std::string&)> restore_prefill_geometry;
    std::function<GenerationAccounting()> accounting;
    std::function<void(bool)> invalidate_optional_cache;
    GenerationProof* proof = nullptr; // null in ordinary serving
    // Parked-capsule disk fallback (opt-in).  A null or invalid directory, or `spill_always` false, means RAM only -
    // exactly today's behaviour, no file opened and no new refusal.  The directory is validated and pinned at startup
    // by the caller; the engine only uses it.
    core::SpillDirectory* spill_dir = nullptr;
    bool spill_always = false;                        // testing override: skip the RAM attempt, keep every real check
    uint64_t spill_min_free_bytes = 4096ull << 20;    // the configured extra reserve beyond the payload
    // Set when the pause is accepted: publishing a disk capture must finish before it (the pause drain and a refused
    // RAM attempt already eat into the budget).  A zero value means "budget from now".
    std::chrono::steady_clock::time_point capture_deadline{};
    // The last successful capture's rate on this directory, for the capture-time prediction.
    core::SpillCaptureRate spill_rate;
    // Coalesced external progress: (bytes completed, payload bytes).  The capture applies the 256 MiB / ten-second
    // gate itself, so this is called at the start and then only when an event is actually owed.
    std::function<void(uint64_t,uint64_t)> spill_progress;
};
enum class GenerationTransfer { complete, refused, transfer_failed };
// Capture copies authoritative device data, then rechecks RAM, then moves the
// driver/checkpoints. Refusal preserves them; no replay or cache live_ok needed.
GenerationTransfer generation_capture(GenerationCapsule&, GenerationDriver&,
    std::vector<core::ConversationCheckpoint>&, GenerationRuntime&, std::string&);
// Validate every main/MTP/driver/visual target before the first destination write.
// A partial transfer poisons runtime and MUST prevent any further inference.
GenerationTransfer generation_restore(GenerationCapsule&, GenerationDriver&,
    std::vector<core::ConversationCheckpoint>&, GenerationRuntime&, std::string&);

} // namespace strata::program
