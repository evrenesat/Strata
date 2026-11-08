#pragma once
#include "strata/program/generation_capsule.hpp"
#include "strata/core/proof_hash.hpp"
#include <functional>
#include <sstream>

namespace strata::program {
enum class ProofFault { none, capture_admission, capture_postcheck, restore_validation, restore_after_main,
                        spill_ram_admission, spill_write, spill_truncate, spill_corrupt_first, spill_corrupt_late,
                        spill_source_copy };
struct GenerationProof {
    // Bound only by --priority-proof, never request parameters or HTTP.
    ProofFault fault=ProofFault::none;
    uint64_t fault_id=0, pause_id=0;
    GenerationPhase pause_phase=GenerationPhase::terminal;
    int64_t pause_min=0;
    bool pause_pending=false;
    std::function<void(const char*,uint64_t,uint64_t,uint64_t,uint64_t)> emit;
    bool take(ProofFault f,uint64_t id) {
        if(fault!=f || fault_id!=id) return false;
        fault=ProofFault::none; fault_id=0; return true;
    }
};
struct ProofCommand {
    enum Kind { reset, fault, pause } kind{};
    uint64_t id=0;
    ProofFault injection=ProofFault::none;
    GenerationPhase phase=GenerationPhase::terminal;
    int64_t minimum=0;
};
inline bool proof_command(const std::string& line,ProofCommand& out) {
    if(line=="PTEST RESET") { out={}; return true; }
    std::istringstream s(line); std::string prefix,verb,id,arg,min,extra;
    if(!(s>>prefix>>verb>>id>>arg) || prefix!="PTEST" || id.empty() || id[0]=='0') return false;
    ProofCommand c;
    for(char ch:id) {
        if(ch<'0'||ch>'9'||c.id>(uint64_t(INT64_MAX)-uint64_t(ch-'0'))/10) return false;
        c.id=c.id*10+uint64_t(ch-'0');
    }
    if(verb=="FAULT") {
        c.kind=ProofCommand::fault;
        if(arg=="capture_admission") c.injection=ProofFault::capture_admission;
        else if(arg=="capture_postcheck") c.injection=ProofFault::capture_postcheck;
        else if(arg=="restore_validation") c.injection=ProofFault::restore_validation;
        else if(arg=="restore_after_main") c.injection=ProofFault::restore_after_main;
        // The parked-capsule disk faults: the same trusted, default-off grammar (never an HTTP parameter).
        else if(arg=="spill_ram_admission") c.injection=ProofFault::spill_ram_admission;
        else if(arg=="spill_write") c.injection=ProofFault::spill_write;
        else if(arg=="spill_truncate") c.injection=ProofFault::spill_truncate;
        else if(arg=="spill_corrupt_first") c.injection=ProofFault::spill_corrupt_first;
        else if(arg=="spill_corrupt_late") c.injection=ProofFault::spill_corrupt_late;
        else if(arg=="spill_source_copy") c.injection=ProofFault::spill_source_copy;
        else return false;
    } else if(verb=="PAUSE") {
        c.kind=ProofCommand::pause;
        if(arg=="prefill") c.phase=GenerationPhase::prefill;
        else if(arg=="decode") c.phase=GenerationPhase::decode;
        else return false;
        if(!(s>>min) || min.empty()) return false;
        for(char ch:min) {
            if(ch<'0'||ch>'9'||c.minimum>(INT64_MAX-(ch-'0'))/10) return false;
            c.minimum=c.minimum*10+(ch-'0');
        }
    } else return false;
    if(s>>extra) return false;
    out=c; return true;
}
inline void proof_checkpoint(core::ProofHash& h,const core::ConversationCheckpoint& c) {
    h.add(c.ids); h.add(c.imgs.size());
    for(const auto& i:c.imgs) { h.add(i.start); h.add(i.hash); }
    h.add(c.gdn); h.add(c.ple); h.add(c.tails); h.add(c.dead); h.add(c.block_pos);
    h.add(c.used); h.add(c.pinned); h.add(c.stage_parts.size());
    for(const auto& p:c.stage_parts) proof_checkpoint(h,p);
}
inline uint64_t proof_checkpoints(const std::vector<core::ConversationCheckpoint>& checks) {
    core::ProofHash h; h.add(checks.size()); for(const auto& c:checks) proof_checkpoint(h,c); return h.value;
}
inline void proof_accounting(core::ProofHash& h,const GenerationAccounting& a) {
    h.add(a.wait);h.add(a.pool);h.add(a.host);h.add(a.plan);h.add(a.actq);h.add(a.jobs);h.add(a.run);
    h.add(a.misses);h.add(a.entries);h.add(a.hits);h.add(a.pcie);h.add(a.ram);h.add(a.files);
    h.add(a.lookups);h.add(a.offload);h.add(a.file_bytes);
}
// Continuation state. Excludes only fields deliberately updated by restore:
// foreground_excluded and paused_*_ms; those have separate arithmetic assertions.
inline uint64_t proof_driver(const GenerationDriver& d) {
    core::ProofHash h;
    h.add(d.version);h.add(d.phase);h.add(d.request_id);h.add(d.model_identity);h.add(d.config_identity);
    h.add(d.original_prompt_len);h.add(d.max_context);h.add(d.max_new);h.add(d.produced);
    h.add(d.position);h.add(d.read_from);h.add(d.reused);h.add(d.reread_to);h.add(d.next_token);
    h.add(d.window_size);h.add(d.mtp_window_size);h.add(d.history_size);
    h.add(d.first_window);h.add(d.cancelled);h.add(d.cvec);h.add(d.checkpoints_enabled);h.add(d.uses_mtp);
    h.add(d.pin);h.add(d.short_read);h.add(d.tail_checkpoint_len);h.add(d.checkpoint_clock);h.add(d.ple_token);
    h.add(d.progress.total);h.add(d.progress.from);h.add(d.progress.reached);h.add(d.progress.next_checkpoint);
    h.add(d.progress.tail_saved);
    h.add(d.pcie_fraction);h.add(d.spec_min_p);h.add(d.sampling);h.add(d.finish);
    h.add(d.prompt);h.add(d.consumed);h.add(d.drafts);h.add(d.window);h.add(d.output);h.add(d.draft_probabilities);
    h.add(d.suffix_buffer);h.add(d.chain_buffer);h.add(d.chain_tail);h.add(d.history_stage);
    h.add(d.output_remainder);h.add(d.output_cursor);h.add(d.images.size());
    for(const auto& i:d.images) { h.add(i.start);h.add(i.hash); }
    h.add(d.image_rows);h.add(d.image_row_offsets);h.add(d.mrope);h.add(d.mrope_identity);
    h.add(d.prefill.begin);h.add(d.prefill.end);h.add(d.prefill.offset);h.add(d.prefill.chunk);h.add(d.prefill.cuts);
    h.add(d.prefill.returned);h.add(d.prefill.prefetch_drained);h.add(d.prefill.loans_returned);h.add(d.prefill.published_ple);
    h.add(d.prefill.geometry.size());for(const auto& g:d.prefill.geometry) {
        h.add(g.chunk);h.add(g.first);h.add(g.slots.size());for(const auto& p:g.slots) {h.add(p.first);h.add(p.second);}
    }
    h.add(d.suffix.has_value());if(d.suffix)d.suffix->proof_hash(h);
    h.add(d.policy.has_value());if(d.policy)d.policy->proof_hash(h);
    h.add(d.rounds);h.add(d.draft_offered);h.add(d.draft_accepted);h.add(d.suffix_windows);h.add(d.suffix_drafts);
    h.add(d.suffix_accepted);h.add(d.chain_windows);h.add(d.chain_drafts);h.add(d.chain_accepted);
    h.add(d.top2_rejected);h.add(d.top2_hit);h.add(d.depth_tested);h.add(d.depth_accepted);
    h.add(d.decode_windows);h.add(d.decode_rows);
    return h.value;
}
} // namespace strata::program
