#include "strata/program/generation_capture.hpp"
#include "strata/spec/draft_source.hpp"
#include "strata/kernels/cvec.hpp"
#include <cuda_runtime.h>
#include <new>
#include <utility>
#include <cstring>
#if defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace strata::program {
namespace {
bool fail(std::string& error,const char* why) { error=why; return false; }
bool add(size_t& n,size_t amount) {
    if (amount>SIZE_MAX-n) return false;
    n+=amount; return true;
}
bool runtime_valid(const GenerationDriver& d,const GenerationRuntime& r,std::string& error,bool restoring=false) {
    if (r.poisoned || !r.engine_poisoned || *r.engine_poisoned || r.batch_slots || r.layer_splits || r.pipeline_windows ||
        r.session.layer_lo!=0 || r.session.layer_hi!=r.geometry.n_layers ||
        r.expert_loans || r.prefetch_live || r.commit_live || r.adaptation_live ||
        !r.model_identity || !r.config_identity || d.model_identity!=r.model_identity ||
        d.config_identity!=r.config_identity || d.max_context>r.session.max_cells ||
        d.uses_mtp!=(r.mtp!=nullptr) || !spec::extra_draft_sources().empty() ||
        !r.verifier || !r.prefill || !r.image_row_pointers || !r.pcie_num ||
        !r.checkpoint_clock || !r.tail_checkpoint_len || !r.progress || !r.current_prompt ||
        !r.request_images || !r.mrope_host || !r.mrope_identity || !r.accounting || !r.invalidate_optional_cache ||
        d.history_stage.size()>r.history_capacity || (d.history_size && !r.history_device) ||
        (!d.prefill.geometry.empty() && (!r.restore_prefill_geometry || !r.validate_prefill_geometry)))
        return fail(error,"generation runtime incompatible or work/loan ownership not drained");
    if (d.mrope.size()!=r.mrope_capacity || (!d.mrope.empty() && !r.mrope_device))
        return fail(error,"generation visual position target incompatible");
    for (int64_t off:d.image_row_offsets)
        if (off>=0 && (uint64_t(off)>d.image_rows.size() ||
            uint64_t(r.geometry.n_embd)>d.image_rows.size()-uint64_t(off)))
            return fail(error,"generation visual embedding row incomplete");
    if (!restoring && d.phase==GenerationPhase::prefill &&
        d.prefill.published_ple!=std::array<int32_t,2>{r.session.ple_prev[0],r.session.ple_prev[1]})
        return fail(error,"generation prefill predecessor was not published");
    if (d.phase==GenerationPhase::prefill && r.validate_prefill_geometry &&
        !r.validate_prefill_geometry(d.prefill,error)) return false;
    return true;
}
std::vector<core::ConversationImageKey> prefix_images(const GenerationDriver& d) {
    std::vector<core::ConversationImageKey> images;
    for (const auto& image:d.images) if (uint64_t(image.start)<d.consumed.size()) images.push_back(image);
    return images;
}
bool prefix_images_equal(const std::vector<core::ConversationImageKey>& saved,const GenerationDriver& d) {
    size_t n=0;
    for (const auto& image:d.images) {
        if (uint64_t(image.start)>=d.consumed.size()) break;
        if (n>=saved.size() || !(saved[n++]==image)) return false;
    }
    return n==saved.size();
}
bool checkpoints_valid(const std::vector<core::ConversationCheckpoint>& checks,
        const GenerationDriver& d,const GenerationRuntime& r,std::string& error) {
    for (const auto& c:checks) {
        if (!c.stage_parts.empty() || c.ids.empty() || c.ids.size()>d.consumed.size() ||
            !std::equal(c.ids.begin(),c.ids.end(),d.consumed.begin()) ||
            !core::conversation_checkpoint_validate(c,r.session,r.geometry,error))
            return fail(error,"generation checkpoint ownership/identity incompatible");
        size_t i=0;
        for (const auto& image:d.images) {
            if (uint64_t(image.start)>=c.ids.size()) break;
            if (i>=c.imgs.size() || !(c.imgs[i++]==image)) return fail(error,"generation checkpoint image differs");
        }
        if (i!=c.imgs.size()) return fail(error,"generation checkpoint image count differs");
    }
    return true;
}
int64_t draft_extent(const GenerationDriver& d,const GenerationRuntime& r) {
    // Include the complete serial catch-up and every speculative step, not only
    // the main prefix. Whole pages are captured by conversation_kv_save.
    return std::min(r.mtp->kv_state().max_cells,d.position+2*int64_t(r.mtp->max_t()));
}
GenerationTransfer fatal(GenerationRuntime& r,std::string& error) {
    r.poisoned=true;
    if (r.engine_poisoned) *r.engine_poisoned=true;
    if (error.empty()) error="generation state transfer failed; inference is poisoned";
    return GenerationTransfer::transfer_failed;
}
constexpr size_t proof_scratch=1024*1024;
bool proof_refusal(GenerationRuntime& r,const GenerationDriver& d,
        const std::vector<core::ConversationCheckpoint>& checks,uint64_t before,
        uint64_t driver_hash,uint64_t checkpoint_hash,const char* event,std::string& error) {
    uint64_t after=0;
    if(!core::conversation_live_fingerprint(r.session,r.geometry,d.position,after,error) ||
       before!=after || proof_driver(d)!=driver_hash || proof_checkpoints(checks)!=checkpoint_hash)
        return fail(error,"native proof: refused capture changed its owner");
    r.proof->emit(event,d.request_id,before,driver_hash,checkpoint_hash);return true;
}
bool proof_device(const void* device,const void* saved,size_t bytes,std::string& error) {
    std::array<uint8_t,65536> scratch;
    for(size_t at=0;at<bytes;) {
        const size_t n=std::min(scratch.size(),bytes-at);
        if(cudaMemcpy(scratch.data(),static_cast<const uint8_t*>(device)+at,n,cudaMemcpyDefault)!=cudaSuccess ||
           std::memcmp(scratch.data(),static_cast<const uint8_t*>(saved)+at,n))
            return fail(error,"native proof: restored visual/history bytes differ");
        at+=n;
    }
    return true;
}
}

namespace {
// The RAM capture.  `capacity_refused` marks the failures a disk attempt may follow (memory admission, its post-copy
// recheck, an allocation that could not be made, a forced postcheck refusal); a validation or ownership refusal never
// becomes a disk capture.
GenerationTransfer capture_ram(GenerationCapsule& out,GenerationDriver& driver,
        std::vector<core::ConversationCheckpoint>& checks,GenerationRuntime& runtime,std::string& error,
        bool& capacity_refused) {
    capacity_refused=false;
    if (out.occupied || !driver.validate(error) || !runtime_valid(driver,runtime,error) ||
        !checkpoints_valid(checks,driver,runtime,error)) return GenerationTransfer::refused;
    try {
        const auto fault_id=driver.request_id;
        const bool refuse_first=runtime.proof && runtime.proof->take(ProofFault::capture_admission,fault_id);
        const bool refuse_post=runtime.proof && runtime.proof->take(ProofFault::capture_postcheck,fault_id);
        uint64_t before=0,driver_hash=0,checkpoint_hash=0;
        if(refuse_first || refuse_post) {
            driver_hash=proof_driver(driver);checkpoint_hash=proof_checkpoints(checks);
            if(!core::conversation_live_fingerprint(runtime.session,runtime.geometry,driver.position,before,error))
                return fatal(runtime,error);
        }
        const auto images=prefix_images(driver);
        const std::vector<core::ConversationCheckpoint> no_copies;
        const core::ConversationView view{driver.consumed,images,no_copies,driver.cvec};
        size_t main_bytes=0,mtp_bytes=0,retained=sizeof(GenerationCapsule),allocation=0;
        if (!core::conversation_snapshot_bytes(view,runtime.session,runtime.geometry,
                static_cast<const core::QsaState*>(nullptr),main_bytes,error) ||
            (runtime.mtp && !runtime.mtp->request_bytes(draft_extent(driver,runtime),mtp_bytes,error)))
            return GenerationTransfer::refused;
        if (!add(allocation,main_bytes) || !add(allocation,mtp_bytes) || !add(retained,allocation) ||
            !add(retained,driver.bytes()) || checks.capacity()>SIZE_MAX/sizeof(core::ConversationCheckpoint) ||
            !add(retained,checks.capacity()*sizeof(core::ConversationCheckpoint))) {
            error="generation capsule byte estimate overflow"; return GenerationTransfer::refused;
        }
        for (const auto& c:checks) if (!add(retained,c.bytes())) {
            error="generation checkpoint byte estimate overflow"; return GenerationTransfer::refused;
        }
        if(runtime.proof && (!add(allocation,proof_scratch) || !add(retained,proof_scratch)))
            return GenerationTransfer::refused;
        if(refuse_first) {
            if(generation_memory_admit(uint64_t(0),allocation,retained,error) ||
               !proof_refusal(runtime,driver,checks,before,driver_hash,checkpoint_hash,"refuse_admission",error))
                return fatal(runtime,error);
            return GenerationTransfer::refused;
        }
        if (!generation_memory_admit(core::conversation_available_memory(),allocation,retained,error)) {
            capacity_refused=true;
            return GenerationTransfer::refused;
        }
        GenerationCapsule image;
        if (cudaDeviceSynchronize()!=cudaSuccess) return fatal(runtime,error);
        if (!core::conversation_snapshot_save(image.main,view,runtime.session,runtime.geometry,
                static_cast<const core::QsaState*>(nullptr),error)) return fatal(runtime,error);
        if (runtime.mtp) {
            image.mtp.emplace();
            if (!runtime.mtp->request_capture(*image.mtp,draft_extent(driver,runtime),error))
                return fatal(runtime,error);
        }
        // Count actual capacities while the original driver/checkpoints still
        // have their original single owner. A refusal destroys only new copies.
        size_t actual=image.bytes();
        if (!add(actual,driver.bytes()) || !add(actual,checks.capacity()*sizeof(core::ConversationCheckpoint)))
            return GenerationTransfer::refused;
        for (const auto& c:checks) if (!add(actual,c.bytes())) return GenerationTransfer::refused;
        if(runtime.proof && !add(actual,proof_scratch)) return GenerationTransfer::refused;
        const auto accounting=runtime.accounting();
        const auto parked_at=std::chrono::steady_clock::now();
        if (!generation_publish(out,driver,checks,std::move(image.main),std::move(image.mtp),
                actual,refuse_post ? std::optional<uint64_t>(0):core::conversation_available_memory(),error)) {
            if(refuse_post && !proof_refusal(runtime,driver,checks,before,driver_hash,checkpoint_hash,"refuse_postcheck",error))
                return fatal(runtime,error);
            capacity_refused=true;   // a post-copy refusal is a capacity refusal; the copies are already destroyed
            return GenerationTransfer::refused;
        }
        out.driver.progress=*runtime.progress;
        out.driver.accounting_at_pause=accounting; out.driver.parked_at=parked_at;
        out.driver.ple_token=runtime.session.ple_token;
        out.driver.checkpoint_clock=*runtime.checkpoint_clock;
        out.driver.tail_checkpoint_len=*runtime.tail_checkpoint_len;
        if(runtime.proof) {
            uint64_t main=0,mtp=0;
            if(!core::conversation_snapshot_verify(out.main,runtime.session,runtime.geometry,main,error) ||
               (runtime.mtp && !runtime.mtp->request_verify(*out.mtp,mtp,error,false))) return fatal(runtime,error);
            runtime.proof->emit("capture",fault_id,main,mtp,proof_driver(out.driver));
            if(out.mtp) {
                runtime.proof->emit("mtp_shape",fault_id,out.mtp->geometry[0],out.mtp->prompt_len,out.mtp->kv_extent);
                runtime.proof->emit("mtp_window",fault_id,out.driver.mtp_window_size,out.mtp->max_drafts,out.driver.window_size);
                runtime.proof->emit("mtp_sampling",fault_id,out.mtp->geometry[11],out.mtp->geometry[12],out.mtp->coupled);
                runtime.proof->emit("mtp_q",fault_id,out.mtp->q.size(),
                    std::count_if(out.mtp->q.begin(),out.mtp->q.end(),[](int32_t x){return x!=0;}),0);
            }
            runtime.proof->emit("checkpoint_capture",fault_id,proof_checkpoints(out.main.checkpoints),
                out.main.checkpoints.size(),actual);
        }
        return GenerationTransfer::complete;
    } catch (const std::bad_alloc&) {
        error="generation capsule host allocation refused"; capacity_refused=true; return GenerationTransfer::refused;
    }
}
}  // namespace


namespace {
// The session's owned K/V layers, in the order a snapshot and a manifest both use.
std::vector<const core::QsaState*> spill_layer_states(const GenerationRuntime& r) {
    std::vector<const core::QsaState*> states;
    const int64_t first=r.session.qsa_ord0, count=r.session.qsa_alloc;
    if (count<0 || first<0 || !r.session.qsa_states) return states;
    for (int64_t j=0;j<count;++j) states.push_back(&r.session.qsa_states[size_t(first+j)]);
    return states;
}
// One manifest entry from a streamed source: the sizes and geometry the source already resolved.
core::SpillLayer spill_entry(int64_t layer,const core::SessionKvSource& src,bool include_index) {
    core::SpillLayer entry;
    entry.geometry.layer=layer;
    entry.geometry.format=src.format;
    entry.geometry.cells=src.cells; entry.geometry.heads=src.heads; entry.geometry.head_dim=src.head_dim;
    entry.geometry.page_size=src.page_size; entry.geometry.pooled_rows=src.pooled_rows; entry.geometry.idx_dim=src.idx_dim;
    entry.geometry.include_index=include_index;
    for (size_t part=0;part<core::kSpillParts;++part) entry.sizes[part]=src.sizes[part];
    return entry;
}
// Stream every part of one source into the capsule, in part order, in whole blocks.  Returns false on a source or
// storage failure with `error` set by whichever side failed.
bool spill_stream_parts(const core::SessionKvSource& src,core::SpillWriter& writer,uint64_t& written,
                        const std::chrono::steady_clock::time_point deadline,const std::function<void(uint64_t)>& progress,
                        std::string& error) {
    for (size_t part=0;part<core::kSpillParts;++part) {
        uint64_t offset=0;
        while (offset<src.sizes[part]) {
            const size_t chunk=size_t(std::min<uint64_t>(core::kSpillBlockBytes,src.sizes[part]-offset));
            if (!writer.append_filled(chunk,[&](uint8_t* dst,size_t n,std::string& why) {
                    if (!src.read || !src.read(part,offset,dst,n)) {
                        why=src.error && !src.error().empty() ? src.error() : "spill source read failed";
                        return false;
                    }
                    return true;
                },error)) return false;
            offset+=chunk; written+=chunk;
            if (progress) progress(written);
            // Between bounded operations, never inside one: a cooperative overrun stops here.
            if (std::chrono::steady_clock::now()>deadline) { error="spill capture exceeded the pause budget"; return false; }
        }
    }
    return true;
}

// The disk capture.  One attempt only: this never retries after a storage fault, and it is reached either because the
// RAM attempt was refused for capacity or because `spill_always` skipped it.
GenerationTransfer capture_spill(GenerationCapsule& out,GenerationDriver& driver,
        std::vector<core::ConversationCheckpoint>& checks,GenerationRuntime& runtime,std::string& error) {
    if (!runtime.spill_dir || !runtime.spill_dir->valid()) {
        error="spill capture has no validated directory"; return GenerationTransfer::refused;
    }
    const auto began=std::chrono::steady_clock::now();
    const auto fault_id=driver.request_id;
    try {
        const auto images=prefix_images(driver);
        const std::vector<core::ConversationCheckpoint> no_copies;
        const core::ConversationView view{driver.consumed,images,no_copies,driver.cvec};
        // Metadata (everything but the K/V) plus one streamed source per owned layer: the K/V never becomes host RAM.
        core::SavedConversation meta;
        std::vector<core::SessionKvSource> sources;
        if (!core::conversation_snapshot_sources(meta,sources,view,runtime.session,runtime.geometry,
                static_cast<const core::QsaState*>(nullptr),error)) return GenerationTransfer::refused;
        const auto states=spill_layer_states(runtime);
        if (states.size()!=sources.size()) { error="spill layer/source count mismatch"; return GenerationTransfer::refused; }
        core::SpillManifest manifest;
        manifest.request_id=fault_id;
        for (size_t i=0;i<states.size();++i) manifest.main.push_back(spill_entry(int64_t(i),sources[i],true));
        std::optional<core::MtpRequestState> mtp_state;
        core::SessionKvSource draft_source;
        const bool with_draft=runtime.mtp!=nullptr;
        const int64_t draft=with_draft ? draft_extent(driver,runtime) : 0;
        if (with_draft) {
            mtp_state.emplace();
            // The resident state and the draft K/V source in one call: the speculative extent (lookahead included) is
            // what gets captured, and the coupled history/q/R/sampling state comes with it.
            if (!runtime.mtp->request_capture_sources(*mtp_state,draft_source,draft,error))
                return fatal(runtime,error);
            // A synthetic ordinal past the owned layers: the draft entry is not a main layer and never validates as one.
            manifest.draft=spill_entry(runtime.session.qsa_ord0+runtime.session.qsa_alloc,draft_source,false);
        }
        if (!core::spill_manifest_layout(manifest,error)) return GenerationTransfer::refused;
        const uint64_t payload=manifest.payload_bytes;
        // Capture-time budget: refuse before opening anything when the prediction does not fit what is left of it.
        const auto deadline=std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            runtime.capture_deadline.time_since_epoch()).count()
            ? runtime.capture_deadline
            : began+std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(core::kSpillCaptureBudgetSeconds));
        const double remaining=std::chrono::duration<double>(deadline-began).count();
        const double estimate=core::spill_capture_estimate_seconds(payload,runtime.spill_rate);
        if (!core::spill_capture_admits(estimate,remaining)) {
            error="spill capture would exceed the remaining pause budget"; return GenerationTransfer::refused;
        }
        // Free space: a preflight, not a reservation; a runtime shortfall is still a storage refusal.
        const auto free=runtime.spill_dir->free_bytes();
        if (!core::spill_disk_preflight(free.has_value(),free ? *free : 0,payload,runtime.spill_min_free_bytes)) {
            error="spill capture does not fit the configured free-space reserve"; return GenerationTransfer::refused;
        }
        // Memory: what this capture keeps resident (driver, checkpoints, the MTP resident state, the manifest) against
        // what it adds (one staging block, the proof scratch).  The K/V itself is not resident at any point.
        size_t retained=sizeof(GenerationCapsule),additional=core::spill_capture_additional_bytes();
        if (!add(retained,driver.bytes()) || !add(retained,checks.capacity()*sizeof(core::ConversationCheckpoint)))
            return GenerationTransfer::refused;
        for (const auto& c:checks) if (!add(retained,c.bytes())) return GenerationTransfer::refused;
        if (mtp_state && !add(retained,mtp_state->bytes())) return GenerationTransfer::refused;
        if (runtime.proof && !add(additional,proof_scratch)) return GenerationTransfer::refused;
        if (!generation_memory_admit(core::conversation_available_memory(),additional,retained,error))
            return GenerationTransfer::refused;
        // The writer: one anonymous file, one 16 MiB staging block.  A storage refusal here leaves the background running.
        const core::SpillFault fault=runtime.proof ? core::SpillFault([&runtime,fault_id](const char* step) -> int {
            if (std::strcmp(step,"write")==0 &&
                runtime.proof->take(ProofFault::spill_write,fault_id)) return ENOSPC;
            return 0;
        }) : core::SpillFault{};
        const core::SpillCaptureRate rate_before=runtime.spill_rate;
        core::SpillWriter writer=core::SpillWriter::create(*runtime.spill_dir,manifest,error,fault);
        if (!writer.valid()) return GenerationTransfer::refused;
        if (cudaDeviceSynchronize()!=cudaSuccess) return fatal(runtime,error);
        // External progress is coalesced by this gate: 256 MiB of new work or ten seconds, plus the start.
        core::SpillProgressGate gate(0);
        uint64_t written=0;
        const auto progress=[&](uint64_t completed) {
            const double now=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
            if (gate.due(now,completed) && runtime.spill_progress) { runtime.spill_progress(completed,payload); gate.emitted(now,completed); }
        };
        if (runtime.spill_progress) { runtime.spill_progress(0,payload); gate.emitted(0,0); }
        bool source_failed=false;
        for (size_t i=0;i<sources.size();++i) {
            // Proof fault: a GPU source read failure is FATAL, never a storage refusal that would keep decoding.
            const bool injected=runtime.proof && runtime.proof->take(ProofFault::spill_source_copy,fault_id);
            const bool copied=!injected && spill_stream_parts(sources[i],writer,written,deadline,progress,error);
            if (!copied) {
                if (injected) {
                    source_failed=true;
                    if (error.empty()) error="spill source copy failed";
                    break;
                }
                return GenerationTransfer::refused;   // a storage or budget failure: the background keeps running
            }
        }
        if (source_failed) return fatal(runtime,error);
        if (with_draft && !spill_stream_parts(draft_source,writer,written,deadline,progress,error))
            return GenerationTransfer::refused;
        if (!writer.finish(error)) return GenerationTransfer::refused;
        // A capture that finished after its budget is never published.
        if (std::chrono::steady_clock::now()>deadline) {
            error="spill capture finished after the pause budget"; return GenerationTransfer::refused;
        }
        if (!generation_memory_postcheck(core::conversation_available_memory(),retained,error))
            return GenerationTransfer::refused;
        CapsuleSpill spill;
        spill.file=writer.release();
        spill.manifest=writer.manifest();
        if (!spill.valid() || spill.spilled_bytes()!=payload) { error="spill capsule is not the one written"; return GenerationTransfer::refused; }
#if defined(__linux__)
        if (runtime.proof && runtime.proof->take(ProofFault::spill_truncate,fault_id)) {
            // Proof fault: a capsule that lost its tail must be refused by the restore, never published as complete.
            (void)::ftruncate(spill.file.fd(),static_cast<off_t>(payload>core::kSpillBlockBytes ? payload-core::kSpillBlockBytes : 0));
        }
        if (runtime.proof) {
            // Proof faults: one corrupted block early (before any live write, so the restore must refuse the capsule)
            // and one late (after the first block was written, so the restore must be fatal).
            const bool corrupt_first=runtime.proof->take(ProofFault::spill_corrupt_first,fault_id);
            const bool corrupt_late=!corrupt_first && runtime.proof->take(ProofFault::spill_corrupt_late,fault_id);
            if (corrupt_first || corrupt_late) {
                const uint64_t at=corrupt_late && payload>core::kSpillBlockBytes*2 ? core::kSpillBlockBytes+16 : 0;
                const uint8_t junk=0x5A;
                if (at<payload) (void)::pwrite(spill.file.fd(),&junk,1,static_cast<off_t>(at));
            }
        }
#endif
        size_t actual=retained;
        if (!add(actual,spill.bytes())) return GenerationTransfer::refused;   // the manifest stays resident
        const auto accounting=runtime.accounting();
        const auto parked_at=std::chrono::steady_clock::now();
        core::SpillManifest published=spill.manifest;
        if (!generation_publish(out,driver,checks,std::move(meta),std::move(mtp_state),actual,
                core::conversation_available_memory(),error,std::move(spill))) {
            return GenerationTransfer::refused;   // a broken capsule is refused with the background still running
        }
        out.driver.progress=*runtime.progress;
        out.driver.accounting_at_pause=accounting; out.driver.parked_at=parked_at;
        out.driver.ple_token=runtime.session.ple_token;
        out.driver.checkpoint_clock=*runtime.checkpoint_clock;
        out.driver.tail_checkpoint_len=*runtime.tail_checkpoint_len;
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
        runtime.spill_rate.sample(seconds,payload);
        if (runtime.proof) {
            runtime.proof->emit("spill_capture",fault_id,payload,written,rate_before.have() ? uint64_t(rate_before.bytes_per_second()) : 0);
            runtime.proof->emit("spill_manifest",fault_id,published.main.size(),
                published.draft ? 1u : 0u,uint64_t(seconds*1000));
        }
        return GenerationTransfer::complete;
    } catch (const std::bad_alloc&) {
        error="spill capture host allocation refused"; return GenerationTransfer::refused;
    }
}
}  // namespace

GenerationTransfer generation_capture(GenerationCapsule& out,GenerationDriver& driver,
        std::vector<core::ConversationCheckpoint>& checks,GenerationRuntime& runtime,std::string& error) {
    const bool spill_available=runtime.spill_dir && runtime.spill_dir->valid();
    if (spill_available) {
        // The trusted fault refuses only the RAM CANDIDATE, so the disk attempt below is the real fallback logic
        // (validation included) rather than a second, test-only path through it.
        const bool forced_ram_refusal=runtime.proof &&
            runtime.proof->take(ProofFault::spill_ram_admission,driver.request_id);
        if (runtime.spill_always || forced_ram_refusal) {
            if (!out.occupied && driver.validate(error) && runtime_valid(driver,runtime,error) &&
                checkpoints_valid(checks,driver,runtime,error))
                return capture_spill(out,driver,checks,runtime,error);
        }
    }
    bool capacity_refused=false;
    const auto result=capture_ram(out,driver,checks,runtime,error,capacity_refused);
    // `fallback`: one disk attempt after a capacity refusal, and never for any other kind of refusal.  A forced RAM
    // refusal (the trusted proof fault) is the way this branch is exercised on real hardware.
    if (result==GenerationTransfer::refused && capacity_refused && spill_available)
        return capture_spill(out,driver,checks,runtime,error);
    return result;
}

namespace {
// The draft entry's layout, for the disk-aware validation: the same geometry the capture wrote.
bool spill_draft_layout(const GenerationRuntime& r,const GenerationDriver& d,core::ConversationKvLayout& layout,
                        std::string& error) {
    return core::conversation_kv_layout(r.mtp->kv_state(),r.geometry,draft_extent(d,r),false,layout,error);
}
// Restore a disk capsule's K/V and resident state in the plan's order: every main layer's K/V, then the main layers'
// recurrent state, then the draft layer's K/V and its working state.  Returns `lost` when the payload proved unusable
// before anything was written, `refused` for an admission refusal (the capsule keeps its file), `transfer_failed`
// once any live write happened.
GenerationTransfer restore_spill_kv(GenerationCapsule& image,GenerationRuntime& runtime,std::string& error) {
    if (!image.spill) { error="generation capsule has no spill to restore"; return GenerationTransfer::refused; }
    std::optional<CapsuleSpill> spill=image.release_spill();
    core::SpillReader reader=core::SpillReader::open(std::move(spill->file),spill->manifest,error);
    if (!reader.valid()) {
        // Validation refused it (or the file is not what the manifest describes): give the capsule its file back so a
        // retry or a cancellation can still close it, exactly as the RAM refusal path allows.
        spill->file=reader.release_file();
        image.spill=std::move(spill);
        return GenerationTransfer::refused;
    }
    spill.reset();   // the reader owns the descriptor from here on
    std::vector<core::ConversationKvTarget> targets;
    for (const auto* state:spill_layer_states(runtime))
        targets.push_back({state,false,image.driver.position});
    const bool has_draft=runtime.mtp!=nullptr;
    if (has_draft) targets.push_back({&runtime.mtp->kv_state(),true,draft_extent(image.driver,runtime)});
    bool main_restored=false;
    const auto restore_main_resident=[&](std::string& why) {
        if (!core::conversation_snapshot_restore_resident(image.main,runtime.session,runtime.geometry,why))
            return false;
        main_restored=true;
        // The existing proof fault keeps firing on this path too: an injected failure after the main transfer.
        if (runtime.proof && runtime.proof->take(ProofFault::restore_after_main,image.driver.request_id)) {
            why="native proof: injected failure after main transfer";
            return false;
        }
        return true;
    };
    // One verified pass over the capsule; the reader has already checked every block against its digest.
    const core::KvLayerDriveFn drive=[&](const core::KvLayerSink& sink,std::string& why) -> core::KvDrive {
        const core::SpillReadResult read=reader.run([&](const core::SpillBlockRef& block,std::string& why_block) {
            return sink(block.layer,block.draft,block.part,block.offset,block.data,block.size,why_block);
        });
        if (read.ok) return core::KvDrive::ok;
        why=read.error;
        return read.admission ? core::KvDrive::refused : core::KvDrive::lost;
    };
    const auto restored=core::conversation_kv_restore_layers(runtime.geometry,targets,drive,
        has_draft ? restore_main_resident : std::function<bool(std::string&)>{});
    if (restored.outcome!=core::ConversationRestore::restored) {
        error=restored.error;
        if (restored.detail==core::KvTransferOutcome::fatal) return fatal(runtime,error);
        return restored.detail==core::KvTransferOutcome::lost ? GenerationTransfer::lost : GenerationTransfer::refused;
    }
    if (!has_draft && !restore_main_resident(error)) return fatal(runtime,error);
    // Main K/V and recurrent state are in place; now the draft layer's working state (its K/V came last in the pass).
    if (has_draft && !runtime.mtp->request_restore_resident(*image.mtp,runtime.session,error))
        return fatal(runtime,error);
    if (!main_restored) { error="generation capsule main state was not restored"; return fatal(runtime,error); }
    return GenerationTransfer::complete;
}
}  // namespace

GenerationTransfer generation_restore(GenerationCapsule& image,GenerationDriver& driver,
        std::vector<core::ConversationCheckpoint>& checks,GenerationRuntime& runtime,std::string& error) {
    // Trusted proof metadata only: categories distinguish prevalidation,
    // byte-count overflow, admission and postcheck refusal. Fatal transfers
    // keep their existing poison path rather than becoming refusals.
    auto refused=[&](uint64_t category) {
        if (runtime.proof) {
            const uint64_t ownership=(runtime.expert_loans ? 1u:0u) |
                (runtime.prefetch_live ? 2u:0u) | (runtime.commit_live ? 4u:0u) |
                (runtime.adaptation_live ? 8u:0u) | (runtime.poisoned ? 16u:0u) |
                (runtime.engine_poisoned && *runtime.engine_poisoned ? 32u:0u);
            runtime.proof->emit("restore_refused",image.occupied ? image.driver.request_id:0,
                category,ownership,checks.size());
        }
        return GenerationTransfer::refused;
    };
    const bool injected_validation=runtime.proof && image.occupied &&
        runtime.proof->take(ProofFault::restore_validation,image.driver.request_id);
    const auto original_version=image.driver.version;
    uint64_t before_validation=0;
    if(injected_validation) {
        if(!core::conversation_live_fingerprint(runtime.session,runtime.geometry,image.driver.position,before_validation,error))
            return fatal(runtime,error);
        image.driver.version=original_version+1;
    }
    // A disk capsule validates its resident manifest and its K/V as LAYOUTS (the payload is in its file); the RAM path
    // keeps validating the payload itself.  Both run before any destination write.
    const bool disk_capsule=image.spill.has_value();
    core::ConversationKvLayout draft_layout;
    const auto state_valid=[&]() -> bool {
        if (disk_capsule)
            return core::conversation_snapshot_validate_resident(image.main,runtime.session,runtime.geometry,error);
        return core::conversation_snapshot_validate(image.main,runtime.session,runtime.geometry,
            static_cast<const core::QsaState*>(nullptr),error);
    };
    const auto mtp_valid=[&]() -> bool {
        if (!runtime.mtp) return true;
        const int64_t extent=draft_extent(image.driver,runtime);
        if (!disk_capsule) return runtime.mtp->request_validate(*image.mtp,image.driver.original_prompt_len,extent,error);
        return core::conversation_kv_layout(runtime.mtp->kv_state(),runtime.geometry,extent,false,draft_layout,error) &&
               runtime.mtp->request_validate_layout(*image.mtp,image.driver.original_prompt_len,extent,draft_layout,error);
    };
    if (!image.occupied || image.version!=GenerationCapsule::schema_version ||
        image.saved_phase!=image.driver.phase || !image.driver.validate(error) ||
        !runtime_valid(image.driver,runtime,error,true) || !checks.empty() ||
        (driver.phase!=GenerationPhase::parked && driver.phase!=GenerationPhase::terminal) ||
        image.main.live.ids!=image.driver.consumed || !prefix_images_equal(image.main.live.imgs,image.driver) ||
        image.main.cvec!=image.driver.cvec || image.mtp.has_value()!=(runtime.mtp!=nullptr) ||
        !state_valid() || !mtp_valid()) {
        if(injected_validation) {
            image.driver.version=original_version;
            uint64_t after=0;
            if(!core::conversation_live_fingerprint(runtime.session,runtime.geometry,image.driver.position,after,error) ||
               before_validation!=after) return fatal(runtime,error);
            runtime.proof->emit("refuse_validation",image.driver.request_id,before_validation,proof_driver(image.driver),
                proof_checkpoints(image.main.checkpoints));
        }
        return refused(1);
    }
    if(injected_validation) {
        image.driver.version=original_version;
        error="native proof: incompatible schema unexpectedly admitted";return fatal(runtime,error);
    }
    // Do not compare the *current* PLE predecessor on restore: Hermes owns it.
    // The image's checkpoint validates/restores it, and its driver binds it to
    // the original coding prefix before any destination writes above.
    try {
        size_t extra=0;
        for (size_t n : {image.driver.prompt.size()*sizeof(int64_t),
                image.driver.images.size()*sizeof(core::ConversationImageKey),
                image.driver.mrope.size()*sizeof(int32_t),
                image.driver.image_row_offsets.size()*sizeof(const float*),
                image.mtp ? (image.mtp->top2.size()+image.mtp->debug_logits.size())*sizeof(int32_t) : 0})
            if (!add(extra,n)) return refused(2);
        size_t retained=image.bytes();
        if(runtime.proof && (!add(extra,proof_scratch) || !add(retained,proof_scratch)))
            return refused(2);
        if (!add(retained,extra) || !generation_memory_admit(core::conversation_available_memory(),extra,retained,error))
            return refused(3);
        const auto accounting=runtime.accounting();
        // Prepare derived pointers before device mutation. Moving the driver's
        // owning vectors afterwards preserves every embedding address.
        std::vector<const float*> rows(image.driver.image_row_offsets.size(),nullptr);
        for (size_t i=0;i<rows.size();++i) if (image.driver.image_row_offsets[i]>=0)
            rows[i]=image.driver.image_rows.data()+image.driver.image_row_offsets[i];
        auto current_prompt=image.driver.prompt;
        auto request_images=image.driver.images;
        auto mrope_host=image.driver.mrope;
        if (!generation_memory_postcheck(core::conversation_available_memory(),retained,error))
            return refused(4);
        if (disk_capsule) {
            // K/V from the file, resident state after it, the draft layer last - and a capsule whose payload proved
            // unusable before any write comes back as `lost`, not as a refusal and not as a poisoned engine.
            const auto disk=restore_spill_kv(image,runtime,error);
            if (disk!=GenerationTransfer::complete) return disk;
        } else {
            if (core::conversation_snapshot_restore(image.main,runtime.session,runtime.geometry,
                    static_cast<const core::QsaState*>(nullptr),error)!=core::ConversationRestore::restored)
                return fatal(runtime,error);
            if(runtime.proof && runtime.proof->take(ProofFault::restore_after_main,image.driver.request_id)) {
                error="native proof: injected failure after main transfer";
                const auto result=fatal(runtime,error);
                runtime.proof->emit("partial_transfer_poison",image.driver.request_id,runtime.poisoned,
                    *runtime.engine_poisoned,0);return result;
            }
            if (runtime.mtp && !runtime.mtp->request_restore(*image.mtp,runtime.session,error))
                return fatal(runtime,error);
        }
        if (!image.driver.mrope.empty() && cudaMemcpy(runtime.mrope_device,image.driver.mrope.data(),
                image.driver.mrope.size()*sizeof(int32_t),cudaMemcpyHostToDevice)!=cudaSuccess)
            return fatal(runtime,error);
        if (cudaDeviceSynchronize()!=cudaSuccess) return fatal(runtime,error);
        if (image.driver.phase==GenerationPhase::prefill && runtime.restore_prefill_geometry &&
            !runtime.restore_prefill_geometry(image.driver.prefill,error)) return fatal(runtime,error);
        if (image.driver.history_size && cudaMemcpy(runtime.history_device,image.driver.history_stage.data(),
                image.driver.history_stage.size()*sizeof(int32_t),cudaMemcpyHostToDevice)!=cudaSuccess)
            return fatal(runtime,error);
        runtime.verifier->set_sampling(image.driver.sampling);
        runtime.verifier->set_history(image.driver.history_size ? runtime.history_device : nullptr,image.driver.history_size);
        runtime.verifier->set_spec_q(nullptr,0); // the next window stages restored MTP q under the original policy
        runtime.verifier->set_head_sampling(true);
        if (kernels::cvec().loaded()) kernels::cvec_set_enabled(image.driver.cvec);
        runtime.invalidate_optional_cache(image.driver.cvec);
        *runtime.pcie_num=std::max(0,std::min(256,int(image.driver.pcie_fraction*256.0+0.5)));
        *runtime.checkpoint_clock=image.driver.checkpoint_clock;
        *runtime.tail_checkpoint_len=image.driver.tail_checkpoint_len;
        runtime.session.ple_token=image.driver.ple_token;
        *runtime.progress=image.driver.progress;
        *runtime.current_prompt=std::move(current_prompt);
        *runtime.request_images=std::move(request_images);
        *runtime.mrope_host=std::move(mrope_host);
        *runtime.mrope_identity=image.driver.mrope_identity;
        const uint64_t driver_before=runtime.proof ? proof_driver(image.driver):0;
        const uint64_t checks_before=runtime.proof ? proof_checkpoints(image.main.checkpoints):0;
        const auto checkpoint_count=image.main.checkpoints.size();
        if(runtime.proof) {
            uint64_t main=0,mtp=0;
            // A disk capsule's K/V was verified block by block as it was read; there is no resident image to re-hash.
            const bool kv_readback=disk_capsule ||
                (core::conversation_snapshot_verify(image.main,runtime.session,runtime.geometry,main,error) &&
                 (!runtime.mtp || runtime.mtp->request_verify(*image.mtp,mtp,error)));
            if(!kv_readback ||
               !proof_device(runtime.mrope_device,image.driver.mrope.data(),image.driver.mrope.size()*sizeof(int32_t),error) ||
               (image.driver.history_size && !proof_device(runtime.history_device,image.driver.history_stage.data(),
                    image.driver.history_stage.size()*sizeof(int32_t),error)) ||
               *runtime.current_prompt!=image.driver.prompt || *runtime.request_images!=image.driver.images ||
               *runtime.mrope_host!=image.driver.mrope || *runtime.mrope_identity!=image.driver.mrope_identity ||
               runtime.session.ple_token!=image.driver.ple_token)
                return fatal(runtime,error);
            runtime.proof->emit("restore",image.driver.request_id,main,mtp,driver_before);
            if(image.mtp) {
                runtime.proof->emit("mtp_shape_restore",image.driver.request_id,image.mtp->geometry[0],image.mtp->prompt_len,image.mtp->kv_extent);
                runtime.proof->emit("mtp_window_restore",image.driver.request_id,image.driver.mtp_window_size,image.mtp->max_drafts,image.driver.window_size);
                runtime.proof->emit("mtp_sampling_restore",image.driver.request_id,image.mtp->geometry[11],image.mtp->geometry[12],image.mtp->coupled);
                runtime.proof->emit("mtp_q_restore",image.driver.request_id,image.mtp->q.size(),
                    std::count_if(image.mtp->q.begin(),image.mtp->q.end(),[](int32_t x){return x!=0;}),0);
            }
        }
        const auto expected_accounting=image.driver.foreground_excluded+accounting-image.driver.accounting_at_pause;
        const double prior_prefill=image.driver.paused_prefill_ms, prior_decode=image.driver.paused_decode_ms;
        image.driver.foreground_excluded=expected_accounting;
        const double paused=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-image.driver.parked_at).count();
        if (image.saved_phase==GenerationPhase::prefill) image.driver.paused_prefill_ms+=paused;
        else image.driver.paused_decode_ms+=paused;
        if(runtime.proof) {
            core::ProofHash a,b;proof_accounting(a,image.driver.foreground_excluded);proof_accounting(b,expected_accounting);
            if(a.value!=b.value || paused<0 ||
               image.driver.paused_prefill_ms!=(image.saved_phase==GenerationPhase::prefill ? prior_prefill+paused:prior_prefill) ||
               image.driver.paused_decode_ms!=(image.saved_phase==GenerationPhase::decode ? prior_decode+paused:prior_decode))
                return fatal(runtime,error);
            runtime.proof->emit("accounting",image.driver.request_id,a.value,b.value,uint64_t(paused));
        }
        generation_take(image,driver,checks);
        *runtime.image_row_pointers=std::move(rows);
        runtime.prefill->embd_rows=driver.image_rows.empty() ? nullptr : runtime.image_row_pointers->data();
        if(runtime.proof) {
            if(image.occupied || proof_driver(driver)!=driver_before || proof_checkpoints(checks)!=checks_before ||
               checks.size()!=checkpoint_count) return fatal(runtime,error);
            for(size_t i=0;i<driver.image_row_offsets.size();++i) {
                const auto offset=driver.image_row_offsets[i];
                if((*runtime.image_row_pointers)[i]!=(offset<0 ? nullptr:driver.image_rows.data()+offset))
                    return fatal(runtime,error);
            }
            runtime.proof->emit("checkpoint_restore",driver.request_id,checks_before,checks.size(),1);
        }
        return GenerationTransfer::complete;
    } catch (const std::bad_alloc&) {
        return fatal(runtime,error);
    }
}
} // namespace strata::program
