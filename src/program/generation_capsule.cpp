#include "strata/program/generation_capsule.hpp"
#include "strata/core/coupled_draft.hpp"
#include "strata/core/spec_prob.hpp"
#include <cmath>
#include <utility>

namespace {
bool fail(std::string& error, const char* why) { error=why; return false; }
void add(size_t& n, size_t amount) { n=amount>SIZE_MAX-n ? SIZE_MAX : n+amount; }
template<class T> void count(size_t& n, const std::vector<T>& v) {
    add(n,v.capacity()>SIZE_MAX/sizeof(T) ? SIZE_MAX : v.capacity()*sizeof(T));
}
bool sample_valid(const strata::kernels::SamplerParams& s) {
    return s.seed && std::isfinite(s.temperature) && std::isfinite(s.top_p) &&
        std::isfinite(s.min_p) && std::isfinite(s.penalty_repeat) &&
        std::isfinite(s.penalty_freq) && std::isfinite(s.penalty_present) && s.penalty_last_n>=0;
}
}

namespace strata::core {
bool MtpRequestState::validate_layout(const std::array<int64_t,13>& expected,
        int64_t expected_prompt, int64_t expected_extent, std::string& error) const {
    if (version!=schema_version || geometry!=expected || prompt_len!=expected_prompt ||
        kv_extent!=expected_extent || prompt_len<=0 || kv_extent<=0 || kv_extent>geometry[4] ||
        max_drafts<0 || (coupled && !geometry[11])) return fail(error,"mtp request identity/extent mismatch");
    const auto T=geometry[0], HCN=geometry[1], NH=geometry[2];
    if (T<1 || T>8 || HCN<1 || HCN>1<<20 || NH<1 || NH>1024)
        return fail(error,"mtp request geometry out of range");
    auto ints=[&](const auto& v, int64_t n) { return n>=0 && v.size()==uint64_t(n); };
    if (!ints(host_tokens,T) || !ints(device_tokens,T) || !ints(host_steps,8*T) ||
        !ints(device_steps,8*T) || !ints(host_positions,2*T*NH) || !ints(device_positions,2*T*NH) ||
        !ints(host_row,2) || !ints(device_row,4) || !ints(host_out,T) || !ints(device_out,T+4) ||
        !ints(host_prob,T) || !ints(device_prob,T+4) || !ints(input_R,T*HCN) ||
        !ints(residual_R,T*HCN) || !ints(window_R,T*HCN) ||
        !ints(history,geometry[11] ? kCoupledHistCap : 0) ||
        !ints(ring,geometry[11] ? kCoupledHistCap+T : 0) ||
        !ints(q,geometry[12] ? T*kSpecQStride : 0) ||
        top2.size()>uint64_t(T) || debug_logits.size()>uint64_t(std::max(geometry[6],geometry[7])))
        return fail(error,"mtp request buffer manifest mismatch");
    if (coupled && !sample_valid(sampling)) return fail(error,"mtp request sampling invalid");
    return true;
}
}

namespace strata::program {
GenerationAccounting operator+(GenerationAccounting a,const GenerationAccounting& b) {
    a.wait+=b.wait; a.pool+=b.pool; a.host+=b.host; a.plan+=b.plan; a.actq+=b.actq; a.jobs+=b.jobs; a.run+=b.run;
    a.misses+=b.misses; a.entries+=b.entries; a.hits+=b.hits; a.pcie+=b.pcie;
    a.ram+=b.ram; a.files+=b.files; a.lookups+=b.lookups; a.offload+=b.offload; a.file_bytes+=b.file_bytes;
    return a;
}
GenerationAccounting operator-(GenerationAccounting a,const GenerationAccounting& b) {
    a.wait-=b.wait; a.pool-=b.pool; a.host-=b.host; a.plan-=b.plan; a.actq-=b.actq; a.jobs-=b.jobs; a.run-=b.run;
    a.misses-=b.misses; a.entries-=b.entries; a.hits-=b.hits; a.pcie-=b.pcie;
    a.ram-=b.ram; a.files-=b.files; a.lookups-=b.lookups; a.offload-=b.offload; a.file_bytes-=b.file_bytes;
    return a;
}
size_t GenerationDriver::bytes() const {
    size_t n=sizeof(*this); add(n,finish.capacity()); count(n,prompt);
    for (const auto* v : {&consumed,&drafts,&window,&output,&suffix_buffer,&chain_buffer,&chain_tail,
                         &history_stage,&output_remainder,&mrope}) count(n,*v);
    count(n,draft_probabilities); count(n,images); count(n,image_rows); count(n,image_row_offsets);
    count(n,prefill.cuts); count(n,prefill.geometry);
    for (const auto& loan:prefill.geometry) count(n,loan.slots);
    if (suffix) add(n,suffix->owned_bytes());
    return n;
}

bool GenerationDriver::validate(std::string& error) const {
    if (version!=schema_version || (phase!=GenerationPhase::prefill && phase!=GenerationPhase::decode) ||
        original_prompt_len<=0 || uint64_t(original_prompt_len)!=prompt.size() || max_context<=0 ||
        max_new<=0 || max_new>max_context || original_prompt_len>max_context-max_new ||
        produced<0 || produced>=max_new || position<0 || position>=max_context ||
        window_size<1 || window_size>8 || mtp_window_size<1 || mtp_window_size>window_size ||
        history_size<0 || history_size>4096 || history_size>sampling.penalty_last_n ||
        history_stage.size()<uint64_t(window_size*history_size) || !sample_valid(sampling) ||
        !std::isfinite(pcie_fraction) || pcie_fraction<0 || pcie_fraction>1 ||
        !std::isfinite(spec_min_p) || spec_min_p<0 || spec_min_p>1 || cancelled ||
        output_cursor!=output_remainder.size() || !suffix || !policy)
        return fail(error,"generation driver manifest/continuation mismatch");
    if (std::any_of(prompt.begin(),prompt.end(),[](int64_t id){return id<0 || id>INT32_MAX;}) ||
        std::any_of(consumed.begin(),consumed.end(),[](int32_t id){return id<0;}))
        return fail(error,"generation token identity invalid");
    if (phase==GenerationPhase::decode) {
        if (uint64_t(position)!=consumed.size() || next_token<0 ||
            position!=original_prompt_len-1+produced ||
            drafts.size()<uint64_t(mtp_window_size) || draft_probabilities.size()!=drafts.size() ||
            window.size()!=uint64_t(window_size) || output.size()!=window.size())
            return fail(error,"generation next-unfed-token boundary mismatch");
    } else {
        if (!prefill.returned || !prefill.prefetch_drained || !prefill.loans_returned ||
            prefill.begin<0 || prefill.end<=prefill.begin || prefill.end-prefill.begin>2048 || prefill.end>original_prompt_len-1 ||
            prefill.offset!=prefill.end || position!=prefill.end ||
            consumed.size()!=uint64_t(position) || prefill.chunk<=0)
            return fail(error,"prefill has not returned/drained at a complete segment");
        const std::array<int32_t,2> prev{position>1 ? consumed[size_t(position-2)] : -1,
                                        position>0 ? consumed[size_t(position-1)] : -1};
        if (prev!=prefill.published_ple) return fail(error,"stale prefill PLE predecessor");
    }
    if (!std::equal(consumed.begin(),consumed.begin()+std::min(consumed.size(),prompt.size()-1),prompt.begin()))
        return fail(error,"generation committed prefix differs from prompt");
    int64_t last=-1;
    for (const auto& image:images) {
        if (image.start<0 || image.start<=last || image.start>=original_prompt_len)
            return fail(error,"generation image identity invalid");
        last=image.start;
    }
    if (!image_rows.empty()) {
        if (image_row_offsets.size()!=prompt.size() || images.empty() || mrope.empty())
            return fail(error,"generation image row manifest invalid");
        for (int64_t off:image_row_offsets)
            if (off<-1 || (off>=0 && uint64_t(off)>=image_rows.size()))
                return fail(error,"generation image row offset invalid");
    } else if (std::any_of(image_row_offsets.begin(),image_row_offsets.end(),[](int64_t off){return off!=-1;}))
        return fail(error,"generation image rows missing");
    if (!mrope.empty() && mrope.size()%3) return fail(error,"generation M-RoPE geometry invalid");
    return true;
}

bool GenerationDriver::next_window(int32_t token,int committed_rows,std::string& error) {
    if (phase!=GenerationPhase::decode || committed_rows<1 || committed_rows>window_size || token<0 ||
        position>max_context-committed_rows || consumed.size()!=uint64_t(position+committed_rows))
        return fail(error,"cannot normalize generation window");
    next_token=token; position+=committed_rows; sampling.counter=uint64_t(position);
    return true;
}

bool GenerationDriver::prefill_return(int64_t end,std::array<int32_t,2> published,
        bool drained,bool loans_returned,std::string& error) {
    // Validate without altering the active driver on refusal.
    if (phase!=GenerationPhase::prefill || end!=prefill.end || end<=prefill.begin ||
        end>=original_prompt_len || end-prefill.begin>2048 || !drained || !loans_returned)
        return fail(error,"unsafe prefill return boundary");
    const std::array<int32_t,2> expected{end>1?int32_t(prompt[size_t(end-2)]):-1,
                                       end>0?int32_t(prompt[size_t(end-1)]):-1};
    if (published!=expected) return fail(error,"stale prefill PLE predecessor");
    std::vector<int32_t> prefix(prompt.begin(),prompt.begin()+end);
    consumed=std::move(prefix); position=end;
    prefill.offset=end; prefill.published_ple=published;
    prefill.returned=true; prefill.prefetch_drained=drained; prefill.loans_returned=loans_returned;
    return true;
}

GenerationLookupLease::GenerationLookupLease(GenerationDriver& d,spec::SuffixDrafter& s,
        spec::DraftPolicy& p,int64_t& rounds,int min_match,size_t capacity)
        :driver_(d),suffix_(s),policy_(p),rounds_(rounds) {
    if (!s.allocated()) s=spec::SuffixDrafter(min_match,64,capacity);
    d.policy.emplace(p); // bounded CPU policy; a parked branch must not learn from Hermes's rounds
    d.suffix.emplace(std::move(s));
    d.rounds=rounds;
}
GenerationLookupLease::~GenerationLookupLease() {
    if (driver_.phase==GenerationPhase::parked) return;
    if (driver_.suffix) suffix_=std::move(*driver_.suffix);
    if (driver_.policy) policy_=*driver_.policy;
    rounds_=driver_.rounds;
}

size_t CapsuleSpill::bytes() const {
    // The retained part is RAM: the manifest itself plus one digest per stored block and the per-layer geometry.
    size_t n=sizeof(*this)+sizeof(manifest);
    for (const auto& layer:manifest.main)
        for (const auto& extent:layer.extents) count(n,extent.blocks);
    if (manifest.draft)
        for (const auto& extent:manifest.draft->extents) count(n,extent.blocks);
    return n;
}

GenerationCapsule::GenerationCapsule(GenerationCapsule&& other) noexcept { *this=std::move(other); }
GenerationCapsule& GenerationCapsule::operator=(GenerationCapsule&& other) noexcept {
    if (this==&other) return *this;
    version=other.version; saved_phase=other.saved_phase;
    driver=std::move(other.driver); main=std::move(other.main); mtp=std::move(other.mtp);
    spill=std::move(other.spill);   // its assignment closes any descriptor this capsule held
    other.spill.reset();            // a moved-from capsule owns nothing, so it reports no bytes
    admitted_bytes=std::exchange(other.admitted_bytes,0); occupied=std::exchange(other.occupied,false);
    return *this;
}
std::optional<CapsuleSpill> GenerationCapsule::release_spill() {
    std::optional<CapsuleSpill> out=std::move(spill);
    spill.reset();
    return out;
}
size_t GenerationCapsule::bytes() const {
    size_t n=sizeof(*this); add(n,driver.bytes()); add(n,main.bytes());
    if (mtp) add(n,mtp->bytes());
    if (spill) add(n,spill->bytes());   // the manifest only: a spilled payload is not resident
    return n;
}
bool generation_memory_admit(std::optional<uint64_t> available,uint64_t allocation,
        uint64_t retained,std::string& error) {
    if (retained>capsule_limit || !core::conversation_memory_admit(available,allocation,capsule_free_floor))
        return fail(error,"generation capsule RAM admission refused");
    return true;
}
bool generation_memory_postcheck(std::optional<uint64_t> available,uint64_t retained,std::string& error) {
    return generation_memory_admit(available,0,retained,error);
}
bool generation_publish(GenerationCapsule& out,GenerationDriver& driver,
        std::vector<core::ConversationCheckpoint>& checks,core::SavedConversation&& main,
        std::optional<core::MtpRequestState>&& mtp,size_t bytes,std::optional<uint64_t> available,
        std::string& error,std::optional<CapsuleSpill>&& spill) {
    if (out.occupied || !driver.validate(error) || main.live.ids!=driver.consumed ||
        !main.checkpoints.empty() || mtp.has_value()!=driver.uses_mtp ||
        !generation_memory_postcheck(available,bytes,error)) return false;
    // A disk capture is part of the same transaction: prove it here, while nothing has moved yet, so a broken file
    // refuses the publish instead of leaving a half-published capsule.
    if (spill) {
        if (!spill->valid()) return fail(error,"generation capsule spill has no open file");
        if (!core::spill_manifest_check(spill->manifest,error)) return false;
        if (spill->manifest.payload_bytes==0) return fail(error,"generation capsule spill is empty");
        if (!main.kv.empty()) return fail(error,"generation capsule spill and resident K/V cannot both be set");
    }
    out.saved_phase=driver.phase;
    main.checkpoints=std::move(checks);
    out.main=std::move(main); out.mtp=std::move(mtp); out.driver=std::move(driver);
    out.spill=std::move(spill);   // moved only after every check, and before the capsule counts as occupied
    out.admitted_bytes=bytes; out.occupied=true; driver.phase=GenerationPhase::parked;
    return true;
}
void generation_take(GenerationCapsule& image,GenerationDriver& driver,
        std::vector<core::ConversationCheckpoint>& checks) noexcept {
    checks=std::move(image.main.checkpoints); driver=std::move(image.driver);
    image=GenerationCapsule{};
}
} // namespace strata::program
