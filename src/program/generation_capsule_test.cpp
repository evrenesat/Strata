#include "strata/program/generation_capsule.hpp"
#include "strata/core/coupled_draft.hpp"
#include "strata/core/spec_prob.hpp"
#include <cstdio>
#include <cstdlib>
#include <type_traits>

using namespace strata::program;
using namespace strata::core;
namespace {
int checks=0;
void check(bool ok,const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr,"FAIL: %s\n",label); std::exit(1); }
}
GenerationDriver driver() {
    GenerationDriver d;
    d.phase=GenerationPhase::decode; d.model_identity=11; d.config_identity=22;
    d.original_prompt_len=3; d.max_context=100; d.max_new=8; d.position=2;
    d.prompt={1,2,3}; d.consumed={1,2}; d.next_token=3;
    d.sampling.seed=424242; d.sampling.temperature=1; d.sampling.top_p=.95f;
    d.window_size=d.mtp_window_size=4;
    d.drafts={4,5,6,7}; d.draft_probabilities={.9f,.8f,.7f,0};
    d.window.resize(4); d.output.resize(4);
    d.suffix.emplace(1,64,32); d.suffix->append(1); d.suffix->append(2); d.suffix->append(3);
    d.policy.emplace(4);
    return d;
}
MtpRequestState mtp() {
    MtpRequestState s;
    s.geometry={4,8,2,32,100,32,1000,20,0,0,0,1,1};
    s.prompt_len=73; s.kv_extent=16; s.max_drafts=3; s.coupled=true; s.sampling.seed=424242;
    s.host_tokens.resize(4); s.device_tokens.resize(4);
    s.host_steps.resize(32); s.device_steps.resize(32);
    s.host_positions.resize(16); s.device_positions.resize(16);
    s.host_row.resize(2); s.device_row.resize(4);
    s.host_out.resize(4); s.device_out.resize(8);
    s.host_prob.resize(4); s.device_prob.resize(8);
    s.input_R.resize(32); s.residual_R.resize(32); s.window_R.resize(32);
    s.history.resize(kCoupledHistCap); s.ring.resize(kCoupledHistCap+4); s.q.resize(4*kSpecQStride);
    s.top2={10,11,12,13}; s.debug_logits.resize(20);
    s.kv.k.resize(64,7); s.kv.v.resize(64,9);
    return s;
}
void boundaries() {
    std::string e;
    auto d=driver(); check(d.validate(e),"initial next-unfed boundary");
    d.consumed.insert(d.consumed.end(),{3,4}); d.produced=2;
    check(d.next_window(5,2,e),"normalize after whole committed window before interrupt");
    check(d.position==4 && d.next_token==5 && d.sampling.counter==4 && d.validate(e),"position/draw/budget agree");
    d.output_remainder={4,5}; d.output_cursor=1;
    check(!d.validate(e),"partial committed emission cannot silently disappear");
    d.output_cursor=2; check(d.validate(e),"fully emitted remainder retained");
    d.sampling.seed=0; check(!d.validate(e),"unresolved seed rejected");
    d=driver(); d.version++; check(!d.validate(e),"driver schema mismatch rejected");
    d=driver(); d.next_token=-1; check(!d.validate(e),"invalid next token rejected");
    d=driver(); d.produced=d.max_new; check(!d.validate(e),"terminal budget cannot be parked");

    d=driver(); d.phase=GenerationPhase::prefill;
    d.prompt={1,2,3,4,5,6,7,8,9,10}; d.original_prompt_len=10;
    d.prefill.begin=0; d.prefill.end=5; d.prefill.chunk=4;
    const auto old=d.consumed; const auto p=d.position;
    check(!d.prefill_return(5,{-1,-1},true,true,e),"on_chunk stale PLE refused");
    check(d.consumed==old && d.position==p && !d.prefill.returned,"stale callback refusal is non-mutating");
    check(!d.prefill_return(5,{4,5},false,true,e),"live prefetch refused");
    check(!d.prefill_return(5,{4,5},true,false,e),"borrowed expert buffers refused");
    check(d.prefill_return(5,{4,5},true,true,e) && d.validate(e),"complete return publishes exact prefix");
    check(d.original_prompt_len==10,"prefill prefix never replaces original full prompt length");
    d.prefill.returned=false; check(!d.validate(e),"callback progress cannot forge completed return");
}
void ownership() {
    std::string e;
    auto d=driver(); auto* tokens=d.consumed.data(); auto* drafts=d.drafts.data();
    std::vector<ConversationCheckpoint> cp(1); cp[0].ids={1}; cp[0].gdn.resize(32,7);
    auto* state=cp[0].gdn.data();
    SavedConversation main; main.live.ids=d.consumed; main.kv.resize(1); main.kv[0].k.resize(32,13);
    std::optional<MtpRequestState> no_mtp;
    GenerationCapsule parked;
    const auto old=d.consumed;
    check(!generation_publish(parked,d,cp,std::move(main),std::move(no_mtp),4096,std::nullopt,e),"unknown post-capture RAM refuses");
    check(d.consumed==old && d.consumed.data()==tokens && cp[0].gdn.data()==state && !parked.occupied,
          "refusal keeps source buffers and checkpoint owner intact");
    check(generation_publish(parked,d,cp,std::move(main),std::move(no_mtp),4096,capsule_free_floor,e),"publish admitted capsule");
    check(d.phase==GenerationPhase::parked && cp.empty(),"active owner relinquished once");
    check(parked.driver.consumed.data()==tokens && parked.driver.drafts.data()==drafts &&
          parked.main.checkpoints[0].gdn.data()==state,"driver/checkpoints moved without copies");
    GenerationCapsule moved=std::move(parked);
    check(moved.occupied && !parked.occupied,"capsule move has one owner");
    auto fg=driver(); fg.phase=GenerationPhase::terminal;
    generation_take(moved,fg,cp);
    check(!moved.occupied && fg.consumed.data()==tokens && cp[0].gdn.data()==state && fg.validate(e),"take restores same buffers");
    check(generation_memory_admit(capsule_free_floor+4096,4096,capsule_limit,e),"exact admission floor/budget");
    check(!generation_memory_admit(capsule_free_floor+4095,4096,4096,e),"transient copy counted before capture");
    check(!generation_memory_admit(UINT64_MAX,1,capsule_limit+1,e),"capsule budget enforced");
    check(!generation_memory_postcheck(capsule_free_floor-1,4096,e),"concurrent consumer postcheck refuses");

    strata::spec::SuffixDrafter reuse(1,64,32); strata::spec::DraftPolicy policy(4); int64_t rounds=9;
    auto live=driver(); live.suffix.reset(); live.policy.reset();
    { GenerationLookupLease lease(live,reuse,policy,rounds,1,32);
      check(live.suffix && !reuse.allocated() && live.rounds==9,"lookup ownership belongs to active request");
      live.suffix->append(42); live.rounds=10; }
    check(reuse.allocated() && reuse.history().back()==42 && rounds==10,"legacy allocation/policy counter returned");
    GenerationAccounting began,paused,resumed,finished;
    began.hits=10; began.file_bytes=100; paused.hits=15; paused.file_bytes=150;
    resumed.hits=115; resumed.file_bytes=1150; finished.hits=120; finished.file_bytes=1200;
    const auto own=finished-began-(resumed-paused);
    check(own.hits==10 && own.file_bytes==100,"Hermes cumulative work excluded from coding accounting");
}
void mtp_layout() {
    std::string e; auto s=mtp(); const auto geometry=s.geometry;
    check(s.validate_layout(geometry,73,16,e),"complete MTP manifest with lookahead extent validates");
    check(!s.validate_layout(geometry,9,16,e),"original full prompt length required, consumed prefix insufficient");
    check(!s.validate_layout(geometry,73,9,e),"draft lookahead cannot be replaced by main extent");
    s.q.pop_back(); check(!s.validate_layout(geometry,73,16,e),"missing q distribution word rejected");
    s=mtp(); s.residual_R.pop_back(); check(!s.validate_layout(geometry,73,16,e),"missing residual rejected");
    s=mtp(); s.history.clear(); check(!s.validate_layout(geometry,73,16,e),"coupled history ownership required");
    s=mtp(); s.version++; check(!s.validate_layout(geometry,73,16,e),"MTP schema mismatch rejected");
    s=mtp(); MtpRequestState moved=std::move(s);
    std::array<uint8_t,64> bytes{};
    check(moved.kv.k.read(bytes.data(),0,bytes.size()) && bytes[0]==7 && bytes.back()==7 &&
          moved.validate_layout(geometry,73,16,e),"MTP owned KV/staging/residual survive move");
}
}
int main() {
    static_assert(!std::is_copy_constructible_v<GenerationCapsule>);
    static_assert(!std::is_copy_constructible_v<GenerationDriver>);
    static_assert(!std::is_copy_constructible_v<MtpRequestState>);
    static_assert(std::is_nothrow_move_constructible_v<GenerationCapsule>);
    boundaries(); ownership(); mtp_layout();
    std::printf("generation capsule: %d checks passed\n",checks);
}
