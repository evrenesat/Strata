#include "strata/program/generation_proof.hpp"
#include <cstdio>
#include <string>
#include <utility>
#include <cstdlib>
using namespace strata::program;
int checks=0;
void check(bool v) { ++checks; if(!v) { std::fprintf(stderr,"FAIL\n"); std::exit(1); } }
int main() {
    ProofCommand c;
    check(proof_command("PTEST RESET",c) && c.kind==ProofCommand::reset);
    check(proof_command("PTEST FAULT 7 restore_after_main",c) && c.id==7 && c.injection==ProofFault::restore_after_main);
    check(proof_command("PTEST PAUSE 2 prefill 245759",c) && c.minimum==245759);
    // The parked-capsule disk faults are armed by the same trusted control line.
    for(const auto& [name,fault]:std::initializer_list<std::pair<const char*,ProofFault>>{
            {"spill_ram_admission",ProofFault::spill_ram_admission},
            {"spill_write",ProofFault::spill_write},{"spill_truncate",ProofFault::spill_truncate},
            {"spill_corrupt_first",ProofFault::spill_corrupt_first},
            {"spill_corrupt_late",ProofFault::spill_corrupt_late},
            {"spill_source_copy",ProofFault::spill_source_copy}}) {
        const std::string line=std::string("PTEST FAULT 9 ")+name;
        const bool parsed=proof_command(line,c) && c.kind==ProofCommand::fault && c.id==9 && c.injection==fault;
        if(!parsed) { std::fprintf(stderr,"FAIL: %s\n",name); std::exit(1); }
    }
    for(const char* bad:{"PTEST FAULT 0 capture_admission","PTEST FAULT +1 capture_postcheck",
        "PTEST FAULT 9223372036854775808 restore_validation","PTEST RESET extra",
        "PTEST PAUSE 1 decode -1","PTEST FAULT 1 unknown","PTEST PAUSE 1 decode 1 extra"})
        check(!proof_command(bad,c));
    GenerationProof proof;
    proof.fault=ProofFault::capture_postcheck;proof.fault_id=7;
    check(!proof.take(ProofFault::capture_postcheck,8));check(proof.take(ProofFault::capture_postcheck,7));
    check(!proof.take(ProofFault::capture_postcheck,7));
    GenerationDriver d;
    d.request_id=7;d.prompt={1,2,3};d.next_token=3;d.sampling.seed=424242;
    d.suffix.emplace();d.suffix->append(3);d.policy.emplace(4);
    auto h=proof_driver(d);
    d.prompt.reserve(1000);check(proof_driver(d)==h); // capacities and addresses are not state
    d.began=std::chrono::steady_clock::now();d.paused_decode_ms=100;d.verify_ms=200;
    check(proof_driver(d)==h); // wall-clock accounting is checked separately
    d.next_token=4;check(proof_driver(d)!=h);d.next_token=3;
    d.sampling.counter=1;check(proof_driver(d)!=h);d.sampling.counter=0;
    d.draft_probabilities={.5f};check(proof_driver(d)!=h);d.draft_probabilities.clear();
    d.mrope={0,1,2};check(proof_driver(d)!=h);d.mrope.clear();
    d.suffix->append(4);check(proof_driver(d)!=h);
    h=proof_driver(d);d.policy->observe(false,4,2,0,10);check(proof_driver(d)!=h);
    std::vector<strata::core::ConversationCheckpoint> cp(1);cp[0].ids={1};cp[0].gdn={2};
    const auto ch=proof_checkpoints(cp);cp[0].gdn[0]=3;check(proof_checkpoints(cp)!=ch);
    cp[0].gdn[0]=2;cp[0].pinned=true;check(proof_checkpoints(cp)!=ch);
    strata::core::ProofHash a,b;
    strata::kernels::SamplerParams p{};a.add(p);b.add(p);check(a.value==b.value);
    p.seed++;strata::core::ProofHash changed;changed.add(p);check(a.value!=changed.value);
    std::puts("generation proof: strict trusted controls, canonical state and fault ownership passed");
}
