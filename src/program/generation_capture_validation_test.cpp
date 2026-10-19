// Native API rejection fixtures: no GPU initialization or model assets required.
// Full transfer/readback tests use the guarded native CP5 build.
#include "strata/program/generation_capture.hpp"
#include <cstdio>
#include <cstdlib>
using namespace strata::program;
int main() {
    strata::core::SessionState ss;
    strata::core::ModelGeometry g;
    GenerationRuntime runtime{ss,g};
    GenerationDriver d;
    GenerationCapsule image;
    std::vector<strata::core::ConversationCheckpoint> checks;
    std::string error;
    auto check=[](bool ok) { if (!ok) std::exit(1); };
    check(generation_capture(image,d,checks,runtime,error)==GenerationTransfer::refused);
    check(!image.occupied && !runtime.poisoned && checks.empty());
    check(generation_restore(image,d,checks,runtime,error)==GenerationTransfer::refused);
    strata::core::MtpDrafter unbound;
    size_t bytes=0;
    check(!unbound.request_bytes(32,bytes,error));
    strata::core::MtpRequestState mtp;
    check(!unbound.request_validate(mtp,73,32,error));
    check(!runtime.poisoned);
    std::puts("generation native validation: invalid state refused without CUDA");
}
