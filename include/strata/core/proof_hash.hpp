#pragma once
// Diagnostic equality fingerprint only, not authentication. Fields are encoded
// explicitly, little endian, with lengths; never hash object padding/pointers.
#include <bit>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>
#include <array>
#include "strata/kernels/sampler.hpp"
namespace strata::core {
struct ProofHash {
    uint64_t value=1469598103934665603ull, bytes=0;
    void raw(const void* p,size_t n) {
        const auto* b=static_cast<const uint8_t*>(p);
        for(size_t i=0;i<n;++i) { value^=b[i]; value*=1099511628211ull; }
        bytes+=n;
    }
    template<class T> void add(T x) requires (std::is_integral_v<T> || std::is_enum_v<T>) {
        uint64_t v=static_cast<uint64_t>(x);
        for(int i=0;i<8;++i) { const uint8_t b=uint8_t(v); raw(&b,1); v>>=8; }
    }
    void add(float x) { add(std::bit_cast<uint32_t>(x)); }
    void add(double x) { add(std::bit_cast<uint64_t>(x)); }
    void add(const std::string& x) { add(x.size()); raw(x.data(),x.size()); }
    void add(const std::vector<uint8_t>& v) { add(v.size());raw(v.data(),v.size()); }
    template<class T> void add(const std::vector<T>& v) { add(v.size()); for(const auto& x:v) add(x); }
    template<class T,size_t N> void add(const std::array<T,N>& v) { add(N); for(const auto& x:v) add(x); }
    void add(const kernels::SamplerParams& p) {
        add(p.top_k); add(p.top_p); add(p.min_p); add(p.temperature); add(p.min_keep);
        add(p.penalty_last_n); add(p.penalty_repeat); add(p.penalty_freq); add(p.penalty_present);
        add(p.seed); add(p.counter); add(p.greedy); add(p.gumbel);
    }
};
} // namespace strata::core
