#pragma once
#include <cstdint>
#include <string>
#include <string_view>

namespace strata::program {
enum class PriorityVerb { generate, image, pause, resume, cancel };
struct PriorityCommand {
    PriorityVerb verb{};
    uint64_t id = 0;
    bool backfill = false;
    std::string generation;
};
// Deliberately strict control grammar: no atoi truncation, signs, zero or
// ambiguous options. Generation payload remains the legacy parser's input.
inline bool priority_command(std::string_view line, PriorityCommand& out) {
    const auto gap=line.find(' ');
    if (gap==line.npos || line.find_first_of("\r\n")!=line.npos) return false;
    const auto verb=line.substr(0,gap);
    PriorityCommand c;
    if (verb=="PGEN") c.verb=PriorityVerb::generate;
    else if (verb=="PGENI") c.verb=PriorityVerb::image;
    else if (verb=="PPAUSE") c.verb=PriorityVerb::pause;
    else if (verb=="PRESUME") c.verb=PriorityVerb::resume;
    else if (verb=="PCANCEL") c.verb=PriorityVerb::cancel;
    else return false;
    const auto end=line.find(' ',gap+1);
    const auto id=line.substr(gap+1,(end==line.npos ? line.size() : end)-gap-1);
    if (id.empty() || id.front()=='0') return false;
    for (char ch:id) {
        if (ch<'0' || ch>'9' || c.id>(uint64_t(INT64_MAX)-uint64_t(ch-'0'))/10) return false;
        c.id=c.id*10+uint64_t(ch-'0');
    }
    if (c.verb==PriorityVerb::generate || c.verb==PriorityVerb::image) {
        if (end==line.npos || end+1==line.size()) return false;
        c.generation=(c.verb==PriorityVerb::image ? "GENI " : "GEN ")+std::string(line.substr(end+1));
    } else if (end!=line.npos) {
        if (c.verb!=PriorityVerb::resume || line.substr(end+1)!="backfill=1") return false;
        c.backfill=true;
    }
    out=std::move(c);
    return true;
}
} // namespace strata::program
