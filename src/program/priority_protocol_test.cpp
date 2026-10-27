#include "strata/program/priority_protocol.hpp"
// This standalone parser test must retain assertions in a Release CMake build.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
int main() {
    using namespace strata::program;
    PriorityCommand c;
    assert(priority_command("PGEN 42 9 seed=42 1,2,3",c));
    assert(c.id==42 && c.generation=="GEN 9 seed=42 1,2,3");
    assert(priority_command("PGENI 43 9 /private/image 1,2",c));
    assert(c.generation=="GENI 9 /private/image 1,2");
    assert(priority_command("PRESUME 42 backfill=1",c) && c.backfill);
    assert(priority_command("PRESUME 42",c) && !c.backfill);
    assert(priority_command("PPAUSE 9223372036854775807",c));
    assert(priority_command("PCANCEL 1",c));
    for (const char* s:{"PPAUSE 0","PPAUSE -1","PPAUSE +1","PPAUSE 01","PPAUSE 1x",
        "PPAUSE 9223372036854775808","PPAUSE 1 extra","PCANCEL 1 backfill=1",
        "PRESUME 1 backfill=0","PRESUME 1 backfill=1 extra","PGEN 1","PGEN 1 ","STOP",
        "PGEN 1 2 1\nPCANCEL 1"}) assert(!priority_command(s,c));
}
