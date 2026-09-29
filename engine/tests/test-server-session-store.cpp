// session identity index of the per-agent session store (tools/server/server-session-store.h)

#include "server-session-store.h"

#include <cassert>
#include <cstdio>
#include <vector>

static std::vector<int32_t> seq(int32_t first, size_t n) {
    std::vector<int32_t> res(n);
    for (size_t i = 0; i < n; ++i) {
        res[i] = first + (int32_t) i;
    }
    return res;
}

static std::vector<int32_t> cat(std::vector<int32_t> a, const std::vector<int32_t> & b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

// the longest shared prefix wins, across block boundaries and within a block
static void test_longest_prefix() {
    server_session_index index;
    const auto sys = seq(1, 1000);
    index.update(1, cat(sys, seq(100000, 3000)));
    index.update(2, cat(sys, seq(200000, 700)));

    auto m = index.find(cat(cat(sys, seq(100000, 3000)), seq(7, 50)));
    assert(m && m->id == 1 && m->n_common == 4000);

    m = index.find(cat(sys, seq(200000, 10)));
    assert(m && m->id == 2 && m->n_common == 1010);

    // a prompt sharing only the system prompt still reports the best candidate with its common length
    m = index.find(cat(sys, seq(300000, 500)));
    assert(m && m->n_common == 1000);

    assert(!index.find(seq(900000, 300)));
}

// updating replaces the tokens, erasing removes the session from every node
static void test_update_erase() {
    server_session_index index;
    index.update(1, seq(1, 2048));
    index.update(1, seq(5000, 2048));
    assert(!index.find(seq(1, 600)));
    auto m = index.find(seq(5000, 600));
    assert(m && m->id == 1 && m->n_common == 600);

    index.erase(1);
    assert(index.size() == 0 && !index.find(seq(5000, 600)));
}

// the candidates are compared token by token: equal block hashes never join different tokens
static void test_tie_most_recent() {
    server_session_index index;
    index.update(1, seq(1, 512));
    index.update(2, seq(1, 512));
    auto m = index.find(seq(1, 512));
    assert(m && m->id == 2);
    index.update(1, seq(1, 512));
    m = index.find(seq(1, 512));
    assert(m && m->id == 1);
}

int main() {
    test_longest_prefix();
    test_update_erase();
    test_tie_most_recent();
    printf("test-server-session-store: OK\n");
    return 0;
}
