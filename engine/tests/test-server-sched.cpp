// KV budget admission and victim rules of llama-server (tools/server/server-sched.h)

#include "server-sched.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

using slot_info = server_kv_budget::slot_info;
using candidate = server_kv_budget::candidate;

static server_kv_budget make_budget(int32_t n_cells, int32_t headroom, std::vector<slot_info> slots) {
    server_kv_budget b;
    b.n_cells  = n_cells;
    b.headroom = headroom;
    b.slots    = std::move(slots);
    return b;
}

static void test_reservation() {
    // bounded request: prompt + max_tokens + draft window + the sampled token
    CHECK(server_kv_budget::reservation(1000, 400, 8192, 3, 100000) == 1000 + 400 + 3 + 1);
    // max_tokens above the cap reserves the cap, the rest grows on demand
    CHECK(server_kv_budget::reservation(1000, 20000, 8192, 0, 100000) == 1000 + 8192 + 1);
    // unlimited generation reserves the cap
    CHECK(server_kv_budget::reservation(1000, -1, 8192, 3, 100000) == 1000 + 8192 + 3 + 1);
    // never more than a slot can hold
    CHECK(server_kv_budget::reservation(99000, 4096, 8192, 3, 100000) == 100000);
    // resuming a request with nothing left to generate still covers the next step
    CHECK(server_kv_budget::reservation(500, 0, 8192, 3, 100000) == 500 + 3 + 1);
}

static void test_committed() {
    // a running slot holds its reservation even before its cells are used
    CHECK(server_kv_budget::committed({ 100, 5000, true,  0 }) == 5000);
    // and may have outgrown it
    CHECK(server_kv_budget::committed({ 6000, 5000, true, 0 }) == 6000);
    // an idle slot only holds its cells, a stale reservation does not count
    CHECK(server_kv_budget::committed({ 100, 5000, false, 0 }) == 100);
}

static void test_plan_fits_without_parking() {
    auto b = make_budget(10000, 100, {
        { 3000, 4000, true,  1 },
        { 2000,    0, false, 2 },
        {    0,    0, false, 3 },
    });
    // 4000 + 2000 committed, 3900 left after headroom
    auto p = b.plan_commit(2, 3900, true);
    CHECK(p.fits && p.victims.empty());

    p = b.plan_commit(2, 3901, true);
    CHECK(p.fits && p.victims.size() == 1 && p.victims[0] == 1);

    // growth of running work may use the headroom
    p = b.plan_commit(2, 4000, false);
    CHECK(p.fits && p.victims.empty());
}

static void test_plan_victims_lru_and_minimal() {
    auto b = make_budget(10000, 0, {
        { 2000, 0, false, 30 },   // most recent
        { 2000, 0, false, 10 },   // least recent
        { 2000, 0, false, 20 },
        { 1000, 5000, true, 5 },  // running: never a victim, even if least recent
        {    0, 0, false, 1 },    // target
    });
    // committed 2000*3 + 5000 = 11000 > 10000 already; need 1000 -> must free 2000
    auto p = b.plan_commit(4, 1000, false);
    CHECK(p.fits);
    CHECK(p.victims.size() == 1 && p.victims[0] == 1);

    // need 3000 -> free 4000: the two least recent idle slots, in LRU order
    p = b.plan_commit(4, 3000, false);
    CHECK(p.fits);
    CHECK(p.victims.size() == 2 && p.victims[0] == 1 && p.victims[1] == 2);

    // 5000 needed, parking all three idle slots leaves exactly 5000
    p = b.plan_commit(4, 5000, false);
    CHECK(p.fits && p.victims.size() == 3);
}

static void test_plan_never_parks_in_vain() {
    auto b = make_budget(10000, 0, {
        { 1000, 0, false, 1 },
        { 1000, 6000, true, 2 },
        { 0, 0, false, 3 },
    });
    // even parking slot 0 leaves only 4000 free
    auto p = b.plan_commit(2, 4500, false);
    CHECK(!p.fits);
    CHECK(p.victims.empty());
}

static void test_plan_target_cells_are_replaced() {
    // the target's own cells do not count, the new commitment replaces them
    auto b = make_budget(10000, 0, {
        { 9000, 0, false, 1 },
        { 1000, 0, false, 2 },
    });
    auto p = b.plan_commit(0, 9000, false);
    CHECK(p.fits && p.victims.empty());

    // an idle slot with no cells is never a victim
    b = make_budget(1000, 0, {
        { 0, 0, false, 1 },
        { 0, 0, false, 2 },
    });
    p = b.plan_commit(0, 1001, false);
    CHECK(!p.fits && p.victims.empty());
}

static void test_plan_lone_request_uses_everything() {
    // with nothing running the caller drops the headroom: a prompt of almost the full cache is admitted
    auto b = make_budget(10000, 256, {
        { 5000, 0, false, 1 },
        { 0, 0, false, 2 },
    });
    auto p = b.plan_commit(1, 10000, false);
    CHECK(p.fits && p.victims.size() == 1 && p.victims[0] == 0);
    p = b.plan_commit(1, 10000, true);
    CHECK(!p.fits);
}

static void test_pick_slot() {
    const float sim = 0.1f;

    // the idle slot holding the longest prefix wins, busy slots are ignored even with a longer prefix
    {
        std::vector<candidate> c = {
            { true,  50000, 40000, 5 },
            { false, 90000, 0,     1 },
            { true,  60000, 55000, 9 },
            { true,  0,     0,     0 },
        };
        CHECK(server_kv_budget::pick_slot(c, 60000, sim) == 2);
    }
    // a shared prefix below the threshold (e.g. only the system prompt) is not affinity:
    // take the empty slot rather than parking someone's session
    {
        std::vector<candidate> c = {
            { true, 120000, 7000, 1 },
            { true, 0,      0,    5 },
        };
        CHECK(server_kv_budget::pick_slot(c, 130000, sim) == 1);
    }
    // the threshold is strict
    {
        std::vector<candidate> c = {
            { true, 100, 10, 1 },
            { true, 100, 0,  0 },
        };
        CHECK(server_kv_budget::pick_slot(c, 100, sim) == 1); // LRU, 10 is not > 0.1 * 100
    }
    // no empty slot: least recently used
    {
        std::vector<candidate> c = {
            { true,  1000, 0, 30 },
            { true,  1000, 0, 10 },
            { false, 1000, 0, 1 },
        };
        CHECK(server_kv_budget::pick_slot(c, 500, sim) == 1);
    }
    // several empty slots: least recently used of them
    {
        std::vector<candidate> c = {
            { true, 0, 0, 30 },
            { true, 0, 0, 10 },
        };
        CHECK(server_kv_budget::pick_slot(c, 500, sim) == 1);
    }
    // nothing idle
    {
        std::vector<candidate> c = {
            { false, 1000, 900, 1 },
        };
        CHECK(server_kv_budget::pick_slot(c, 1000, sim) == -1);
    }
}

int main() {
    test_reservation();
    test_committed();
    test_plan_fits_without_parking();
    test_plan_victims_lru_and_minimal();
    test_plan_never_parks_in_vain();
    test_plan_target_cells_are_replaced();
    test_plan_lone_request_uses_everything();
    test_pick_slot();

    if (n_fail > 0) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("all server-sched tests passed\n");
    return 0;
}
