// Disk tier of the session store (tools/server/server-sessions.cpp): the sessions of queued requests come back from
// disk in queue order and stay in the arena until they are admitted.
//
// 6 x 150k replay of 2026-09-29 (bench/results/c3-seats): the arena held one parked mirror next to the resident
// ones. poll() promoted the head, then promoted the next in line, whose make_room() demoted the head (its file was
// current, so the demotion was instant); the next poll() promoted the head again, which demoted the next in line,
// and so on: 691 promotions (3.8 TB read) in 35 minutes, and the head was never in the arena when admission ran, so
// the GPU sat idle for up to 23 s with requests waiting.
//
// Here: one resident mirror and room for one more; two parked sessions on disk are queued. The head must be promoted
// once and stay; the next in line must wait on disk (it cannot push the head out).

#include "server-sessions.h"

#include "ggml.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <thread>

#include <unistd.h>

static constexpr size_t CHUNK = 1 << 20;

static void fill(common_state_arena & arena, server_session & s, size_t n_chunks) {
    const size_t n = n_chunks * CHUNK;
    GGML_ASSERT(s.log.grow(arena, n));
    s.log.size = n;
    std::vector<llama_state_chunk> v;
    s.log.views(arena, 0, n, v);
    for (auto & c : v) {
        memset(c.data, (int) s.id, c.size);
    }
    s.tokens.assign(16, (llama_token) s.id);
    s.n_kv = 16;
    s.t_used_us = ggml_time_us();
}

static bool wait_tier(server_session_store & store, server_session & s, server_session_tier tier,
        const std::vector<uint64_t> & wanted) {
    for (int i = 0; i < 2000 && s.tier != tier; i++) {
        store.poll(wanted);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return s.tier == tier;
}

int main() {
    ggml_time_init();

    const auto dir = std::filesystem::temp_directory_path() / ("test-server-session-tiers-" + std::to_string(getpid()));
    std::filesystem::remove_all(dir);

    int rc = 0;
    {
        common_state_arena arena(16 * CHUNK, CHUNK);
        GGML_ASSERT(arena.n_total() == 16);

        server_session_store::params_t params;
        params.dir = dir.string();
        server_session_store store(arena, nullptr, nullptr, params, nullptr);

        // two parked sessions, demoted to disk
        auto & a = store.create(1);
        auto & b = store.create(2);
        fill(arena, a, 6);
        fill(arena, b, 6);
        GGML_ASSERT(store.make_room(16, /* wait */ true));
        GGML_ASSERT(wait_tier(store, a, SERVER_SESSION_DISK, {}) && wait_tier(store, b, SERVER_SESSION_DISK, {}));

        // a resident session: 6 chunks, 10 left, room for one parked mirror
        auto & r = store.create(3);
        fill(arena, r, 6);
        r.id_slot = 0;

        const uint64_t n_promote0 = store.n_promote_total;
        const std::vector<uint64_t> wanted = { a.id, b.id }; // a is the head, b next in line

        // the server loop polls the store every iteration
        for (int i = 0; i < 1000; i++) {
            store.poll(wanted);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        const uint64_t n_promote = store.n_promote_total - n_promote0;
        printf("head: %s, next: %s, promotions: %llu\n", server_session_tier_name(a.tier), server_session_tier_name(b.tier),
                (unsigned long long) n_promote);
        if (a.tier != SERVER_SESSION_RAM) {
            fprintf(stderr, "FAIL: the head is not in the arena\n");
            rc = 1;
        }
        if (b.tier != SERVER_SESSION_DISK) {
            fprintf(stderr, "FAIL: the next in line pushed the head out\n");
            rc = 1;
        }
        if (n_promote != 1) {
            fprintf(stderr, "FAIL: %llu promotions, expected 1 (queued sessions evict each other)\n", (unsigned long long) n_promote);
            rc = 1;
        }

        // the resident session leaves and the head is gone from the queue: the next in line comes back
        r.id_slot = -1;
        store.erase(r.id);
        if (!wait_tier(store, b, SERVER_SESSION_RAM, { b.id })) {
            fprintf(stderr, "FAIL: the new head was not promoted\n");
            rc = 1;
        }

        // the old head changed (a turn ran) and is parked again, not queued; the arena has no room for another
        // mirror: the store writes it ahead, it stays in the arena, and making room then needs no write
        a.file_current = false;
        for (int i = 0; i < 2000 && !a.file_current; i++) {
            store.poll({ b.id });
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!a.file_current || a.tier != SERVER_SESSION_RAM) {
            fprintf(stderr, "FAIL: the next victim was not written ahead (tier %s)\n", server_session_tier_name(a.tier));
            rc = 1;
        }
        if (!store.make_room(10, /* wait */ false) || a.tier != SERVER_SESSION_DISK) {
            fprintf(stderr, "FAIL: making room waited for a write\n");
            rc = 1;
        }
    }

    // 6 x 150k replay with the fix above: the third in line was promoted into the arena reserve, and the reserve
    // demoted it right away (it is not protected), every 0.75 s. Only the head and the next in line come back.
    {
        common_state_arena arena(20 * CHUNK, CHUNK);
        server_session_store::params_t params;
        params.dir     = dir.string();
        params.reserve = 4 * CHUNK;
        server_session_store store(arena, nullptr, nullptr, params, nullptr);

        auto & c = store.create(13);
        fill(arena, c, 6);
        GGML_ASSERT(store.make_room(20, /* wait */ true));
        GGML_ASSERT(wait_tier(store, c, SERVER_SESSION_DISK, {}));
        auto & a = store.create(11);
        auto & b = store.create(12);
        fill(arena, a, 6);
        fill(arena, b, 6);

        const uint64_t n_promote0 = store.n_promote_total;
        for (int i = 0; i < 500; i++) {
            store.poll({ a.id, b.id, c.id });
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const uint64_t n_promote = store.n_promote_total - n_promote0;
        printf("third in line: %s, promotions: %llu\n", server_session_tier_name(c.tier), (unsigned long long) n_promote);
        if (n_promote != 0 || a.tier != SERVER_SESSION_RAM || b.tier != SERVER_SESSION_RAM) {
            fprintf(stderr, "FAIL: the third in line was read back (%llu promotions)\n", (unsigned long long) n_promote);
            rc = 1;
        }
    }
    std::filesystem::remove_all(dir);

    if (rc == 0) {
        printf("PASS\n");
    }
    return rc;
}
