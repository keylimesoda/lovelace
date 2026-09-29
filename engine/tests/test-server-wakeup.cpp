// Lost wake-up of KV budget admission (tools/server/server-queue.cpp, tools/server/server-sched.h)
//
// Production hang of 2026-09-29: slot 0 released a 202917-token request and the mirror sync of its session was in
// flight; the next ~200k first contacts waited for KV (376832 cells) and were admissible only once slot 0 could be
// parked, i.e. after the sync. The sync completed while every slot was idle, admission was not retried, and the
// server loop waited for a new task: the requests stalled until an unrelated request arrived 10+ minutes later.
//
// The pass below follows update_slots() with --kv-budget: complete the finished transfers, admit, and with no slot
// running wait for the transfers and admit again. The sync completes without marking the admission gate (an event
// the scheduler misses, as sched_finish_xfer() did for syncs until the fix), and no task ever arrives: the server
// loop (server_queue) and the backstop of the gate (server_sched_gate) must retry the admission by themselves.

#include "server-queue.h"
#include "server-sched.h"

#include "ggml.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

int main() {
    ggml_time_init();

    constexpr int32_t n_cells   = 376832;
    constexpr int32_t n_victim  = 202917; // cells of the idle session of slot 0
    constexpr int64_t t_sync_us = 100000; // the sync completes this long after the start
    constexpr int     t_limit_s = 5;      // the task counts as stalled after this long

    server_kv_budget budget;
    budget.n_cells = n_cells;
    budget.slots.assign(3, {});
    budget.slots[0] = { n_victim, 0, false, 1 };

    const int32_t need = server_kv_budget::reservation(199942, -1, 8192, 3, n_cells);

    server_sched_gate gate;
    gate.backstop_us = 200000;
    gate.mark(); // the task was just queued

    server_queue queue;

    const int64_t t_start = ggml_time_us();

    bool syncing   = true; // slot 0: the mirror sync of its cells is in flight, it cannot be parked yet
    int  n_waiting = 1;

    std::atomic<int64_t> t_sync_done  { 0 };
    std::atomic<int64_t> t_admitted   { 0 };
    std::atomic<int>     n_passes     { 0 };
    std::atomic<int>     n_tasks      { 0 };

    auto sync_complete = [&]() {
        return ggml_time_us() - t_start >= t_sync_us;
    };

    // sched_poll_xfers(): true if a transfer was in progress
    auto poll_xfers = [&](bool wait) {
        if (!syncing) {
            return false;
        }
        while (wait && !sync_complete()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (sync_complete()) {
            syncing = false;
            t_sync_done = ggml_time_us();
        }
        return true;
    };

    // sched_admit() of the queued first contact
    auto admit = [&]() {
        if (!gate.take(ggml_time_us(), n_waiting > 0) || n_waiting == 0) {
            return;
        }
        budget.slots[0].active = syncing; // sched_sync(): a slot with a transfer in flight is not a victim
        const auto plan = budget.plan_commit(1, need, false);
        if (!plan.fits) {
            return;
        }
        for (int id : plan.victims) {
            budget.slots[id].n_used = 0;
        }
        budget.slots[1] = { 0, need, true, 2 };
        n_waiting  = 0;
        t_admitted = ggml_time_us();
    };

    queue.on_new_task([&](server_task &&, bool) {
        n_tasks++;
        return true;
    });
    queue.on_update_slots([&]() {
        n_passes++;
        poll_xfers(false);
        admit();
        const bool all_idle = !budget.slots[1].active;
        if (all_idle && poll_xfers(true)) {
            admit();
        }
        queue.update_slots_after(gate.wait_ms(ggml_time_us(), n_waiting > 0));
    });

    std::thread loop([&]() { queue.start_loop(-1); });

    while (t_admitted.load() == 0 && ggml_time_us() - t_start < t_limit_s * 1000000LL) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    queue.terminate();
    loop.join();

    int n_fail = 0;
    if (t_admitted.load() == 0) {
        fprintf(stderr, "FAIL: the task waiting for KV was not admitted within %d s of the sync (%d passes, %d tasks)\n",
                t_limit_s, n_passes.load(), n_tasks.load());
        n_fail++;
    } else {
        const double gap_ms = (t_admitted.load() - t_sync_done.load()) / 1e3;
        printf("admitted %.1f ms after the sync completed, %d passes, %d tasks\n", gap_ms, n_passes.load(), n_tasks.load());
        if (t_sync_done.load() == 0 || t_admitted.load() < t_sync_done.load()) {
            fprintf(stderr, "FAIL: admitted before the sync completed\n");
            n_fail++;
        }
        if (gap_ms > gate.backstop_us / 1e3 + 100.0) {
            fprintf(stderr, "FAIL: admitted %.1f ms after the sync, the backstop is %.1f ms\n", gap_ms, gate.backstop_us / 1e3);
            n_fail++;
        }
    }
    if (n_tasks.load() != 0) {
        fprintf(stderr, "FAIL: a task arrived\n");
        n_fail++;
    }

    if (n_fail > 0) {
        return 1;
    }
    printf("all server-wakeup tests passed\n");
    return 0;
}
