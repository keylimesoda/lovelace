#pragma once

// KV budget scheduling for a unified KV cache shared by all slots (see README.md, "KV budget scheduling").
//
// Pure bookkeeping: no llama.cpp calls, so that the admission and victim rules can be unit tested
// (tests/test-server-sched.cpp). The server mirrors each slot into slot_info before planning.

#include <algorithm>
#include <cstdint>
#include <vector>

struct server_kv_budget {
    struct slot_info {
        int32_t n_used      = 0;     // cells held by the slot's sequence now
        int32_t n_reserved  = 0;     // cells promised to the running task (0 when idle)
        bool    active      = false; // runs (or holds a preempted) task: not parkable
        int64_t t_last_used = -1;    // victim order, least recently used first
    };

    int32_t n_cells  = 0; // total cells of the unified KV cache
    int32_t headroom = 0; // cells kept free when admitting new work

    std::vector<slot_info> slots;

    // cells a slot holds or may still grow into
    static int32_t committed(const slot_info & s) {
        return s.active ? std::max(s.n_used, s.n_reserved) : s.n_used;
    }

    int64_t n_committed(int exclude = -1) const {
        int64_t res = 0;
        for (int i = 0; i < (int) slots.size(); i++) {
            if (i != exclude) {
                res += committed(slots[i]);
            }
        }
        return res;
    }

    int64_t n_used() const {
        int64_t res = 0;
        for (const auto & s : slots) {
            res += s.n_used;
        }
        return res;
    }

    int64_t n_reserved() const {
        int64_t res = 0;
        for (const auto & s : slots) {
            res += s.active ? s.n_reserved : 0;
        }
        return res;
    }

    // cells to reserve for a task: its prompt, the generation budget and the speculative draft window
    //   n_predict < 0 means unlimited, the reservation then covers gen_cap tokens and grows on demand
    //   the result is capped at the slot context, a slot can never hold more
    static int32_t reservation(int32_t n_prompt, int32_t n_predict, int32_t gen_cap, int32_t n_draft, int32_t n_ctx_slot) {
        int64_t n_gen = gen_cap > 0 ? gen_cap : 0;
        if (n_predict >= 0) {
            n_gen = gen_cap > 0 ? std::min<int64_t>(n_predict, gen_cap) : n_predict;
        }
        const int64_t res = (int64_t) n_prompt + n_gen + std::max(0, n_draft) + 1;
        return (int32_t) std::min<int64_t>(res, n_ctx_slot);
    }

    struct plan {
        bool fits = false;
        std::vector<int> victims; // idle slots to park first, in order
    };

    // plan to commit `need` cells to slot `target` (its own current commitment is replaced)
    //   use_headroom: new work keeps `headroom` cells free, growth of running work may use them
    //   victims are idle slots holding cells, least recently used first; only as many as needed,
    //   and none at all when parking every candidate would still not make room
    plan plan_commit(int target, int32_t need, bool use_headroom) const {
        plan res;

        const int64_t limit = (int64_t) n_cells - (use_headroom ? headroom : 0);

        int64_t load = n_committed(target) + need;
        if (load <= limit) {
            res.fits = true;
            return res;
        }

        std::vector<int> idle;
        for (int i = 0; i < (int) slots.size(); i++) {
            if (i != target && !slots[i].active && slots[i].n_used > 0) {
                idle.push_back(i);
            }
        }
        std::stable_sort(idle.begin(), idle.end(), [&](int a, int b) {
            return slots[a].t_last_used < slots[b].t_last_used;
        });

        for (int i : idle) {
            res.victims.push_back(i);
            load -= slots[i].n_used;
            if (load <= limit) {
                res.fits = true;
                return res;
            }
        }

        res.victims.clear();
        return res;
    }

    struct candidate {
        bool    idle        = false;
        int32_t n_used      = 0;  // cells held now
        int32_t n_lcp       = 0;  // common prefix of the held prompt with the new prompt
        int64_t t_last_used = -1;
    };

    // session-affine slot choice for a new prompt of n_prompt tokens, -1 if no slot is idle
    //   1. the idle slot holding the longest common prefix, if it covers more than sim_min of the prompt
    //   2. an empty idle slot, so that no resident session has to be parked
    //   3. the least recently used idle slot
    static int pick_slot(const std::vector<candidate> & cands, int32_t n_prompt, float sim_min) {
        int best = -1;
        for (int i = 0; i < (int) cands.size(); i++) {
            const auto & c = cands[i];
            if (!c.idle || c.n_lcp <= 0 || (float) c.n_lcp <= sim_min * (float) n_prompt) {
                continue;
            }
            if (best < 0 || c.n_lcp > cands[best].n_lcp) {
                best = i;
            }
        }
        if (best >= 0) {
            return best;
        }

        int empty = -1;
        int lru   = -1;
        for (int i = 0; i < (int) cands.size(); i++) {
            const auto & c = cands[i];
            if (!c.idle) {
                continue;
            }
            if (c.n_used == 0 && (empty < 0 || c.t_last_used < cands[empty].t_last_used)) {
                empty = i;
            }
            if (lru < 0 || c.t_last_used < cands[lru].t_last_used) {
                lru = i;
            }
        }
        return empty >= 0 ? empty : lru;
    }
};
