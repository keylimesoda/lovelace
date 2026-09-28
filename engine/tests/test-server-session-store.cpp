#include "server-session-store.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static std::vector<int32_t> make_tokens(size_t n, int32_t start) {
    std::vector<int32_t> tokens(n);
    for (size_t i = 0; i < n; ++i) {
        tokens[i] = start + static_cast<int32_t>(i);
    }
    return tokens;
}

static void test_index_join_and_new_session() {
    server_session_index index;
    auto alpha = make_tokens(700, 1000);
    auto beta = alpha;
    beta[300] = 900001;
    auto unrelated = alpha;
    unrelated[0] = 900002;

    index.update(1, alpha);
    const auto new_agent = index.find(beta);
    CHECK(!new_agent); // same common preamble, but no history beyond the shared prefix

    index.update(2, beta);
    auto alpha_turn = alpha;
    alpha_turn.insert(alpha_turn.end(), {7001, 7002});
    const auto joined_alpha = index.find(alpha_turn);
    CHECK(joined_alpha.has_value());
    CHECK(joined_alpha && joined_alpha->id == 1);
    CHECK(joined_alpha && joined_alpha->n_common_tokens == alpha.size());

    auto beta_turn = beta;
    beta_turn.push_back(8001);
    const auto joined_beta = index.find(beta_turn);
    CHECK(joined_beta.has_value());
    CHECK(joined_beta && joined_beta->id == 2);

    CHECK(!index.find(unrelated));
}

static void test_index_ttl() {
    server_session_index index;
    auto first = make_tokens(300, 100);
    auto second = first;
    second[200] = 5000;
    index.update(1, first, 100);
    index.update(2, second, 101);
    CHECK(index.expire(10, 109).empty());
    CHECK(index.expire(10, 110) == std::vector<uint64_t>({1}));
    CHECK(index.expire(10, 111) == std::vector<uint64_t>({2}));
    CHECK(!index.find(first));
}

static void test_index_update_and_hash_collision_check() {
    server_session_index index(256);
    auto old = make_tokens(300, 1);
    auto current = old;
    current[17] = 777;
    index.update(9, old);
    index.update(9, current);

    CHECK(!index.find(old));
    auto continuation = current;
    continuation.push_back(42);
    const auto found = index.find(continuation);
    CHECK(found.has_value());
    CHECK(found && found->id == 9);

    auto same_block_different_prefix = current;
    same_block_different_prefix[18] = 888;
    CHECK(!index.find(same_block_different_prefix));
}

static std::filesystem::path temp_dir() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() / ("llama-session-store-" + std::to_string(nonce));
}

static std::vector<uint8_t> make_bytes(uint8_t first, size_t n) {
    return std::vector<uint8_t>(n, first);
}

static void test_tiers_demote_promote_without_dropping() {
    const auto dir = temp_dir();
    {
        server_session_tiers tiers(dir, 6, 0, 0);
        const auto a = make_bytes(0x11, 4);
        const auto b = make_bytes(0x22, 4);
        const auto c = make_bytes(0x33, 8);
        tiers.put(1, a, 10);
        tiers.put(2, b, 11);
        CHECK(tiers.get_tier(1) == server_session_tiers::tier::disk);
        CHECK(tiers.get_tier(2) == server_session_tiers::tier::ram);
        CHECK(tiers.contains(1) && tiers.contains(2));

        CHECK(tiers.promote(1, 12));
        CHECK(tiers.get_tier(1) == server_session_tiers::tier::ram);
        CHECK(tiers.get_tier(2) == server_session_tiers::tier::disk);
        const auto * restored = tiers.ram_snapshot(1, 13);
        CHECK(restored && *restored == a);

        tiers.put(3, c, 14);
        CHECK(tiers.get_tier(3) == server_session_tiers::tier::disk);
        CHECK(!tiers.promote(3, 15));
        CHECK(tiers.get_tier(3) == server_session_tiers::tier::disk);
        CHECK(tiers.contains(1) && tiers.contains(2) && tiers.contains(3));

        const auto before = tiers.usage();
        CHECK(before.sessions == 3);
        CHECK(before.ram_bytes <= before.ram_limit);
        CHECK(before.ram_sessions >= 1);
        CHECK(before.disk_sessions >= 1);
    }
    {
        server_session_tiers reopened(dir, 6, 0, 0);
        CHECK(reopened.contains(2) && reopened.contains(3));
        CHECK(reopened.get_tier(2) == server_session_tiers::tier::disk);
        CHECK(reopened.promote(2, 20));
        const auto * restored = reopened.ram_snapshot(2, 21);
        CHECK(restored && *restored == make_bytes(0x22, 4));
        CHECK(!reopened.promote(3, 22));
    }
    std::filesystem::remove_all(dir);
}

static void test_tiers_disk_budget_reports_deletion() {
    const auto dir = temp_dir();
    {
        server_session_tiers tiers(dir, 4, 4, 0);
        tiers.put(20, make_bytes(0x20, 4), 1);
        tiers.put(21, make_bytes(0x21, 4), 2);
        std::vector<uint64_t> dropped;
        tiers.put(22, make_bytes(0x22, 4), 3, &dropped);
        CHECK(dropped == std::vector<uint64_t>({20}));
        CHECK(!tiers.contains(20));
        CHECK(tiers.contains(21) && tiers.contains(22));
        CHECK(tiers.usage().disk_bytes <= tiers.usage().disk_limit);
    }
    std::filesystem::remove_all(dir);
}

static void test_tiers_ttl() {
    const auto dir = temp_dir();
    {
        server_session_tiers tiers(dir, 4, 0, 10);
        tiers.put(10, make_bytes(0x10, 4), 100);
        tiers.put(11, make_bytes(0x11, 4), 101);
        const auto not_expired = tiers.expire(109);
        CHECK(not_expired.empty());
        const auto first_expired = tiers.expire(110);
        CHECK(first_expired == std::vector<uint64_t>({10}));
        const auto last_expired = tiers.expire(111);
        CHECK(last_expired == std::vector<uint64_t>({11}));
        CHECK(!tiers.contains(10) && !tiers.contains(11));
        CHECK(tiers.usage().sessions == 0);
    }
    std::filesystem::remove_all(dir);
}

int main() {
    test_index_join_and_new_session();
    test_index_ttl();
    test_index_update_and_hash_collision_check();
    test_tiers_demote_promote_without_dropping();
    test_tiers_disk_budget_reports_deletion();
    test_tiers_ttl();
    if (n_fail) {
        fprintf(stderr, "%d session-store checks failed\n", n_fail);
        return EXIT_FAILURE;
    }
    printf("all server-session-store tests passed\n");
    return EXIT_SUCCESS;
}
