#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Prefix index of the agent sessions: the session whose tokens share the longest prefix with a prompt.
// Blocks of block_tokens tokens are hashed into a radix tree; the hashes only select the candidates, whose tokens are
// then compared, so a hash collision can never join the wrong session.
struct server_session_index {
    static constexpr size_t block_tokens = 256;
    using id_type = uint64_t;

    struct match {
        id_type id = 0;
        size_t  n_common = 0; // tokens the prompt shares with the session
    };

    void update(id_type id, const std::vector<int32_t> & tokens);
    void erase(id_type id);

    // the session sharing the longest prefix with tokens (at least one token), the most recently updated on a tie
    std::optional<match> find(const std::vector<int32_t> & tokens) const;

    size_t size() const { return entries.size(); }

private:
    struct radix_node {
        std::unordered_map<uint64_t, std::unique_ptr<radix_node>> children;
        std::unordered_set<id_type> sessions;
    };

    struct entry {
        std::vector<int32_t> tokens;
        uint64_t             t_update = 0;
    };

    uint64_t   clock = 0;
    radix_node root;
    std::unordered_map<id_type, entry> entries;

    static uint64_t block_hash(const std::vector<int32_t> & tokens, size_t begin);
};
