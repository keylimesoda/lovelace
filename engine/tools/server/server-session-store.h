#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Prefix index for identifying agent sessions without a client-provided session id.
// Hash blocks are only an index: candidate token streams are compared before joining.
struct server_session_index {
    static constexpr size_t block_tokens = 256;
    using id_type = uint64_t;

    struct match {
        id_type id = 0;
        size_t n_common_tokens = 0;
    };

    explicit server_session_index(size_t shared_prefix_tokens = static_cast<size_t>(-1));

    void update(id_type id, const std::vector<int32_t> & tokens, uint64_t now_seconds = 0);
    void erase(id_type id);
    void touch(id_type id, uint64_t now_seconds);
    std::vector<id_type> expire(uint64_t ttl_seconds, uint64_t now_seconds);
    std::vector<id_type> expired(uint64_t ttl_seconds, uint64_t now_seconds) const;
    std::optional<match> find(const std::vector<int32_t> & tokens, id_type preferred = 0, uint64_t now_seconds = 0);

private:
    struct radix_node {
        std::unordered_map<uint64_t, std::unique_ptr<radix_node>> children;
        std::unordered_set<id_type> sessions;
    };

    struct session_entry {
        std::vector<int32_t> tokens;
        uint64_t last_used = 0;
        uint64_t last_used_seconds = 0;
    };

    size_t shared_prefix_tokens;
    uint64_t clock = 0;
    radix_node root;
    std::unordered_map<id_type, session_entry> entries;

    static uint64_t block_hash(const std::vector<int32_t> & tokens, size_t begin);
    std::vector<uint64_t> path_for(const std::vector<int32_t> & tokens) const;
};

// Lossless RAM/NVMe tier manager for opaque, serialized session snapshots.
// Files are replaced atomically. A failed write leaves the prior snapshot intact.
struct server_session_tiers {
    using id_type = server_session_index::id_type;

    enum class tier { ram, disk };

    struct stats {
        size_t ram_bytes = 0;
        size_t ram_limit = 0;
        size_t disk_bytes = 0;
        size_t disk_limit = 0;
        size_t sessions = 0;
        size_t ram_sessions = 0;
        size_t disk_sessions = 0;
    };

    server_session_tiers(std::filesystem::path directory, size_t ram_limit, size_t disk_limit, uint64_t ttl_seconds);
    ~server_session_tiers();

    server_session_tiers(const server_session_tiers &) = delete;
    server_session_tiers & operator=(const server_session_tiers &) = delete;

    // Add or replace the durable mirror. This demotes least-recently-used mirrors as needed.
    void put(id_type id, std::vector<uint8_t> snapshot, uint64_t now_seconds,
            std::vector<id_type> * disk_budget_drops = nullptr);

    // Promote a disk mirror into RAM, demoting other mirrors first. Returns false if this session
    // exceeds the RAM limit; the disk copy remains intact in that case.
    bool promote(id_type id, uint64_t now_seconds, std::vector<id_type> * disk_budget_drops = nullptr);

    // Update LRU time and return the in-RAM snapshot, or nullptr when the session is disk-tiered.
    std::vector<uint8_t> read_snapshot(id_type id, uint64_t now_seconds);
    std::vector<uint8_t> read_prefix(id_type id, size_t max_bytes) const;
    std::vector<id_type> ids() const;
    const std::vector<uint8_t> * ram_snapshot(id_type id, uint64_t now_seconds);

    // Remove only sessions whose configured TTL has elapsed. Explicitly returns IDs for logging.
    std::vector<id_type> expire(uint64_t now_seconds);

    bool contains(id_type id) const;
    tier get_tier(id_type id) const;
    stats usage() const;
    size_t bytes(id_type id) const;
    uint64_t last_sync_seconds(id_type id) const;
    void touch(id_type id, uint64_t now_seconds);
    void erase(id_type id);

private:
    struct item {
        std::vector<uint8_t> ram;
        size_t bytes = 0;
        uint64_t last_used = 0;
        uint64_t last_sync = 0;
        bool on_disk = false;
        std::filesystem::path path;
    };

    std::filesystem::path directory;
    size_t ram_limit;
    size_t disk_limit;
    uint64_t ttl_seconds;
    size_t ram_bytes = 0;
    size_t disk_bytes = 0;
    std::unordered_map<id_type, item> items;

    bool demote_one(id_type except, std::vector<id_type> * disk_budget_drops);
    void make_disk_room(size_t incoming_bytes, id_type except, std::vector<id_type> * disk_budget_drops);
    void write_disk(id_type id, const std::vector<uint8_t> & snapshot, const std::filesystem::path & path);
    std::vector<uint8_t> read_disk(const item & value) const;
    void erase_item(id_type id);
};
