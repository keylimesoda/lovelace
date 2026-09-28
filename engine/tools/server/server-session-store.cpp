#include "server-session-store.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>

server_session_index::server_session_index(size_t shared_prefix_tokens) : shared_prefix_tokens(shared_prefix_tokens) {
}

uint64_t server_session_index::block_hash(const std::vector<int32_t> & tokens, size_t begin) {
    uint64_t hash = 14695981039346656037ull;
    const size_t end = std::min(tokens.size(), begin + block_tokens);
    for (size_t i = begin; i < end; ++i) {
        uint32_t token = static_cast<uint32_t>(tokens[i]);
        for (int byte = 0; byte < 4; ++byte) {
            hash ^= (token >> (byte * 8)) & 0xff;
            hash *= 1099511628211ull;
        }
    }
    return hash;
}

std::vector<uint64_t> server_session_index::path_for(const std::vector<int32_t> & tokens) const {
    std::vector<uint64_t> path;
    path.reserve(tokens.size() / block_tokens);
    for (size_t begin = 0; begin + block_tokens <= tokens.size(); begin += block_tokens) {
        path.push_back(block_hash(tokens, begin));
    }
    return path;
}

void server_session_index::erase(id_type id) {
    const auto entry = entries.find(id);
    if (entry == entries.end()) {
        return;
    }

    std::vector<std::pair<radix_node *, uint64_t>> path;
    radix_node * node = &root;
    node->sessions.erase(id);
    for (uint64_t hash : path_for(entry->second.tokens)) {
        const auto child = node->children.find(hash);
        if (child == node->children.end()) {
            break;
        }
        radix_node * next = child->second.get();
        next->sessions.erase(id);
        path.emplace_back(node, hash);
        node = next;
    }

    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        auto child = it->first->children.find(it->second);
        if (child == it->first->children.end() || !child->second->sessions.empty() || !child->second->children.empty()) {
            break;
        }
        it->first->children.erase(child);
    }

    entries.erase(entry);
}

void server_session_index::update(id_type id, const std::vector<int32_t> & tokens, uint64_t now_seconds) {
    erase(id);
    auto & entry = entries[id];
    entry.tokens.assign(tokens.begin(), tokens.end());
    entry.last_used = ++clock;
    entry.last_used_seconds = now_seconds;

    radix_node * node = &root;
    node->sessions.insert(id);
    for (uint64_t hash : path_for(entry.tokens)) {
        auto & child = node->children[hash];
        if (!child) {
            child = std::make_unique<radix_node>();
        }
        node = child.get();
        node->sessions.insert(id);
    }
}
void server_session_index::touch(id_type id, uint64_t now_seconds) {
    const auto it = entries.find(id);
    if (it == entries.end()) {
        return;
    }
    it->second.last_used = ++clock;
    it->second.last_used_seconds = now_seconds;
}

std::vector<server_session_index::id_type> server_session_index::expired(uint64_t ttl_seconds, uint64_t now_seconds) const {
    std::vector<id_type> result;
    if (ttl_seconds == 0) {
        return result;
    }
    for (const auto & [id, entry] : entries) {
        if (entry.last_used_seconds != 0 && now_seconds >= entry.last_used_seconds &&
                now_seconds - entry.last_used_seconds >= ttl_seconds) {
            result.push_back(id);
        }
    }
    return result;
}

std::vector<server_session_index::id_type> server_session_index::expire(uint64_t ttl_seconds, uint64_t now_seconds) {
    auto result = expired(ttl_seconds, now_seconds);
    for (id_type id : result) {
        erase(id);
    }
    return result;
}

std::optional<server_session_index::match> server_session_index::find(const std::vector<int32_t> & tokens, id_type preferred, uint64_t now_seconds) {
    if (entries.empty()) {
        return std::nullopt;
    }

    std::unordered_set<id_type> candidates = root.sessions;
    radix_node * node = &root;
    for (uint64_t hash : path_for(tokens)) {
        const auto child = node->children.find(hash);
        if (child == node->children.end()) {
            break;
        }
        node = child->second.get();
        candidates.insert(node->sessions.begin(), node->sessions.end());
    }

    size_t common_prefix = shared_prefix_tokens;
    if (common_prefix == std::numeric_limits<size_t>::max()) {
        if (entries.size() > 1) {
            const auto & first = entries.begin()->second.tokens;
            common_prefix = first.size();
            for (const auto & [id, entry] : entries) {
                (void) id;
                common_prefix = std::min(common_prefix, entry.tokens.size());
                size_t n = 0;
                while (n < common_prefix && first[n] == entry.tokens[n]) {
                    ++n;
                }
                common_prefix = n;
            }
        } else {
            common_prefix = entries.begin()->second.tokens.size();
        }
    }

    std::optional<match> best;
    uint64_t best_touch = 0;
    for (id_type id : candidates) {
        const auto it = entries.find(id);
        if (it == entries.end()) {
            continue;
        }
        const auto & current = it->second.tokens;
        const size_t limit = std::min(tokens.size(), current.size());
        size_t common = 0;
        while (common < limit && tokens[common] == current[common]) {
            ++common;
        }
        const bool same_single_session_prefix = entries.size() == 1 && common == current.size();
        if (common <= common_prefix && !same_single_session_prefix) {
            continue;
        }
        if (!best || common > best->n_common_tokens ||
                (common == best->n_common_tokens && id == preferred && best->id != preferred) ||
                (common == best->n_common_tokens && (id == preferred) == (best->id == preferred) && it->second.last_used > best_touch)) {
            best = match{ id, common };
            best_touch = it->second.last_used;
        }
    }

    if (best) {
        auto & entry = entries[best->id];
        entry.last_used = ++clock;
        if (now_seconds != 0) {
            entry.last_used_seconds = now_seconds;
        }
    }
    return best;
}

server_session_tiers::server_session_tiers(std::filesystem::path directory, size_t ram_limit, size_t disk_limit, uint64_t ttl_seconds)
        : directory(std::move(directory)), ram_limit(ram_limit), disk_limit(disk_limit), ttl_seconds(ttl_seconds) {
    std::filesystem::create_directories(this->directory);
    const uint64_t now = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    for (const auto & file : std::filesystem::directory_iterator(this->directory)) {
        if (!file.is_regular_file()) {
            continue;
        }
        const std::string name = file.path().filename().string();
        if (name.size() <= 12 || name.compare(0, 8, "session-") != 0 || name.compare(name.size() - 4, 4, ".bin") != 0) {
            continue;
        }
        const std::string id_text = name.substr(8, name.size() - 12);
        id_type id = 0;
        const auto parsed = std::from_chars(id_text.data(), id_text.data() + id_text.size(), id);
        if (parsed.ec != std::errc{} || parsed.ptr != id_text.data() + id_text.size()) {
            continue;
        }

        item value;
        value.bytes = static_cast<size_t>(file.file_size());
        value.path = file.path();
        value.on_disk = true;
        const auto age = std::chrono::duration_cast<std::chrono::seconds>(
                std::filesystem::file_time_type::clock::now() - file.last_write_time()).count();
        value.last_used = age > 0 && static_cast<uint64_t>(age) < now ? now - static_cast<uint64_t>(age) : now;
        value.last_sync = value.last_used;
        items.emplace(id, std::move(value));
        disk_bytes += static_cast<size_t>(file.file_size());
    }
}

server_session_tiers::~server_session_tiers() = default;

void server_session_tiers::write_disk(id_type id, const std::vector<uint8_t> & snapshot, const std::filesystem::path & path) {
    const auto temp = path.string() + ".tmp";
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out || (snapshot.size() && !out.write(reinterpret_cast<const char *>(snapshot.data()), static_cast<std::streamsize>(snapshot.size())))) {
            std::filesystem::remove(temp);
            throw std::runtime_error("failed to write session snapshot " + std::to_string(id));
        }
        out.flush();
        if (!out) {
            std::filesystem::remove(temp);
            throw std::runtime_error("failed to flush session snapshot " + std::to_string(id));
        }
    }
    std::error_code ec;
    std::filesystem::rename(temp, path, ec);
    if (ec) {
        std::filesystem::remove(temp);
        throw std::runtime_error("failed to publish session snapshot " + std::to_string(id) + ": " + ec.message());
    }
}

std::vector<uint8_t> server_session_tiers::read_disk(const item & value) const {
    std::ifstream in(value.path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("failed to open session snapshot " + value.path.string());
    }
    const auto end = in.tellg();
    if (end < 0 || static_cast<uint64_t>(end) != value.bytes) {
        throw std::runtime_error("session snapshot size mismatch " + value.path.string());
    }
    std::vector<uint8_t> result(value.bytes);
    in.seekg(0);
    if (value.bytes && !in.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(value.bytes))) {
        throw std::runtime_error("failed to read session snapshot " + value.path.string());
    }
    return result;
}

void server_session_tiers::make_disk_room(size_t incoming_bytes, id_type except, std::vector<id_type> * disk_budget_drops) {
    while (disk_limit && disk_bytes + incoming_bytes > disk_limit) {
        auto victim = items.end();
        for (auto it = items.begin(); it != items.end(); ++it) {
            if (it->first == except || !it->second.on_disk) {
                continue;
            }
            if (victim == items.end() || it->second.last_used < victim->second.last_used) {
                victim = it;
            }
        }
        if (victim == items.end()) {
            throw std::runtime_error("session disk limit cannot fit snapshot");
        }
        const id_type id = victim->first;
        erase_item(id);
        if (disk_budget_drops) {
            disk_budget_drops->push_back(id);
        }
    }
}

bool server_session_tiers::demote_one(id_type except, std::vector<id_type> * disk_budget_drops) {
    auto victim = items.end();
    for (auto it = items.begin(); it != items.end(); ++it) {
        if (it->first == except || it->second.ram.empty()) {
            continue;
        }
        if (victim == items.end() || it->second.last_used < victim->second.last_used) {
            victim = it;
        }
    }
    if (victim == items.end()) {
        return false;
    }

    const id_type id = victim->first;
    const size_t size = victim->second.ram.size();
    make_disk_room(size, except, disk_budget_drops);
    const auto path = directory / ("session-" + std::to_string(id) + ".bin");
    write_disk(id, victim->second.ram, path);
    disk_bytes += size;
    ram_bytes -= size;
    victim->second.bytes = size;
    victim->second.ram.clear();
    victim->second.ram.shrink_to_fit();
    victim->second.path = path;
    victim->second.on_disk = true;
    return true;
}

void server_session_tiers::put(id_type id, std::vector<uint8_t> snapshot, uint64_t now_seconds, std::vector<id_type> * disk_budget_drops) {
    const size_t size = snapshot.size();
    const auto existing = items.find(id);
    const size_t old_ram = existing != items.end() ? existing->second.ram.size() : 0;
    const bool old_on_disk = existing != items.end() && existing->second.on_disk;
    const size_t old_disk = old_on_disk ? existing->second.bytes : 0;
    const auto old_path = old_on_disk ? existing->second.path : std::filesystem::path{};
    const bool use_ram = size <= ram_limit;

    if (use_ram) {
        while (ram_bytes - old_ram + size > ram_limit) {
            if (!demote_one(id, disk_budget_drops)) {
                throw std::runtime_error("session RAM limit cannot fit snapshot");
            }
        }
    }

    const auto path = directory / ("session-" + std::to_string(id) + ".bin");
    if (!use_ram) {
        while (disk_limit && disk_bytes - old_disk + size > disk_limit) {
            auto victim = items.end();
            for (auto it = items.begin(); it != items.end(); ++it) {
                if (it->first == id || !it->second.on_disk) {
                    continue;
                }
                if (victim == items.end() || it->second.last_used < victim->second.last_used) {
                    victim = it;
                }
            }
            if (victim == items.end()) {
                throw std::runtime_error("session disk limit cannot fit snapshot");
            }
            const id_type dropped = victim->first;
            erase_item(dropped);
            if (disk_budget_drops) {
                disk_budget_drops->push_back(dropped);
            }
        }
        write_disk(id, snapshot, path);
    }

    if (old_on_disk) {
        if (use_ram) {
            std::error_code ec;
            std::filesystem::remove(old_path, ec);
            if (ec) {
                throw std::runtime_error("failed to remove replaced session snapshot: " + ec.message());
            }
        }
        disk_bytes -= old_disk;
    }

    auto & value = items[id];
    ram_bytes -= value.ram.size();
    value.ram.clear();
    value.ram.shrink_to_fit();
    value.on_disk = !use_ram;
    value.bytes = size;
    value.last_used = now_seconds;
    value.last_sync = now_seconds;
    value.path = path;
    if (use_ram) {
        value.ram = std::move(snapshot);
        ram_bytes += size;
    } else {
        disk_bytes += size;
    }
}

bool server_session_tiers::promote(id_type id, uint64_t now_seconds, std::vector<id_type> * disk_budget_drops) {
    auto it = items.find(id);
    if (it == items.end()) {
        return false;
    }
    if (!it->second.on_disk) {
        it->second.last_used = now_seconds;
        return true;
    }
    if (it->second.bytes > ram_limit) {
        return false;
    }

    std::vector<uint8_t> snapshot = read_disk(it->second);
    while (ram_bytes + snapshot.size() > ram_limit) {
        if (!demote_one(id, disk_budget_drops)) {
            return false;
        }
    }
    std::error_code ec;
    std::filesystem::remove(it->second.path, ec);
    if (ec) {
        throw std::runtime_error("failed to remove promoted session snapshot: " + ec.message());
    }
    disk_bytes -= it->second.bytes;
    it->second.ram = std::move(snapshot);
    it->second.on_disk = false;
    it->second.last_used = now_seconds;
    ram_bytes += it->second.bytes;
    return true;
}

const std::vector<uint8_t> * server_session_tiers::ram_snapshot(id_type id, uint64_t now_seconds) {
    const auto it = items.find(id);
    if (it == items.end()) {
        return nullptr;
    }
    it->second.last_used = now_seconds;
    return it->second.on_disk ? nullptr : &it->second.ram;
}
std::vector<uint8_t> server_session_tiers::read_snapshot(id_type id, uint64_t now_seconds) {
    auto it = items.find(id);
    if (it == items.end()) {
        throw std::out_of_range("unknown session snapshot");
    }
    it->second.last_used = now_seconds;
    return it->second.on_disk ? read_disk(it->second) : it->second.ram;
}

std::vector<uint8_t> server_session_tiers::read_prefix(id_type id, size_t max_bytes) const {
    const auto it = items.find(id);
    if (it == items.end()) {
        throw std::out_of_range("unknown session snapshot");
    }
    const size_t n_bytes = std::min(max_bytes, it->second.bytes);
    if (!it->second.on_disk) {
        return std::vector<uint8_t>(it->second.ram.begin(), it->second.ram.begin() + n_bytes);
    }
    std::ifstream in(it->second.path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open session snapshot " + it->second.path.string());
    }
    std::vector<uint8_t> result(n_bytes);
    if (n_bytes && !in.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(n_bytes))) {
        throw std::runtime_error("failed to read session snapshot prefix " + it->second.path.string());
    }
    return result;
}

std::vector<server_session_tiers::id_type> server_session_tiers::ids() const {
    std::vector<id_type> result;
    result.reserve(items.size());
    for (const auto & [id, value] : items) {
        (void) value;
        result.push_back(id);
    }
    return result;
}


std::vector<server_session_tiers::id_type> server_session_tiers::expire(uint64_t now_seconds) {
    std::vector<id_type> expired;
    if (ttl_seconds == 0) {
        return expired;
    }
    for (auto it = items.begin(); it != items.end();) {
        if (now_seconds >= it->second.last_used && now_seconds - it->second.last_used >= ttl_seconds) {
            const id_type id = it->first;
            ++it;
            erase_item(id);
            expired.push_back(id);
        } else {
            ++it;
        }
    }
    return expired;
}

bool server_session_tiers::contains(id_type id) const {
    return items.find(id) != items.end();
}
void server_session_tiers::erase(id_type id) {
    erase_item(id);
}

server_session_tiers::tier server_session_tiers::get_tier(id_type id) const {
    const auto it = items.find(id);
    if (it == items.end()) {
        throw std::out_of_range("unknown session tier");
    }
    return it->second.on_disk ? tier::disk : tier::ram;
}
size_t server_session_tiers::bytes(id_type id) const {
    const auto it = items.find(id);
    if (it == items.end()) {
        throw std::out_of_range("unknown session tier");
    }
    return it->second.bytes;
}

uint64_t server_session_tiers::last_sync_seconds(id_type id) const {
    const auto it = items.find(id);
    if (it == items.end()) {
        throw std::out_of_range("unknown session tier");
    }
    return it->second.last_sync;
}

void server_session_tiers::touch(id_type id, uint64_t now_seconds) {
    const auto it = items.find(id);
    if (it != items.end()) {
        it->second.last_used = now_seconds;
    }
}

server_session_tiers::stats server_session_tiers::usage() const {
    stats result;
    result.ram_bytes = ram_bytes;
    result.ram_limit = ram_limit;
    result.disk_bytes = disk_bytes;
    result.disk_limit = disk_limit;
    result.sessions = items.size();
    for (const auto & [id, value] : items) {
        (void) id;
        (value.on_disk ? result.disk_sessions : result.ram_sessions)++;
    }
    return result;
}

void server_session_tiers::erase_item(id_type id) {
    const auto it = items.find(id);
    if (it == items.end()) {
        return;
    }
    ram_bytes -= it->second.ram.size();
    if (it->second.on_disk) {
        std::error_code ec;
        std::filesystem::remove(it->second.path, ec);
        if (!ec) {
            disk_bytes -= it->second.bytes;
        }
    }
    items.erase(it);
}
