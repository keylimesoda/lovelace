#include "server-session-store.h"

#include <algorithm>

uint64_t server_session_index::block_hash(const std::vector<int32_t> & tokens, size_t begin) {
    uint64_t hash = 14695981039346656037ull;
    const size_t end = std::min(tokens.size(), begin + block_tokens);
    for (size_t i = begin; i < end; ++i) {
        const uint32_t token = static_cast<uint32_t>(tokens[i]);
        for (int byte = 0; byte < 4; ++byte) {
            hash ^= (token >> (byte * 8)) & 0xff;
            hash *= 1099511628211ull;
        }
    }
    return hash;
}

void server_session_index::erase(id_type id) {
    const auto it = entries.find(id);
    if (it == entries.end()) {
        return;
    }

    const auto & tokens = it->second.tokens;

    std::vector<std::pair<radix_node *, uint64_t>> path;
    radix_node * node = &root;
    node->sessions.erase(id);
    for (size_t begin = 0; begin + block_tokens <= tokens.size(); begin += block_tokens) {
        const uint64_t hash = block_hash(tokens, begin);
        const auto child = node->children.find(hash);
        if (child == node->children.end()) {
            break;
        }
        child->second->sessions.erase(id);
        path.emplace_back(node, hash);
        node = child->second.get();
    }

    // prune the nodes no session passes through anymore
    for (auto p = path.rbegin(); p != path.rend(); ++p) {
        const auto child = p->first->children.find(p->second);
        if (!child->second->sessions.empty() || !child->second->children.empty()) {
            break;
        }
        p->first->children.erase(child);
    }

    entries.erase(it);
}

void server_session_index::update(id_type id, const std::vector<int32_t> & tokens) {
    erase(id);

    auto & e = entries[id];
    e.tokens   = tokens;
    e.t_update = ++clock;

    radix_node * node = &root;
    node->sessions.insert(id);
    for (size_t begin = 0; begin + block_tokens <= e.tokens.size(); begin += block_tokens) {
        auto & child = node->children[block_hash(e.tokens, begin)];
        if (!child) {
            child = std::make_unique<radix_node>();
        }
        node = child.get();
        node->sessions.insert(id);
    }
}

std::optional<server_session_index::match> server_session_index::find(const std::vector<int32_t> & tokens) const {
    // the sessions of the deepest node share the most whole blocks with the prompt, no other one can share more
    const radix_node * node = &root;
    for (size_t begin = 0; begin + block_tokens <= tokens.size(); begin += block_tokens) {
        const auto child = node->children.find(block_hash(tokens, begin));
        if (child == node->children.end()) {
            break;
        }
        node = child->second.get();
    }

    std::optional<match> best;
    uint64_t best_update = 0;
    for (id_type id : node->sessions) {
        const auto & e = entries.at(id);
        const size_t lim = std::min(tokens.size(), e.tokens.size());
        size_t n = 0;
        while (n < lim && tokens[n] == e.tokens[n]) {
            ++n;
        }
        if (n == 0) {
            continue;
        }
        if (!best || n > best->n_common || (n == best->n_common && e.t_update > best_update)) {
            best = match { id, n };
            best_update = e.t_update;
        }
    }

    // a hash collision can leave the deepest node without a real match, the root holds every session
    if (!best && node != &root) {
        for (id_type id : root.sessions) {
            const auto & e = entries.at(id);
            const size_t lim = std::min(tokens.size(), e.tokens.size());
            size_t n = 0;
            while (n < lim && tokens[n] == e.tokens[n]) {
                ++n;
            }
            if (n > 0 && (!best || n > best->n_common || (n == best->n_common && e.t_update > best_update))) {
                best = match { id, n };
                best_update = e.t_update;
            }
        }
    }

    return best;
}
