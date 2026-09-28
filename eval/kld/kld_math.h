// Distribution math for lovelace-kld: log-softmax normalizer, top-k, sparse KL with a tail bucket.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace kld {

constexpr double TAIL_FLOOR = 1e-10;

struct TopEntry {
    int32_t id;
    float   logprob;
};

// log(sum(exp(x))) over n logits; max-shifted, accumulated in double.
inline double logsumexp(const float * x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, x[i]);
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += std::exp(double(x[i]) - double(mx));
    return double(mx) + std::log(s);
}

inline int32_t argmax(const float * x, int n) {
    int32_t best = 0;
    for (int i = 1; i < n; ++i)
        if (x[i] > x[best]) best = i;  // strict: the lowest id wins ties
    return best;
}

// The k largest logits, descending (ties: lower id first), as log-probs under normalizer lse.
// `heap` is scratch reused across calls.
inline void top_k(const float * x, int n, int k, double lse, std::vector<TopEntry> & out,
                  std::vector<std::pair<float, int32_t>> & heap) {
    k = std::min(k, n);
    // "a before b" in output order; the heap keeps the worst kept entry at the front
    auto better = [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    };
    heap.clear();
    for (int32_t i = 0; i < k; ++i) heap.emplace_back(x[i], i);
    std::make_heap(heap.begin(), heap.end(), better);
    for (int32_t i = k; i < n; ++i) {
        // later ids never beat an equal logit, so strict > is the full comparison
        if (x[i] > heap.front().first) {
            std::pop_heap(heap.begin(), heap.end(), better);
            heap.back() = {x[i], i};
            std::push_heap(heap.begin(), heap.end(), better);
        }
    }
    std::sort(heap.begin(), heap.end(), better);
    out.resize(k);
    for (int j = 0; j < k; ++j) out[j] = {heap[j].second, float(double(heap[j].first) - lse)};
}

struct SparseKl {
    double kl;
    double p_tail;  // clamped
    double q_tail;  // clamped
};

// KL(P || Q) over P's top-k ids plus one tail bucket holding the remaining mass of each side.
// p: base top-k log-probs; q_logits/q_lse: the full candidate distribution.
inline SparseKl sparse_kl(const TopEntry * p, int k, const float * q_logits, double q_lse) {
    double kl = 0.0, p_sum = 0.0, q_sum = 0.0;
    for (int j = 0; j < k; ++j) {
        const double lp = p[j].logprob;
        const double lq = double(q_logits[p[j].id]) - q_lse;
        const double pj = std::exp(lp);
        kl += pj * (lp - lq);
        p_sum += pj;
        q_sum += std::exp(lq);
    }
    const double p_tail = std::max(1.0 - p_sum, TAIL_FLOOR);
    const double q_tail = std::max(1.0 - q_sum, TAIL_FLOOR);
    kl += p_tail * (std::log(p_tail) - std::log(q_tail));
    return {kl, p_tail, q_tail};
}

// Linear-interpolated quantile of a sorted sample (numpy's default method).
inline double quantile(const std::vector<double> & sorted, double q) {
    if (sorted.empty()) return NAN;
    const double idx = q * double(sorted.size() - 1);
    const size_t lo = size_t(std::floor(idx));
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (idx - double(lo)) * (sorted[hi] - sorted[lo]);
}

// FNV-1a 64-bit.
inline uint64_t fnv1a(const void * data, size_t n, uint64_t h = 0xcbf29ce484222325ULL) {
    const auto * b = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

}  // namespace kld
