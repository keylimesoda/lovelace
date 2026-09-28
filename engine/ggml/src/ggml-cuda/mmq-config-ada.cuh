// Ada Lovelace (tuned on RTX 4090): same tiles as Ampere, but the small-J tiles are memory-bound and a single
// 8-warp block per SM cannot keep enough loads in flight, so they target two resident blocks per SM.
// Q4_K/Q5_K with J = 24..32 already stream well with one block per SM and lose from the finer split (observation).
static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_ada(ggml_type type, int J, bool fallback) {
    ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config_ampere(type, J, fallback);
    const bool q_k_wide = (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) && config.J > 16;
    if (config.type != GGML_TYPE_COUNT && config.J <= 32 && !q_k_wide) {
        config.occupancy = 2;
    }
    return config;
}
