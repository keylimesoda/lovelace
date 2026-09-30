#include "server-observe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <sstream>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

static int64_t obs_steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

int64_t server_observer::now_unix_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
}

// round for readability, the records are meant to be read by people too
static double r3(double x) { return std::round(x * 1e3) / 1e3; }
static double r6(double x) { return std::round(x * 1e6) / 1e6; }

static json opt_num(bool valid, double x) {
    return valid ? json(x) : json(nullptr);
}

static std::vector<std::string> g_obs_argv;

void server_obs_set_argv(int argc, char ** argv) {
    g_obs_argv.clear();
    bool redact = false;
    for (int i = 0; i < argc; i++) {
        const std::string a = argv[i] ? argv[i] : "";
        g_obs_argv.push_back(redact ? std::string("<redacted>") : a);
        redact = a == "--api-key" || a == "--api-key-file";
        if (a.rfind("--api-key=", 0) == 0) {
            g_obs_argv.back() = "--api-key=<redacted>";
        }
    }
}

const std::vector<std::string> & server_obs_argv() {
    return g_obs_argv;
}

const char * server_obs_phase_name(server_obs_phase phase) {
    switch (phase) {
        case SERVER_OBS_PHASE_IDLE:    return "idle";
        case SERVER_OBS_PHASE_HOST:    return "host";
        case SERVER_OBS_PHASE_PREFILL: return "prefill";
        case SERVER_OBS_PHASE_DECODE:  return "decode";
        default:                       return "unknown";
    }
}

//
// server_obs_clock
//

void server_obs_clock::init() {
    w_t_ns = obs_steady_ns();
    t_last.store(w_t_ns, std::memory_order_relaxed);
}

void server_obs_clock::commit(server_obs_phase next, bool compute, uint32_t permille) {
    const int64_t now = obs_steady_ns();
    const int64_t dt  = now - w_t_ns;

    if (w_compute) {
        const int64_t p = dt * (int64_t) w_permille / 1000;
        w_ns[SERVER_OBS_PHASE_PREFILL] += p;
        w_ns[SERVER_OBS_PHASE_DECODE]  += dt - p;
    } else {
        w_ns[w_phase] += dt;
    }

    w_t_ns     = now;
    w_phase    = next;
    w_compute  = compute;
    w_permille = permille;

    const uint32_t s = seq.load(std::memory_order_relaxed);
    seq.store(s + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    for (int i = 0; i < SERVER_OBS_PHASE_COUNT; i++) {
        acc[i].store(w_ns[i], std::memory_order_relaxed);
    }
    t_last.store(now, std::memory_order_relaxed);
    cur.store((uint32_t) next | ((uint32_t) compute << 8) | (permille << 16), std::memory_order_relaxed);
    seq.store(s + 2, std::memory_order_release);
}

server_obs_clock::snapshot server_obs_clock::read() const {
    snapshot res;
    int64_t  t   = 0;
    uint32_t c   = 0;
    while (true) {
        const uint32_t s1 = seq.load(std::memory_order_acquire);
        if (s1 & 1) {
            std::this_thread::yield();
            continue;
        }
        for (int i = 0; i < SERVER_OBS_PHASE_COUNT; i++) {
            res.ns[i] = acc[i].load(std::memory_order_relaxed);
        }
        t = t_last.load(std::memory_order_relaxed);
        c = cur.load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (seq.load(std::memory_order_relaxed) == s1) {
            break;
        }
    }

    const int64_t now = obs_steady_ns();
    const int64_t dt  = std::max<int64_t>(0, now - t);

    const auto     phase    = (server_obs_phase) (c & 0xff);
    const bool     compute  = (c >> 8) & 1;
    const uint32_t permille = c >> 16;

    if (compute) {
        const int64_t p = dt * (int64_t) permille / 1000;
        res.ns[SERVER_OBS_PHASE_PREFILL] += p;
        res.ns[SERVER_OBS_PHASE_DECODE]  += dt - p;
    } else {
        res.ns[phase] += dt;
    }
    res.t_ns  = now;
    res.phase = phase;
    return res;
}

//
// server_obs_subscriber
//

void server_obs_subscriber::publish(const std::string & line) {
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (closed) {
            return;
        }
        if (queue.size() >= capacity) {
            queue.pop_front();
            dropped++;
        }
        queue.push_back(line);
    }
    cv.notify_one();
}

bool server_obs_subscriber::next(std::string & line, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return closed || !queue.empty(); });
    if (queue.empty()) {
        return false;
    }
    line = std::move(queue.front());
    queue.pop_front();
    return true;
}

void server_obs_subscriber::close() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        closed = true;
    }
    cv.notify_all();
}

//
// NVML, loaded at runtime so that the server has no link-time dependency on the driver
//

struct server_nvml {
    using nvml_ret = int;
    using nvml_dev = void *;

    struct nvml_memory      { unsigned long long total, free, used; };
    struct nvml_utilization { unsigned int gpu, memory; };

    static constexpr int NVML_SUCCESS          = 0;
    static constexpr int NVML_TEMPERATURE_GPU  = 0;
    static constexpr int NVML_CLOCK_SM         = 1;
    static constexpr int NVML_CLOCK_MEM        = 2;

    void * lib = nullptr;
    nvml_dev dev = nullptr;
    bool ok = false;
    bool energy_ok = false;
    std::string error;

    // static identity
    std::string name;
    std::string uuid;
    std::string driver_version;
    uint64_t    memory_total = 0;
    unsigned    sm_clock_max = 0;
    double      power_limit_w = 0.0;

    nvml_ret     (*init_v2)() = nullptr;
    nvml_ret     (*shutdown)() = nullptr;
    const char * (*error_string)(nvml_ret) = nullptr;
    nvml_ret (*get_handle_by_index)(unsigned, nvml_dev *) = nullptr;
    nvml_ret (*get_handle_by_pci)(const char *, nvml_dev *) = nullptr;
    nvml_ret (*get_name)(nvml_dev, char *, unsigned) = nullptr;
    nvml_ret (*get_uuid)(nvml_dev, char *, unsigned) = nullptr;
    nvml_ret (*get_driver_version)(char *, unsigned) = nullptr;
    nvml_ret (*get_memory)(nvml_dev, nvml_memory *) = nullptr;
    nvml_ret (*get_max_clock)(nvml_dev, int, unsigned *) = nullptr;
    nvml_ret (*get_clock)(nvml_dev, int, unsigned *) = nullptr;
    nvml_ret (*get_temperature)(nvml_dev, int, unsigned *) = nullptr;
    nvml_ret (*get_fan)(nvml_dev, unsigned *) = nullptr;
    nvml_ret (*get_power)(nvml_dev, unsigned *) = nullptr;
    nvml_ret (*get_power_limit)(nvml_dev, unsigned *) = nullptr;
    nvml_ret (*get_energy)(nvml_dev, unsigned long long *) = nullptr;
    nvml_ret (*get_utilization)(nvml_dev, nvml_utilization *) = nullptr;
    nvml_ret (*get_clock_reasons)(nvml_dev, unsigned long long *) = nullptr;
    nvml_ret (*get_pstate)(nvml_dev, int *) = nullptr;
    nvml_ret (*get_cuda_cc)(nvml_dev, int *, int *) = nullptr;
    nvml_ret (*get_cuda_driver)(int *) = nullptr;

    int cc_major = 0, cc_minor = 0;
    int cuda_driver = 0; // e.g. 13010

    template <typename F>
    void sym(F & fn, const char * name_primary, const char * name_fallback = nullptr) {
#if !defined(_WIN32)
        void * p = dlsym(lib, name_primary);
        if (!p && name_fallback) {
            p = dlsym(lib, name_fallback);
        }
        fn = reinterpret_cast<F>(p);
#else
        GGML_UNUSED(name_primary);
        GGML_UNUSED(name_fallback);
        fn = nullptr;
#endif
    }

    std::string err_str(nvml_ret r) const {
        return error_string ? std::string(error_string(r)) : std::to_string(r);
    }

    explicit server_nvml(const std::string & pci_bus_id) {
#if defined(_WIN32)
        GGML_UNUSED(pci_bus_id);
        error = "NVML is not supported on this platform";
#else
        lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!lib) {
            const char * e = dlerror();
            error = std::string("dlopen libnvidia-ml.so.1: ") + (e ? e : "unknown error");
            return;
        }
        sym(init_v2,             "nvmlInit_v2");
        sym(shutdown,            "nvmlShutdown");
        sym(error_string,        "nvmlErrorString");
        sym(get_handle_by_index, "nvmlDeviceGetHandleByIndex_v2");
        sym(get_handle_by_pci,   "nvmlDeviceGetHandleByPciBusId_v2");
        sym(get_name,            "nvmlDeviceGetName");
        sym(get_uuid,            "nvmlDeviceGetUUID");
        sym(get_driver_version,  "nvmlSystemGetDriverVersion");
        sym(get_memory,          "nvmlDeviceGetMemoryInfo");
        sym(get_max_clock,       "nvmlDeviceGetMaxClockInfo");
        sym(get_clock,           "nvmlDeviceGetClockInfo");
        sym(get_temperature,     "nvmlDeviceGetTemperature");
        sym(get_fan,             "nvmlDeviceGetFanSpeed");
        sym(get_power,           "nvmlDeviceGetPowerUsage");
        sym(get_power_limit,     "nvmlDeviceGetEnforcedPowerLimit");
        sym(get_energy,          "nvmlDeviceGetTotalEnergyConsumption");
        sym(get_utilization,     "nvmlDeviceGetUtilizationRates");
        sym(get_clock_reasons,   "nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons");
        sym(get_pstate,          "nvmlDeviceGetPerformanceState");
        sym(get_cuda_cc,         "nvmlDeviceGetCudaComputeCapability");
        sym(get_cuda_driver,     "nvmlSystemGetCudaDriverVersion_v2", "nvmlSystemGetCudaDriverVersion");

        if (!init_v2 || !shutdown || !get_handle_by_index) {
            error = "libnvidia-ml.so.1 lacks nvmlInit_v2 / nvmlDeviceGetHandleByIndex_v2";
            dlclose(lib);
            lib = nullptr;
            return;
        }
        nvml_ret r = init_v2();
        if (r != NVML_SUCCESS) {
            error = "nvmlInit_v2: " + err_str(r);
            dlclose(lib);
            lib = nullptr;
            return;
        }
        r = -1;
        if (!pci_bus_id.empty() && get_handle_by_pci) {
            r = get_handle_by_pci(pci_bus_id.c_str(), &dev);
        }
        if (r != NVML_SUCCESS) {
            r = get_handle_by_index(0, &dev);
        }
        if (r != NVML_SUCCESS) {
            error = "nvmlDeviceGetHandle: " + err_str(r);
            shutdown();
            dlclose(lib);
            lib = nullptr;
            return;
        }
        ok = true;

        char buf[256];
        if (get_name && get_name(dev, buf, sizeof(buf)) == NVML_SUCCESS) {
            name = std::string(buf, strnlen(buf, sizeof(buf)));
        }
        if (get_uuid && get_uuid(dev, buf, sizeof(buf)) == NVML_SUCCESS) {
            uuid = std::string(buf, strnlen(buf, sizeof(buf)));
        }
        if (get_driver_version && get_driver_version(buf, sizeof(buf)) == NVML_SUCCESS) {
            driver_version = std::string(buf, strnlen(buf, sizeof(buf)));
        }
        nvml_memory mem {};
        if (get_memory && get_memory(dev, &mem) == NVML_SUCCESS) {
            memory_total = mem.total;
        }
        if (get_max_clock) {
            get_max_clock(dev, NVML_CLOCK_SM, &sm_clock_max);
        }
        unsigned limit_mw = 0;
        if (get_power_limit && get_power_limit(dev, &limit_mw) == NVML_SUCCESS) {
            power_limit_w = limit_mw / 1000.0;
        }
        // the energy counter is a fixed capability of the board, probe it once
        unsigned long long mj = 0;
        energy_ok = get_energy && get_energy(dev, &mj) == NVML_SUCCESS;
        if (get_cuda_cc) {
            get_cuda_cc(dev, &cc_major, &cc_minor);
        }
        if (get_cuda_driver) {
            get_cuda_driver(&cuda_driver);
        }
#endif
    }

    ~server_nvml() {
#if !defined(_WIN32)
        if (ok) {
            shutdown();
        }
        if (lib) {
            dlclose(lib);
        }
#endif
    }

    // board energy since driver load, in mJ; costs a few ms, never call it on the server loop
    bool energy_mj(uint64_t & out) const {
        unsigned long long mj = 0;
        if (!ok || !energy_ok || get_energy(dev, &mj) != NVML_SUCCESS) {
            return false;
        }
        out = mj;
        return true;
    }

    // GET /telemetry `gpu`, tensorninja's shape: every field present, 0 where a read failed
    // server_joules / idle_watts: the sampler's board energy since server start and idle calibration (< 0 = none)
    json live_json(double server_joules, double w_idle, int & pstate_out) const {
        pstate_out = -1;
        if (!ok) {
            return json { {"available", false}, {"error", error} };
        }
        unsigned temp = 0, fan = 0, power_mw = 0, sm = 0, memclk = 0;
        if (get_temperature) { get_temperature(dev, NVML_TEMPERATURE_GPU, &temp); }
        if (get_fan)         { get_fan(dev, &fan); }
        // trailing-average board power, the right statistic for a 1 Hz gauge
        if (get_power)       { get_power(dev, &power_mw); }
        if (get_clock)       { get_clock(dev, NVML_CLOCK_SM, &sm); get_clock(dev, NVML_CLOCK_MEM, &memclk); }
        nvml_utilization util {};
        if (get_utilization) { get_utilization(dev, &util); }
        nvml_memory mem {};
        if (!get_memory || get_memory(dev, &mem) != NVML_SUCCESS) {
            mem.total = memory_total;
        }
        uint64_t mj = 0;
        const bool have_mj = energy_mj(mj);
        int pstate = 0;
        if (get_pstate && get_pstate(dev, &pstate) == NVML_SUCCESS) {
            pstate_out = pstate;
        }
        json reasons_arr = json::array();
        unsigned long long bits = 0;
        if (get_clock_reasons && get_clock_reasons(dev, &bits) == NVML_SUCCESS) {
            static const struct { unsigned long long bit; const char * name; } reasons[] = {
                { 0x001ULL, "gpu_idle"                    },
                { 0x002ULL, "applications_clocks_setting" },
                { 0x004ULL, "sw_power_cap"                },
                { 0x008ULL, "hw_slowdown"                 },
                { 0x010ULL, "sync_boost"                  },
                { 0x020ULL, "sw_thermal_slowdown"         },
                { 0x040ULL, "hw_thermal_slowdown"         },
                { 0x080ULL, "hw_power_brake_slowdown"     },
                { 0x100ULL, "display_clock_setting"       },
            };
            for (const auto & r : reasons) {
                if (bits & r.bit) {
                    reasons_arr.push_back(r.name);
                }
            }
        }
        return json {
            {"available",                  true},
            {"name",                       name},
            {"uuid",                       uuid},
            {"driver_version",             driver_version},
            {"temperature_c",              temp},
            {"fan_percent",                fan},
            {"power_watts",                r3(power_mw / 1000.0)},
            {"power_limit_watts",          r3(power_limit_w)},
            {"energy_available",           energy_ok},
            {"energy_joules_total",        have_mj ? json(r3(mj / 1000.0)) : json(nullptr)},
            {"server_energy_joules",       energy_ok ? json(r3(server_joules)) : json(nullptr)},
            {"idle_watts",                 energy_ok ? json(r3(std::max(0.0, w_idle))) : json(nullptr)},
            {"utilization_gpu_percent",    util.gpu},
            {"utilization_memory_percent", util.memory},
            {"sm_clock_mhz",               sm},
            {"sm_clock_max_mhz",           sm_clock_max},
            {"memory_clock_mhz",           memclk},
            {"memory_used_bytes",          (uint64_t) mem.used},
            {"memory_total_bytes",         (uint64_t) mem.total},
            // nvmlDeviceGetPcieThroughput is deliberately not read: it samples for ~20 ms inside the driver and
            // measurably slowed decode (~1-2.5 %) when polled at 1 Hz on an RTX 4090; reported as 0
            {"pcie_rx_bytes_per_second",   0},
            {"pcie_tx_bytes_per_second",   0},
            {"throttle_reasons",           reasons_arr},
        };
    }

    // server_start.lovelace.gpu: the static identity
    json static_json() const {
        json res = { {"available", ok} };
        if (!ok) {
            res["error"] = error;
            return res;
        }
        res["name"]               = name;
        res["uuid"]               = uuid;
        res["driver_version"]     = driver_version;
        res["memory_total_bytes"] = memory_total;
        res["sm_clock_max_mhz"]   = sm_clock_max;
        res["power_limit_watts"]  = r3(power_limit_w);
        res["energy_available"]   = energy_ok;
        return res;
    }
};

//
// server_observer
//

static std::string obs_random_hex(size_t n) {
    std::random_device rd;
    std::mt19937_64 rng(((uint64_t) rd() << 32) ^ rd() ^ (uint64_t) obs_steady_ns());
    static const char * hex = "0123456789abcdef";
    std::string res(n, '0');
    for (auto & c : res) {
        c = hex[rng() & 0xf];
    }
    return res;
}

static std::string obs_hex16(uint64_t v) {
    if (v == 0) {
        return "";
    }
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) v);
    return buf;
}

static constexpr int OBS_NINFER_SCHEMA   = 20;
static constexpr int OBS_LOVELACE_SCHEMA = 2;
static constexpr const char * OBS_ARTIFACT_TYPE = "ninfer_serve_request_log";

static const char * obs_protocol(const server_obs_record & rec) {
    const std::string_view ep = rec.endpoint;
    if (ep == "oai_chat")  { return "openai_chat_completions"; }
    if (ep == "oai_cmpl")  { return "openai_completions"; }
    if (ep == "oai_resp")  { return "openai_responses"; }
    if (ep == "anthropic") { return "anthropic_messages"; }
    if (ep == "oai_embd")  { return "openai_embeddings"; }
    if (ep == "oai_asr")   { return "openai_transcriptions"; }
    const std::string_view tt = rec.task_type;
    if (tt == "infill")    { return "infill"; }
    if (tt == "embedding") { return "embeddings"; }
    if (tt == "rerank")    { return "rerank"; }
    return "native";
}

// the continuation-cache tier a request's reuse came from: 0 = l1 (GPU-resident), 1 = l2 (RAM), 2 = l3 (NVMe), -1 = none
static int obs_source_tier(const server_obs_record & rec) {
    if (rec.n_cached_tokens == 0) {
        return -1;
    }
    if (rec.session_tier == nullptr) {
        return 0; // slot KV / prompt cache reuse without a session store
    }
    const std::string_view t = rec.session_tier;
    if (t == "ram")  { return 1; }
    if (t == "disk") { return 2; }
    return 0; // resident in VRAM (or a new session that still reused slot KV)
}

server_observer::server_observer(const server_observer_config & cfg) : cfg(cfg) {
    server_instance_id = obs_random_hex(16);
    t_start_ms         = now_unix_ms();
    clock.init();

    snap.slots.resize(std::max(0, cfg.n_slots));
    snap.sessions.reserve(256);

    nvml = std::make_unique<server_nvml>(cfg.gpu_pci_bus_id);
    energy_available = nvml->ok && nvml->energy_ok;
    if (!nvml->ok) {
        SRV_INF("telemetry: GPU board sensors unavailable (%s)\n", nvml->error.c_str());
    } else {
        SRV_INF("telemetry: NVML attached to %s, energy counter %s\n", nvml->name.c_str(), energy_available ? "available" : "unavailable");
    }

    if (!cfg.jsonl_path.empty()) {
        jsonl = fopen(cfg.jsonl_path.c_str(), "ab");
        if (!jsonl) {
            throw std::runtime_error("failed to open --request-log-jsonl file: " + cfg.jsonl_path);
        }
    }
}

server_observer::~server_observer() {
    stop();
}

void server_observer::start(server_obs_static s) {
    st = std::move(s);

    server_obs_record rec;
    rec.kind      = SERVER_OBS_REC_RAW;
    rec.event     = "server_start";
    rec.t_unix_ms = t_start_ms;
    rec.body      = server_start_json();
    rec.ext       = {
        {"build_info", st.build_info},
        {"model", {
            {"path",        st.model_path},
            {"name",        st.model_desc},
            {"ftype",       st.model_ftype},
            {"n_params",    st.model_n_params},
            {"size_bytes",  st.model_size},
            {"n_ctx_train", st.n_ctx_train},
        }},
        {"config", {
            {"n_ctx",         st.n_ctx},
            {"n_ctx_slot",    st.n_ctx_slot},
            {"n_parallel",    st.n_slots},
            {"n_batch",       st.n_batch},
            {"n_ubatch",      st.n_ubatch},
            {"kv_unified",    st.kv_unified},
            {"cache_type_k",  st.cache_type_k},
            {"cache_type_v",  st.cache_type_v},
            {"flash_attn",    st.flash_attn},
            {"cache_ram_mib", st.cache_ram_mib},
            {"n_gpu_layers",  st.n_gpu_layers},
            {"speculative", {
                {"types", st.spec_types},
                {"n_max", st.spec_n_max},
                {"n_min", st.spec_n_min},
            }},
            {"throughput_interval_ms", cfg.throughput_interval_ms},
            {"energy_sample_hz",       cfg.sample_hz},
            {"request_log_jsonl",      cfg.jsonl_path.empty() ? json(nullptr) : json(cfg.jsonl_path)},
        }},
        {"gpu",  nvml->static_json()},
        {"vram", st.vram},
    };
    emit(std::move(rec));

    writer_thread  = std::thread([this] { writer_loop(); });
    sampler_thread = std::thread([this] { sampler_loop(); });
}

void server_observer::stop() {
    if (sampler_thread.joinable()) {
        {
            std::lock_guard<std::mutex> lock(mtx_sampler_stop);
            sampler_stop = true;
        }
        cv_sampler_stop.notify_all();
        sampler_thread.join();
    }
    if (writer_thread.joinable()) {
        {
            std::lock_guard<std::mutex> lock(mtx_queue);
            stopping = true;
        }
        cv_queue.notify_all();
        writer_thread.join();
    }
    {
        std::lock_guard<std::mutex> lock(mtx_subs);
        for (auto & s : subs) {
            s->close();
        }
    }
    if (jsonl) {
        fclose(jsonl);
        jsonl = nullptr;
    }
}

void server_observer::emit(server_obs_record && rec) {
    {
        std::lock_guard<std::mutex> lock(mtx_queue);
        queue.push_back(std::move(rec));
    }
    cv_queue.notify_one();
}

void server_observer::writer_loop() {
    std::vector<server_obs_record> batch;
    while (true) {
        {
            std::unique_lock<std::mutex> lock(mtx_queue);
            cv_queue.wait(lock, [&] { return stopping || !queue.empty(); });
            if (queue.empty() && stopping) {
                break;
            }
            batch.swap(queue);
        }
        for (auto & rec : batch) {
            const bool retain = rec.kind == SERVER_OBS_REC_RAW && std::string_view(rec.event) == "server_start";
            std::string line = format(rec, seq_next++);
            if (jsonl) {
                fwrite(line.data(), 1, line.size(), jsonl);
                fputc('\n', jsonl);
            }
            publish_line(std::move(line), retain);
        }
        batch.clear();
        if (jsonl) {
            fflush(jsonl);
        }
    }
}

void server_observer::publish_line(std::string && line, bool retain) {
    std::lock_guard<std::mutex> lock(mtx_subs);
    for (auto & s : subs) {
        s->publish(line);
    }
    if (retain) {
        server_start_line = std::move(line);
    } else {
        replay.push_back(std::move(line));
        while (replay.size() > cfg.replay_capacity) {
            replay.pop_front();
        }
    }
    n_emitted.fetch_add(1, std::memory_order_relaxed);
}

std::shared_ptr<server_obs_subscriber> server_observer::subscribe(std::vector<std::string> & backlog) {
    auto sub = std::make_shared<server_obs_subscriber>(cfg.subscriber_capacity);
    std::lock_guard<std::mutex> lock(mtx_subs);
    backlog.clear();
    backlog.reserve(replay.size() + 1);
    if (!server_start_line.empty()) {
        backlog.push_back(server_start_line);
    }
    backlog.insert(backlog.end(), replay.begin(), replay.end());
    subs.push_back(sub);
    return sub;
}

void server_observer::unsubscribe(const std::shared_ptr<server_obs_subscriber> & sub) {
    sub->close();
    std::lock_guard<std::mutex> lock(mtx_subs);
    auto it = std::find(subs.begin(), subs.end(), sub);
    if (it != subs.end()) {
        std::lock_guard<std::mutex> lock_sub(sub->mtx);
        n_dropped_closed.fetch_add(sub->dropped, std::memory_order_relaxed);
        subs.erase(it);
    }
}

std::string server_observer::record_event(const std::string & line) {
    // the envelope is formatted first, so the first `"event":"` is the record's own
    static const std::string key = "\"event\":\"";
    const size_t k = line.find(key);
    if (k == std::string::npos) {
        return "message";
    }
    const size_t b = k + key.size();
    const size_t e = line.find('"', b);
    return e == std::string::npos ? std::string("message") : line.substr(b, e - b);
}

//
// records
//

static json obs_arena(uint64_t bytes) {
    return json { {"capacity_bytes", bytes}, {"used_bytes", bytes}, {"peak_used_bytes", bytes} };
}

json server_observer::server_start_json() const {
    const uint64_t mib = 1024ull * 1024ull;
    json lora_names = json::array();
    for (const auto & n : st.lora_names) {
        lora_names.push_back(n);
    }
    const json preset = {
        {"temperature",       st.temperature},
        {"top_p",             st.top_p},
        {"top_k",             st.top_k},
        {"min_p",             st.min_p},
        {"presence_penalty",  st.presence_penalty},
        {"frequency_penalty", st.frequency_penalty},
    };
    const char * tiers = !st.sessions ? "l1" : st.session_dir.empty() ? "l1-l2" : "l1-l2-l3";
    const std::string kv_cache = st.cache_type_k == st.cache_type_v ? st.cache_type_k : st.cache_type_k + "/" + st.cache_type_v;
    const std::string cuda_driver = nvml->cuda_driver > 0
        ? std::to_string(nvml->cuda_driver / 1000) + "." + std::to_string((nvml->cuda_driver % 1000) / 10)
        : std::string();

    return json {
        {"server", {
            {"host",                      st.host},
            {"port",                      st.port},
            {"public_model_id",           st.model_id},
            {"api_key_configured",        st.api_key},
            {"cors_enabled",              true},
            {"max_request_bytes",         0},
            {"request_log_jsonl",         cfg.jsonl_path},
            {"slot_save_path",            st.slot_save_path},
            {"default_output_tokens",     st.default_n_predict},
            {"default_thinking",          st.default_thinking},
            {"default_preserve_thinking", st.preserve_thinking},
        }},
        {"artifact", {
            {"path",                 st.model_path},
            {"size_bytes",           st.file_bytes >= 0 ? json(st.file_bytes) : json(nullptr)},
            {"target",               st.model_arch},
            {"weights_id",           st.model_desc},
            {"bytes_read",           st.model_size},
            {"host_to_device_bytes", st.model_bytes},
            {"peak_staging_bytes",   0},
            {"tensor_count",         st.tensor_count},
            {"resource_count",       0},
            {"load_seconds",         r3(st.load_seconds)},
            {"upload_seconds",       0.0},
        }},
        {"adapters", {
            {"count",        st.lora_names.size()},
            {"names",        lora_names},
            {"rank",         0},
            {"slots",        0},
            {"device_bytes", 0},
            {"file_bytes",   0},
        }},
        {"engine", {
            {"device",                      0},
            {"max_context",                 st.n_ctx_slot},
            {"kv_capacity_mode",            "explicit"},
            {"kv_capacity",                 st.n_ctx},
            {"kv_capacity_page_groups",     st.n_ctx},
            {"kv_capacity_max_page_groups", st.n_ctx},
            {"max_concurrency",             st.n_slots},
            {"max_pending_requests",        0},
            {"pending_timeout_ms",          0},
            {"prefill_chunk",               st.n_batch},
            {"log_stats_interval_ms",       cfg.throughput_interval_ms},
            {"kv_cache",                    kv_cache},
            {"vision",                      st.vision},
            {"cuda_graph",                  st.cuda_graph},
            {"prefix_reuse",                true},
            {"prefix_checkpoint_policy",    "ctx-checkpoints:" + std::to_string(st.ctx_checkpoints)},
            {"speculative_backend",         st.spec_backend},
            {"speculative_draft_window",    st.spec_n_max},
            {"proposal_head",               "full"},
            {"continuation_cache", {
                {"tiers",                     tiers},
                {"policy",                    "session-store"},
                {"l1_capacity_mib",           st.context_bytes / mib},
                {"l2_capacity_mib",           st.session_ram_bytes / mib},
                {"l3_capacity_mib",           st.session_disk_limit / mib},
                {"directory",                 st.session_dir},
                {"namespace",                 ""},
                {"l1_idle_ttl_seconds",       0},
                {"l2_idle_ttl_seconds",       st.session_ttl_s},
                {"l3_idle_ttl_seconds",       st.session_ttl_s},
                {"persist_interval_seconds",  0},
                {"persist_min_tokens",        0},
                {"filesystem_reserve_mib",    0},
                {"prefix_checkpoint_history", st.ctx_checkpoints},
            }},
        }},
        {"sampling_defaults", {
            {"thinking",     preset},
            {"non_thinking", preset},
            {"server_overrides", {
                {"temperature", nullptr}, {"top_p", nullptr}, {"top_k", nullptr}, {"min_p", nullptr},
                {"presence_penalty", nullptr}, {"frequency_penalty", nullptr}, {"seed", nullptr},
            }},
            {"omitted_seed", "random"},
            {"greedy",       false},
        }},
        {"memory", {
            {"weights",                           obs_arena(st.model_bytes)},
            {"sequence",                          obs_arena(st.context_bytes + st.draft_context_bytes)},
            {"workspace",                         obs_arena(st.compute_bytes)},
            {"request_transient",                 obs_arena(0)},
            {"minimum_runtime_reservation_bytes", 0},
            {"kv_capacity_increment_bytes",       0},
            {"runtime_reservation_bytes",         0},
            {"available_after_weights_bytes",     st.device_total > st.model_bytes ? st.device_total - st.model_bytes : 0},
            {"available_after_startup_bytes",     st.device_free},
            {"kv_capacity_headroom_bytes",        0},
            {"planned_slack_bytes",               0},
            {"cuda_graph_allowance_bytes",        0},
            {"cuda_graph_observed_bytes",         0},
            {"kv_payload_bytes",                  st.context_bytes},
            {"lora_bank_bytes",                   0},
        }},
        {"environment", {
            {"device",                    0},
            {"gpu_name",                  nvml->name},
            {"gpu_uuid",                  nvml->uuid},
            {"total_device_memory_bytes", nvml->memory_total},
            {"compute_capability_major",  nvml->cc_major},
            {"compute_capability_minor",  nvml->cc_minor},
            {"cuda_compile_version",      ""},
            {"cuda_runtime_version",      ""},
            {"cuda_driver_version",       cuda_driver},
        }},
        {"argv", st.argv},
    };
}

json server_observer::request_json(const server_obs_record & rec) const {
    return json {
        {"request_id",                        std::max(0, rec.id_task)},
        {"x_request_id",                      rec.id_task >= 0 ? std::to_string(rec.id_task) : std::string()},
        {"protocol",                          obs_protocol(rec)},
        {"model",                             st.model_id},
        {"stream",                            rec.stream},
        {"message_count",                     0},
        {"requested_output_tokens",           rec.n_predict},
        {"requested_output_tokens_source",    rec.n_predict_client ? "client" : "server_default"},
        {"tool_count",                        0},
        {"tool_choice",                       "auto"},
        {"has_tool_history",                  false},
        {"enable_thinking",                   rec.thinking},
        {"preserve_thinking",                 st.preserve_thinking},
        {"preserve_thinking_semantic_change", false},
        {"adapter",                           ""},
        {"prompt_cache_key_digest",           obs_hex16(rec.session_id)},
        {"sampling", {
            {"temperature",       rec.temperature},
            {"top_p",             rec.top_p},
            {"top_k",             rec.top_k},
            {"min_p",             rec.min_p},
            {"presence_penalty",  rec.presence_penalty},
            {"frequency_penalty", rec.frequency_penalty},
            {"seed",              rec.seed},
        }},
    };
}

std::string server_observer::format(server_obs_record & rec, uint64_t seq) {
    json res = {
        {"artifact_type",      OBS_ARTIFACT_TYPE},
        {"schema_version",     OBS_NINFER_SCHEMA},
        {"event",              ""},
        {"timestamp_unix_ms",  rec.t_unix_ms},
        {"server_instance_id", server_instance_id},
    };
    json ext = {
        {"schema_version", OBS_LOVELACE_SCHEMA},
        {"seq",            seq},
    };
    const auto opt_id = [](int32_t v) { return v >= 0 ? json(v) : json(nullptr); };

    switch (rec.kind) {
        case SERVER_OBS_REC_RAW:
            {
                res["event"] = rec.event;
                if (std::string_view(rec.event) == "throughput") {
                    throughput_body(rec.body, rec.ext);
                }
                for (const auto & [key, val] : rec.body.items()) {
                    res[key] = val;
                }
                if (rec.ext.is_object()) {
                    for (const auto & [key, val] : rec.ext.items()) {
                        ext[key] = val;
                    }
                }
            } break;
        case SERVER_OBS_REC_REQUEST_START:
            {
                res["event"]   = "request_start";
                res["request"] = request_json(rec);
                ext["id_task"]           = opt_id(rec.id_task);
                ext["id_slot"]           = opt_id(rec.id_slot);
                ext["id_parent"]         = opt_id(rec.id_parent);
                ext["task_type"]         = rec.task_type;
                ext["endpoint"]          = rec.endpoint;
                ext["n_prompt_tokens"]   = rec.n_prompt_tokens;
                ext["n_predict"]         = rec.n_predict;
                ext["n_ctx_slot"]        = rec.n_ctx_slot;
                ext["queue_wait_ms"]     = r3(std::max(0.0, rec.queue_wait_ms));
                ext["wait_reason"]       = rec.wait_reason;
                ext["n_unparked_tokens"] = rec.n_unparked_tokens;

                std::lock_guard<std::mutex> lock(mtx_cc);
                cc.admitted++;
                cc.queue_s += std::max(0.0, rec.queue_wait_ms) / 1e3;
            } break;
        case SERVER_OBS_REC_REQUEST_DONE:
            {
                const int  tier   = obs_source_tier(rec);
                const bool known  = rec.session_tier != nullptr && std::string_view(rec.session_tier) != "new";
                const bool is_new = rec.session_tier != nullptr && std::string_view(rec.session_tier) == "new";
                static const char * tier_names[] = { "l1", "l2", "l3" };
                const char * miss = tier >= 0 ? "none" : is_new ? "no_alias" : known ? "not_deeper" : "not_attempted";
                const double restore_s = rec.restore_ready_ms > 0.0 ? rec.restore_ready_ms / 1e3 : 0.0;

                res["event"]   = "request_done";
                res["request"] = request_json(rec);
                res["result"]  = {
                    {"finish_reason",                rec.finish_reason},
                    {"prompt_tokens",                rec.n_prompt_tokens},
                    {"completion_tokens",            rec.n_completion_tokens},
                    {"computed_prefill_tokens",      rec.n_prefill_tokens},
                    {"prefix_cache_hit_tokens",      rec.n_cached_tokens},
                    {"prefix_reuse_path",            rec.n_cached_tokens == 0 ? "full_reset" : tier > 0 ? "restore_turn_checkpoint" : "append_frontier"},
                    {"tool_call_count",              0},
                    {"tool_calls_incomplete",        0},
                    {"tool_calls_invalid_arguments", 0},
                    {"tool_call_discarded_bytes",    0},
                };
                res["timings_seconds"] = {
                    {"prepare", 0.0},
                    {"ttft",    rec.t_ttft_ms >= 0.0 ? r6(rec.t_ttft_ms / 1e3) : 0.0},
                    {"vision",  0.0},
                    {"queue",   r6(std::max(0.0, rec.queue_wait_ms) / 1e3)},
                    {"restore", r6(restore_s)},
                    {"publish", 0.0},
                    {"prefill", r6(rec.t_prefill_ms / 1e3)},
                    {"decode",  r6(rec.t_decode_ms / 1e3)},
                    {"total",   r6(rec.t_total_ms / 1e3)},
                };
                res["speculative"] = {
                    {"backend",               st.spec_backend},
                    {"draft_window",          st.spec_n_max},
                    {"rounds",                rec.n_verif_steps},
                    {"drafted_tokens",        rec.n_drafted},
                    {"accepted_tokens",       rec.n_accepted},
                    {"fallback_steps",        0},
                    {"accepted_per_position", rec.accepted_per_pos},
                };
                res["continuation_cache"] = {
                    {"deepest_candidate_agreement",   nullptr},
                    {"source",                        tier >= 0 ? tier_names[tier] : "none"},
                    {"alias_kind",                    known ? "routed_session" : "none"},
                    {"final_miss_reason",             miss},
                    {"restore_failure",               "none"},
                    {"lookup_microseconds",           0},
                    {"preflight_microseconds",        0},
                    {"restore_microseconds",          (uint64_t) std::llround(rec.restore_ms * 1e3)},
                    {"restored_tokens",               rec.n_cached_tokens},
                    {"restored_bytes",                rec.restore_bytes},
                    {"cache_write_tokens",            0},
                    {"restore_gate_checks",           0},
                    {"restore_gate_passes",           0},
                    {"restore_target_lane",           std::max(0, rec.id_slot)},
                    {"admitted_lane",                 std::max(0, rec.id_slot)},
                    {"destructive_rollback",          false},
                    {"completion_publication_queued", rec.sync_queued},
                };
                ext["id_task"]           = opt_id(rec.id_task);
                ext["id_slot"]           = opt_id(rec.id_slot);
                ext["id_parent"]         = opt_id(rec.id_parent);
                ext["task_type"]         = rec.task_type;
                ext["endpoint"]          = rec.endpoint;
                ext["n_predict"]         = rec.n_predict;
                ext["n_ctx_slot"]        = rec.n_ctx_slot;
                ext["n_ctx_used"]        = rec.n_ctx_used;
                ext["truncated"]         = rec.truncated;
                ext["wait_reason"]       = rec.wait_reason;
                ext["n_unparked_tokens"] = rec.n_unparked_tokens;
                ext["queue_wait_ms"]     = r3(std::max(0.0, rec.queue_wait_ms));
                ext["session"] = rec.session_tier == nullptr ? json(nullptr) : json {
                    {"id",               rec.session_id},
                    {"tier",             rec.session_tier},
                    {"restore_bytes",    rec.restore_bytes},
                    {"restore_ms",       r3(rec.restore_ms)},
                    {"park_ms",          r3(rec.park_ms)},
                    {"restore_issue_ms", r3(rec.restore_issue_ms)},
                    {"restore_ready_ms", opt_num(rec.restore_ready_ms >= 0.0, r3(rec.restore_ready_ms))},
                    {"prompt_start_ms",  opt_num(rec.prompt_start_ms >= 0.0, r3(rec.prompt_start_ms))},
                    {"prefetch_hit",     rec.prefetch_hit},
                };
                ext["rates"] = {
                    {"prefill_tps", opt_num(rec.prefill_tps > 0.0, r3(rec.prefill_tps))},
                    {"decode_tps",  opt_num(rec.decode_tps  > 0.0, r3(rec.decode_tps))},
                };
                ext["speculative"] = { {"n_verif_steps", rec.n_verif_steps} };
                ext["energy"] = !rec.energy_valid ? json(nullptr) : json {
                    {"prefill_joules", r3(rec.prefill_joules)},
                    {"decode_joules",  r3(rec.decode_joules)},
                    {"total_joules",   r3(rec.prefill_joules + rec.decode_joules)},
                };

                std::lock_guard<std::mutex> lock(mtx_cc);
                cc.requests_done++;
                cc.prefix_hit_tokens += rec.n_cached_tokens;
                cc.drafted           += rec.n_drafted;
                cc.accepted          += rec.n_accepted;
                cc.lookup_hits       += known;
                cc.lookup_misses     += is_new;
                cc.no_alias          += is_new;
                if (tier >= 0) {
                    cc.restores++;
                    cc.restored_tokens += rec.n_cached_tokens;
                    cc.restored_bytes  += rec.restore_bytes;
                    cc.tier_restores[tier]++;
                    cc.tier_tokens[tier] += rec.n_cached_tokens;
                    cc.tier_bytes[tier]  += rec.restore_bytes;
                    if (rec.restore_bytes > 0) {
                        cc.tier_restore_us[tier]  += (uint64_t) std::llround(rec.restore_ms * 1e3);
                        cc.tier_restore_ops[tier] += 1;
                    }
                }
            } break;
        case SERVER_OBS_REC_SCHED:
            {
                res["event"] = "lovelace_sched";
                ext["action"]   = rec.event;
                ext["id_slot"]  = rec.id_slot;
                ext["id_task"]  = opt_id(rec.id_task);
                ext["n_tokens"] = rec.sched_tokens;
                ext["bytes"]    = rec.restore_bytes;
                ext["ms"]       = r3(rec.restore_ms);
                ext["host_ms"]  = r3(rec.park_ms);
                ext["gbps"]     = rec.restore_ms > 0.0 ? r3(rec.restore_bytes / 1e6 / rec.restore_ms) : 0.0;
            } break;
        case SERVER_OBS_REC_REQUEST_ERROR:
            {
                res["event"]   = "request_error";
                res["request"] = request_json(rec);
                res["error"]   = { {"message", rec.error_msg} };
                ext["id_task"]         = opt_id(rec.id_task);
                ext["id_slot"]         = opt_id(rec.id_slot);
                ext["task_type"]       = rec.task_type;
                ext["endpoint"]        = rec.endpoint;
                ext["stage"]           = rec.stage;
                ext["error_type"]      = rec.error_type;
                ext["n_prompt_tokens"] = rec.n_prompt_tokens;
                ext["queue_wait_ms"]   = opt_num(rec.queue_wait_ms >= 0.0, r3(rec.queue_wait_ms));
            } break;
    }

    res["lovelace"] = std::move(ext);
    return res.dump_safe();
}

//
// continuation cache and scheduler views, shared by the throughput records, /telemetry and /metrics
//

namespace {

// a consistent copy of the loop's fixed-size view, taken under the snapshot lock (no allocation)
struct obs_loop_view {
    int32_t n_busy = 0, n_prompt = 0, n_gen = 0, n_preempted = 0, n_retained = 0;
    int32_t queue_deferred = 0, sched_waiting = 0;
    int64_t cells_used = 0, cells_reserved = 0;
    server_obs_sched_totals sched;
    bool    sessions_on = false;
    server_session_store::stats_t ss;
    bool    pc_enabled = false;
    size_t  pc_bytes = 0, pc_limit = 0, pc_tokens = 0, pc_tokens_limit = 0, pc_entries = 0;
};

// caller holds snap.mtx
obs_loop_view obs_read_view_locked(const server_obs_snapshot & snap) {
    obs_loop_view v;
    v.n_busy = snap.n_busy; v.n_prompt = snap.n_prompt; v.n_gen = snap.n_gen;
    v.n_preempted = snap.n_preempted; v.n_retained = snap.n_retained;
    v.queue_deferred = snap.queue_deferred; v.sched_waiting = snap.sched_waiting;
    v.cells_used = snap.cells_used; v.cells_reserved = snap.cells_reserved;
    v.sched = snap.sched;
    v.sessions_on = snap.sessions_on;
    v.ss = snap.sessions_stats;
    v.pc_enabled = snap.pc_enabled; v.pc_bytes = snap.pc_bytes; v.pc_limit = snap.pc_limit;
    v.pc_tokens = snap.pc_tokens; v.pc_tokens_limit = snap.pc_tokens_limit; v.pc_entries = snap.pc_entries;
    return v;
}

} // namespace

// the continuation-cache tier occupancy: l1 = GPU-resident sessions, l2 = RAM mirrors, l3 = NVMe files
static server_obs_occupancy obs_occupancy_of(const obs_loop_view & v, const server_obs_static & st) {
    server_obs_occupancy o;
    o.l1_entries   = v.sessions_on ? v.ss.n_gpu : (uint64_t) (v.n_busy + v.n_retained);
    o.l1_bytes     = (uint64_t) std::llround(v.cells_used * st.bytes_per_cell);
    if (v.sessions_on) {
        o.l2_entries = v.ss.n_ram + v.ss.n_io;
        o.l2_bytes   = v.ss.ram_bytes;
        o.l3_entries = v.ss.n_disk;
        o.l3_bytes   = v.ss.disk_bytes;
        o.l2_evictions     = v.ss.n_demote;
        o.l2_evicted_bytes = v.ss.io_bytes_written;
        o.publication_failures = v.ss.n_sync_stale;
        o.persisted  = v.ss.n_demote;
        o.persist_us = v.ss.io_write_s * 1e6;
    } else {
        o.l2_entries = v.pc_entries;
        o.l2_bytes   = v.pc_bytes;
    }
    o.l1_evictions = v.sched.n_park;
    o.l1_demotions = v.sched.n_park - std::min(v.sched.n_park, v.sched.n_park_drop);
    o.grows        = v.sched.n_grow;
    o.spills       = v.sched.n_preempt;
    o.publications = v.sched.n_sync;
    return o;
}

// writer thread: the sections of a throughput record that come from the request records and the loop's view
void server_observer::throughput_body(json & body, json & ext) {
    obs_loop_view v;
    {
        std::lock_guard<std::mutex> lock(snap.mtx);
        v = obs_read_view_locked(snap);
    }
    const server_obs_occupancy o = obs_occupancy_of(v, st);

    cc_totals c;
    {
        std::lock_guard<std::mutex> lock(mtx_cc);
        c = cc;
    }
    const cc_totals & p = cc_prev;
    const server_obs_occupancy & q = occ_prev;

    body["scheduler"] = {
        {"running",                 v.n_busy},
        {"prefilling",              v.n_prompt},
        {"decode_ready",            v.n_gen},
        {"waiting",                 v.queue_deferred + v.sched_waiting},
        {"admitted_requests",       c.admitted},
        {"queue_seconds_total",     r6(c.queue_s)},
        {"delta_admitted_requests", c.admitted - p.admitted},
        {"delta_queue_seconds",     r6(c.queue_s - p.queue_s)},
        {"rejected_overloaded",     0},
        {"rejected_queue_timeout",  0},
    };

    json cache = {
        {"lookup_hits",                          c.lookup_hits},
        {"lookup_misses",                        c.lookup_misses},
        {"preflight_candidate_rejections",       0},
        {"restore_successes",                    c.restores},
        {"restore_failures",                     0},
        {"restore_deferrals",                    0},
        {"delta_restore_deferrals",              0},
        {"delta_lookup_hits",                    c.lookup_hits - p.lookup_hits},
        {"delta_lookup_misses",                  c.lookup_misses - p.lookup_misses},
        {"delta_preflight_candidate_rejections", 0},
        {"delta_restore_failures",               0},
        {"delta_restore_successes",              c.restores - p.restores},
        {"delta_restored_tokens",                c.restored_tokens - p.restored_tokens},
        {"delta_restored_bytes",                 c.restored_bytes - p.restored_bytes},
        {"publication_successes",                o.publications},
        {"publication_failures",                 o.publication_failures},
        {"publication_superseded",               0},
        {"delta_publication_successes",          o.publications - q.publications},
        {"delta_publication_failures",           o.publication_failures - q.publication_failures},
        {"delta_publication_superseded",         0},
        {"restored_tokens",                      c.restored_tokens},
        {"restored_bytes",                       c.restored_bytes},
    };
    static const char * tier_l[3] = { "l1", "l2", "l3" };
    json tiers = json::object();
    for (int i = 0; i < 3; i++) {
        tiers[std::string(tier_l[i]) + "_restore_successes"] = c.tier_restores[i];
    }
    for (int i = 0; i < 3; i++) {
        tiers[std::string("delta_") + tier_l[i] + "_restore_successes"] = c.tier_restores[i] - p.tier_restores[i];
    }
    for (int i = 0; i < 3; i++) {
        tiers[std::string(tier_l[i]) + "_restored_tokens"] = c.tier_tokens[i];
    }
    for (int i = 0; i < 3; i++) {
        tiers[std::string("delta_") + tier_l[i] + "_restored_tokens"] = c.tier_tokens[i] - p.tier_tokens[i];
    }
    for (int i = 0; i < 3; i++) {
        tiers[std::string(tier_l[i]) + "_restored_bytes"] = c.tier_bytes[i];
    }
    for (int i = 0; i < 3; i++) {
        tiers[std::string("delta_") + tier_l[i] + "_restored_bytes"] = c.tier_bytes[i] - p.tier_bytes[i];
    }
    cache["tiers"]   = std::move(tiers);
    cache["aliases"] = {
        {"routed_session_restores",       c.restores},
        {"stable_prefix_restores",        0},
        {"delta_routed_session_restores", c.restores - p.restores},
        {"delta_stable_prefix_restores",  0},
    };
    static const char * miss_names[] = { "disabled", "no_alias", "entry_unavailable_or_corrupt", "not_deeper",
                                         "preflight_rejected", "rollback_conflict", "no_lane", "restore_failed" };
    json miss = json::object();
    for (const char * name : miss_names) {
        miss[name] = std::string_view(name) == "no_alias" ? c.no_alias : 0;
    }
    for (const char * name : miss_names) {
        miss[std::string("delta_") + name] = std::string_view(name) == "no_alias" ? c.no_alias - p.no_alias : 0;
    }
    cache["miss_reasons"] = std::move(miss);
    cache["latency_microseconds"] = {
        {"l2_lookup_total",                  0},
        {"l3_lookup_total",                  0},
        {"l2_restore_total",                 c.tier_restore_us[1]},
        {"l3_restore_total",                 c.tier_restore_us[2]},
        {"preflight_total",                  0},
        {"l2_lookup_delta",                  0},
        {"l3_lookup_delta",                  0},
        {"l2_restore_delta",                 c.tier_restore_us[1] - p.tier_restore_us[1]},
        {"l3_restore_delta",                 c.tier_restore_us[2] - p.tier_restore_us[2]},
        {"preflight_delta",                  0},
        {"l2_admission_total",               0},
        {"l3_persistence_total",             (uint64_t) std::llround(o.persist_us)},
        {"l2_admission_delta",               0},
        {"l3_persistence_delta",             (uint64_t) std::llround(std::max(0.0, o.persist_us - q.persist_us))},
        {"export_total",                     0},
        {"export_delta",                     0},
        {"export_operations_total",          0},
        {"export_operations_delta",          0},
        {"l2_lookup_operations_total",       0},
        {"l3_lookup_operations_total",       0},
        {"l2_restore_operations_total",      c.tier_restore_ops[1]},
        {"l3_restore_operations_total",      c.tier_restore_ops[2]},
        {"preflight_operations_total",       0},
        {"l2_admission_operations_total",    0},
        {"l3_persistence_operations_total",  o.persisted},
        {"l2_lookup_operations_delta",       0},
        {"l3_lookup_operations_delta",       0},
        {"l2_restore_operations_delta",      c.tier_restore_ops[1] - p.tier_restore_ops[1]},
        {"l3_restore_operations_delta",      c.tier_restore_ops[2] - p.tier_restore_ops[2]},
        {"preflight_operations_delta",       0},
        {"l2_admission_operations_delta",    0},
        {"l3_persistence_operations_delta",  o.persisted - q.persisted},
    };
    cache["occupancy"] = {
        {"l1_entries",                 o.l1_entries},
        {"l1_bytes",                   o.l1_bytes},
        {"l2_entries",                 o.l2_entries},
        {"l2_bytes",                   o.l2_bytes},
        {"l3_entries",                 o.l3_entries},
        {"l3_bytes",                   o.l3_bytes},
        {"l1_evictions",               o.l1_evictions},
        {"l1_demotions",               o.l1_demotions},
        {"kv_restore_reclaimed_lanes", 0},
        {"kv_growth_attempts",         o.grows},
        {"kv_growth_forced_spills",    o.spills},
        {"kv_growth_curtailed",        0},
        {"delta_l1_evictions",         o.l1_evictions - q.l1_evictions},
        {"delta_l1_demotions",         o.l1_demotions - q.l1_demotions},
        {"l2_evictions",               o.l2_evictions},
        {"l2_evicted_bytes",           o.l2_evicted_bytes},
        {"l3_evictions",               0},
        {"l3_evicted_bytes",           0},
        {"delta_l2_evictions",         o.l2_evictions - q.l2_evictions},
        {"delta_l2_evicted_bytes",     o.l2_evicted_bytes - q.l2_evicted_bytes},
        {"delta_l3_evictions",         0},
        {"delta_l3_evicted_bytes",     0},
    };
    cache["persistence_delta"] = {
        {"queued",    o.persisted - q.persisted},
        {"coalesced", 0},
        {"successes", o.persisted - q.persisted},
        {"failures",  0},
    };
    cache["persistence_total"] = {
        {"queued",    o.persisted},
        {"coalesced", 0},
        {"successes", o.persisted},
        {"failures",  0},
    };
    body["continuation_cache"] = std::move(cache);

    // lovelace: the session store over the interval
    if (v.sessions_on) {
        const auto & s  = v.ss;
        const auto & sp = sessions_prev;
        ext["sessions"] = {
            {"n_gpu",            s.n_gpu},
            {"n_ram",            s.n_ram},
            {"n_disk",           s.n_disk},
            {"n_io",             s.n_io},
            {"ram_bytes",        s.ram_bytes},
            {"disk_bytes",       s.disk_bytes},
            {"promotions",       s.n_promote - sp.n_promote},
            {"demotions",        s.n_demote - sp.n_demote},
            {"prefetches",       s.n_prefetch - sp.n_prefetch},
            {"prefetch_hits",    s.n_prefetch_hit - sp.n_prefetch_hit},
            {"io_bytes_read",    s.io_bytes_read - sp.io_bytes_read},
            {"io_bytes_written", s.io_bytes_written - sp.io_bytes_written},
        };
        sessions_prev = s;
    } else {
        ext["sessions"] = nullptr;
    }

    cc_prev  = c;
    occ_prev = o;
}

//
// sampler: board energy attribution, rolling window, throughput records
//

void server_observer::sampler_loop() {
    const int64_t period_ns   = 1000000000LL / std::max(1, cfg.sample_hz);
    const int64_t window_ns   = 10LL * 1000000000LL;
    const int64_t interval_ns = cfg.throughput_interval_ms * 1000000LL;

    // previous sample
    server_obs_clock::snapshot prev = clock.read();
    uint64_t prev_mj = 0;
    bool     prev_ok = nvml->energy_mj(prev_mj);

    // host time is split between prefill and decode pro rata; this keeps the last known split
    // for sampling periods that ran host code only while requests were in flight
    double ratio_prefill = 0.0;

    // idle baseline, calibrated over fully idle sampling periods
    double idle_acc_j = 0.0;
    double idle_acc_s = 0.0;

    // throughput interval base
    struct base_t {
        server_obs_clock::snapshot clk;
        uint64_t prefill = 0, decode = 0, drafted = 0, accepted = 0, calls = 0, busy = 0, rounds = 0, rows = 0;
        uint64_t started = 0, done = 0, rejected = 0;
        double   e_board = 0.0, e_prefill = 0.0, e_decode = 0.0, e_idle = 0.0;
        server_obs_sched_totals sched;
    };
    auto take_base = [&](const server_obs_clock::snapshot & clk) {
        base_t b;
        b.clk      = clk;
        b.prefill  = counters.prefill_tokens.load(std::memory_order_relaxed);
        b.decode   = counters.decode_tokens.load(std::memory_order_relaxed);
        b.drafted  = counters.drafted.load(std::memory_order_relaxed);
        b.accepted = counters.accepted.load(std::memory_order_relaxed);
        b.calls    = counters.decode_calls.load(std::memory_order_relaxed);
        b.busy     = counters.busy_slot_calls.load(std::memory_order_relaxed);
        b.rounds   = counters.decode_rounds.load(std::memory_order_relaxed);
        b.rows     = counters.decode_rows.load(std::memory_order_relaxed);
        b.started  = counters.req_started.load(std::memory_order_relaxed);
        b.done     = counters.req_done.load(std::memory_order_relaxed);
        b.rejected = counters.req_rejected.load(std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(snap.mtx);
            b.sched = snap.sched;
        }
        std::lock_guard<std::mutex> lock(mtx_state);
        b.e_board   = energy_board;
        b.e_prefill = energy_prefill;
        b.e_decode  = energy_decode;
        b.e_idle    = energy_idle;
        return b;
    };
    base_t base = take_base(prev);

    {
        std::lock_guard<std::mutex> lock(mtx_state);
        window.push_back({prev.t_ns, prev.ns, 0.0});
        last_clock   = prev;
        mark.valid   = energy_available;
    }

    while (true) {
        {
            std::unique_lock<std::mutex> lock(mtx_sampler_stop);
            if (cv_sampler_stop.wait_for(lock, std::chrono::nanoseconds(period_ns), [&] { return sampler_stop; })) {
                break;
            }
        }

        const server_obs_clock::snapshot cur_clk = clock.read();
        uint64_t mj = 0;
        const bool cur_ok = nvml->energy_mj(mj);

        std::array<int64_t, SERVER_OBS_PHASE_COUNT> dt;
        int64_t total = 0;
        for (int i = 0; i < SERVER_OBS_PHASE_COUNT; i++) {
            dt[i]  = cur_clk.ns[i] - prev.ns[i];
            total += dt[i];
        }

        double ep = 0.0, ed = 0.0, ei = 0.0;
        // a driver reload resets the counter, skip the period instead of producing a negative delta
        const bool have_delta = cur_ok && prev_ok && mj >= prev_mj && total > 0;
        if (have_delta) {
            const double de = (mj - prev_mj) / 1000.0;

            const int64_t active = dt[SERVER_OBS_PHASE_PREFILL] + dt[SERVER_OBS_PHASE_DECODE];
            if (active > 0) {
                ratio_prefill = (double) dt[SERVER_OBS_PHASE_PREFILL] / active;
            }
            // host work with no request in flight (e.g. answering /telemetry) is idle-time work
            const bool serving = active > 0 || counters.slots_busy.load(std::memory_order_relaxed) > 0;
            const double host = serving ? (double) dt[SERVER_OBS_PHASE_HOST] : 0.0;
            const double tp   = dt[SERVER_OBS_PHASE_PREFILL] + host * ratio_prefill;
            const double td   = dt[SERVER_OBS_PHASE_DECODE]  + host * (1.0 - ratio_prefill);

            ep = de * tp / total;
            ed = de * td / total;
            ei = de - ep - ed;

            if (!serving && dt[SERVER_OBS_PHASE_IDLE] * 100 >= total * 99) {
                idle_acc_j += de;
                idle_acc_s += total / 1e9;
            }
        }

        const uint64_t c_prefill = counters.prefill_tokens.load(std::memory_order_relaxed);
        const uint64_t c_decode  = counters.decode_tokens.load(std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lock(mtx_state);
            if (have_delta) {
                energy_prefill += ep;
                energy_decode  += ed;
                energy_idle    += ei;
                energy_board   += ep + ed + ei;
                energy_samples++;
            }
            if (idle_acc_s >= 2.0) {
                const double w = idle_acc_j / idle_acc_s;
                idle_watts = idle_watts < 0.0 ? w : 0.8 * idle_watts + 0.2 * w;
                idle_acc_j = 0.0;
                idle_acc_s = 0.0;
            }
            window.push_back({cur_clk.t_ns, cur_clk.ns, energy_board});
            // keep one sample at or beyond the window edge, so the window always spans it fully
            while (window.size() > 2 && window[1].t_ns <= cur_clk.t_ns - window_ns) {
                window.pop_front();
            }
            last_clock = cur_clk;

            mark.prefill_joules = energy_prefill;
            mark.decode_joules  = energy_decode;
            mark.prefill_tokens = c_prefill;
            mark.decode_tokens  = c_decode;
        }

        prev    = cur_clk;
        if (cur_ok) {
            prev_mj = mj;
            prev_ok = true;
        }

        if (cur_clk.t_ns - base.clk.t_ns < interval_ns) {
            continue;
        }

        base_t cur = take_base(cur_clk);

        const uint64_t n_prefill = cur.prefill - base.prefill;
        const uint64_t n_decode  = cur.decode  - base.decode;
        const bool active = n_prefill > 0 || n_decode > 0 ||
                            cur.started != base.started || cur.done != base.done || cur.rejected != base.rejected;
        if (!active) {
            // fold the idle time into the next record
            continue;
        }

        const double interval_s = (cur_clk.t_ns - base.clk.t_ns) / 1e9;
        double loop_s[SERVER_OBS_PHASE_COUNT];
        for (int i = 0; i < SERVER_OBS_PHASE_COUNT; i++) {
            loop_s[i] = (cur_clk.ns[i] - base.clk.ns[i]) / 1e9;
        }
        const uint64_t n_calls  = cur.calls  - base.calls;
        const uint64_t n_rounds = cur.rounds - base.rounds;
        const uint64_t n_rows   = cur.rows   - base.rows;

        json body = {
            {"interval_seconds", r6(interval_s)},
            {"tokens", {
                {"computed_prefill", n_prefill},
                {"committed_decode", n_decode},
            }},
            {"throughput_tokens_per_second", {
                {"prefill", r3(n_prefill / interval_s)},
                {"decode",  r3(n_decode  / interval_s)},
            }},
        };

        double w_idle = -1.0;
        {
            std::lock_guard<std::mutex> lock(mtx_state);
            w_idle = idle_watts;
        }

        if (energy_available) {
            const double jb = cur.e_board   - base.e_board;
            const double jp = cur.e_prefill - base.e_prefill;
            const double jd = cur.e_decode  - base.e_decode;
            const double ji = cur.e_idle    - base.e_idle;
            const uint64_t n_all = n_prefill + n_decode;
            body["energy"] = {
                {"board_joules",      r3(jb)},
                {"prefill_joules",    r3(jp)},
                {"decode_joules",     r3(jd)},
                {"idle_joules",       r3(ji)},
                {"idle_watts",        r3(std::max(0.0, w_idle))},
                {"accounted_seconds", r6(loop_s[SERVER_OBS_PHASE_PREFILL] + loop_s[SERVER_OBS_PHASE_DECODE] + loop_s[SERVER_OBS_PHASE_HOST])},
                {"residual_joules",   0.0},
                {"residual_fraction", 0.0},
                {"joules_per_token", {
                    {"served",  opt_num(n_all     > 0, r6(jb / std::max<uint64_t>(1, n_all)))},
                    {"active",  opt_num(n_all     > 0, r6((jb - ji) / std::max<uint64_t>(1, n_all)))},
                    {"prefill", opt_num(n_prefill > 0, r6(jp / std::max<uint64_t>(1, n_prefill)))},
                    {"decode",  opt_num(n_decode  > 0, r6(jd / std::max<uint64_t>(1, n_decode)))},
                }},
            };

            std::lock_guard<std::mutex> lock(mtx_state);
            if (n_prefill > 0) {
                mark.prefill_j_per_token = jp / n_prefill;
            }
            if (n_decode > 0) {
                mark.decode_j_per_token = jd / n_decode;
            }
        } else {
            body["energy"] = nullptr;
        }
        body["decode_batch"] = {
            {"rounds",       n_rounds},
            {"row_rounds",   n_rows},
            {"average_size", opt_num(n_rounds > 0, r3((double) n_rows / std::max<uint64_t>(1, n_rounds)))},
        };

        const auto & s0 = base.sched;
        const auto & s1 = cur.sched;
        json ext = {
            {"tokens", {
                {"drafted",  cur.drafted  - base.drafted},
                {"accepted", cur.accepted - base.accepted},
            }},
            {"phase_tokens_per_second", {
                {"prefill", opt_num(loop_s[SERVER_OBS_PHASE_PREFILL] > 0.0 && n_prefill > 0, r3(n_prefill / loop_s[SERVER_OBS_PHASE_PREFILL]))},
                {"decode",  opt_num(loop_s[SERVER_OBS_PHASE_DECODE]  > 0.0 && n_decode  > 0, r3(n_decode  / loop_s[SERVER_OBS_PHASE_DECODE]))},
            }},
            {"requests", {
                {"started",  cur.started  - base.started},
                {"done",     cur.done     - base.done},
                {"rejected", cur.rejected - base.rejected},
            }},
            {"scheduler", {
                {"slots_total",             cfg.n_slots},
                {"slots_busy",              counters.slots_busy.load(std::memory_order_relaxed)},
                {"slots_processing_prompt", counters.slots_prompt.load(std::memory_order_relaxed)},
                {"queue_deferred",          counters.queue_deferred.load(std::memory_order_relaxed)},
                {"queue_waiting",           counters.sched_waiting.load(std::memory_order_relaxed)},
                {"slots_preempted",         counters.sched_preempted.load(std::memory_order_relaxed)},
                {"waited_slot",             s1.n_wait_slot - s0.n_wait_slot},
                {"waited_kv",               s1.n_wait_kv   - s0.n_wait_kv},
            }},
            {"kv", {
                {"cells_total",    counters.kv_cells_total.load(std::memory_order_relaxed)},
                {"cells_used",     counters.kv_cells_used.load(std::memory_order_relaxed)},
                {"cells_reserved", counters.kv_cells_reserved.load(std::memory_order_relaxed)},
            }},
            {"parking", {
                {"sessions_parked",  counters.sessions_parked.load(std::memory_order_relaxed)},
                {"parks",            s1.n_park    - s0.n_park},
                {"unparks",          s1.n_unpark  - s0.n_unpark},
                {"preempts",         s1.n_preempt - s0.n_preempt},
                {"resumes",          s1.n_resume  - s0.n_resume},
                {"park_bytes",       s1.bytes_park   - s0.bytes_park},
                {"unpark_bytes",     s1.bytes_unpark - s0.bytes_unpark},
                {"park_seconds",     r6((s1.us_park   - s0.us_park)   / 1e6)},
                {"unpark_seconds",   r6((s1.us_unpark - s0.us_unpark) / 1e6)},
                {"recompute_tokens", s1.n_recompute_tokens - s0.n_recompute_tokens},
            }},
            {"loop_seconds", {
                {"prefill", r6(loop_s[SERVER_OBS_PHASE_PREFILL])},
                {"decode",  r6(loop_s[SERVER_OBS_PHASE_DECODE])},
                {"host",    r6(loop_s[SERVER_OBS_PHASE_HOST])},
                {"idle",    r6(loop_s[SERVER_OBS_PHASE_IDLE])},
            }},
            {"batch", {
                {"decode_calls",    n_calls},
                {"mean_busy_slots", opt_num(n_calls > 0, r3((double) (cur.busy - base.busy) / std::max<uint64_t>(1, n_calls)))},
            }},
        };

        server_obs_record rec;
        rec.kind      = SERVER_OBS_REC_RAW;
        rec.event     = "throughput";
        rec.t_unix_ms = now_unix_ms();
        rec.body      = std::move(body);
        rec.ext       = std::move(ext);
        emit(std::move(rec));

        base = cur;
    }
}

server_obs_energy_mark server_observer::energy_mark() const {
    std::lock_guard<std::mutex> lock(mtx_state);
    return mark;
}

bool server_observer::request_energy(const server_obs_energy_mark & m0, uint64_t n_prefill, uint64_t n_decode, double & e_prefill, double & e_decode) const {
    const server_obs_energy_mark m1 = energy_mark();
    if (!m0.valid || !m1.valid) {
        return false;
    }
    // mean rate over the request's lifetime; a request shorter than one sampling period falls back
    // to the last reported interval rate
    const uint64_t dp = m1.prefill_tokens - m0.prefill_tokens;
    const uint64_t dd = m1.decode_tokens  - m0.decode_tokens;
    const double rate_p = dp > 0 ? (m1.prefill_joules - m0.prefill_joules) / dp : m1.prefill_j_per_token;
    const double rate_d = dd > 0 ? (m1.decode_joules  - m0.decode_joules)  / dd : m1.decode_j_per_token;
    e_prefill = n_prefill * rate_p;
    e_decode  = n_decode  * rate_d;
    return true;
}

//
// HTTP side: /telemetry, /slots, /metrics, built from the snapshot, never from the server loop
//

json server_observer::energy_ext_json() const {
    std::lock_guard<std::mutex> lock(mtx_state);

    double watts_1s = -1.0;
    if (energy_available && window.size() >= 2) {
        const sample & last = window.back();
        const sample * ref  = &window.front();
        for (auto it = window.rbegin(); it != window.rend(); ++it) {
            if (last.t_ns - it->t_ns >= 1000000000LL) {
                ref = &*it;
                break;
            }
        }
        if (last.t_ns > ref->t_ns) {
            watts_1s = (last.board_joules - ref->board_joules) / ((last.t_ns - ref->t_ns) / 1e9);
        }
    }

    json res = {
        {"available", energy_available},
    };
    if (energy_available) {
        res["joules"] = {
            {"board",   r3(energy_board)},
            {"prefill", r3(energy_prefill)},
            {"decode",  r3(energy_decode)},
            {"idle",    r3(energy_idle)},
        };
    } else {
        res["joules"] = nullptr;
    }
    res["idle_watts"]     = opt_num(idle_watts >= 0.0, r3(idle_watts));
    res["board_watts_1s"] = opt_num(watts_1s   >= 0.0, r3(watts_1s));
    return res;
}

static json obs_phase_seconds(const std::array<int64_t, SERVER_OBS_PHASE_COUNT> & ns, double scale) {
    return json {
        {"prefill", r6(ns[SERVER_OBS_PHASE_PREFILL] * scale)},
        {"decode",  r6(ns[SERVER_OBS_PHASE_DECODE]  * scale)},
        {"host",    r6(ns[SERVER_OBS_PHASE_HOST]    * scale)},
        {"idle",    r6(ns[SERVER_OBS_PHASE_IDLE]    * scale)},
    };
}

json server_observer::loop_json() const {
    const server_obs_clock::snapshot now = clock.read();

    server_obs_clock::snapshot ref;
    {
        std::lock_guard<std::mutex> lock(mtx_state);
        if (!window.empty()) {
            ref.t_ns = window.front().t_ns;
            ref.ns   = window.front().ns;
        } else {
            ref = now;
        }
    }

    std::array<int64_t, SERVER_OBS_PHASE_COUNT> d;
    for (int i = 0; i < SERVER_OBS_PHASE_COUNT; i++) {
        d[i] = std::max<int64_t>(0, now.ns[i] - ref.ns[i]);
    }
    const int64_t span = std::max<int64_t>(1, now.t_ns - ref.t_ns);

    return json {
        {"window_seconds", r6(span / 1e9)},
        {"phase",          server_obs_phase_name(now.phase)},
        {"seconds",        obs_phase_seconds(d, 1e-9)},
        {"fraction",       obs_phase_seconds(d, 1.0 / span)},
    };
}

json server_observer::telemetry_json(bool sleeping) {
    const int64_t t_now_ms = now_unix_ms();
    const int64_t t_now_us = ggml_time_us();

    // the loop's view: the fixed-size fields and the per-slot / per-session lists, copied under one short lock
    obs_loop_view v;
    std::vector<server_obs_slot_view>    slot_views;
    std::vector<server_obs_session_view> session_views;
    uint64_t m_prompt_count = 0, m_prompt_us = 0, m_predict_count = 0, m_predict_us = 0, m_cached = 0, m_decode = 0, m_busy = 0;
    {
        std::lock_guard<std::mutex> lock(snap.mtx);
        v = obs_read_view_locked(snap);
        slot_views      = snap.slots;
        session_views   = snap.sessions;
        m_prompt_count  = snap.prompt_count;
        m_prompt_us     = snap.prompt_time_us;
        m_predict_count = snap.predict_count;
        m_predict_us    = snap.predict_time_us;
        m_cached        = snap.n_prompt_cached;
        m_decode        = snap.n_decode;
        m_busy          = snap.n_busy_slots;
    }
    cc_totals c;
    {
        std::lock_guard<std::mutex> lock(mtx_cc);
        c = cc;
    }
    const server_obs_occupancy o = obs_occupancy_of(v, st);
    const server_obs_clock::snapshot clk = clock.read();

    const double loop_prefill = clk.ns[SERVER_OBS_PHASE_PREFILL] / 1e9;
    const double loop_decode  = clk.ns[SERVER_OBS_PHASE_DECODE] / 1e9;
    const double loop_host    = clk.ns[SERVER_OBS_PHASE_HOST] / 1e9;
    const double admission_s  = (v.sched.us_park_host + v.sched.us_unpark_host) / 1e6;

    double joules = 0.0, w_idle = -1.0;
    {
        std::lock_guard<std::mutex> lock(mtx_state);
        joules = energy_board;
        w_idle = idle_watts;
    }
    int pstate = -1;

    json out;
    out["timestamp_unix_ms"]  = t_now_ms;
    out["server_instance_id"] = server_instance_id;
    out["uptime_seconds"]     = r3((t_now_ms - t_start_ms) / 1e3);
    out["attached"]           = true;
    out["model_id"]           = st.model_id;
    out["gpu"]                = nvml->live_json(joules, w_idle, pstate);

    out["scheduler"] = {
        {"running",                 v.n_busy},
        {"prefilling",              v.n_prompt},
        {"decode_ready",            v.n_gen},
        {"waiting",                 v.queue_deferred + v.sched_waiting},
        {"max_concurrency",         st.n_slots},
        {"max_pending_requests",    0},
        {"admitted_requests",       c.admitted},
        {"queue_seconds_total",     r6(c.queue_s)},
        {"rejected_overloaded",     0},
        {"rejected_queue_timeout",  0},
        {"decode_rounds",           counters.decode_rounds.load(std::memory_order_relaxed)},
        {"decode_row_rounds",       counters.decode_rows.load(std::memory_order_relaxed)},
        {"decode_rounds_abandoned", 0},
        {"worker_seconds", {
            {"decode",    r6(loop_decode)},
            {"prefill",   r6(loop_prefill)},
            {"admission", r6(admission_s)},
            {"publish",   0.0},
            {"upkeep",    r6(std::max(0.0, loop_host - admission_s))},
        }},
        {"worker_admission", {
            {"calls",           c.admitted},
            {"plan_seconds",    0.0},
            {"restore_seconds", r6(admission_s)},
            {"commit_seconds",  0.0},
        }},
        {"worker_decode_rounds", counters.decode_rounds.load(std::memory_order_relaxed)},
        {"worker_prefill_steps", counters.prefill_steps.load(std::memory_order_relaxed)},
    };

    const uint64_t host_bytes    = v.sessions_on ? v.ss.ram_bytes : v.pc_bytes;
    const uint64_t host_capacity = v.sessions_on ? st.session_ram_bytes : v.pc_limit;
    out["cache"] = {
        {"l1", {
            {"entries",        o.l1_entries},
            {"bytes",          o.l1_bytes},
            {"capacity_bytes", st.context_bytes},
            {"evictions",      o.l1_evictions},
            {"demotions",      o.l1_demotions},
        }},
        {"l2", {
            {"entries",        o.l2_entries},
            {"bytes",          o.l2_bytes},
            {"capacity_bytes", host_capacity},
            {"evictions",      o.l2_evictions},
            {"evicted_bytes",  o.l2_evicted_bytes},
        }},
        {"l3", {
            {"entries",        o.l3_entries},
            {"bytes",          o.l3_bytes},
            {"capacity_bytes", st.session_disk_limit},
            {"evictions",      0},
            {"evicted_bytes",  0},
        }},
        {"kv_restore_reclaimed_lanes", 0},
        {"kv_growth", {
            {"attempts",      o.grows},
            {"forced_spills", o.spills},
            {"curtailed",     0},
        }},
        {"restore_successes",     c.restores},
        {"restore_failures",      0},
        {"restore_deferrals",     0},
        {"lookup_hits",           c.lookup_hits},
        {"lookup_misses",         c.lookup_misses},
        {"publication_successes", o.publications},
        {"publication_failures",  o.publication_failures},
        {"persistence_successes", o.persisted},
        {"persistence_failures",  0},
        // the keys of keylimesoda/ninfer-4090-lin-hybrid's /telemetry that the omarchy-ninfer widget reads (page = cell)
        {"kv_capacity",                      st.n_ctx},
        {"kv_capacity_page_groups",          st.n_ctx},
        {"kv_capacity_max_page_groups",      st.n_ctx},
        {"device_main_kv_occupied_pages",    v.cells_used},
        {"device_backend_kv_occupied_pages", 0},
        {"device_state_occupied_slots",      o.l1_entries},
        {"host_state_occupied_slots",        v.sessions_on ? v.ss.n_ram : v.pc_entries},
        {"host_state_capacity_slots",        0},
        {"host_kv_occupied_bytes",           host_bytes},
        {"host_kv_capacity_bytes",           host_capacity},
        {"pressure", {
            {"spill_pages",                 v.n_preempted},
            {"partial_tail_cow_pages",      0},
            {"private_owners_degraded",     0},
            {"private_owners_evicted",      v.sched.n_park_drop},
            {"shared_owners_degraded",      0},
            {"shared_owners_evicted",       0},
            {"checkpoints_dropped",         0},
            {"searches",                    0},
            {"search_budget_exhaustions",   0},
            {"maximal_fallback_selections", 0},
        }},
    };

    json slots_arr = json::array();
    for (const auto & sv : slot_views) {
        slots_arr.push_back({
            {"processing",     sv.is_processing},
            {"retained",       !sv.is_processing && sv.n_ctx_used > 0},
            {"prompt_tokens",  sv.is_processing ? sv.n_prompt_tokens : sv.n_ctx_used},
            {"cached_tokens",  sv.is_processing ? sv.n_cached : sv.n_ctx_used},
            {"session_digest", obs_hex16(sv.session_id)},
            {"checkpoints",    sv.n_ckpt},
        });
    }
    out["slots"] = std::move(slots_arr);

    json lora_names = json::array();
    for (const auto & n : st.lora_names) {
        lora_names.push_back(n);
    }
    out["adapters"] = {
        {"count",        st.lora_names.size()},
        {"names",        lora_names},
        {"model_ids",    lora_names},
        {"rank",         0},
        {"slots",        0},
        {"device_bytes", 0},
        {"file_bytes",   0},
    };

    out["memory"] = {
        {"device",                        0},
        {"max_context",                   st.n_ctx_slot},
        {"kv_capacity",                   st.n_ctx},
        {"kv_capacity_page_groups",       st.n_ctx},
        {"kv_capacity_max_page_groups",   st.n_ctx},
        {"weights",                       obs_arena(st.model_bytes)},
        {"sequence",                      obs_arena(st.context_bytes + st.draft_context_bytes)},
        {"workspace",                     obs_arena(st.compute_bytes)},
        {"request_transient",             obs_arena(0)},
        {"kv_payload_bytes",              st.context_bytes},
        {"text_kv_bytes",                 st.context_bytes},
        {"mtp_kv_bytes",                  st.draft_context_bytes},
        {"gdn_state_bytes",               0},
        {"dflash_kv_bytes",               0},
        {"replay_records_bytes",          0},
        {"cuda_graph_observed_bytes",     0},
        {"cuda_graph_allowance_bytes",    0},
        {"available_after_startup_bytes", st.device_free},
        {"available_after_weights_bytes", st.device_total > st.model_bytes ? st.device_total - st.model_bytes : 0},
        {"lora_bank_bytes",               0},
    };

    uint64_t dropped = n_dropped_closed.load(std::memory_order_relaxed);
    size_t   n_subs  = 0;
    {
        std::lock_guard<std::mutex> lock(mtx_subs);
        n_subs = subs.size();
        for (const auto & s : subs) {
            std::lock_guard<std::mutex> lock_sub(s->mtx);
            dropped += s->dropped;
        }
    }
    out["events"] = {
        {"jsonl_enabled", jsonl != nullptr},
        {"subscribers",   n_subs},
    };

    // lovelace: the sections of lovelace's previous (schema 1) /telemetry
    json ext_slots = json::array();
    for (const auto & sv : slot_views) {
        ext_slots.push_back({
            {"id",                sv.id},
            {"state",             sv.state},
            {"is_processing",     sv.is_processing},
            {"n_ctx",             sv.n_ctx},
            {"n_ctx_used",        sv.n_ctx_used},
            {"n_reserved",        sv.n_reserved},
            {"wait_reason",       sv.wait_reason ? json(sv.wait_reason) : json(nullptr)},
            {"n_unparked_tokens", sv.n_unparked},
            {"id_task",           sv.id_task >= 0 ? json(sv.id_task) : json(nullptr)},
            {"n_prompt_tokens",   sv.id_task >= 0 ? sv.n_prompt_tokens : 0},
            {"n_cached_tokens",   sv.n_cached},
            {"n_prefill_tokens",  sv.n_prefill},
            {"n_decoded",         sv.n_decoded},
            {"speculative",       sv.speculative},
            {"session_id",        sv.session_id ? json(sv.session_id) : json(nullptr)},
        });
    }

    json sessions_json = nullptr;
    if (v.sessions_on) {
        json list = json::array();
        for (const auto & s : session_views) {
            list.push_back({
                {"id",                   s.id},
                {"tier",                 s.tier},
                {"slot",                 s.slot},
                {"resident_cells",       s.resident_cells},
                {"mirror_tokens",        s.mirror_tokens},
                {"mirror_bytes",         s.mirror_bytes},
                {"mirror_segments",      s.mirror_segments},
                {"checkpoints",          s.checkpoints},
                {"file_bytes",           s.file_bytes},
                {"stale",                s.stale},
                {"sync_ms",              s.t_synced_us > s.t_sync_us ? json((s.t_synced_us - s.t_sync_us) / 1e3) : json(nullptr)},
                {"since_sync_s",         s.t_sync_us > 0 ? json((t_now_us - s.t_sync_us) / 1e6) : json(nullptr)},
                {"syncs",                s.syncs},
                {"truncations",          s.truncations},
                {"restores",             s.restores},
                {"last_restore_bytes",   s.last_restore_bytes},
                {"last_restore_ms",      s.last_restore_ms},
                {"last_restore_wall_ms", s.last_restore_wall_ms},
                {"prefetches",           s.prefetches},
                {"prefetch_hits",        s.prefetch_hits},
                {"demotions",            s.demotions},
                {"promotions",           s.promotions},
            });
        }
        const auto & ss = v.ss;
        sessions_json = {
            {"arena_bytes",      ss.arena_bytes},
            {"arena_free_bytes", ss.arena_free},
            {"ram_bytes",        ss.ram_bytes},
            {"disk_bytes",       ss.disk_bytes},
            {"disk_limit_bytes", st.session_disk_limit},
            {"n_sessions",       ss.n_sessions},
            {"n_gpu",            ss.n_gpu},
            {"n_ram",            ss.n_ram},
            {"n_disk",           ss.n_disk},
            {"n_io",             ss.n_io},
            {"demotions",        ss.n_demote},
            {"promotions",       ss.n_promote},
            {"deleted",          ss.n_deleted},
            {"stale_syncs",      ss.n_sync_stale},
            {"prefetches",       ss.n_prefetch},
            {"prefetch_hits",    ss.n_prefetch_hit},
            {"syncs",            v.sched.n_sync},
            {"sync_bytes",       v.sched.bytes_sync},
            {"io_bytes_written", ss.io_bytes_written},
            {"io_bytes_read",    ss.io_bytes_read},
            {"io_write_s",       ss.io_write_s},
            {"io_read_s",        ss.io_read_s},
            {"sessions",         list},
        };
    }

    const auto & sc = v.sched;
    out["lovelace"] = {
        {"schema_version",      OBS_LOVELACE_SCHEMA},
        {"build_info",          st.build_info},
        {"sleeping",            sleeping},
        {"energy",              energy_ext_json()},
        {"gpu",                 { {"pstate", pstate >= 0 ? json(pstate) : json(nullptr)} }},
        {"loop",                loop_json()},
        {"loop_totals_seconds", obs_phase_seconds(clk.ns, 1e-9)},
        {"scheduler", {
            {"slots_total",             st.n_slots},
            {"slots_busy",              v.n_busy},
            {"slots_idle",              st.n_slots - v.n_busy},
            {"slots_processing_prompt", v.n_prompt},
            {"slots_generating",        v.n_gen},
            {"queue_deferred",          v.queue_deferred},
            {"kv_budget", {
                {"enabled",           st.kv_budget},
                {"queue_waiting",     v.sched_waiting},
                {"slots_preempted",   v.n_preempted},
                {"gen_reserve",       st.kv_budget_gen},
                {"headroom",          st.kv_budget_headroom},
                {"prefill_chunk_gen", st.prefill_chunk_gen},
                {"timeslice",         st.kv_timeslice},
                {"totals", {
                    {"parks",               sc.n_park},
                    {"park_drops",          sc.n_park_drop},
                    {"unparks",             sc.n_unpark},
                    {"preempts",            sc.n_preempt},
                    {"resumes",             sc.n_resume},
                    {"grows",               sc.n_grow},
                    {"park_bytes",          sc.bytes_park},
                    {"unpark_bytes",        sc.bytes_unpark},
                    {"park_seconds",        sc.us_park / 1e6},
                    {"unpark_seconds",      sc.us_unpark / 1e6},
                    {"park_host_seconds",   sc.us_park_host / 1e6},
                    {"unpark_host_seconds", sc.us_unpark_host / 1e6},
                    {"park_gbps",           sc.us_park   > 0 ? sc.bytes_park   / 1e3 / sc.us_park   : 0.0},
                    {"unpark_gbps",         sc.us_unpark > 0 ? sc.bytes_unpark / 1e3 / sc.us_unpark : 0.0},
                    {"waited_slot",         sc.n_wait_slot},
                    {"waited_kv",           sc.n_wait_kv},
                    {"rollback_fallbacks",  sc.n_rollback_fallback},
                    {"recompute_tokens",    sc.n_recompute_tokens},
                    {"kv_overflows",        sc.n_kv_overflow},
                }},
            }},
            {"totals", {
                {"prompt_tokens",           m_prompt_count},
                {"prompt_tokens_cached",    m_cached},
                {"predicted_tokens",        m_predict_count},
                {"prompt_seconds",          m_prompt_us / 1e6},
                {"predicted_seconds",       m_predict_us / 1e6},
                {"n_decode",                m_decode},
                {"n_busy_slots_per_decode", (double) m_busy / std::max<double>(1.0, (double) m_decode)},
                {"drafted",                 counters.drafted.load(std::memory_order_relaxed)},
                {"accepted",                counters.accepted.load(std::memory_order_relaxed)},
                {"requests_started",        counters.req_started.load(std::memory_order_relaxed)},
                {"requests_done",           counters.req_done.load(std::memory_order_relaxed)},
                {"requests_rejected",       counters.req_rejected.load(std::memory_order_relaxed)},
            }},
        }},
        {"kv", {
            {"cells_total",    st.n_ctx},
            {"cells_used",     v.cells_used},
            {"cells_reserved", v.cells_reserved},
            {"unified",        st.kv_unified},
            {"n_ctx_slot",     st.n_ctx_slot},
            {"cache_type_k",   st.cache_type_k},
            {"cache_type_v",   st.cache_type_v},
        }},
        {"prompt_cache", {
            {"enabled",      v.pc_enabled},
            {"bytes_used",   v.pc_bytes},
            {"bytes_limit",  v.pc_limit},
            {"tokens",       v.pc_tokens},
            {"tokens_limit", v.pc_tokens_limit},
            {"entries",      v.pc_entries},
        }},
        {"sessions", sessions_json},
        {"vram",     st.vram},
        {"slots",    ext_slots},
        {"events", {
            {"jsonl_path",      cfg.jsonl_path.empty() ? json(nullptr) : json(cfg.jsonl_path)},
            {"records_emitted", n_emitted.load(std::memory_order_relaxed)},
            {"records_dropped", dropped},
        }},
    };
    return out;
}

json server_observer::slots_json() {
    std::vector<server_obs_slot_view> views;
    {
        std::lock_guard<std::mutex> lock(snap.mtx);
        views = snap.slots;
    }
    json arr = json::array();
    for (const auto & sv : views) {
        const std::string digest = obs_hex16(sv.session_id);
        json ckpts = json::array();
        for (int i = 0; i < sv.n_ckpt && i < server_obs_slot_view::MAX_CKPT; i++) {
            ckpts.push_back({ {"frontier", sv.ckpt_pos[i]}, {"session_digest", digest} });
        }
        json res = {
            {"id",                    sv.id},
            {"n_ctx",                 sv.n_ctx},
            {"speculative",           sv.speculative},
            {"is_processing",         sv.is_processing},
            {"retained",              !sv.is_processing && sv.n_ctx_used > 0},
            {"session_digest",        digest},
            {"checkpoints",           ckpts},
            {"n_prompt_tokens",       sv.n_ctx_used},
            {"n_prompt_tokens_cache", sv.n_cached},
        };
        if (sv.task) {
            // llama.cpp's slot fields; the task is immutable once launched. `generated` is not reported: the text
            // grows on the server loop per token and is never copied into the snapshot
            res["id_task"]                   = sv.task->id;
            res["n_prompt_tokens_processed"] = sv.n_prefill;
            res["params"]                    = sv.task->params.to_json(false);
            res["next_token"] = json::array({
                {
                    {"has_next_token", sv.has_next_token},
                    {"has_new_line",   sv.has_new_line},
                    {"n_remain",       sv.n_remain},
                    {"n_decoded",      sv.n_decoded},
                }
            });
            if (st.ctx) {
                res["prompt"] = sv.task->tokens.detokenize(st.ctx, true);
            }
        }
        arr.push_back(std::move(res));
    }
    return arr;
}

std::string server_observer::metrics_text(int64_t & t_start) {
    // llama.cpp's own series from the snapshot; its rate gauges cover the time since the previous scrape
    server_task_result_metrics m;
    obs_loop_view v;
    {
        std::lock_guard<std::mutex> lock(snap.mtx);
        v = obs_read_view_locked(snap);
        m.metrics.t_start             = snap.metrics_t_start;
        t_start                       = snap.metrics_t_start;
        m.metrics.prompt.count        = snap.prompt_count;
        m.metrics.prompt.steps        = snap.prompt_steps;
        m.metrics.prompt.time         = snap.prompt_time_us;
        m.metrics.predict.count       = snap.predict_count;
        m.metrics.predict.steps       = snap.predict_steps;
        m.metrics.predict.time        = snap.predict_time_us;
        m.metrics.n_prompt_cached     = snap.n_prompt_cached;
        m.metrics.n_tokens_max        = snap.n_tokens_max;
        m.metrics.n_decode            = snap.n_decode;
        m.metrics.n_busy_slots        = snap.n_busy_slots;
        m.metrics.n_draft_tokens      = snap.n_draft_tokens;
        m.metrics.n_draft_accepted    = snap.n_draft_accepted;
        m.metrics.n_draft_verif_steps = snap.n_draft_verif_steps;
    }
    {
        std::lock_guard<std::mutex> lock(mtx_scrape);
        m.metrics.prompt_bucket.steps  = m.metrics.prompt.steps  - std::min<uint64_t>(m.metrics.prompt.steps,  scrape_prompt_steps);
        m.metrics.prompt_bucket.time   = m.metrics.prompt.time   - std::min<uint64_t>(m.metrics.prompt.time,   scrape_prompt_us);
        m.metrics.predict_bucket.steps = m.metrics.predict.steps - std::min<uint64_t>(m.metrics.predict.steps, scrape_predict_steps);
        m.metrics.predict_bucket.time  = m.metrics.predict.time  - std::min<uint64_t>(m.metrics.predict.time,  scrape_predict_us);
        scrape_prompt_steps  = m.metrics.prompt.steps;
        scrape_prompt_us     = m.metrics.prompt.time;
        scrape_predict_steps = m.metrics.predict.steps;
        scrape_predict_us    = m.metrics.predict.time;
    }
    m.n_processing_slots = v.n_busy;
    m.n_tasks_deferred   = v.queue_deferred + v.sched_waiting;

    std::string out = m.to_metrics();

    cc_totals c;
    {
        std::lock_guard<std::mutex> lock(mtx_cc);
        c = cc;
    }
    const server_obs_occupancy o = obs_occupancy_of(v, st);
    const server_obs_clock::snapshot clk = clock.read();
    const double admission_s = (v.sched.us_park_host + v.sched.us_unpark_host) / 1e6;

    // tensorninja's series (serve_metrics.cpp): same names, types and order; N/A ones are 0
    char line[256];
    const auto type_of = [](const char * name) {
        const std::string_view n(name);
        return n.size() > 6 && n.substr(n.size() - 6) == "_total" ? "counter" : "gauge";
    };
    const auto addf = [&](const char * name, double value) {
        snprintf(line, sizeof(line), "# TYPE %s %s\n%s %.6f\n", name, type_of(name), name, value);
        out += line;
    };
    const auto addu = [&](const char * name, uint64_t value) {
        snprintf(line, sizeof(line), "# TYPE %s %s\n%s %llu\n", name, type_of(name), name, (unsigned long long) value);
        out += line;
    };
    addu("ninfer:requests_total",                    c.requests_done);
    addu("ninfer:prefix_cache_hit_tokens_total",     c.prefix_hit_tokens);
    addu("ninfer:draft_tokens_total",                c.drafted);
    addu("ninfer:draft_accepted_tokens_total",       c.accepted);
    addu("ninfer:continuation_lookup_hits_total",    c.lookup_hits);
    addu("ninfer:continuation_lookup_misses_total",  c.lookup_misses);
    for (const char * name : { "ninfer:continuation_preflight_candidate_rejections_total", "ninfer:continuation_preparation_hits_total",
                               "ninfer:continuation_preparation_decoded_total", "ninfer:continuation_preparation_inline_total" }) {
        addu(name, 0);
    }
    addu("ninfer:continuation_restore_successes_total",     c.restores);
    addu("ninfer:continuation_restore_failures_total",      0);
    addu("ninfer:continuation_restore_deferrals_total",     0);
    addu("ninfer:continuation_publication_successes_total", o.publications);
    addu("ninfer:continuation_publication_failures_total",  o.publication_failures);
    for (const char * name : { "ninfer:continuation_publication_superseded_total", "ninfer:continuation_publication_coalesced_total",
                               "ninfer:continuation_publication_failed_capacity_total", "ninfer:continuation_publication_failed_evicted_total",
                               "ninfer:continuation_publication_failed_alias_moved_total", "ninfer:continuation_publication_failed_lineage_total",
                               "ninfer:continuation_publication_failed_error_total" }) {
        addu(name, 0);
    }
    addu("ninfer:continuation_restored_tokens_total", c.restored_tokens);
    addu("ninfer:continuation_restored_bytes_total",  c.restored_bytes);
    static const char * tier_succ[3]   = { "ninfer:continuation_l1_restore_successes_total", "ninfer:continuation_l2_restore_successes_total", "ninfer:continuation_l3_restore_successes_total" };
    static const char * tier_tokens[3] = { "ninfer:continuation_l1_restored_tokens_total", "ninfer:continuation_l2_restored_tokens_total", "ninfer:continuation_l3_restored_tokens_total" };
    static const char * tier_bytes[3]  = { "ninfer:continuation_l1_restored_bytes_total", "ninfer:continuation_l2_restored_bytes_total", "ninfer:continuation_l3_restored_bytes_total" };
    for (int i = 0; i < 3; i++) { addu(tier_succ[i],   c.tier_restores[i]); }
    for (int i = 0; i < 3; i++) { addu(tier_tokens[i], c.tier_tokens[i]); }
    for (int i = 0; i < 3; i++) { addu(tier_bytes[i],  c.tier_bytes[i]); }
    addu("ninfer:continuation_session_restores_total",       c.restores);
    addu("ninfer:continuation_stable_prefix_restores_total", 0);
    addu("ninfer:continuation_miss_disabled_total",          0);
    addu("ninfer:continuation_miss_no_alias_total",          c.no_alias);
    for (const char * name : { "ninfer:continuation_miss_not_attempted_total", "ninfer:continuation_miss_entry_unavailable_or_corrupt_total",
                               "ninfer:continuation_miss_not_deeper_total", "ninfer:continuation_miss_preflight_rejected_total",
                               "ninfer:continuation_miss_rollback_conflict_total", "ninfer:continuation_miss_no_lane_total",
                               "ninfer:continuation_miss_restore_failed_total", "ninfer:continuation_restore_failed_kv_reservation_total",
                               "ninfer:continuation_restore_failed_verify_depth_total", "ninfer:continuation_restore_failed_inventory_total",
                               "ninfer:continuation_restore_failed_metadata_total", "ninfer:continuation_restore_failed_decode_total",
                               "ninfer:continuation_restore_failed_lane_total", "ninfer:admission_rejected_overloaded_total",
                               "ninfer:admission_rejected_queue_timeout_total" }) {
        addu(name, 0);
    }
    addu("ninfer:admitted_requests_total",                c.admitted);
    addf("ninfer:queue_seconds_total",                    c.queue_s);
    addf("ninfer:worker_decode_seconds_total",            clk.ns[SERVER_OBS_PHASE_DECODE] / 1e9);
    addf("ninfer:worker_prefill_seconds_total",           clk.ns[SERVER_OBS_PHASE_PREFILL] / 1e9);
    addf("ninfer:worker_admission_seconds_total",         admission_s);
    addu("ninfer:worker_admission_calls_total",           c.admitted);
    addf("ninfer:worker_admission_plan_seconds_total",    0.0);
    addf("ninfer:worker_admission_restore_seconds_total", admission_s);
    addf("ninfer:worker_admission_commit_seconds_total",  0.0);
    addf("ninfer:worker_publish_seconds_total",           0.0);
    addf("ninfer:worker_upkeep_seconds_total",            std::max(0.0, clk.ns[SERVER_OBS_PHASE_HOST] / 1e9 - admission_s));
    addu("ninfer:worker_decode_rounds_total",             counters.decode_rounds.load(std::memory_order_relaxed));
    addu("ninfer:worker_prefill_steps_total",             counters.prefill_steps.load(std::memory_order_relaxed));
    addu("ninfer:continuation_l2_lookup_microseconds_total",      0);
    addu("ninfer:continuation_l2_lookup_operations_total",        0);
    addu("ninfer:continuation_l3_lookup_microseconds_total",      0);
    addu("ninfer:continuation_l3_lookup_operations_total",        0);
    addu("ninfer:continuation_preflight_microseconds_total",      0);
    addu("ninfer:continuation_preflight_operations_total",        0);
    addu("ninfer:continuation_l2_restore_microseconds_total",     c.tier_restore_us[1]);
    addu("ninfer:continuation_l2_restore_operations_total",       c.tier_restore_ops[1]);
    addu("ninfer:continuation_l3_restore_microseconds_total",     c.tier_restore_us[2]);
    addu("ninfer:continuation_l3_restore_operations_total",       c.tier_restore_ops[2]);
    addu("ninfer:continuation_l2_admission_microseconds_total",   0);
    addu("ninfer:continuation_l2_admission_operations_total",     0);
    addu("ninfer:continuation_export_microseconds_total",         0);
    addu("ninfer:continuation_export_operations_total",           0);
    addu("ninfer:continuation_l3_persistence_microseconds_total", (uint64_t) std::llround(o.persist_us));
    addu("ninfer:continuation_l3_persistence_operations_total",   o.persisted);
    addu("ninfer:continuation_persistence_queued_total",          o.persisted);
    addu("ninfer:continuation_persistence_coalesced_total",       0);
    addu("ninfer:continuation_persistence_successes_total",       o.persisted);
    addu("ninfer:continuation_persistence_failures_total",        0);
    addu("ninfer:continuation_l2_entries",                        o.l2_entries);
    addu("ninfer:continuation_l2_bytes",                          o.l2_bytes);
    addu("ninfer:continuation_l3_entries",                        o.l3_entries);
    addu("ninfer:continuation_l3_bytes",                          o.l3_bytes);
    addu("ninfer:continuation_l2_evictions_total",                o.l2_evictions);
    addu("ninfer:continuation_l2_evicted_bytes_total",            o.l2_evicted_bytes);
    addu("ninfer:continuation_l3_evictions_total",                0);
    addu("ninfer:continuation_l3_evicted_bytes_total",            0);
    addu("ninfer:l1_evictions_total",                             o.l1_evictions);
    addu("ninfer:l1_demotions_total",                             o.l1_demotions);
    addu("ninfer:kv_restore_reclaimed_lanes_total",               0);
    addu("ninfer:kv_growth_attempts_total",                       o.grows);
    addu("ninfer:kv_growth_forced_spills_total",                  o.spills);
    addu("ninfer:kv_growth_curtailed_total",                      0);
    addu("ninfer:l1_resident_entries",                            o.l1_entries);
    addu("ninfer:l1_resident_bytes",                              o.l1_bytes);

    double e_board, e_prefill, e_decode, e_idle, w_idle;
    uint64_t n_samples;
    {
        std::lock_guard<std::mutex> lock(mtx_state);
        e_board   = energy_board;
        e_prefill = energy_prefill;
        e_decode  = energy_decode;
        e_idle    = energy_idle;
        w_idle    = idle_watts;
        n_samples = energy_samples;
    }
    if (energy_available) {
        addf("ninfer:board_energy_joules_total",      e_board);
        addf("ninfer:board_idle_watts",               std::max(0.0, w_idle));
        addf("ninfer:prefill_energy_joules_total",    e_prefill);
        addf("ninfer:decode_energy_joules_total",     e_decode);
        addf("ninfer:energy_accounted_seconds_total", (clk.ns[SERVER_OBS_PHASE_PREFILL] + clk.ns[SERVER_OBS_PHASE_DECODE] + clk.ns[SERVER_OBS_PHASE_HOST]) / 1e9);
        addu("ninfer:energy_samples_total",           n_samples);
    }

    // lovelace-only series
    std::stringstream ext;
    ext << "# HELP lovelace:loop_seconds_total Server loop wall clock by phase\n"
        << "# TYPE lovelace:loop_seconds_total counter\n";
    for (int i = 0; i < SERVER_OBS_PHASE_COUNT; i++) {
        ext << "lovelace:loop_seconds_total{phase=\"" << server_obs_phase_name((server_obs_phase) i) << "\"} " << clk.ns[i] / 1e9 << "\n";
    }
    if (energy_available) {
        const json energy = energy_ext_json();
        ext << "# HELP lovelace:energy_joules_total GPU board energy attributed to the server loop phase\n"
            << "# TYPE lovelace:energy_joules_total counter\n"
            << "lovelace:energy_joules_total{phase=\"prefill\"} " << e_prefill << "\n"
            << "lovelace:energy_joules_total{phase=\"decode\"} "  << e_decode  << "\n"
            << "lovelace:energy_joules_total{phase=\"idle\"} "    << e_idle    << "\n";
        if (!energy.at("board_watts_1s").is_null()) {
            ext << "# HELP lovelace:board_power_watts GPU board power, mean over the last second\n"
                << "# TYPE lovelace:board_power_watts gauge\n"
                << "lovelace:board_power_watts " << energy.at("board_watts_1s").get<double>() << "\n";
        }
    }
    const auto metric = [&](const char * name, const char * type, const char * help, double value) {
        ext << "# HELP lovelace:" << name << " " << help << "\n"
            << "# TYPE lovelace:" << name << " " << type << "\n"
            << "lovelace:" << name << " " << value << "\n";
    };
    const auto & sc = v.sched;
    metric("kv_cells_total",    "gauge", "KV cells of the unified cache",                             st.n_ctx);
    metric("kv_cells_used",     "gauge", "KV cells held by all slots",                                v.cells_used);
    metric("kv_cells_reserved", "gauge", "KV cells held or reserved by running requests (KV budget)", v.cells_reserved);
    metric("sched_waiting",     "gauge", "Requests waiting for admission (KV budget)",                v.sched_waiting);
    metric("sched_preempted",   "gauge", "Slots paused to host memory (KV budget)",                   v.n_preempted);
    metric("sessions_parked",   "gauge", "Sessions held in the host prompt cache",                    v.pc_entries);
    metric("sched_parks_total",        "counter", "Idle sessions moved from VRAM to host memory",     sc.n_park);
    metric("sched_park_drops_total",   "counter", "Idle sessions evicted without a new host copy",    sc.n_park_drop);
    metric("sched_unparks_total",      "counter", "Sessions restored from host memory",               sc.n_unpark);
    metric("sched_preempts_total",     "counter", "Generating slots paused to host memory",           sc.n_preempt);
    metric("sched_resumes_total",      "counter", "Paused slots resumed",                             sc.n_resume);
    metric("sched_grows_total",        "counter", "KV reservations extended in flight",               sc.n_grow);
    metric("sched_park_bytes_total",   "counter", "Bytes copied device to host by parking",           sc.bytes_park);
    metric("sched_unpark_bytes_total", "counter", "Bytes copied host to device by unparking",         sc.bytes_unpark);
    metric("sched_park_seconds_total",   "counter", "Time spent parking",                             sc.us_park / 1e6);
    metric("sched_unpark_seconds_total", "counter", "Time spent unparking",                           sc.us_unpark / 1e6);
    metric("sched_wait_slot_total",    "counter", "Requests that waited for a free slot",             sc.n_wait_slot);
    metric("sched_wait_kv_total",      "counter", "Requests that waited for KV cells",                sc.n_wait_kv);
    metric("sched_rollback_fallbacks_total", "counter", "Sequence truncations the memory refused (recovered)", sc.n_rollback_fallback);
    metric("sched_recompute_tokens_total",   "counter", "Tokens re-prefilled to recover a sequence",           sc.n_recompute_tokens);
    metric("sched_kv_overflows_total",       "counter", "Decodes that found no free KV cells",                 sc.n_kv_overflow);
    if (v.sessions_on) {
        const auto & ss = v.ss;
        metric("sessions",                     "gauge",   "Sessions in the session store",                      ss.n_sessions);
        metric("sessions_gpu",                 "gauge",   "Sessions resident in VRAM",                          ss.n_gpu);
        metric("sessions_ram",                 "gauge",   "Sessions with a mirror in the RAM arena",            ss.n_ram);
        metric("sessions_disk",                "gauge",   "Sessions only on disk",                              ss.n_disk);
        metric("sessions_ram_bytes",           "gauge",   "Bytes of session mirrors in the RAM arena",          ss.ram_bytes);
        metric("sessions_disk_bytes",          "gauge",   "Bytes of session files on disk",                     ss.disk_bytes);
        metric("sessions_demotions_total",     "counter", "Session mirrors written to disk and freed",          ss.n_demote);
        metric("sessions_promotions_total",    "counter", "Session mirrors read back from disk",                ss.n_promote);
        metric("sessions_prefetches_total",    "counter", "Session restores started while the request queued", ss.n_prefetch);
        metric("sessions_prefetch_hits_total", "counter", "Prefetched restores used by their request",          ss.n_prefetch_hit);
    }
    out += ext.str();
    return out;
}
