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

    json static_json() const {
        json res = {
            {"available", ok},
        };
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

    json live_json() const {
        json res = static_json();
        if (!ok) {
            return res;
        }
        unsigned u = 0;
        if (get_temperature && get_temperature(dev, NVML_TEMPERATURE_GPU, &u) == NVML_SUCCESS) {
            res["temperature_c"] = u;
        }
        if (get_fan && get_fan(dev, &u) == NVML_SUCCESS) {
            res["fan_percent"] = u;
        }
        // trailing-average board power, the right statistic for a 1 Hz gauge
        if (get_power && get_power(dev, &u) == NVML_SUCCESS) {
            res["power_watts"] = r3(u / 1000.0);
        }
        uint64_t mj = 0;
        res["energy_joules_total"] = energy_mj(mj) ? json(r3(mj / 1000.0)) : json(nullptr);
        nvml_utilization util {};
        if (get_utilization && get_utilization(dev, &util) == NVML_SUCCESS) {
            res["utilization_gpu_percent"]    = util.gpu;
            res["utilization_memory_percent"] = util.memory;
        }
        if (get_clock && get_clock(dev, NVML_CLOCK_SM, &u) == NVML_SUCCESS) {
            res["sm_clock_mhz"] = u;
        }
        if (get_clock && get_clock(dev, NVML_CLOCK_MEM, &u) == NVML_SUCCESS) {
            res["memory_clock_mhz"] = u;
        }
        nvml_memory mem {};
        if (get_memory && get_memory(dev, &mem) == NVML_SUCCESS) {
            res["memory_used_bytes"]  = mem.used;
            res["memory_total_bytes"] = mem.total;
        }
        // nvmlDeviceGetPcieThroughput is deliberately not read: it samples for ~20 ms inside the driver
        // and measurably slowed decode (~1-2.5 %) when polled at 1 Hz on an RTX 4090
        int pstate = 0;
        if (get_pstate && get_pstate(dev, &pstate) == NVML_SUCCESS) {
            res["pstate"] = pstate;
        }
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
            json arr = json::array();
            for (const auto & r : reasons) {
                if (bits & r.bit) {
                    arr.push_back(r.name);
                }
            }
            res["throttle_reasons"] = arr;
        }
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

server_observer::server_observer(const server_observer_config & cfg) : cfg(cfg) {
    server_instance_id = obs_random_hex(16);
    t_start_ms         = now_unix_ms();
    clock.init();

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

void server_observer::start(json server_start) {
    server_start["gpu"] = gpu_static_json();

    server_obs_record rec;
    rec.kind      = SERVER_OBS_REC_RAW;
    rec.event     = "server_start";
    rec.t_unix_ms = t_start_ms;
    rec.body      = std::move(server_start);
    rec.retain    = true;
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
            std::string line = format(rec, seq_next++);
            if (jsonl) {
                fwrite(line.data(), 1, line.size(), jsonl);
                fputc('\n', jsonl);
            }
            publish_line(std::move(line), rec.retain);
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

static json obs_request_info(const server_obs_record & rec, bool with_detail) {
    json res = {
        {"id_task",   rec.id_task >= 0 ? json(rec.id_task) : json(nullptr)},
        {"id_slot",   rec.id_slot >= 0 ? json(rec.id_slot) : json(nullptr)},
        {"session_id", rec.session_id != 0 ? json(rec.session_id) : json(nullptr)},
    };
    if (with_detail) {
        res["id_parent"] = rec.id_parent >= 0 ? json(rec.id_parent) : json(nullptr);
    }
    res["task_type"] = rec.task_type;
    res["endpoint"]  = rec.endpoint;
    if (with_detail) {
        res["stream"]    = rec.stream;
    }
    res["n_prompt_tokens"] = rec.n_prompt_tokens;
    if (with_detail) {
        res["n_predict"] = rec.n_predict;
    }
    res["n_ctx_slot"] = rec.n_ctx_slot;
    return res;
}

std::string server_observer::format(const server_obs_record & rec, uint64_t seq) const {
    json res = {
        {"schema_version",     1},
        {"event",              ""},
        {"server_instance_id", server_instance_id},
        {"seq",                seq},
        {"timestamp_unix_ms",  rec.t_unix_ms},
    };

    switch (rec.kind) {
        case SERVER_OBS_REC_RAW:
            {
                res["event"] = rec.event;
                for (const auto & [key, val] : rec.body.items()) {
                    res[key] = val;
                }
            } break;
        case SERVER_OBS_REC_REQUEST_START:
            {
                res["event"]         = "request_start";
                res["request"]       = obs_request_info(rec, true);
                res["queue_wait_ms"] = r3(rec.queue_wait_ms);
                res["wait_reason"]   = rec.wait_reason;
                res["n_unparked_tokens"] = rec.n_unparked_tokens;
            } break;
        case SERVER_OBS_REC_REQUEST_DONE:
            {
                res["event"]   = "request_done";
                res["request"] = obs_request_info(rec, true);
                res["result"]  = {
                    {"finish_reason",       rec.finish_reason},
                    {"truncated",           rec.truncated},
                    {"n_prompt_tokens",     rec.n_prompt_tokens},
                    {"n_cached_tokens",     rec.n_cached_tokens},
                    {"n_prefill_tokens",    rec.n_prefill_tokens},
                    {"n_completion_tokens", rec.n_completion_tokens},
                    {"n_ctx_used",          rec.n_ctx_used},
                    {"n_unparked_tokens",   rec.n_unparked_tokens},
                    {"wait_reason",         rec.wait_reason},
                    {"error",               rec.error_msg.empty() ? json(nullptr) : json(rec.error_msg)},
                };
                res["session"] = rec.session_tier == nullptr ? json(nullptr) : json {
                    {"id",              rec.session_id},
                    {"tier",            rec.session_tier},
                    {"restore_bytes",   rec.restore_bytes},
                    {"restore_ms",      r3(rec.restore_ms)},
                    {"park_ms",          r3(rec.park_ms)},
                    {"restore_issue_ms", r3(rec.restore_issue_ms)},
                    {"restore_ready_ms", opt_num(rec.restore_ready_ms >= 0.0, r3(rec.restore_ready_ms))},
                    {"prompt_start_ms",  opt_num(rec.prompt_start_ms >= 0.0, r3(rec.prompt_start_ms))},
                    {"prefetch_hit",    rec.prefetch_hit},
                };
                res["timings_ms"] = {
                    {"queue_wait", r3(std::max(0.0, rec.queue_wait_ms))},
                    {"prefill",    r3(rec.t_prefill_ms)},
                    {"decode",     r3(rec.t_decode_ms)},
                    {"ttft",       opt_num(rec.t_ttft_ms >= 0.0, r3(rec.t_ttft_ms))},
                    {"total",      r3(rec.t_total_ms)},
                };
                res["rates"] = {
                    {"prefill_tps", opt_num(rec.prefill_tps > 0.0, r3(rec.prefill_tps))},
                    {"decode_tps",  opt_num(rec.decode_tps  > 0.0, r3(rec.decode_tps))},
                };
                if (rec.n_drafted > 0) {
                    res["speculative"] = {
                        {"n_drafted",        rec.n_drafted},
                        {"n_accepted",       rec.n_accepted},
                        {"n_verif_steps",    rec.n_verif_steps},
                        {"accept_rate",      r6((double) rec.n_accepted / rec.n_drafted)},
                        {"accepted_per_pos", rec.accepted_per_pos},
                    };
                } else {
                    res["speculative"] = nullptr;
                }
                if (rec.energy_valid) {
                    res["energy"] = {
                        {"prefill_joules", r3(rec.prefill_joules)},
                        {"decode_joules",  r3(rec.decode_joules)},
                        {"total_joules",   r3(rec.prefill_joules + rec.decode_joules)},
                    };
                } else {
                    res["energy"] = nullptr;
                }
            } break;
        case SERVER_OBS_REC_REQUEST_REJECTED:
            {
                res["event"]   = "request_rejected";
                res["request"] = obs_request_info(rec, false);
                res["stage"]   = rec.stage;
                res["error"]   = {
                    {"type",    rec.error_type},
                    {"message", rec.error_msg},
                };
                res["queue_wait_ms"] = opt_num(rec.queue_wait_ms >= 0.0, r3(rec.queue_wait_ms));
            } break;
    }

    return res.dump_safe();
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
        uint64_t prefill = 0, decode = 0, drafted = 0, accepted = 0, calls = 0, busy = 0;
        uint64_t started = 0, done = 0, rejected = 0;
        double   e_board = 0.0, e_prefill = 0.0, e_decode = 0.0, e_idle = 0.0;
        uint64_t parks = 0, unparks = 0, preempts = 0, resumes = 0, park_bytes = 0, unpark_bytes = 0, park_us = 0, unpark_us = 0;
        uint64_t wait_slot = 0, wait_kv = 0, recompute = 0;
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
        b.started  = counters.req_started.load(std::memory_order_relaxed);
        b.done     = counters.req_done.load(std::memory_order_relaxed);
        b.rejected = counters.req_rejected.load(std::memory_order_relaxed);
        b.parks        = counters.sched_parks.load(std::memory_order_relaxed);
        b.unparks      = counters.sched_unparks.load(std::memory_order_relaxed);
        b.preempts     = counters.sched_preempts.load(std::memory_order_relaxed);
        b.resumes      = counters.sched_resumes.load(std::memory_order_relaxed);
        b.park_bytes   = counters.sched_park_bytes.load(std::memory_order_relaxed);
        b.unpark_bytes = counters.sched_unpark_bytes.load(std::memory_order_relaxed);
        b.park_us      = counters.sched_park_us.load(std::memory_order_relaxed);
        b.unpark_us    = counters.sched_unpark_us.load(std::memory_order_relaxed);
        b.wait_slot    = counters.sched_wait_slot.load(std::memory_order_relaxed);
        b.wait_kv      = counters.sched_wait_kv.load(std::memory_order_relaxed);
        b.recompute    = counters.sched_recompute_tokens.load(std::memory_order_relaxed);
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

        const server_obs_clock::snapshot snap = clock.read();
        uint64_t mj = 0;
        const bool cur_ok = nvml->energy_mj(mj);

        std::array<int64_t, SERVER_OBS_PHASE_COUNT> dt;
        int64_t total = 0;
        for (int i = 0; i < SERVER_OBS_PHASE_COUNT; i++) {
            dt[i]  = snap.ns[i] - prev.ns[i];
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
            }
            if (idle_acc_s >= 2.0) {
                const double w = idle_acc_j / idle_acc_s;
                idle_watts = idle_watts < 0.0 ? w : 0.8 * idle_watts + 0.2 * w;
                idle_acc_j = 0.0;
                idle_acc_s = 0.0;
            }
            window.push_back({snap.t_ns, snap.ns, energy_board});
            // keep one sample at or beyond the window edge, so the window always spans it fully
            while (window.size() > 2 && window[1].t_ns <= snap.t_ns - window_ns) {
                window.pop_front();
            }
            last_clock = snap;

            mark.prefill_joules = energy_prefill;
            mark.decode_joules  = energy_decode;
            mark.prefill_tokens = c_prefill;
            mark.decode_tokens  = c_decode;
        }

        prev    = snap;
        if (cur_ok) {
            prev_mj = mj;
            prev_ok = true;
        }

        if (snap.t_ns - base.clk.t_ns < interval_ns) {
            continue;
        }

        base_t cur = take_base(snap);

        const uint64_t n_prefill = cur.prefill - base.prefill;
        const uint64_t n_decode  = cur.decode  - base.decode;
        const bool active = n_prefill > 0 || n_decode > 0 ||
                            cur.started != base.started || cur.done != base.done || cur.rejected != base.rejected;
        if (!active) {
            // fold the idle time into the next record
            continue;
        }

        const double interval_s = (snap.t_ns - base.clk.t_ns) / 1e9;
        double loop_s[SERVER_OBS_PHASE_COUNT];
        for (int i = 0; i < SERVER_OBS_PHASE_COUNT; i++) {
            loop_s[i] = (snap.ns[i] - base.clk.ns[i]) / 1e9;
        }
        const uint64_t n_calls = cur.calls - base.calls;

        json body = {
            {"interval_seconds", r6(interval_s)},
            {"tokens", {
                {"prefill",  n_prefill},
                {"decode",   n_decode},
                {"drafted",  cur.drafted  - base.drafted},
                {"accepted", cur.accepted - base.accepted},
            }},
            {"throughput_tokens_per_second", {
                {"prefill", r3(n_prefill / interval_s)},
                {"decode",  r3(n_decode  / interval_s)},
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
                {"waited_slot",             cur.wait_slot - base.wait_slot},
                {"waited_kv",               cur.wait_kv   - base.wait_kv},
            }},
            {"kv", {
                {"cells_total",    counters.kv_cells_total.load(std::memory_order_relaxed)},
                {"cells_used",     counters.kv_cells_used.load(std::memory_order_relaxed)},
                {"cells_reserved", counters.kv_cells_reserved.load(std::memory_order_relaxed)},
            }},
            {"parking", {
                {"sessions_parked", counters.sessions_parked.load(std::memory_order_relaxed)},
                {"parks",           cur.parks    - base.parks},
                {"unparks",         cur.unparks  - base.unparks},
                {"preempts",        cur.preempts - base.preempts},
                {"resumes",         cur.resumes  - base.resumes},
                {"park_bytes",      cur.park_bytes   - base.park_bytes},
                {"unpark_bytes",    cur.unpark_bytes - base.unpark_bytes},
                {"park_seconds",    r6((cur.park_us   - base.park_us)   / 1e6)},
                {"unpark_seconds",  r6((cur.unpark_us - base.unpark_us) / 1e6)},
                {"recompute_tokens", cur.recompute - base.recompute},
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
                {"board_joules",   r3(jb)},
                {"prefill_joules", r3(jp)},
                {"decode_joules",  r3(jd)},
                {"idle_joules",    r3(ji)},
                {"idle_watts",     opt_num(w_idle >= 0.0, r3(w_idle))},
                {"joules_per_token", {
                    {"served",  opt_num(n_all     > 0, r6(jb / std::max<uint64_t>(1, n_all)))},
                    {"active",  opt_num(n_all     > 0, r6((jp + jd) / std::max<uint64_t>(1, n_all)))},
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

        server_obs_record rec;
        rec.kind      = SERVER_OBS_REC_RAW;
        rec.event     = "throughput";
        rec.t_unix_ms = now_unix_ms();
        rec.body      = std::move(body);
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
// /telemetry and /metrics sections
//

json server_observer::gpu_static_json() const {
    return nvml->static_json();
}

json server_observer::gpu_json() const {
    return nvml->live_json();
}

json server_observer::energy_json() const {
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

json server_observer::loop_totals_json() const {
    return obs_phase_seconds(clock.read().ns, 1e-9);
}

json server_observer::events_json() const {
    std::lock_guard<std::mutex> lock(mtx_subs);
    uint64_t dropped = n_dropped_closed.load(std::memory_order_relaxed);
    for (const auto & s : subs) {
        std::lock_guard<std::mutex> lock_sub(s->mtx);
        dropped += s->dropped;
    }
    return json {
        {"jsonl_enabled",   jsonl != nullptr},
        {"jsonl_path",      cfg.jsonl_path.empty() ? json(nullptr) : json(cfg.jsonl_path)},
        {"subscribers",     subs.size()},
        {"records_emitted", n_emitted.load(std::memory_order_relaxed)},
        {"records_dropped", dropped},
    };
}

std::string server_observer::to_prometheus() const {
    const server_obs_clock::snapshot clk = clock.read();

    double e_board, e_prefill, e_decode, e_idle, w_idle;
    {
        std::lock_guard<std::mutex> lock(mtx_state);
        e_board   = energy_board;
        e_prefill = energy_prefill;
        e_decode  = energy_decode;
        e_idle    = energy_idle;
        w_idle    = idle_watts;
    }
    const json energy = energy_json();

    std::stringstream out;
    out << "# HELP llamacpp:loop_seconds_total Server loop wall clock by phase\n"
        << "# TYPE llamacpp:loop_seconds_total counter\n";
    for (int i = 0; i < SERVER_OBS_PHASE_COUNT; i++) {
        out << "llamacpp:loop_seconds_total{phase=\"" << server_obs_phase_name((server_obs_phase) i) << "\"} " << clk.ns[i] / 1e9 << "\n";
    }

    if (energy_available) {
        out << "# HELP llamacpp:board_energy_joules_total GPU board energy since server start\n"
            << "# TYPE llamacpp:board_energy_joules_total counter\n"
            << "llamacpp:board_energy_joules_total " << e_board << "\n"
            << "# HELP llamacpp:energy_joules_total GPU board energy attributed to the server loop phase\n"
            << "# TYPE llamacpp:energy_joules_total counter\n"
            << "llamacpp:energy_joules_total{phase=\"prefill\"} " << e_prefill << "\n"
            << "llamacpp:energy_joules_total{phase=\"decode\"} "  << e_decode  << "\n"
            << "llamacpp:energy_joules_total{phase=\"idle\"} "    << e_idle    << "\n";
        if (!energy.at("board_watts_1s").is_null()) {
            out << "# HELP llamacpp:board_power_watts GPU board power, mean over the last second\n"
                << "# TYPE llamacpp:board_power_watts gauge\n"
                << "llamacpp:board_power_watts " << energy.at("board_watts_1s").get<double>() << "\n";
        }
        if (w_idle >= 0.0) {
            out << "# HELP llamacpp:idle_power_watts GPU board power calibrated while the server is idle\n"
                << "# TYPE llamacpp:idle_power_watts gauge\n"
                << "llamacpp:idle_power_watts " << w_idle << "\n";
        }
    }

    const auto metric = [&](const char * name, const char * type, const char * help, double value) {
        out << "# HELP llamacpp:" << name << " " << help << "\n"
            << "# TYPE llamacpp:" << name << " " << type << "\n"
            << "llamacpp:" << name << " " << value << "\n";
    };
    const auto ld = [](const auto & a) { return (double) a.load(std::memory_order_relaxed); };

    const auto & c = counters;
    metric("kv_cells_total",    "gauge", "KV cells of the unified cache",                               ld(c.kv_cells_total));
    metric("kv_cells_used",     "gauge", "KV cells held by all slots",                                  ld(c.kv_cells_used));
    metric("kv_cells_reserved", "gauge", "KV cells held or reserved by running requests (KV budget)",   ld(c.kv_cells_reserved));
    metric("sched_waiting",     "gauge", "Requests waiting for admission (KV budget)",                  ld(c.sched_waiting));
    metric("sched_preempted",   "gauge", "Slots paused to host memory (KV budget)",                     ld(c.sched_preempted));
    metric("sessions_parked",   "gauge", "Sessions held in the host prompt cache",                      ld(c.sessions_parked));
    metric("sched_parks_total",       "counter", "Idle sessions moved from VRAM to host memory",        ld(c.sched_parks));
    metric("sched_park_drops_total",  "counter", "Idle sessions evicted without a new host copy",       ld(c.sched_park_drops));
    metric("sched_unparks_total",     "counter", "Sessions restored from host memory",                  ld(c.sched_unparks));
    metric("sched_preempts_total",    "counter", "Generating slots paused to host memory",              ld(c.sched_preempts));
    metric("sched_resumes_total",     "counter", "Paused slots resumed",                                ld(c.sched_resumes));
    metric("sched_grows_total",       "counter", "KV reservations extended in flight",                  ld(c.sched_grows));
    metric("sched_park_bytes_total",  "counter", "Bytes copied device to host by parking",              ld(c.sched_park_bytes));
    metric("sched_unpark_bytes_total","counter", "Bytes copied host to device by unparking",            ld(c.sched_unpark_bytes));
    metric("sched_park_seconds_total",  "counter", "Time spent parking",                                ld(c.sched_park_us) / 1e6);
    metric("sched_unpark_seconds_total","counter", "Time spent unparking",                              ld(c.sched_unpark_us) / 1e6);
    metric("sched_wait_slot_total",   "counter", "Requests that waited for a free slot",                ld(c.sched_wait_slot));
    metric("sched_wait_kv_total",     "counter", "Requests that waited for KV cells",                   ld(c.sched_wait_kv));
    metric("sched_rollback_fallbacks_total", "counter", "Sequence truncations the memory refused (recovered)", ld(c.sched_rollback_fallbacks));
    metric("sched_recompute_tokens_total",   "counter", "Tokens re-prefilled to recover a sequence",           ld(c.sched_recompute_tokens));
    metric("sched_kv_overflows_total",       "counter", "Decodes that found no free KV cells",                 ld(c.sched_kv_overflows));

    return out.str();
}
