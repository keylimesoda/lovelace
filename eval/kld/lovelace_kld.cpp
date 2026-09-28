// lovelace-kld: quantization error at long context with a sparse (top-k) base file.
//
//   base     -> run the reference model over windows.jsonl, store top-k log-probs per scored position
//   compare  -> run a candidate model over the same windows, KL(P_base || Q) over top-k + tail bucket
//   tokcount -> exact token depths for build_corpus.py (vocab-only load, no weights)
//   selftest -> checks for the math and the base-file round trip
//
// See README.md for the workflow and file format.

#include "kld_math.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <list>
#include <map>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

constexpr char     BASE_MAGIC[8] = {'L', 'V', 'K', 'L', 'D', 'B', 'A', 'S'};
constexpr uint32_t BASE_VERSION  = 1;

[[noreturn]] void die(const std::string & msg) {
    fprintf(stderr, "lovelace-kld: error: %s\n", msg.c_str());
    exit(1);
}

// ---------------------------------------------------------------------------------------------
// arguments

struct Args {
    std::string cmd, model, windows, out, base;
    int         topk       = 64;
    int         max_scored = 1024;
    int         n_ctx      = 0;  // 0 = size from the longest window
    int         ngl        = -1;
    std::string ctk = "f16", ctv = "f16", fa = "on";
    std::string load_mode = "auto";  // llama load mode: auto|none|mmap|mlock|mmap+mlock|dio
    int         n_ubatch    = 512;
    int         n_batch     = 2048;
    int         max_outputs = 256;  // logit rows per llama_decode call (llama n_outputs_max)
    int         n_threads   = 8;
    bool        verbose     = false;
    std::vector<std::string> overrides;  // -ot "<tensor regex>=<buffer type>"
    std::vector<std::string> buckets;    // --buckets: only these depth buckets (empty = all)
};

const char * USAGE =
    "usage:\n"
    "  lovelace-kld base     --model M --windows W.jsonl --out base.bin [--topk 64] [ctx args]\n"
    "  lovelace-kld compare  --model M --windows W.jsonl --base base.bin --out result.json [ctx args]\n"
    "  lovelace-kld tokcount --model M --windows W.jsonl --out counts.jsonl\n"
    "  lovelace-kld selftest\n"
    "ctx args: -c N (0 = longest window) -ngl N -ctk T -ctv T -fa on|off|auto -ub N -b N -t N\n"
    "          -ot REGEX=BUFT[,...] (repeatable; BUFT e.g. CPU, CUDA_Host, CUDA0)\n"
    "          --max-outputs N (logit rows per decode call, default 256) --buckets B[,B...]\n"
    "          --load-mode auto|none|mmap|... (none: -ot=CPU weights go to pinned host memory)\n"
    "          --max-scored N (base only, default 1024) --verbose\n";

std::vector<std::string> split_commas(const std::string & s) {
    std::vector<std::string> out;
    size_t                   start = 0;
    while (start <= s.size()) {
        const size_t end = std::min(s.find(',', start), s.size());
        if (end > start) out.push_back(s.substr(start, end - start));
        start = end + 1;
    }
    return out;
}

Args parse_args(int argc, char ** argv) {
    if (argc < 2) {
        fputs(USAGE, stderr);
        exit(2);
    }
    Args a;
    a.cmd = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value for " + k);
            return argv[++i];
        };
        auto ival = [&]() -> int {
            const std::string v = val();
            char * end = nullptr;
            const long n = strtol(v.c_str(), &end, 10);
            if (end == v.c_str() || *end) die("bad integer for " + k + ": " + v);
            return int(n);
        };
        if (k == "--model" || k == "-m") a.model = val();
        else if (k == "--windows") a.windows = val();
        else if (k == "--out" || k == "-o") a.out = val();
        else if (k == "--base") a.base = val();
        else if (k == "--topk") a.topk = ival();
        else if (k == "--max-scored") a.max_scored = ival();
        else if (k == "-c" || k == "--ctx-size") a.n_ctx = ival();
        else if (k == "-ngl" || k == "--n-gpu-layers") a.ngl = ival();
        else if (k == "-ctk" || k == "--cache-type-k") a.ctk = val();
        else if (k == "-ctv" || k == "--cache-type-v") a.ctv = val();
        else if (k == "-fa" || k == "--flash-attn") a.fa = val();
        else if (k == "-ub" || k == "--ubatch-size") a.n_ubatch = ival();
        else if (k == "-b" || k == "--batch-size") a.n_batch = ival();
        else if (k == "-t" || k == "--threads") a.n_threads = ival();
        else if (k == "-ot" || k == "--override-tensor") {
            for (auto & o : split_commas(val())) a.overrides.push_back(o);
        } else if (k == "--buckets") a.buckets = split_commas(val());
        else if (k == "--max-outputs") a.max_outputs = ival();
        else if (k == "--load-mode") a.load_mode = val();
        else if (k == "--verbose") a.verbose = true;
        else if (k == "-h" || k == "--help") {
            fputs(USAGE, stdout);
            exit(0);
        } else die("unknown argument " + k + "\n" + USAGE);
    }
    auto need = [&](const std::string & v, const char * name) {
        if (v.empty()) die(std::string("missing ") + name + " for " + a.cmd);
    };
    if (a.cmd == "base" || a.cmd == "compare" || a.cmd == "tokcount") {
        need(a.model, "--model");
        need(a.windows, "--windows");
        need(a.out, "--out");
    }
    if (a.cmd == "compare") need(a.base, "--base");
    if (a.topk < 1) die("--topk must be >= 1");
    if (a.max_scored < 1) die("--max-scored must be >= 1");
    if (a.n_batch < 1 || a.n_ubatch < 1 || a.n_ubatch > a.n_batch) die("need 1 <= -ub <= -b");
    if (a.max_outputs < 1) die("--max-outputs must be >= 1");
    try {
        llama_load_mode_from_str(a.load_mode.c_str());
    } catch (const std::exception & e) {
        die(e.what());
    }
    if (a.fa != "on" && a.fa != "off" && a.fa != "auto") die("-fa must be on|off|auto");
    return a;
}

ggml_type parse_kv_type(const std::string & s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * name = ggml_type_name(ggml_type(t));
        if (name && s == name) return ggml_type(t);
    }
    die("unknown KV cache type " + s);
}

// ---------------------------------------------------------------------------------------------
// windows

struct Window {
    std::string id, session, bucket, text;
    size_t      start_byte = 0;  // UTF-8 byte offset of score_start_char
};

// Python code-point offset -> UTF-8 byte offset.
size_t utf8_offset(const std::string & s, int64_t cp) {
    int64_t seen = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80) {
            if (seen == cp) return i;
            ++seen;
        }
    }
    if (seen == cp) return s.size();
    die("score_start_char beyond end of text");
}

std::vector<Window> load_windows(const std::string & path) {
    std::ifstream f(path);
    if (!f) die("cannot open " + path);
    std::vector<Window> out;
    std::string         line;
    size_t              lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        json j;
        try {
            j = json::parse(line);
        } catch (const std::exception & e) {
            die(path + ":" + std::to_string(lineno) + ": " + e.what());
        }
        Window w;
        w.id         = j.at("id").get<std::string>();
        w.session    = j.value("session", "");
        w.bucket     = j.value("depth_bucket", "all");
        w.text       = j.at("text").get<std::string>();
        w.start_byte = utf8_offset(w.text, j.at("score_start_char").get<int64_t>());
        out.push_back(std::move(w));
    }
    if (out.empty()) die("no windows in " + path);
    return out;
}

bool bucket_selected(const Args & a, const std::string & bucket) {
    return a.buckets.empty() || std::find(a.buckets.begin(), a.buckets.end(), bucket) != a.buckets.end();
}

// --buckets names must exist in the windows file (catches typos that would silently select nothing)
void check_buckets(const Args & a, const std::vector<Window> & windows) {
    for (const auto & b : a.buckets)
        if (std::none_of(windows.begin(), windows.end(), [&](const Window & w) { return w.bucket == b; }))
            die("--buckets: no windows in bucket " + b + " in " + a.windows);
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const char * text, size_t len) {
    if (len > size_t(INT32_MAX)) die("text too long to tokenize");
    std::vector<llama_token> tok(len + 16);
    int32_t n = llama_tokenize(vocab, text, int32_t(len), tok.data(), int32_t(tok.size()), false, true);
    if (n < 0) {
        tok.resize(size_t(-n));
        n = llama_tokenize(vocab, text, int32_t(len), tok.data(), int32_t(tok.size()), false, true);
        if (n < 0) die("tokenization failed");
    }
    tok.resize(size_t(n));
    return tok;
}

// The model's view of a window: prefix tokens (context) followed by the scored reply tokens.
// Prefix and reply are tokenized separately, as at serving time (prompt tokenized, reply generated).
struct Tokenized {
    std::vector<llama_token> tokens;  // n_prefix + n_scored
    int32_t                  n_prefix = 0;
    int32_t                  n_scored = 0;
    uint64_t                 hash     = 0;

    // tokens that must be decoded: the last scored token is only a target
    int32_t n_decode() const { return n_prefix + n_scored - 1; }
};

Tokenized tokenize_window(const llama_vocab * vocab, const Window & w, int max_scored) {
    Tokenized t;
    t.tokens  = tokenize(vocab, w.text.data(), w.start_byte);
    auto tail = tokenize(vocab, w.text.data() + w.start_byte, w.text.size() - w.start_byte);
    if (t.tokens.empty()) die(w.id + ": empty prefix");
    if (tail.empty()) die(w.id + ": empty scored span");
    t.n_prefix = int32_t(t.tokens.size());
    t.n_scored = int32_t(std::min<size_t>(tail.size(), size_t(max_scored)));
    t.tokens.insert(t.tokens.end(), tail.begin(), tail.begin() + t.n_scored);
    t.hash = kld::fnv1a(&t.n_prefix, sizeof(t.n_prefix));
    t.hash = kld::fnv1a(t.tokens.data(), t.tokens.size() * sizeof(llama_token), t.hash);
    return t;
}

// ---------------------------------------------------------------------------------------------
// llama

int g_log_level = GGML_LOG_LEVEL_WARN;

// warnings/errors, plus llama.cpp's per-device model/KV/compute "buffer size" lines (VRAM check)
void log_cb(ggml_log_level level, const char * text, void *) {
    static int last = GGML_LOG_LEVEL_INFO;
    if (level != GGML_LOG_LEVEL_CONT) last = level;
    if (last >= g_log_level || (level == GGML_LOG_LEVEL_INFO && strstr(text, "buffer size"))) fputs(text, stderr);
}

// -ot "<regex>=<buffer type>" -> llama's NULL-terminated override list. Buffer types are the device
// buffer types (CPU, CUDA0) as in llama.cpp's -ot, plus the devices' pinned host types (CUDA_Host).
std::vector<llama_model_tensor_buft_override> tensor_overrides(const std::vector<std::string> & specs) {
    static std::list<std::string> patterns;  // outlive the model load; list: stable c_str()
    std::map<std::string, ggml_backend_buffer_type_t> bufts;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        for (auto * buft : {ggml_backend_dev_buffer_type(dev), ggml_backend_dev_host_buffer_type(dev)})
            if (buft) bufts[ggml_backend_buft_name(buft)] = buft;
    }
    std::vector<llama_model_tensor_buft_override> out;
    for (const auto & s : specs) {
        const size_t eq = s.rfind('=');
        if (eq == std::string::npos || eq == 0) die("-ot needs <tensor regex>=<buffer type>: " + s);
        const auto it = bufts.find(s.substr(eq + 1));
        if (it == bufts.end()) {
            std::string known;
            for (const auto & b : bufts) known += " " + b.first;
            die("unknown buffer type in -ot " + s + " (available:" + known + ")");
        }
        patterns.push_back(s.substr(0, eq));
        out.push_back({patterns.back().c_str(), it->second});
    }
    if (!out.empty()) out.push_back({nullptr, nullptr});
    return out;
}

llama_model * load_model(const Args & a, bool vocab_only) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers       = vocab_only ? 0 : a.ngl;
    mp.vocab_only         = vocab_only;
    mp.load_mode          = llama_load_mode_from_str(a.load_mode.c_str());
    const auto ot         = tensor_overrides(vocab_only ? std::vector<std::string>{} : a.overrides);
    if (!ot.empty()) mp.tensor_buft_overrides = ot.data();
    llama_model * m = llama_model_load_from_file(a.model.c_str(), mp);
    if (!m) die("failed to load " + a.model);
    return m;
}

llama_context * make_context(llama_model * model, const Args & a, uint32_t n_ctx) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = n_ctx;
    cp.n_batch              = uint32_t(a.n_batch);
    cp.n_ubatch             = uint32_t(a.n_ubatch);
    cp.n_outputs_max        = uint32_t(a.max_outputs);
    cp.n_seq_max            = 1;
    cp.n_threads            = a.n_threads;
    cp.n_threads_batch      = a.n_threads;
    cp.type_k               = parse_kv_type(a.ctk);
    cp.type_v               = parse_kv_type(a.ctv);
    cp.flash_attn_type      = a.fa == "on"    ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                              : a.fa == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED
                                              : LLAMA_FLASH_ATTN_TYPE_AUTO;
    cp.no_perf              = true;
    llama_context * ctx     = llama_init_from_model(model, cp);
    if (!ctx) die("failed to create context (n_ctx " + std::to_string(n_ctx) + ")");
    return ctx;
}

// n_ctx covering every window up to the model's training context; windows beyond it are dropped.
uint32_t context_size(const Args & a, const llama_model * model, const std::vector<int32_t> & need) {
    const int32_t train = llama_model_n_ctx_train(model);
    if (a.n_ctx > 0) return uint32_t(a.n_ctx);
    int32_t mx = 0;
    for (int32_t n : need)
        if (n <= train) mx = std::max(mx, n);
    if (mx == 0) die("no window fits the model's training context " + std::to_string(train));
    return uint32_t(std::min<int64_t>(train, (int64_t(mx) + 255) / 256 * 256));
}

// Decode the window and call on_row(j, logits) for the distribution predicting tokens[n_prefix + j].
// Only scored positions request outputs, at most max_outputs per llama_decode call (the context's
// n_outputs_max), which bounds the host logits buffer and the logits part of the compute buffer.
void run_window(llama_context * ctx, const Tokenized & t, int n_batch, int max_outputs,
                const std::function<void(int32_t, const float *)> & on_row) {
    llama_memory_clear(llama_get_memory(ctx), true);
    const int32_t first_out = t.n_prefix - 1;
    const int32_t end       = t.n_decode();
    llama_batch   batch     = llama_batch_init(n_batch, 0, 1);
    for (int32_t pos = 0; pos < end;) {
        int32_t n = std::min(n_batch, end - pos);
        if (pos + n > first_out) n = std::min(n, std::max(first_out - pos, 0) + max_outputs);
        batch.n_tokens = n;
        for (int32_t i = 0; i < n; ++i) {
            batch.token[i]     = t.tokens[size_t(pos + i)];
            batch.pos[i]       = pos + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (pos + i >= first_out) ? 1 : 0;
        }
        const int32_t rc = llama_decode(ctx, batch);
        if (rc != 0) die("llama_decode failed (" + std::to_string(rc) + ") at pos " + std::to_string(pos));
        for (int32_t i = std::max(0, first_out - pos); i < n; ++i) {
            const float * row = llama_get_logits_ith(ctx, i);
            if (!row) die("missing logits at batch index " + std::to_string(i));
            on_row(pos + i - first_out, row);
        }
        pos += n;
    }
    llama_batch_free(batch);
}

// ---------------------------------------------------------------------------------------------
// base file
//
// header : char magic[8] "LVKLDBAS" | u32 version | u32 topk | u32 n_windows | u32 n_vocab
//          | u64 fnv1a64(model path) | u32 meta_len | meta_len bytes of JSON (run parameters)
// window : u32 id_len | id bytes | i32 n_prefix | i32 n_scored | u64 token hash
//          | n_scored x position
// position (16 + 8*topk bytes): i32 next_id | f32 next_logprob | f64 logsumexp
//          | topk x (i32 id, f32 logprob), descending
// All values little-endian (native x86-64).

struct BasePos {
    int32_t                    next_id;
    float                      next_lp;
    double                     lse;
    std::vector<kld::TopEntry> top;
};

struct BaseWindow {
    std::string          id;
    int32_t              n_prefix = 0, n_scored = 0;
    uint64_t             hash = 0;
    std::vector<BasePos> pos;
};

struct BaseFile {
    uint32_t                topk = 0, n_vocab = 0;
    uint64_t                model_hash = 0;
    json                    meta;
    std::vector<BaseWindow> windows;
};

class Writer {
  public:
    explicit Writer(const std::string & path) : f_(fopen(path.c_str(), "wb")), path_(path) {
        if (!f_) die("cannot write " + path);
    }
    ~Writer() {
        if (f_) fclose(f_);
    }
    template <typename T> void put(const T & v) { bytes(&v, sizeof(T)); }
    void bytes(const void * p, size_t n) {
        if (fwrite(p, 1, n, f_) != n) die("write failed: " + path_);
    }
    void seek(long off) {
        if (fseek(f_, off, SEEK_SET) != 0) die("seek failed: " + path_);
    }
    void close() {
        if (fclose(f_) != 0) die("close failed: " + path_);
        f_ = nullptr;
    }

  private:
    FILE *      f_;
    std::string path_;
};

constexpr long N_WINDOWS_OFFSET = 8 + 4 + 4;

void write_header(Writer & w, uint32_t topk, uint32_t n_windows, uint32_t n_vocab, uint64_t model_hash,
                  const json & meta) {
    const std::string m = meta.dump();
    w.bytes(BASE_MAGIC, sizeof(BASE_MAGIC));
    w.put(BASE_VERSION);
    w.put(topk);
    w.put(n_windows);
    w.put(n_vocab);
    w.put(model_hash);
    w.put(uint32_t(m.size()));
    w.bytes(m.data(), m.size());
}

void write_window_head(Writer & w, const std::string & id, const Tokenized & t) {
    w.put(uint32_t(id.size()));
    w.bytes(id.data(), id.size());
    w.put(t.n_prefix);
    w.put(t.n_scored);
    w.put(t.hash);
}

void write_pos(Writer & w, const BasePos & p) {
    w.put(p.next_id);
    w.put(p.next_lp);
    w.put(p.lse);
    w.bytes(p.top.data(), p.top.size() * sizeof(kld::TopEntry));
}

BaseFile read_base(const std::string & path) {
    static_assert(sizeof(kld::TopEntry) == 8, "TopEntry must be packed (i32, f32)");
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot open " + path);
    auto get = [&](void * p, size_t n) {
        if (!f.read(static_cast<char *>(p), std::streamsize(n))) die(path + ": truncated");
    };
    char magic[8];
    get(magic, 8);
    if (memcmp(magic, BASE_MAGIC, 8) != 0) die(path + ": not a lovelace-kld base file");
    uint32_t version, n_windows, meta_len;
    BaseFile b;
    get(&version, 4);
    if (version != BASE_VERSION) die(path + ": unsupported version " + std::to_string(version));
    get(&b.topk, 4);
    get(&n_windows, 4);
    get(&b.n_vocab, 4);
    get(&b.model_hash, 8);
    get(&meta_len, 4);
    std::string m(meta_len, '\0');
    get(m.data(), meta_len);
    b.meta = json::parse(m);
    b.windows.resize(n_windows);
    for (auto & w : b.windows) {
        uint32_t id_len;
        get(&id_len, 4);
        w.id.resize(id_len);
        get(w.id.data(), id_len);
        get(&w.n_prefix, 4);
        get(&w.n_scored, 4);
        get(&w.hash, 8);
        w.pos.resize(size_t(w.n_scored));
        for (auto & p : w.pos) {
            get(&p.next_id, 4);
            get(&p.next_lp, 4);
            get(&p.lse, 8);
            p.top.resize(b.topk);
            get(p.top.data(), b.topk * sizeof(kld::TopEntry));
        }
    }
    if (f.peek() != EOF) die(path + ": trailing data (window count mismatch)");
    return b;
}

json run_meta(const Args & a, uint32_t n_ctx) {
    char   when[32];
    time_t now = time(nullptr);
    strftime(when, sizeof(when), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
    return json{{"model", a.model},         {"windows", a.windows},   {"buckets", a.buckets},
                {"n_ctx", n_ctx},           {"ngl", a.ngl},           {"overrides", a.overrides},
                {"load_mode", a.load_mode},
                {"ctk", a.ctk},             {"ctv", a.ctv},           {"fa", a.fa},
                {"n_batch", a.n_batch},     {"n_ubatch", a.n_ubatch}, {"max_outputs", a.max_outputs},
                {"topk", a.topk},           {"max_scored", a.max_scored}, {"created", when}};
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// ---------------------------------------------------------------------------------------------
// base

int cmd_base(const Args & a) {
    auto windows = load_windows(a.windows);
    check_buckets(a, windows);
    windows.erase(std::remove_if(windows.begin(), windows.end(),
                                 [&](const Window & w) { return !bucket_selected(a, w.bucket); }),
                  windows.end());
    if (windows.empty()) die("no windows selected");
    llama_backend_init();
    llama_model *       model   = load_model(a, false);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<Tokenized> toks;
    std::vector<int32_t>   need;
    for (const auto & w : windows) {
        toks.push_back(tokenize_window(vocab, w, a.max_scored));
        need.push_back(toks.back().n_decode());
    }
    const uint32_t n_ctx = context_size(a, model, need);
    llama_context * ctx  = make_context(model, a, n_ctx);
    fprintf(stderr, "base: %zu windows, n_ctx %u, n_vocab %d, topk %d\n", windows.size(), n_ctx, n_vocab, a.topk);

    const uint32_t topk = uint32_t(std::min(a.topk, n_vocab));
    Writer         out(a.out);
    write_header(out, topk, 0, uint32_t(n_vocab), kld::fnv1a(a.model.data(), a.model.size()), run_meta(a, n_ctx));

    uint32_t                                done = 0;
    BasePos                                 p;
    std::vector<std::pair<float, int32_t>>  heap;
    const auto                              t_all = std::chrono::steady_clock::now();
    for (size_t wi = 0; wi < windows.size(); ++wi) {
        const Tokenized & t = toks[wi];
        if (uint32_t(t.n_decode()) > n_ctx) {
            fprintf(stderr, "[%zu/%zu] %s: SKIP, needs %d > n_ctx %u\n", wi + 1, windows.size(),
                    windows[wi].id.c_str(), t.n_decode(), n_ctx);
            continue;
        }
        write_window_head(out, windows[wi].id, t);
        const auto t0 = std::chrono::steady_clock::now();
        run_window(ctx, t, a.n_batch, a.max_outputs, [&](int32_t j, const float * logits) {
            p.next_id = t.tokens[size_t(t.n_prefix + j)];
            p.lse     = kld::logsumexp(logits, n_vocab);
            p.next_lp = float(double(logits[p.next_id]) - p.lse);
            kld::top_k(logits, n_vocab, int(topk), p.lse, p.top, heap);
            write_pos(out, p);
        });
        ++done;
        const double dt = seconds_since(t0);
        fprintf(stderr, "[%zu/%zu] %s: depth %d, scored %d, %.1fs (%.0f tok/s)\n", wi + 1, windows.size(),
                windows[wi].id.c_str(), t.n_prefix, t.n_scored, dt, t.n_decode() / dt);
    }
    out.seek(N_WINDOWS_OFFSET);
    out.put(done);
    out.close();
    fprintf(stderr, "base: wrote %u windows to %s in %.0fs\n", done, a.out.c_str(), seconds_since(t_all));

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// compare

struct PosResult {
    double kl, p_tail, nll_base, nll_q;
    bool   top1;
};

struct Agg {
    std::vector<double> kl;
    size_t              top1 = 0;
    double              nll_base = 0, nll_q = 0, p_tail = 0;

    void add(const PosResult & r) {
        kl.push_back(r.kl);
        top1 += r.top1;
        nll_base += r.nll_base;
        nll_q += r.nll_q;
        p_tail += r.p_tail;
    }
    void merge(const Agg & o) {
        kl.insert(kl.end(), o.kl.begin(), o.kl.end());
        top1 += o.top1;
        nll_base += o.nll_base;
        nll_q += o.nll_q;
        p_tail += o.p_tail;
    }
    json summary() const {
        std::vector<double> s = kl;
        std::sort(s.begin(), s.end());
        const double n    = double(s.size());
        double       mean = 0;
        for (double v : s) mean += v;
        mean /= n;
        return json{{"positions", s.size()},
                    {"kl_mean", mean},
                    {"kl_median", kld::quantile(s, 0.5)},
                    {"kl_p99", kld::quantile(s, 0.99)},
                    {"kl_p999", kld::quantile(s, 0.999)},
                    {"kl_max", s.back()},
                    {"top1_agree_pct", 100.0 * double(top1) / n},
                    {"nll_base", nll_base / n},
                    {"nll_q", nll_q / n},
                    {"delta_nll", (nll_q - nll_base) / n},
                    {"p_tail_mean", p_tail / n}};
    }
};

void print_row(const char * label, int label_w, const json & s, const std::string & extra) {
    printf("%-*s %7zu %9.5f %9.5f %9.4f %9.4f %7.2f %8.4f %8.4f %+9.5f %8.5f%s\n", label_w, label,
           s["positions"].get<size_t>(), s["kl_mean"].get<double>(), s["kl_median"].get<double>(),
           s["kl_p99"].get<double>(), s["kl_p999"].get<double>(), s["top1_agree_pct"].get<double>(),
           s["nll_base"].get<double>(), s["nll_q"].get<double>(), s["delta_nll"].get<double>(),
           s["p_tail_mean"].get<double>(), extra.c_str());
}

void print_head(const char * label, int label_w, const char * extra) {
    printf("%-*s %7s %9s %9s %9s %9s %7s %8s %8s %9s %8s%s\n", label_w, label, "pos", "KL mean", "KL med", "KL p99",
           "KL p99.9", "top1 %", "NLL ref", "NLL q", "dNLL", "p_tail", extra);
}

int cmd_compare(const Args & a) {
    BaseFile base = read_base(a.base);
    if (base.windows.empty()) die(a.base + ": no windows");
    auto                          windows = load_windows(a.windows);
    std::map<std::string, size_t> by_id;
    for (size_t i = 0; i < windows.size(); ++i) by_id[windows[i].id] = i;
    check_buckets(a, windows);
    for (const auto & bw : base.windows)
        if (!by_id.count(bw.id)) die("base window " + bw.id + " not in " + a.windows);
    const size_t n_base_windows = base.windows.size();
    base.windows.erase(std::remove_if(base.windows.begin(), base.windows.end(),
                                      [&](const BaseWindow & bw) {
                                          return !bucket_selected(a, windows[by_id[bw.id]].bucket);
                                      }),
                       base.windows.end());
    if (base.windows.empty()) die("no base windows in the selected buckets");
    // kernel choice depends on these; a mismatch adds numerical noise to every KL
    for (const auto & [key, mine] : std::vector<std::pair<const char *, json>>{
             {"n_ubatch", a.n_ubatch}, {"fa", a.fa}, {"max_outputs", a.max_outputs}}) {
        if (base.meta.contains(key) && base.meta[key] != mine)
            fprintf(stderr, "compare: warning: %s %s differs from the base run's %s\n", key, mine.dump().c_str(),
                    base.meta[key].dump().c_str());
    }

    llama_backend_init();
    llama_model *       model   = load_model(a, false);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);
    if (uint32_t(n_vocab) != base.n_vocab)
        die("vocab size " + std::to_string(n_vocab) + " != base " + std::to_string(base.n_vocab));
    const int max_scored = base.meta.value("max_scored", a.max_scored);

    std::vector<Tokenized> toks;
    std::vector<int32_t>   need;
    for (const auto & bw : base.windows) {
        toks.push_back(tokenize_window(vocab, windows[by_id[bw.id]], max_scored));
        const Tokenized & t = toks.back();
        if (t.n_prefix != bw.n_prefix || t.n_scored != bw.n_scored || t.hash != bw.hash)
            die(bw.id + ": tokens differ from the base run (different tokenizer or windows file)");
        need.push_back(t.n_decode());
    }
    const uint32_t n_ctx = context_size(a, model, need);
    for (size_t i = 0; i < need.size(); ++i)
        if (uint32_t(need[i]) > n_ctx)
            die(base.windows[i].id + " needs " + std::to_string(need[i]) + " > n_ctx " + std::to_string(n_ctx));
    llama_context * ctx = make_context(model, a, n_ctx);
    fprintf(stderr, "compare: %zu of %zu base windows, n_ctx %u, base topk %u (%s)\n", base.windows.size(),
            n_base_windows, n_ctx, base.topk, base.meta.value("model", "?").c_str());

    std::vector<std::string> bucket_order;
    std::map<std::string, Agg> buckets;
    Agg                        overall;
    json                       jwindows = json::array();
    std::vector<std::pair<std::string, json>> rows;
    const auto                                t_all = std::chrono::steady_clock::now();
    for (size_t wi = 0; wi < base.windows.size(); ++wi) {
        const BaseWindow & bw = base.windows[wi];
        const Window &     w  = windows[by_id[bw.id]];
        const Tokenized &  t  = toks[wi];
        Agg                agg;
        const auto         t0 = std::chrono::steady_clock::now();
        run_window(ctx, t, a.n_batch, a.max_outputs, [&](int32_t j, const float * logits) {
            const BasePos & p     = bw.pos[size_t(j)];
            const double    q_lse = kld::logsumexp(logits, n_vocab);
            const auto      kl    = kld::sparse_kl(p.top.data(), int(p.top.size()), logits, q_lse);
            agg.add({kl.kl, kl.p_tail, -double(p.next_lp), -(double(logits[p.next_id]) - q_lse),
                     kld::argmax(logits, n_vocab) == p.top[0].id});
        });
        const double dt = seconds_since(t0);
        fprintf(stderr, "[%zu/%zu] %s: depth %d, scored %d, %.1fs (%.0f tok/s)\n", wi + 1, base.windows.size(),
                bw.id.c_str(), t.n_prefix, t.n_scored, dt, t.n_decode() / dt);
        json s = agg.summary();
        rows.emplace_back(bw.id, s);
        jwindows.push_back(json{{"id", bw.id},
                                {"session", w.session},
                                {"depth_bucket", w.bucket},
                                {"depth_tokens", t.n_prefix},
                                {"scored_tokens", t.n_scored},
                                {"seconds", dt},
                                {"stats", s}});
        if (!buckets.count(w.bucket)) bucket_order.push_back(w.bucket);
        buckets[w.bucket].merge(agg);
        overall.merge(agg);
    }

    json run          = run_meta(a, n_ctx);
    run["topk"]       = base.topk;  // both come from the base run
    run["max_scored"] = max_scored;
    json jb = json::object();
    for (const auto & b : bucket_order) jb[b] = buckets[b].summary();
    const json jo = overall.summary();
    json       result{{"model", a.model},
                      {"model_hash", kld::fnv1a(a.model.data(), a.model.size())},
                      {"run", run},
                      {"base", {{"path", a.base}, {"model_hash", base.model_hash}, {"topk", base.topk}, {"meta", base.meta}}},
                      {"overall", jo},
                      {"buckets", jb},
                      {"windows", jwindows}};
    std::ofstream f(a.out);
    if (!f) die("cannot write " + a.out);
    f << result.dump(2) << "\n";
    if (!f) die("write failed: " + a.out);

    size_t idw = 6;
    for (const auto & r : rows) idw = std::max(idw, r.first.size());
    idw = std::min<size_t>(idw, 60);
    printf("\n%s  vs base %s  (ctk %s, ctv %s, %.0fs)\n\n", a.model.c_str(), base.meta.value("model", "?").c_str(),
           a.ctk.c_str(), a.ctv.c_str(), seconds_since(t_all));
    print_head("window", int(idw), "");
    for (const auto & r : rows) print_row(r.first.substr(0, idw).c_str(), int(idw), r.second, "");
    printf("\n");
    print_head("bucket", 8, "");
    for (const auto & b : bucket_order) print_row(b.c_str(), 8, jb[b], "");
    print_row("overall", 8, jo, "");
    fprintf(stderr, "compare: wrote %s\n", a.out.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// tokcount: exact depths for build_corpus.py --counts (vocab only, no weights, no GPU)

int cmd_tokcount(const Args & a) {
    auto windows = load_windows(a.windows);
    llama_backend_init();
    llama_model *       model = load_model(a, true);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::ofstream       f(a.out);
    if (!f) die("cannot write " + a.out);
    size_t     unstable = 0;
    const auto t0       = std::chrono::steady_clock::now();
    for (size_t i = 0; i < windows.size(); ++i) {
        const Window & w      = windows[i];
        const auto     prefix = tokenize(vocab, w.text.data(), w.start_byte);
        const auto     reply  = tokenize(vocab, w.text.data() + w.start_byte, w.text.size() - w.start_byte);
        const auto     whole  = tokenize(vocab, w.text.data(), w.text.size());
        // does tokenizing the full text reproduce the prefix/reply split?
        const bool stable = whole.size() == prefix.size() + reply.size() &&
                            std::equal(prefix.begin(), prefix.end(), whole.begin());
        unstable += !stable;
        f << json{{"id", w.id}, {"depth_tokens", prefix.size()}, {"reply_tokens", reply.size()},
                  {"boundary_stable", stable}}
                 .dump()
          << "\n";
        if ((i + 1) % 50 == 0 || i + 1 == windows.size())
            fprintf(stderr, "tokcount: %zu/%zu (%.0fs)\n", i + 1, windows.size(), seconds_since(t0));
    }
    if (!f) die("write failed: " + a.out);
    fprintf(stderr, "tokcount: wrote %s (%zu windows, %zu with unstable prefix/reply boundary)\n", a.out.c_str(),
            windows.size(), unstable);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// selftest

int g_fail = 0;

void check(bool ok, const char * what, double got, double want) {
    printf("  [%s] %-58s got %.12g want %.12g\n", ok ? "ok" : "FAIL", what, got, want);
    g_fail += !ok;
}

void check_near(const char * what, double got, double want, double tol) {
    check(std::fabs(got - want) <= tol, what, got, want);
}

std::vector<float> logits_of(const std::vector<double> & probs) {
    std::vector<float> l;
    for (double p : probs) l.push_back(p > 0 ? float(std::log(p)) : -1000.0f);
    return l;
}

// base side exactly as cmd_base stores it
std::vector<kld::TopEntry> base_top(const std::vector<float> & l, int k, double * lse_out = nullptr) {
    std::vector<kld::TopEntry>             top;
    std::vector<std::pair<float, int32_t>> heap;
    const double                           lse = kld::logsumexp(l.data(), int(l.size()));
    kld::top_k(l.data(), int(l.size()), k, lse, top, heap);
    if (lse_out) *lse_out = lse;
    return top;
}

double kl_of(const std::vector<float> & p, const std::vector<float> & q, int k) {
    const auto top = base_top(p, k);
    return kld::sparse_kl(top.data(), int(top.size()), q.data(), kld::logsumexp(q.data(), int(q.size()))).kl;
}

int cmd_selftest() {
    printf("logsumexp / top-k\n");
    {
        std::vector<float> big(1000, 1000.0f);
        check_near("lse of 1000 equal logits of 1000 = 1000 + ln 1000",
                   kld::logsumexp(big.data(), int(big.size())), 1000.0 + std::log(1000.0), 1e-9);
        const std::vector<float> l = {1, 5, 3, 5, -2, 4, 5};
        const auto               t = base_top(l, 4);
        check(t[0].id == 1 && t[1].id == 3 && t[2].id == 6 && t[3].id == 5,
              "top-4 of {1,5,3,5,-2,4,5}: ids 1,3,6,5 (ties by id)", t[0].id * 1000 + t[1].id * 100 + t[2].id * 10 + t[3].id,
              1365);
        check(kld::argmax(l.data(), int(l.size())) == 1, "argmax ties -> lowest id", kld::argmax(l.data(), 7), 1);
        double lse;
        const auto all = base_top(l, 100, &lse);
        double     s   = 0;
        for (const auto & e : all) s += std::exp(double(e.logprob));
        check_near("k > n clamps to n, probs sum to 1", s, 1.0, 1e-6);
    }

    printf("KL(P || P) = 0\n");
    {
        uint64_t           x = 42;
        std::vector<float> l(248320);
        for (auto & v : l) {
            x = x * 6364136223846793005ULL + 1442695040888963407ULL;
            v = float(double(x >> 11) / double(1ULL << 53) * 30.0 - 15.0);
        }
        check_near("random 248k-vocab logits, k=64", kl_of(l, l, 64), 0.0, 1e-9);
        std::vector<float> shifted = l;
        for (auto & v : shifted) v += 7.25f;
        check_near("logits + constant (same softmax), k=64", kl_of(l, shifted, 64), 0.0, 1e-6);
        const std::vector<float> peaked = logits_of({0.999, 0.0005, 0.0003, 0.0002});
        check_near("peaked 4-way, k=1 (tail bucket only)", kl_of(peaked, peaked, 1), 0.0, 1e-9);
    }

    printf("hand-computed: P = (.5 .3 .15 .05), Q = (.4 .4 .1 .1)\n");
    const auto P = logits_of({0.5, 0.3, 0.15, 0.05});
    const auto Q = logits_of({0.4, 0.4, 0.1, 0.1});
    // .5 ln 1.25 + .3 ln .75 + .15 ln 1.5 + .05 ln .5
    const double full = 0.5 * std::log(1.25) + 0.3 * std::log(0.75) + 0.15 * std::log(1.5) + 0.05 * std::log(0.5);
    check_near("k=4 (full support) = 0.0514295611", kl_of(P, Q, 4), 0.0514295611, 1e-6);
    check_near("  closed form agrees", full, 0.0514295611, 1e-9);
    printf("tail bucket\n");
    {
        // top-2 = {0,1}: .5 ln 1.25 + .3 ln .75 + tail (.2 vs .2) -> 0
        check_near("k=2: tail .2 vs .2 = 0.0252671543", kl_of(P, Q, 2), 0.0252671543, 1e-6);
        check(kl_of(P, Q, 2) <= kl_of(P, Q, 4) + 1e-12, "k=2 bound <= full KL (coarsening never adds)",
              kl_of(P, Q, 2), kl_of(P, Q, 4));
        // top-1 = {0}: .5 ln(.5/.4) + .5 ln(.5/.6)
        check_near("k=1: .5 ln 1.25 + .5 ln(.5/.6) = 0.0204109973", kl_of(P, Q, 1), 0.0204109973, 1e-6);
        // Q has (numerically) no mass outside P's top-2 -> q_tail clamps at 1e-10
        const auto P2 = logits_of({0.6, 0.2, 0.2});
        const auto Q2 = logits_of({0.5, 0.5, 0.0});
        const double want = 0.6 * std::log(1.2) + 0.2 * std::log(0.4) + 0.2 * std::log(0.2 / 1e-10);
        check_near("q_tail -> 0 clamps to 1e-10: 4.2094174", kl_of(P2, Q2, 2), want, 1e-5);
        const auto   top = base_top(P, 4);
        const auto   r   = kld::sparse_kl(top.data(), 4, Q.data(), kld::logsumexp(Q.data(), 4));
        // p_tail keeps the float rounding of the stored log-probs (~1e-8 here); q_tail is exact
        check(r.p_tail < 1e-6 && r.q_tail == kld::TAIL_FLOOR, "full support: tails ~0, q_tail clamps to 1e-10",
              r.q_tail, kld::TAIL_FLOOR);
    }

    printf("quantiles (numpy linear)\n");
    {
        std::vector<double> v;
        for (int i = 1; i <= 100; ++i) v.push_back(i);
        check_near("median of 1..100", kld::quantile(v, 0.5), 50.5, 1e-12);
        check_near("p99 of 1..100", kld::quantile(v, 0.99), 99.01, 1e-9);
        check_near("p99.9 of 1..100", kld::quantile(v, 0.999), 99.901, 1e-9);
    }

    printf("utf-8 offsets\n");
    {
        const std::string s = "a\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80z";  // a é € 😀 z
        check(utf8_offset(s, 4) == 10 && utf8_offset(s, 5) == s.size() && utf8_offset(s, 2) == 3,
              "code points -> bytes (a e-acute euro emoji z)", double(utf8_offset(s, 4)), 10);
    }

    printf("base file round trip\n");
    {
        char tmpl[] = "/tmp/lovelace-kld-selftest-XXXXXX";
        const int fd = mkstemp(tmpl);
        if (fd < 0) die("mkstemp failed");
        ::close(fd);
        const std::string path = tmpl;
        Tokenized         t;
        t.tokens   = {7, 8, 9, 10};
        t.n_prefix = 2;
        t.n_scored = 2;
        t.hash     = 0x1234abcdULL;
        BasePos p;
        p.next_id = 3;
        p.next_lp = -1.5f;
        p.lse     = 12.25;
        p.top     = {{3, -0.25f}, {1, -2.0f}};
        {
            Writer w(path);
            write_header(w, 2, 0, 4, kld::fnv1a("m", 1), json{{"model", "m"}});
            write_window_head(w, "w0", t);
            write_pos(w, p);
            p.next_id = 1;
            write_pos(w, p);
            w.seek(N_WINDOWS_OFFSET);
            w.put(uint32_t(1));
            w.close();
        }
        const BaseFile b = read_base(path);
        remove(path.c_str());
        const bool ok = b.topk == 2 && b.n_vocab == 4 && b.windows.size() == 1 && b.windows[0].id == "w0" &&
                        b.windows[0].n_scored == 2 && b.windows[0].hash == t.hash && b.windows[0].pos[1].next_id == 1 &&
                        b.windows[0].pos[0].top[1].id == 1 && b.windows[0].pos[0].lse == 12.25 &&
                        b.meta["model"] == "m" && b.model_hash == kld::fnv1a("m", 1);
        check(ok, "write -> read preserves header, ids, positions", ok, 1);
    }

    printf("\nselftest: %s (%d failure%s)\n", g_fail ? "FAILED" : "passed", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

}  // namespace

int main(int argc, char ** argv) {
    const Args a = parse_args(argc, argv);
    if (a.verbose) g_log_level = GGML_LOG_LEVEL_INFO;
    llama_log_set(log_cb, nullptr);
    try {
        if (a.cmd == "base") return cmd_base(a);
        if (a.cmd == "compare") return cmd_compare(a);
        if (a.cmd == "tokcount") return cmd_tokcount(a);
        if (a.cmd == "selftest") return cmd_selftest();
    } catch (const std::exception & e) {
        die(e.what());
    }
    die("unknown command " + a.cmd + "\n" + USAGE);
}
