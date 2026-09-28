#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_tensor;

// Asynchronous state transfers (llama_state_seq_get/set_data_async) copy tensor data in order with the compute
// stream by default, since a later graph may move it (e.g. recurrent rows are swapped between cells). A memory
// whose cells of other sequences never move (attention KV) marks its tensor data as concurrent: those copies may
// then overlap with graphs of other sequences on a separate copy stream.
class llama_io_concurrent_i {
public:
    void set_concurrent(bool value) { concurrent = value; }
    bool get_concurrent() const { return concurrent; }

protected:
    bool concurrent = false;
};

// marks the tensor data of the enclosing scope as concurrent
template <typename T>
struct llama_io_concurrent_scope {
    T & io;

    explicit llama_io_concurrent_scope(T & io) : io(io) { io.set_concurrent(true); }
    ~llama_io_concurrent_scope() { io.set_concurrent(false); }
};

class llama_io_write_i : public llama_io_concurrent_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class llama_io_read_i : public llama_io_concurrent_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // drop tensor data that has been read but not yet applied (e.g. when a restore fails)
    virtual void discard() {}

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);
};
