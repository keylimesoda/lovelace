#include "llama-io.h"

#include <vector>

void llama_io_write_i::write_string(const std::string & str) {
    uint32_t str_size = str.size();

    write(&str_size,  sizeof(str_size));
    write(str.data(), str_size);
}

void llama_io_read_i::read_string(std::string & str) {
    uint32_t str_size;
    read(&str_size, sizeof(str_size));

    std::vector<char> buf(str_size);
    read(buf.data(), str_size);

    str.assign(buf.data(), str_size);
}

void llama_io_read_i::skip(size_t size) {
    uint8_t buf[4096];
    while (size > 0) {
        const size_t n = size < sizeof(buf) ? size : sizeof(buf);
        read(buf, n);
        size -= n;
    }
}
