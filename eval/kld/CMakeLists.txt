cmake_minimum_required(VERSION 3.16)
project(lovelace-kld CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(LLAMA_ROOT "$ENV{HOME}/Work/src/llama.cpp" CACHE PATH "llama.cpp source tree")
set(LLAMA_LIB_DIR "${LLAMA_ROOT}/build-sm89/bin" CACHE PATH "directory with libllama.so / libggml*.so")
option(KLD_NATIVE "compile the host-side softmax/top-k loops with -march=native" ON)

foreach(lib llama ggml ggml-base)
    if(NOT EXISTS "${LLAMA_LIB_DIR}/lib${lib}.so")
        message(FATAL_ERROR "lib${lib}.so not found in LLAMA_LIB_DIR=${LLAMA_LIB_DIR}")
    endif()
endforeach()

add_executable(lovelace-kld lovelace_kld.cpp)
target_include_directories(lovelace-kld PRIVATE
    "${LLAMA_ROOT}/include" "${LLAMA_ROOT}/ggml/include" "${LLAMA_ROOT}/vendor")
target_link_libraries(lovelace-kld PRIVATE
    "${LLAMA_LIB_DIR}/libllama.so" "${LLAMA_LIB_DIR}/libggml.so" "${LLAMA_LIB_DIR}/libggml-base.so")
target_compile_options(lovelace-kld PRIVATE -Wall -Wextra -O3)
if(KLD_NATIVE)
    target_compile_options(lovelace-kld PRIVATE -march=native)
endif()
# libggml-cuda.so needs libcuda.so.1 at link time: pass -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,<stub dir>.
# No rpath: the llama.cpp libs carry RUNPATH=/src/build-sm89/bin from their own build container, which
# overrides any rpath of ours for their dependencies, so run with LD_LIBRARY_PATH=${LLAMA_LIB_DIR}.
