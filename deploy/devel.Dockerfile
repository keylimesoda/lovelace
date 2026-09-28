FROM nvidia/cuda:13.1.2-devel-ubuntu24.04
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends cmake ninja-build git ccache \
      libcurl4-openssl-dev libssl-dev python3 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /opt/cuda-stubs && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /opt/cuda-stubs/libcuda.so.1
