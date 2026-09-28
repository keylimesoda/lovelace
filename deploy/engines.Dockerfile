FROM nvidia/cuda:13.1.2-runtime-ubuntu24.04
RUN mkdir -p /opt/cuda-stubs && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /opt/cuda-stubs/libcuda.so.1
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends libcurl4 libgomp1 libssl3 ca-certificates && rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH=/e
WORKDIR /e
