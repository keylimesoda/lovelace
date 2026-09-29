FROM lovelace-engine-runtime:cuda13.1

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends python3 python3-httpx \
    && rm -rf /var/lib/apt/lists/*

COPY engine/build-sm89/bin/ /opt/lovelace/bin/
COPY apps/web/dist/ /opt/lovelace/web/
COPY bench/replay/capture_proxy.py bench/replay/corpusfmt.py /opt/lovelace/proxy/
COPY deploy/draft-vocab/qwen3.8-27b.txt /opt/lovelace/draft-vocab/

ENV LD_LIBRARY_PATH=/opt/lovelace/bin
WORKDIR /opt/lovelace
