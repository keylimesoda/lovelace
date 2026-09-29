# Devel image plus libdw, which the Nsight Systems QdstrmImporter bundled with Nsight Compute needs to turn a capture
# into an .nsys-rep (bench/step_profile.sh).
FROM lovelace-engine-devel:cuda13.1
RUN apt-get update && apt-get install -y --no-install-recommends libdw1 && rm -rf /var/lib/apt/lists/*
ENV NSYS_IMPORTER=/opt/nvidia/nsight-compute/2025.4.1/host/linux-desktop-glibc_2_11_3-x64/QdstrmImporter
