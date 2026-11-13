# landscape-md:1.1 — pinned environment for the odour-ensemble analyses
FROM mambaorg/micromamba:1.5.8

LABEL org.opencontainers.image.title="landscape-md" \
      org.opencontainers.image.version="1.1" \
      org.opencontainers.image.description="Conformer generation, GFN2-xTB thermochemistry and neural olfactory benchmarks"

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY environment.lock.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

ENV PATH=/opt/conda/bin:$PATH
ENV XTBPATH=/opt/conda/share/xtb

WORKDIR /workspace
CMD ["/bin/bash"]
