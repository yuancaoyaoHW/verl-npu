FROM quay.io/ascend/vllm-ascend:v0.18.0

# Copy verl source code (without checkpoints)
COPY verl-src/ /opt/verl/

# Copy patches
COPY patches/ /opt/patches/
COPY apply-patches.sh /opt/

# Copy training scripts
COPY run_opd.sh /root/
COPY cascade/ /root/cascade/

# Install verl, apply patches, and install reward dependencies
RUN cd /opt/verl && pip install -e . --no-deps && \
    bash /opt/apply-patches.sh && \
    pip install "math-verify[antlr4_11_0]==0.7.0" sympy requests \
    tensordict omegaconf hydra-core codetiming datasets torchdata peft

# Set working directory
WORKDIR /opt/verl

# Default command: start Ray and keep container running
CMD ["bash", "-c", "ray start --head --port=6379 --disable-usage-stats && sleep infinity"]
