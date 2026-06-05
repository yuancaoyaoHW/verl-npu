FROM quay.io/ascend/vllm-ascend:v0.18.0

# Copy verl source code (without checkpoints)
COPY verl-src/ /opt/verl/

# Copy patches
COPY patches/ /opt/patches/
COPY apply-patches.sh /opt/

# Copy training script
COPY run_opd.sh /root/

# Install verl and apply patches
RUN cd /opt/verl && pip install -e . --no-deps && \
    bash /opt/apply-patches.sh

# Set working directory
WORKDIR /opt/verl

# Default command: start Ray and keep container running
CMD ["bash", "-c", "ray start --head --port=6379 --disable-usage-stats && sleep infinity"]
