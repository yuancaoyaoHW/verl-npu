#!/usr/bin/env bash
# Launch the verl-vllm container for Haidass OPD training
# Image: quay.io/ascend/vllm-ascend:v0.18.0
# Contains: vLLM 0.18.0 + vllm-ascend 0.18.0 + torch 2.9.0 + torch_npu 2.9.0 + CANN 8.5.1

set -euo pipefail

docker run -d --name verl-vllm \
  --network host --privileged --ipc=host --shm-size 32g \
  --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /home:/home -v /mnt:/mnt -v /tmp:/tmp \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e NPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e NCCL_P2P_DISABLE=1 \
  -e HCCL_BUFFSIZE=300 \
  -e NCCL_P2P_DISABLE=1 \
  -e HCCL_BUFFSIZE=300 \
  -e NCCL_P2P_DISABLE=1 \
  -e HCCL_BUFFSIZE=300 \
  -e NCCL_P2P_DISABLE=1 \
  -e HCCL_BUFFSIZE=300 \
  -e NCCL_P2P_DISABLE=1 \
  -e HCCL_BUFFSIZE=300 \
  -e NCCL_P2P_DISABLE=1 \
  -e HCCL_BUFFSIZE=300 \
  -e NCCL_P2P_DISABLE=1 \
  -e HCCL_BUFFSIZE=300 \
  -e NCCL_P2P_DISABLE=1 \
  -e HCCL_BUFFSIZE=300 \
  -e HCCL_P2P_DISABLE=1 \
  quay.io/ascend/vllm-ascend:v0.18.0 \
  sleep infinity

echo "Container verl-vllm started. Verify with: docker exec verl-vllm npu-smi info"
