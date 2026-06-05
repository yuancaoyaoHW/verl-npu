#!/bin/bash
# 在目标服务器上部署 Haidass OPD 训练环境
# 前提：已有 quay.io/ascend/vllm-ascend:v0.18.0 镜像

set -e

echo "=== Haidass OPD NPU 部署 ==="

# 1. 解压部署包
echo "[1/5] 解压部署包..."
tar xzf haidass-opd-deploy.tar.gz

# 2. 启动容器
echo "[2/5] 启动容器..."
docker run -d --name verl-vllm \
  --network host --privileged --shm-size 32g \
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
  quay.io/ascend/vllm-ascend:v0.18.0 \
  sleep infinity

# 3. 复制代码到容器
echo "[3/5] 复制 verl 源码和 patches..."
docker cp verl-src verl-vllm:/opt/verl
docker cp patches verl-vllm:/opt/patches
docker cp apply-patches.sh verl-vllm:/opt/
docker cp run_opd.sh verl-vllm:/root/

# 4. 安装 verl 并应用 patches
echo "[4/5] 安装 verl 并应用 patches..."
docker exec verl-vllm bash -c "cd /opt/verl && pip install -e . --no-deps && bash /opt/apply-patches.sh"

# 5. 启动 Ray
echo "[5/5] 启动 Ray..."
docker exec verl-vllm bash -c "ray start --head --port=6379 --disable-usage-stats"

echo ""
echo "=== 部署完成 ==="
echo "启动训练: docker exec -d verl-vllm bash -c 'nohup bash /root/run_opd.sh > /home/admin/train_opd.log 2>&1 &'"
echo "查看日志: docker exec verl-vllm tail -f /home/admin/train_opd.log"
echo "NPU 状态: docker exec verl-vllm npu-smi info"
