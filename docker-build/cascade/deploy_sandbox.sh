#!/bin/bash
# Deploy SandboxFusion code execution sandbox
# Used by Stage 3 (code RL) for LiveCodeBench test execution
#
# Usage:
#   bash scripts/deploy_sandbox.sh              # default port 8080
#   SANDBOX_PORT=9090 bash scripts/deploy_sandbox.sh  # custom port
#
# After deployment, configure verl:
#   reward.sandbox_fusion.url=http://localhost:8080/run_code
#
# API docs: http://localhost:8080/docs
# Playground: http://localhost:8080/playground/

set -euo pipefail

SANDBOX_PORT="${SANDBOX_PORT:-8080}"
CONTAINER_NAME="${CONTAINER_NAME:-sandbox-fusion}"
IMAGE="${SANDBOX_IMAGE:-volcengine/sandbox-fusion:server-20250609}"
MIRROR_IMAGE="vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609"

echo "=== SandboxFusion Deployment ==="
echo "  Port:      $SANDBOX_PORT"
echo "  Container: $CONTAINER_NAME"
echo "  Image:     $IMAGE"
echo ""

# Stop existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing $CONTAINER_NAME container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# Try primary image, fallback to China mirror
echo "Pulling image..."
if ! docker pull "$IMAGE" 2>/dev/null; then
    echo "Primary image unavailable, trying China mirror..."
    IMAGE="$MIRROR_IMAGE"
    docker pull "$IMAGE"
fi

# Run container
echo "Starting SandboxFusion..."
docker run -d \
    --name "$CONTAINER_NAME" \
    --privileged \
    --restart unless-stopped \
    -p "${SANDBOX_PORT}:8080" \
    "$IMAGE" \
    make run-online

echo ""
echo "Waiting for service to start..."
sleep 5

# Health check
echo "Health check..."
for i in $(seq 1 10); do
    if curl -s "http://localhost:${SANDBOX_PORT}/docs" > /dev/null 2>&1; then
        echo "  ✅ SandboxFusion is running!"
        break
    fi
    if [ "$i" -eq 10 ]; then
        echo "  ❌ Service not responding after 50s. Check logs:"
        echo "     docker logs $CONTAINER_NAME"
        exit 1
    fi
    echo "  Waiting... ($((i*5))s)"
    sleep 5
done

# Test execution
echo ""
echo "Testing code execution..."
RESULT=$(curl -s -X POST "http://localhost:${SANDBOX_PORT}/run_code" \
    -H "Content-Type: application/json" \
    -d '{
        "code": "print(2 + 2)",
        "language": "python",
        "compile_timeout": 10,
        "run_timeout": 10,
        "memory_limit_MB": 1024
    }')

if echo "$RESULT" | grep -q '"4"'; then
    echo "  ✅ Code execution works! (2+2=4)"
else
    echo "  ⚠️  Unexpected result: $RESULT"
fi

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "  Playground:  http://localhost:${SANDBOX_PORT}/playground/"
echo "  API Docs:    http://localhost:${SANDBOX_PORT}/docs"
echo "  Run Code:    http://localhost:${SANDBOX_PORT}/run_code"
echo ""
echo "  Configure verl with:"
echo "    reward.sandbox_fusion.url=http://localhost:${SANDBOX_PORT}/run_code"
echo "    reward.sandbox_fusion.max_concurrent=64"
echo "    reward.sandbox_fusion.memory_limit_mb=1024"
echo ""
echo "  Management:"
echo "    docker logs $CONTAINER_NAME        # view logs"
echo "    docker restart $CONTAINER_NAME     # restart"
echo "    docker stop $CONTAINER_NAME        # stop"
