#!/bin/bash
OUTPUT=${1:-/tmp/aicore_trace.log}
INTERVAL=${2:-2}
npu-smi info watch -d ${INTERVAL} > ${OUTPUT} &
echo "AICore monitoring started: PID=$!, output=${OUTPUT}, interval=${INTERVAL}s"
