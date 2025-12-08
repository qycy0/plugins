#!/bin/bash
# Disable v1 engine to avoid deep_gemm compatibility issues

# 设置日志级别为 DEBUG
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_LOG_LEVEL=DEBUG

WORK_DIR="$(dirname "$0")"
MODEL_DIR=...
PORT=...

vllm serve MODEL_DIR \
    --served-model-name mirothinker \
    --port PORT \
    --trust-remote-code \
    --chat-template {WORK_DIR}/chat_template.jinja \
    --tool-parser-plugin {WORK_DIR}/MirothinkerToolParser_vllm_0.11.0.py \
    --tool-call-parser mirothinker \
    --enable-auto-tool-choice \
    --enable-log-outputs \
    --max-model-len 64k \
    --enable-log-requests