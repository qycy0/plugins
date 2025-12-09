#!/bin/bash

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