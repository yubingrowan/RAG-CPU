#!/bin/bash
# Start llama.cpp server with Qwen2.5-7B-Instruct model

MODEL_PATH="/home/developer/ai-infra-rag/models/qwen2.5-7b-instruct-q4_k_m.gguf"
LLAMA_SERVER="/home/developer/llama.cpp/build/bin/llama-server"
PORT=8080
HOST="0.0.0.0"
THREADS=16

echo "Starting llama.cpp server..."
echo "Model: $MODEL_PATH"
echo "Listening on: $HOST:$PORT"
echo "Threads: $THREADS"

LOG_FILE="/home/developer/ai-infra-rag/logs/llama_server.log"
mkdir -p "$(dirname "$LOG_FILE")"

$LLAMA_SERVER -m "$MODEL_PATH" --port "$PORT" --host "$HOST" --threads "$THREADS" --verbose > "$LOG_FILE" 2>&1 &
