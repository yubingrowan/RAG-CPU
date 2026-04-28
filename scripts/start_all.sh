#!/bin/bash
# Manage RAG infrastructure services: start, stop, status

set -e

PROJECT_DIR="/home/developer/ai-infra-rag"
LLAMA_SERVER="/home/developer/llama.cpp/build/bin/llama-server"
MODEL_PATH="$PROJECT_DIR/models/qwen2.5-7b-instruct-q4_k_m.gguf"
PORT=8080
HOST="0.0.0.0"
API_PORT=8000
UI_PORT=8501
DOCKER_COMPOSE_CMD=""

is_port_in_use() {
    if command -v ss >/dev/null 2>&1; then
        ss -ltn "( sport = :$1 )" 2>/dev/null | grep -q LISTEN
    else
        netstat -ltn 2>/dev/null | grep -q ":$1 "
    fi
}

kill_port() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltnp "sport = :$port" 2>/dev/null | awk 'NR>1 {print $NF}' | sed -n 's#pid=\([0-9]*\),.*#\1#p' | sort -u | xargs -r kill -TERM
    else
        netstat -ltnp 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d'/' -f1 | sort -u | xargs -r kill -TERM
    fi
}

check_api_health() {
    curl -fsS "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1
}

check_ui_health() {
    curl -fsS "http://127.0.0.1:$UI_PORT" >/dev/null 2>&1
}

print_header() {
    echo "================================"
    echo "$1"
    echo "================================"
}

verify_commands() {
    for cmd in docker python3 streamlit curl; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            echo "ERROR: $cmd not found in PATH"
            exit 1
        fi
    done

    if command -v docker-compose >/dev/null 2>&1; then
        DOCKER_COMPOSE_CMD="docker-compose"
    elif docker compose version >/dev/null 2>&1; then
        DOCKER_COMPOSE_CMD="docker compose"
    else
        echo "ERROR: docker-compose not found in PATH and 'docker compose' is unavailable"
        exit 1
    fi
}

start_docker() {
    echo "1. Starting Docker services..."
    cd "$PROJECT_DIR" || exit 1
    $DOCKER_COMPOSE_CMD up -d
    echo "Waiting for services to start..."
    sleep 5
    echo ""
    echo "2. Checking Docker services..."
    $DOCKER_COMPOSE_CMD ps
}

create_kafka_topics() {
    echo ""
    echo "3. Creating Kafka topics..."
    if docker ps --format '{{.Names}}' | grep -q '^ai-infra-rag-kafka-1$'; then
        docker exec ai-infra-rag-kafka-1 /opt/kafka/bin/kafka-topics.sh --create --topic user_queries --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 --if-not-exists 2>/dev/null
        docker exec ai-infra-rag-kafka-1 /opt/kafka/bin/kafka-topics.sh --create --topic index_tasks --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 --if-not-exists 2>/dev/null
        echo "Kafka topics ready"
    else
        echo "WARNING: Kafka container ai-infra-rag-kafka-1 not found. Skipping topic creation."
    fi
}

start_llama() {
    echo ""
    echo "4. Starting llama.cpp server..."
    echo "Model: $MODEL_PATH"
    echo "Listening on: $HOST:$PORT"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "ERROR: Model not found at $MODEL_PATH"
        exit 1
    fi

    if is_port_in_use "$PORT"; then
        echo "llama.cpp appears to already be listening on port $PORT. Skipping llama.cpp startup."
    else
        # Optimized parameters for i7-1260P CPU
        $LLAMA_SERVER \
    -m "$MODEL_PATH" \
    --port "$PORT" \
    --host "$HOST" \
    -t 12 \
    -c 1024 \
    --n-predict 128 \
    -b 64 \
    -ub 64 \
    -np 4 \
    --flash-attn auto \
    --no-mmap \
    -ngl 0 \
    --override-kv qwen2.context_length=int:1024 &
        LLAMA_PID=$!
        echo "llama.cpp server started with PID: $LLAMA_PID"
        echo "Waiting for llama.cpp server to be ready..."
        sleep 10
        if ps -p "$LLAMA_PID" > /dev/null; then
            echo "✓ llama.cpp server is running"
        else
            echo "✗ llama.cpp server failed to start"
            exit 1
        fi
    fi
}

start_api() {
    if is_port_in_use "$API_PORT"; then
        if check_api_health; then
            echo "API server already running on port $API_PORT. Skipping startup."
            return
        fi
        echo "WARNING: Port $API_PORT is in use but API health check failed. Please free the port or stop the conflicting process."
        return
    fi

    echo ""
    echo "5. Starting API server..."
    
    
    # 使用日期命名日志文件，避免覆盖
    LOG_FILE="/home/developer/ai-infra-rag/logs/api_server.log"
    # 或者固定文件名，但追加模式： LOG_FILE="$LOG_DIR/api_server.log"
    
    cd "$PROJECT_DIR" || exit 1
    # 将 stdout 和 stderr 都重定向到日志文件，并在后台运行
    nohup python3 -u src/api/api_server.py >> "$LOG_FILE" 2>&1 &
    API_PID=$!
    
    sleep 5
    if ps -p "$API_PID" > /dev/null; then
        echo "✓ API server is running with PID: $API_PID"
        echo "✓ Logs are being written to: $LOG_FILE"
    else
        echo "✗ API server failed to start. Check log file: $LOG_FILE"
        exit 1
    fi
}

start_ui() {
    if is_port_in_use "$UI_PORT"; then
        if check_ui_health; then
            echo "Streamlit UI already running on port $UI_PORT. Skipping startup."
            return
        fi
        echo "WARNING: Port $UI_PORT is in use but Streamlit health check failed. Please free the port or stop the conflicting process."
        return
    fi

    echo ""
    echo "6. Starting Streamlit UI..."
    cd "$PROJECT_DIR" || exit 1
    python3 -m streamlit run ui/chat_ui.py --server.port "$UI_PORT" --server.headless true &
    UI_PID=$!
    sleep 5
    if ps -p "$UI_PID" > /dev/null; then
        echo "✓ Streamlit UI is running with PID: $UI_PID"
    else
        echo "✗ Streamlit UI failed to start"
        exit 1
    fi
}

status_docker() {
    cd "$PROJECT_DIR" || exit 1
    if [ -n "$DOCKER_COMPOSE_CMD" ]; then
        $DOCKER_COMPOSE_CMD ps
    elif command -v docker-compose >/dev/null 2>&1; then
        docker-compose ps
    elif docker compose version >/dev/null 2>&1; then
        docker compose ps
    else
        echo "docker compose not available"
    fi
}

status_service() {
    local name="$1"
    local port="$2"
    if is_port_in_use "$port"; then
        echo "$name is listening on port $port"
    else
        echo "$name is not running on port $port"
    fi
}

stop() {
    echo "Stopping services..."
    if command -v docker-compose >/dev/null 2>&1; then
        cd "$PROJECT_DIR" || exit 1
        docker-compose down
    fi
    kill_port "$PORT" || true
    kill_port "$API_PORT" || true
    kill_port "$UI_PORT" || true
    echo "Stopped services and freed ports if processes were running."
}

status() {
    echo "Service status:"
    status_service "llama.cpp" "$PORT"
    status_service "API server" "$API_PORT"
    status_service "Streamlit UI" "$UI_PORT"
    echo ""
    echo "Docker status:"
    status_docker
}

usage() {
    cat <<EOF
Usage: $0 {start|stop|status}

Commands:
  start    Start Docker services, llama.cpp, API server, and Streamlit UI
  stop     Stop Docker compose services and any local servers on ports 8080, 8000, 8501
  status   Show the current process/port status for llama.cpp, API server, Streamlit UI, and Docker
EOF
}

case "$1" in
    start|"")
        print_header "Starting RAG Infrastructure"
        verify_commands
        start_docker
        create_kafka_topics
        start_llama
        start_api
        start_ui
        echo ""
        print_header "All services started successfully!"
        echo "Services:"
        echo "  - Redis: localhost:6379"
        echo "  - Kafka: localhost:9092"
        echo "  - Qdrant: localhost:6333"
        echo "  - llama.cpp: http://localhost:$PORT"
        echo "  - API server: http://localhost:$API_PORT"
        echo "  - Streamlit UI: http://localhost:$UI_PORT"
        ;;
    stop)
        print_header "Stopping RAG Infrastructure"
        stop
        ;;
    status)
        print_header "RAG Infrastructure Status"
        status
        ;;
    *)
        usage
        exit 1
        ;;
esac
