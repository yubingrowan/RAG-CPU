#!/bin/bash
# Progressive GPU Cluster Deployment Script
# Phase 1: Single Node -> Phase 2: Multi-Node Cluster

set -e

echo "=== Progressive GPU Cluster Deployment ==="
echo "Phase 1: Single GPU Node Deployment"
echo "Phase 2: Multi-Node Cluster Expansion"
echo "========================================"

# Configuration
REGION="cn-hangzhou"
ZONE="cn-hangzhou-h"
INSTANCE_TYPE="ecs.gn7i-c8g1.2xlarge"  # 1x A100 (80GB)
IMAGE_ID="centos_8_3_x64_20G_alibase_20240430.vhd"
SECURITY_GROUP_ID="sg-xxxxxxxxx"
VSWITCH_ID="vsw-xxxxxxxxx"

# Phase 1: Deploy Single Node
deploy_single_node() {
    echo "=== Phase 1: Deploying Single GPU Node ==="
    
    # Create single instance
    echo "Creating GPU instance..."
    INSTANCE_ID=$(aliyun ecs CreateInstance \
        --RegionId $REGION \
        --ZoneId $ZONE \
        --ImageId $IMAGE_ID \
        --InstanceType $INSTANCE_TYPE \
        --SecurityGroupId $SECURITY_GROUP_ID \
        --VSwitchId $VSWITCH_ID \
        --InstanceName "ray-gpu-single" \
        --Description "Ray GPU Single Node - 1x A100" \
        --InternetMaxBandwidthOut 10 \
        --Tag.1.Key "Role" \
        --Tag.1.Value "RaySingle" \
        --Tag.2.Key "Project" \
        --Tag.2.Value "AI-Infra-RAG" \
        --Tag.3.Key "Phase" \
        --Tag.3.Value "Phase1" \
        | jq -r '.InstanceId')
    
    echo "Instance created: $INSTANCE_ID"
    
    # Wait for instance to start
    echo "Waiting for instance to start..."
    aliyun ecs WaitInstanceStatus --InstanceId $INSTANCE_ID --Status "Running"
    
    # Get instance IP
    INSTANCE_IP=$(aliyun ecs DescribeInstances \
        --InstanceIds '["'$INSTANCE_ID'"]' \
        | jq -r '.Instances.Instance[0].PublicIpAddress.IpAddress[0]')
    
    echo "Instance IP: $INSTANCE_IP"
    
    # Save instance info
    cat > single_node_info.json << EOF
{
    "instance_id": "$INSTANCE_ID",
    "instance_ip": "$INSTANCE_IP",
    "instance_type": "$INSTANCE_TYPE",
    "phase": "single_node",
    "created_at": "$(date)"
}
EOF
    
    echo "Single node deployment completed!"
    echo "Instance info saved to: single_node_info.json"
    
    # Generate setup script
    generate_single_node_setup "$INSTANCE_IP"
    
    return $INSTANCE_ID
}

# Phase 2: Expand to Multi-Node Cluster
expand_to_cluster() {
    echo "=== Phase 2: Expanding to Multi-Node Cluster ==="
    
    # Read single node info
    if [ ! -f "single_node_info.json" ]; then
        echo "Error: single_node_info.json not found. Please run Phase 1 first."
        exit 1
    fi
    
    HEAD_IP=$(cat single_node_info.json | jq -r '.instance_ip')
    HEAD_ID=$(cat single_node_info.json | jq -r '.instance_id')
    
    echo "Head node IP: $HEAD_IP"
    echo "Head node ID: $HEAD_ID"
    
    # Create worker node
    echo "Creating worker node..."
    WORKER_ID=$(aliyun ecs CreateInstance \
        --RegionId $REGION \
        --ZoneId $ZONE \
        --ImageId $IMAGE_ID \
        --InstanceType $INSTANCE_TYPE \
        --SecurityGroupId $SECURITY_GROUP_ID \
        --VSwitchId $VSWITCH_ID \
        --InstanceName "ray-gpu-worker" \
        --Description "Ray GPU Worker Node - 1x A100" \
        --InternetMaxBandwidthOut 10 \
        --Tag.1.Key "Role" \
        --Tag.1.Value "RayWorker" \
        --Tag.2.Key "Project" \
        --Tag.2.Value "AI-Infra-RAG" \
        --Tag.3.Key "Phase" \
        --Tag.3.Value "Phase2" \
        | jq -r '.InstanceId')
    
    echo "Worker node created: $WORKER_ID"
    
    # Wait for worker to start
    aliyun ecs WaitInstanceStatus --InstanceId $WORKER_ID --Status "Running"
    
    # Get worker IP
    WORKER_IP=$(aliyun ecs DescribeInstances \
        --InstanceIds '["'$WORKER_ID'"]' \
        | jq -r '.Instances.Instance[0].PublicIpAddress.IpAddress[0]')
    
    echo "Worker node IP: $WORKER_IP"
    
    # Save cluster info
    cat > cluster_info.json << EOF
{
    "head_node": {
        "instance_id": "$HEAD_ID",
        "instance_ip": "$HEAD_IP"
    },
    "worker_nodes": [{
        "instance_id": "$WORKER_ID",
        "instance_ip": "$WORKER_IP"
    }],
    "total_nodes": 2,
    "phase": "multi_node",
    "created_at": "$(date)"
}
EOF
    
    echo "Cluster expansion completed!"
    echo "Cluster info saved to: cluster_info.json"
    
    # Generate cluster setup script
    generate_cluster_setup "$HEAD_IP" "$WORKER_IP"
    
    return $WORKER_ID
}

# Generate single node setup script
generate_single_node_setup() {
    local INSTANCE_IP=$1
    
    cat > setup_single_node.sh << EOF
#!/bin/bash
# Single Node GPU Setup Script

echo "=== Setting up Single GPU Node ==="
echo "Instance IP: $INSTANCE_IP"

# Update system
sudo yum update -y

# Install dependencies
sudo yum install -y git python3 python3-pip docker htop

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# Install Python packages
pip3 install --upgrade pip
pip3 install ray[default] torch transformers accelerate
pip3 install vllm
pip3 install sentence-transformers
pip3 install qdrant-client
pip3 install redis

# Clone project
cd /root
git clone https://github.com/your-repo/ai-infra-rag.git
cd ai-infra-rag
pip3 install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY='sk-a8aa6b05ed4444a88017ff3bb3061388'
export CUDA_VISIBLE_DEVICES=0

# Start Ray Head Node
ray start --head \\
    --dashboard-host=0.0.0.0 \\
    --dashboard-port=8265 \\
    --num-cpus=32 \\
    --num-gpus=1 \\
    --object-store-memory=50000000000

# Start services
echo "Starting services..."
docker run -d --name redis -p 6379:6379 redis:7-alpine
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest

# Test GPU
echo "Testing GPU..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0)}')"

# Test Ray
echo "Testing Ray..."
python3 -c "import ray; ray.init(address='auto'); print(f'Ray cluster resources: {ray.cluster_resources()}')"

echo "=== Single Node Setup Complete ==="
echo "Dashboard: http://$INSTANCE_IP:8265"
echo "Redis: localhost:6379"
echo "Qdrant: localhost:6333"
EOF

    chmod +x setup_single_node.sh
    echo "Single node setup script generated: setup_single_node.sh"
}

# Generate cluster setup script
generate_cluster_setup() {
    local HEAD_IP=$1
    local WORKER_IP=$2
    
    cat > setup_cluster.sh << EOF
#!/bin/bash
# Multi-Node Cluster Setup Script

echo "=== Setting up Multi-Node GPU Cluster ==="
echo "Head Node: $HEAD_IP"
echo "Worker Node: $WORKER_IP"

# Head Node Setup
echo "Setting up Head Node..."
ssh root@$HEAD_IP << 'EOF'
# Install dependencies
sudo yum update -y
sudo yum install -y git python3 python3-pip docker htop
sudo systemctl start docker
sudo systemctl enable docker

# Install Python packages
pip3 install --upgrade pip
pip3 install ray[default] torch transformers accelerate
pip3 install vllm
pip3 install sentence-transformers
pip3 install qdrant-client
pip3 install redis

# Clone project
cd /root
git clone https://github.com/your-repo/ai-infra-rag.git
cd ai-infra-rag
pip3 install -r requirements.txt

# Start Ray Head Node
ray start --head \\
    --dashboard-host=0.0.0.0 \\
    --dashboard-port=8265 \\
    --num-cpus=32 \\
    --num-gpus=1 \\
    --object-store-memory=50000000000

# Start services
docker run -d --name redis -p 6379:6379 redis:7-alpine
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
EOF

# Worker Node Setup
echo "Setting up Worker Node..."
ssh root@$WORKER_IP << EOF
# Install dependencies
sudo yum update -y
sudo yum install -y git python3 python3-pip docker htop
sudo systemctl start docker
sudo systemctl enable docker

# Install Python packages
pip3 install --upgrade pip
pip3 install ray[default] torch transformers accelerate
pip3 install vllm
pip3 install sentence-transformers
pip3 install qdrant-client
pip3 install redis

# Clone project
cd /root
git clone https://github.com/your-repo/ai-infra-rag.git
cd ai-infra-rag
pip3 install -r requirements.txt

# Start Ray Worker Node
ray start --address=$HEAD_IP:6379 \\
    --num-cpus=32 \\
    --num-gpus=1 \\
    --object-store-memory=50000000000
EOF

echo "=== Cluster Setup Complete ==="
echo "Head Node Dashboard: http://$HEAD_IP:8265"
echo "Cluster Status: ray status"
EOF

    chmod +x setup_cluster.sh
    echo "Cluster setup script generated: setup_cluster.sh"
}

# Generate test scripts
generate_test_scripts() {
    echo "Generating test scripts..."
    
    # Single node test
    cat > test_single_node.py << EOF
#!/usr/bin/env python3
"""Single Node GPU Test"""

import ray
import torch
import time

def test_single_node():
    print("=== Single Node GPU Test ===")
    
    # Test GPU
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test Ray
    ray.init(address='auto')
    print(f"Ray cluster resources: {ray.cluster_resources()}")
    
    # Test inference
    @ray.remote(num_gpus=1)
    class GPUInference:
        def __init__(self):
            self.device = torch.device("cuda:0")
            print(f"Worker initialized on {self.device}")
        
        def generate(self, prompt):
            # Simulate inference
            time.sleep(1)
            return f"Generated response for: {prompt}"
    
    # Create worker
    worker = GPUInference.remote()
    
    # Test inference
    prompts = ["Hello", "How are you?", "What is AI?"]
    futures = [worker.generate.remote(p) for p in prompts]
    results = ray.get(futures)
    
    print("Inference results:")
    for prompt, result in zip(prompts, results):
        print(f"  {prompt} -> {result}")
    
    print("=== Single Node Test Complete ===")

if __name__ == "__main__":
    test_single_node()
EOF

    # Multi-node test
    cat > test_cluster.py << EOF
#!/usr/bin/env python3
"""Multi-Node Cluster Test"""

import ray
import torch
import time

def test_cluster():
    print("=== Multi-Node Cluster Test ===")
    
    # Test Ray cluster
    ray.init(address='auto')
    print(f"Ray cluster resources: {ray.cluster_resources()}")
    print(f"Ray nodes: {ray.nodes()}")
    
    # Test distributed inference
    @ray.remote(num_gpus=1)
    class DistributedInference:
        def __init__(self, worker_id):
            self.worker_id = worker_id
            self.device = torch.device("cuda:0")
            print(f"Worker {worker_id} initialized on {self.device}")
        
        def generate(self, prompt):
            # Simulate inference
            time.sleep(1)
            return f"Worker {self.worker_id}: {prompt}"
    
    # Create workers on different nodes
    workers = []
    for i in range(2):  # 2 workers for 2 nodes
        worker = DistributedInference.remote(i)
        workers.append(worker)
    
    # Test distributed inference
    prompts = [f"Task {i}" for i in range(10)]
    
    # Distribute tasks
    futures = []
    for i, prompt in enumerate(prompts):
        worker = workers[i % len(workers)]
        futures.append(worker.generate.remote(prompt))
    
    results = ray.get(futures)
    
    print("Distributed inference results:")
    for prompt, result in zip(prompts, results):
        print(f"  {prompt} -> {result}")
    
    print("=== Cluster Test Complete ===")

if __name__ == "__main__":
    test_cluster()
EOF

    chmod +x test_single_node.py test_cluster.py
    echo "Test scripts generated: test_single_node.py, test_cluster.py"
}

# Main execution
main() {
    case "$1" in
        "phase1")
            echo "Starting Phase 1: Single Node Deployment"
            deploy_single_node
            generate_test_scripts
            echo "Phase 1 completed! Next: bash $0 phase2"
            ;;
        "phase2")
            echo "Starting Phase 2: Multi-Node Cluster Expansion"
            expand_to_cluster
            echo "Phase 2 completed! Ready for distributed testing"
            ;;
        "test-single")
            echo "Testing single node..."
            if [ -f "single_node_info.json" ]; then
                HEAD_IP=$(cat single_node_info.json | jq -r '.instance_ip')
                scp setup_single_node.sh root@$HEAD_IP:/root/
                ssh root@$HEAD_IP "bash /root/setup_single_node.sh"
                scp test_single_node.py root@$HEAD_IP:/root/ai-infra-rag/
                ssh root@$HEAD_IP "cd /root/ai-infra-rag && python3 test_single_node.py"
            else
                echo "Error: single_node_info.json not found"
            fi
            ;;
        "test-cluster")
            echo "Testing cluster..."
            if [ -f "cluster_info.json" ]; then
                HEAD_IP=$(cat cluster_info.json | jq -r '.head_node.instance_ip')
                scp setup_cluster.sh root@$HEAD_IP:/root/
                ssh root@$HEAD_IP "bash /root/setup_cluster.sh"
                scp test_cluster.py root@$HEAD_IP:/root/ai-infra-rag/
                ssh root@$HEAD_IP "cd /root/ai-infra-rag && python3 test_cluster.py"
            else
                echo "Error: cluster_info.json not found"
            fi
            ;;
        "cleanup")
            echo "Cleaning up instances..."
            if [ -f "single_node_info.json" ]; then
                HEAD_ID=$(cat single_node_info.json | jq -r '.instance_id')
                echo "Terminating head node: $HEAD_ID"
                aliyun ecs DeleteInstance --InstanceId $HEAD_ID --Force true
            fi
            if [ -f "cluster_info.json" ]; then
                WORKER_ID=$(cat cluster_info.json | jq -r '.worker_nodes[0].instance_id')
                echo "Terminating worker node: $WORKER_ID"
                aliyun ecs DeleteInstance --InstanceId $WORKER_ID --Force true
            fi
            rm -f *.json *.sh *.py
            echo "Cleanup completed"
            ;;
        *)
            echo "Usage: $0 {phase1|phase2|test-single|test-cluster|cleanup}"
            echo "  phase1     - Deploy single GPU node"
            echo "  phase2     - Expand to multi-node cluster"
            echo "  test-single- Test single node setup"
            echo "  test-cluster- Test multi-node cluster"
            echo "  cleanup    - Clean up all instances"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
