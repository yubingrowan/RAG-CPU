#!/bin/bash
# 阿里云GPU集群部署脚本
# 推荐方案：两个1/2 GPU分片 + Ray集群

set -e

echo "🚀 阿里云GPU集群部署脚本"
echo "================================"

# 配置参数
REGION="cn-hangzhou"
ZONE="cn-hangzhou-h"
INSTANCE_TYPE="ecs.gn6i-c4g1.large"  # 1/2 A100
IMAGE_ID="centos_8_3_x64_20G_alibase_20240430.vhd"
SECURITY_GROUP_ID="sg-xxxxxxxxx"
VSWITCH_ID="vsw-xxxxxxxxx"

# 创建Ray Head节点
echo "📦 创建Ray Head节点..."
HEAD_INSTANCE_ID=$(aliyun ecs CreateInstance \
    --RegionId $REGION \
    --ZoneId $ZONE \
    --ImageId $IMAGE_ID \
    --InstanceType $INSTANCE_TYPE \
    --SecurityGroupId $SECURITY_GROUP_ID \
    --VSwitchId $VSWITCH_ID \
    --InstanceName "ray-gpu-head" \
    --Description "Ray Head Node with 1/2 GPU" \
    --InternetMaxBandwidthOut 10 \
    --Tag.1.Key "Role" \
    --Tag.1.Value "RayHead" \
    --Tag.2.Key "Project" \
    --Tag.2.Value "AI-Infra-RAG" \
    | jq -r '.InstanceId')

echo "✅ Head节点创建成功: $HEAD_INSTANCE_ID"

# 创建Ray Worker节点
echo "📦 创建Ray Worker节点..."
WORKER_INSTANCE_ID=$(aliyun ecs CreateInstance \
    --RegionId $REGION \
    --ZoneId $ZONE \
    --ImageId $IMAGE_ID \
    --InstanceType $INSTANCE_TYPE \
    --SecurityGroupId $SECURITY_GROUP_ID \
    --VSwitchId $VSWITCH_ID \
    --InstanceName "ray-gpu-worker" \
    --Description "Ray Worker Node with 1/2 GPU" \
    --InternetMaxBandwidthOut 10 \
    --Tag.1.Key "Role" \
    --Tag.1.Value "RayWorker" \
    --Tag.2.Key "Project" \
    --Tag.2.Value "AI-Infra-RAG" \
    | jq -r '.InstanceId')

echo "✅ Worker节点创建成功: $WORKER_INSTANCE_ID"

# 等待实例启动
echo "⏳ 等待实例启动..."
for instance in $HEAD_INSTANCE_ID $WORKER_INSTANCE_ID; do
    echo "等待实例 $instance 启动..."
    aliyun ecs WaitInstanceStatus --InstanceId $instance --Status "Running"
    echo "✅ 实例 $instance 已启动"
done

# 获取实例IP地址
echo "🌐 获取实例IP地址..."
HEAD_IP=$(aliyun ecs DescribeInstances \
    --InstanceIds '["'$HEAD_INSTANCE_ID'"]' \
    | jq -r '.Instances.Instance[0].PublicIpAddress.IpAddress[0]')

WORKER_IP=$(aliyun ecs DescribeInstances \
    --InstanceIds '["'$WORKER_INSTANCE_ID'"]' \
    | jq -r '.Instances.Instance[0].PublicIpAddress.IpAddress[0]')

echo "📍 Head节点IP: $HEAD_IP"
echo "📍 Worker节点IP: $WORKER_IP"

# 生成部署配置
echo "📄 生成Ray集群配置..."
cat > ray_cluster_config.yaml << EOF
cluster_name: "alibaba-gpu-cluster"
min_workers: 1
max_workers: 2
initial_workers: 2
autoscaling_mode: "default"

provider:
  type: "aws"
  region: $REGION
  availability_zone: $ZONE

head_node:
  InstanceType: $INSTANCE_TYPE
  ImageId: $IMAGE_ID
  InstanceName: "ray-gpu-head"
  # GPU配置
  BlockDeviceMappings:
    - DeviceName: /dev/xvda
      Ebs:
        VolumeSize: 100
        VolumeType: gp3

worker_nodes:
  InstanceType: $INSTANCE_TYPE
  ImageId: $IMAGE_ID
  InstanceName: "ray-gpu-worker"
  BlockDeviceMappings:
    - DeviceName: /dev/xvda
      Ebs:
        VolumeSize: 100
        VolumeType: gp3

# Ray配置
ray_init:
  dashboard_host: "0.0.0.0"
  dashboard_port: 8265
  num_cpus: 16
  num_gpus: 0.5  # 每个节点使用半个GPU

# 初始化脚本
setup_commands:
  - sudo yum update -y
  - sudo yum install -y git python3 python3-pip docker
  - sudo systemctl start docker
  - sudo systemctl enable docker
  - pip3 install ray[default] torch transformers accelerate
  - pip3 install vllm
  - git clone https://github.com/your-repo/ai-infra-rag.git
  - cd ai-infra-rag
  - pip3 install -r requirements.txt

# Head节点额外配置
head_setup_commands:
  - ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-cpus=16 --num-gpus=0.5

# Worker节点配置
worker_setup_commands:
  - ray start --address=$HEAD_IP:6379 --num-cpus=16 --num-gpus=0.5
EOF

echo "✅ Ray集群配置生成完成: ray_cluster_config.yaml"

# 生成连接脚本
cat > connect_to_cluster.sh << EOF
#!/bin/bash
# 连接到Ray集群脚本

echo "🔗 连接到Ray集群..."
echo "Head节点: $HEAD_IP:8265"
echo "Worker节点: $WORKER_IP"

# SSH连接到Head节点
echo "🌐 连接到Head节点..."
ssh root@$HEAD_IP

# 在Head节点上启动Ray Dashboard
# ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-cpus=16 --num-gpus=0.5

# 访问Dashboard: http://$HEAD_IP:8265
EOF

chmod +x connect_to_cluster.sh

echo "🎉 GPU集群部署脚本生成完成！"
echo "================================"
echo "📋 部署信息:"
echo "  Head节点: $HEAD_IP"
echo "  Worker节点: $WORKER_IP"
echo "  Dashboard: http://$HEAD_IP:8265"
echo "  配置文件: ray_cluster_config.yaml"
echo "  连接脚本: connect_to_cluster.sh"
echo ""
echo "🚀 下一步操作:"
echo "  1. 执行: bash connect_to_cluster.sh"
echo "  2. 在Head节点启动Ray集群"
echo "  3. 部署AI基础设施RAG系统"
echo "  4. 测试分布式推理"
