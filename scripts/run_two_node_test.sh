#!/bin/bash
# 双节点GPU集群分布式测试运行脚本

echo "=========================================="
echo "双节点GPU集群分布式测试"
echo "=========================================="

# 配置节点信息（使用私网IP进行集群通信）
HEAD_NODE_PUBLIC_IP="8.133.168.240"
HEAD_NODE_PRIVATE_IP="172.31.84.221"
WORKER_NODE_PUBLIC_IP="47.116.136.33"
WORKER_NODE_PRIVATE_IP="172.31.84.222"

# 在head节点启动Ray（使用私网IP作为集群地址）
echo "在Head节点 ($HEAD_NODE_PUBLIC_IP) 启动Ray..."
ssh -i /home/developer/private.pem root@$HEAD_NODE_PUBLIC_IP "cd /root && source ~/yubing/venv/bin/activate && ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-gpus=1 --node-ip-address=$HEAD_NODE_PRIVATE_IP"

# 等待head节点启动
sleep 5

# 在worker节点连接到head节点（使用私网IP连接）
echo "在Worker节点 ($WORKER_NODE_PUBLIC_IP) 连接到Ray集群..."
ssh -i /home/developer/private.pem root@$WORKER_NODE_PUBLIC_IP "cd /root && source ~/yubing/venv/bin/activate && ray start --address=$HEAD_NODE_PRIVATE_IP:6379 --num-gpus=1 --node-ip-address=$WORKER_NODE_PRIVATE_IP"

# 等待worker节点连接
sleep 5

# 检查集群状态
echo "Ray集群状态:"
ssh -i /home/developer/private.pem root@$HEAD_NODE_PUBLIC_IP "source ~/yubing/venv/bin/activate && ray status"

echo ""
echo "开始双节点分布式测试..."
ssh -i /home/developer/private.pem root@$HEAD_NODE_PUBLIC_IP "cd /root && source ~/yubing/venv/bin/activate && python3 two_node_distributed_test.py"

echo ""
echo "测试完成！"
echo "访问Ray Dashboard: http://$HEAD_NODE_PUBLIC_IP:8265"
