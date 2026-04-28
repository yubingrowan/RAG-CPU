#!/bin/bash
# 明天GPU集群测试运行脚本

echo "=========================================="
echo "明天GPU集群综合测试"
echo "=========================================="

# 检查Ray是否运行
if ! ray status &> /dev/null; then
    echo "启动Ray集群..."
    ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

echo "Ray集群状态:"
ray status

echo ""
echo "开始综合测试..."
python3 /home/developer/ai-infra-rag/scripts/tomorrow_comprehensive_test.py

echo ""
echo "测试完成！"
echo "访问Ray Dashboard: http://localhost:8265"
