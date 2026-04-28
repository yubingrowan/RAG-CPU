# GPU集群重新部署完整指南

## 明天重新部署步骤

### 1. 阿里云ECS实例创建
```bash
# 选择相同配置
# 地域：华东2（上海）
# 实例规格：ecs.gn7i-c8g1.2xlarge
# 镜像：Ubuntu 22.04 with GPU Driver (NVIDIA Driver + CUDA 12.8)
# 存储：100GB ESSD
# 安全组：开放SSH(22)端口
```

### 2. 系统环境验证
```bash
# 连接到服务器
ssh root@your_ip

# 检查GPU状态
nvidia-smi

# 检查CUDA版本
nvcc --version

# 检查系统信息
uname -a
```

### 3. Python环境配置
```bash
# 安装Python和基础工具
apt update
apt install -y python3 python3-pip python3-venv git curl wget

# 创建虚拟环境
python3 -m venv /opt/venv
source /opt/venv/bin/activate

# 升级pip
pip install --upgrade pip
```

### 4. 核心依赖安装
```bash
# 安装Ray和PyTorch
pip install ray[default] ray[runtime] torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install fastapi uvicorn transformers accelerate bitsandbytes modelscope

# 安装vLLM（明天测试用）
pip install vllm
```

### 5. Ray集群启动
```bash
# 启动Ray集群
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265

# 验证Ray状态
ray status

# 检查集群资源
python -c "import ray; ray.init(); print(ray.cluster_resources())"
```

### 6. 基础测试
```bash
# 创建GPU测试脚本
cat > /opt/ray_test.py <<'EOF'
import ray
import torch

@ray.remote(num_gpus=1)
def gpu_task():
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}"

if __name__ == "__main__":
    ray.init()
    result = ray.get(gpu_task.remote())
    print(result)
    print("集群资源:", ray.cluster_resources())
    ray.shutdown()
EOF

python /opt/ray_test.py
```

### 7. vLLM PagedAttention测试（明天重点）
```bash
# 创建vLLM测试脚本
cat > /opt/vllm_paged_test.py <<'EOF'
from vllm import LLM, SamplingParams
import time

print("=== vLLM PagedAttention测试 ===")

# 初始化LLM
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=50
)

# 测试提示
prompts = [
    "什么是人工智能？",
    "机器学习的原理是什么？",
    "深度学习有什么应用？"
]

print(f"开始推理 {len(prompts)} 个提示...")

# 推理测试
start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
total_time = time.time() - start_time

print(f"\n=== 推理结果 ===")
for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"\n提示 {i+1}: {prompt}")
    print(f"回答: {generated_text}")

print(f"\n=== 性能统计 ===")
print(f"总时间: {total_time:.3f}秒")
print(f"平均时间: {total_time/len(prompts):.3f}秒/请求")
print(f"吞吐量: {len(prompts)/total_time:.1f} 请求/秒")

print(f"\n=== PagedAttention特性 ===")
print(f"✅ 自动内存管理")
print(f"✅ 连续批处理")
print(f"✅ KV缓存优化")
EOF

python /opt/vllm_paged_test.py
```

### 8. 从项目文件夹复制测试代码
```bash
# 如果本地有项目代码，可以上传到服务器
# 或者重新创建测试脚本

# 创建测试目录
mkdir -p /opt/tests
cd /opt/tests

# 这里可以复制之前保存的测试脚本
# 或者重新创建基础测试
```

### 9. Ray Dashboard访问
```bash
# 方法1：开放安全组8265端口，直接访问
# http://your_ip:8265

# 方法2：SSH隧道（推荐）
ssh -L 8265:127.0.0.1:8265 root@your_ip
# 然后访问 http://localhost:8265
```

### 10. 清理和成本控制
```bash
# 使用完毕后停止Ray
ray stop

# 停止ECS实例（在阿里云控制台操作）
# 释放实例如果不再需要
```

## 明天重点测试项目

### 1. vLLM PagedAttention
- 对比传统推理vs PagedAttention性能
- 观察内存使用情况
- 测试连续批处理效果

### 2. GPU性能优化
- 不同模型大小对比（0.5B vs 1.5B vs 3B）
- 半精度vs全精度
- 批量大小优化

### 3. Ray高级特性
- Ray Serve部署API服务
- 分布式任务调度优化
- 资源管理策略

## 故障排除

### 常见问题
1. **CUDA版本不匹配**: 使用预装GPU驱动的镜像
2. **内存不足**: 使用更小模型或减少批量大小
3. **网络问题**: 使用ModelScope镜像源
4. **Ray启动失败**: 检查端口占用
5. **vLLM安装失败**: 确保CUDA和PyTorch版本兼容

### 快速检查清单
- [ ] GPU驱动正常（nvidia-smi）
- [ ] CUDA版本正确（nvcc --version）
- [ ] Python虚拟环境激活
- [ ] Ray集群运行（ray status）
- [ ] 网络连接正常
- [ ] 模型下载源配置

## 关键命令速查

```bash
# Ray操作
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265
ray status
ray stop

# GPU检查
nvidia-smi
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"

# Python环境
source /opt/venv/bin/activate
pip list

# 测试运行
python /opt/ray_test.py
python /opt/vllm_paged_test.py
```

## 明天学习目标

1. **深入理解PagedAttention**: 内存管理和批处理优化
2. **vLLM性能分析**: 对比传统推理方法
3. **GPU调优实践**: 找到最佳配置参数
4. **Ray高级应用**: 探索更多分布式特性

---

*按此步骤重新部署，明天重点测试vLLM PagedAttention性能！*
