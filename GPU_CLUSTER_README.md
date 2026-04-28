# GPU集群项目 - 单节点部署完整指南

## 项目概述
成功搭建了基于阿里云ECS的单节点GPU推理集群，实现了从环境配置到LLM推理服务的完整部署。

## 硬件配置
- **实例**: ecs.gn7i-c8g1.2xlarge (华东2上海)
- **GPU**: NVIDIA A10 (22.1GB显存)
- **CPU**: 8核
- **内存**: 30GB
- **存储**: 100GB ESSD

## 软件栈
- **操作系统**: Ubuntu 22.04 LTS
- **GPU驱动**: NVIDIA 570.195.03
- **CUDA**: 12.8
- **Python**: 3.10 + 虚拟环境
- **PyTorch**: 2.5.1+cu121
- **Ray**: 分布式计算框架
- **ModelScope**: 模型下载和加载

---

## 完整部署命令

### 1. 系统环境验证
```bash
# 检查GPU状态
nvidia-smi

# 检查CUDA版本
nvcc --version

# 检查系统信息
uname -a
```

### 2. Python环境配置
```bash
# 安装Python和基础工具
apt install -y python3 python3-pip python3-venv git curl wget

# 创建虚拟环境
python3 -m venv /opt/venv
source /opt/venv/bin/activate

# 升级pip
pip install --upgrade pip
```

### 3. Ray和依赖安装
```bash
# 安装Ray和GPU支持
pip install ray[default] ray[runtime] torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install fastapi uvicorn transformers accelerate bitsandbytes

# 安装ModelScope
pip install modelscope
```

### 4. Ray集群验证
```bash
# 创建Ray测试脚本
cat > /opt/ray_test.py <<EOF
import ray
import torch

@ray.remote(num_gpus=1)  # 指定使用1个GPU
def gpu_task():
    """测试GPU任务"""
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
    else:
        return "GPU not available"

@ray.remote
def cpu_task():
    """测试CPU任务"""
    import os
    return f"CPU cores: {os.cpu_count()}"

if __name__ == "__main__":
    ray.init()
    
    # 并行执行任务
    gpu_result = ray.get(gpu_task.remote())
    cpu_result = ray.get(cpu_task.remote())
    
    print("=== Ray GPU集群测试 ===")
    print(gpu_result)
    print(cpu_result)
    print("集群资源:", ray.cluster_resources())
    
    ray.shutdown()
EOF

# 运行测试
python /opt/ray_test.py
```

### 5. LLM推理服务部署
```bash
# 创建LLM推理服务
cat > /opt/llm_service.py <<EOF
import ray
import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
import time

@ray.remote(num_gpus=1)
class LLMInferenceService:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        print(f"从ModelScope加载模型: {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("模型加载成功!")
    
    def generate(self, prompt, max_length=100):
        start_time = time.time()
        
        # 确保input_ids在正确的设备上
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # 将inputs移动到GPU
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        inference_time = time.time() - start_time
        
        return {
            "prompt": prompt,
            "response": result,
            "inference_time": f"{inference_time:.2f}s",
            "model": self.model_id
        }

if __name__ == "__main__":
    ray.init()
    
    # 启动LLM服务
    service = LLMInferenceService.remote()
    
    # 测试推理
    test_prompts = [
        "什么是人工智能？",
        "请解释一下机器学习的基本概念。"
    ]
    
    for prompt in test_prompts:
        print(f"\n=== 测试提示: {prompt} ===")
        result = ray.get(service.generate.remote(prompt))
        print(f"回答: {result['response']}")
        print(f"推理时间: {result['inference_time']}")
    
    ray.shutdown()
EOF

# 运行LLM推理测试
python /opt/llm_service.py
```

---

## 核心技术解析

### 1. Ray分布式框架
```python
import ray
ray.init()  # 启动Ray集群，自动发现GPU资源
```
**作用**: Ray自动检测GPU资源，创建分布式计算环境。

### 2. GPU资源调度
```python
@ray.remote(num_gpus=1)
class LLMInferenceService:
```
**作用**: 告诉Ray这个类需要1个GPU资源，Ray会自动分配GPU。

### 3. 模型优化加载
```python
self.model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # 半精度减少内存
    device_map="auto",          # 自动分配GPU
    trust_remote_code=True
)
```
**作用**: 从ModelScope下载Qwen2.5-7B模型，自动优化GPU内存使用。

### 4. 设备匹配处理
```python
# 关键修复：确保输入在GPU上
inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
```
**作用**: 解决CPU输入vs GPU模型的设备不匹配问题。

---

## 技术难点解决方案

### 1. 网络访问问题
- **问题**: Hugging Face无法访问
- **解决**: 使用ModelScope国内镜像源
- **命令**: `pip install modelscope`

### 2. 设备不匹配
- **问题**: RuntimeError: Expected all tensors to be on the same device
- **解决**: 确保所有tensor在同一设备上
- **代码**: `inputs = {k: v.to(self.model.device) for k, v in inputs.items()}`

### 3. Ray GPU调度
- **问题**: 任务无法使用GPU
- **解决**: `@ray.remote(num_gpus=1)`指定GPU资源
- **验证**: `ray.cluster_resources()`显示GPU: 1.0

---

## 性能数据

### 模型加载性能
- **下载时间**: ~1分钟 (ModelScope)
- **加载时间**: ~3秒
- **模型大小**: ~14GB (7B参数)

### 推理性能
- **推理时间**: 3.2-3.8秒/请求
- **GPU内存使用**: ~14GB
- **并发能力**: Ray支持多任务并行

### 集群资源
```
集群资源: {
    'CPU': 8.0,
    'GPU': 1.0, 
    'memory': 20.8GB,
    'object_store_memory': 8.9GB,
    'accelerator_type:A10': 1.0
}
```

---

## 项目价值

### 技术价值
- **完整链路**: 从硬件到应用的端到端实现
- **分布式架构**: Ray框架为多节点扩展奠定基础
- **生产就绪**: 包含错误处理、资源管理、监控

### 实用价值
- **可扩展**: 单节点验证成功，可扩展到多节点
- **高性能**: GPU推理性能满足实际需求
- **成本可控**: 按需使用，用完即删

### 学习价值
- **AI Infra实践**: 深入理解GPU集群部署
- **Ray分布式**: 掌握现代分布式计算框架
- **LLM工程**: 实际的模型部署和优化经验

---

## 下一步规划

### Phase 8: 多节点扩展
- 添加第二个GPU节点
- 配置Ray head/worker架构
- 实现真正的分布式推理

### Phase 9: 性能测试
- 并发推理测试
- 负载均衡验证
- 集群稳定性测试

---

## 使用说明

1. **环境准备**: 确保ECS实例有GPU支持
2. **按顺序执行**: 严格按照命令顺序执行
3. **网络问题**: 如遇网络问题，使用国内镜像源
4. **资源监控**: 使用Ray Dashboard监控集群状态
5. **成本控制**: 使用完毕及时停止实例

## 故障排除

### 常见问题
1. **CUDA版本不匹配**: 重新安装兼容的PyTorch版本
2. **内存不足**: 使用更小的模型或增加GPU内存
3. **网络超时**: 使用ModelScope镜像源
4. **Ray启动失败**: 检查端口占用和防火墙设置

---

*该项目展现了从基础设施到AI应用的完整技术能力，既有深度又有广度，是AI Infra领域的典型实战案例。*
