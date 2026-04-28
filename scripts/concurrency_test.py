import ray
import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
import time
import threading
 
@ray.remote(num_gpus=1)
class ConcurrentLLMService:
    def __init__(self, service_id):
        self.service_id = service_id
        self.request_count = 0
        self.total_time = 0
        # 加载轻量级模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"并发服务 {service_id} 初始化完成")
        
    def concurrent_inference(self, prompt, request_id):
        """并发推理"""
        start_time = time.time()
        self.request_count += 1
        
        # 简化推理过程
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,  # 减少token数量
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        input_length = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        processing_time = time.time() - start_time
        self.total_time += processing_time
        
        return {
            "service_id": self.service_id,
            "request_id": request_id,
            "response": response,
            "processing_time": f"{processing_time:.3f}s",
            "total_requests": self.request_count
        }
    
    def get_stats(self):
        """获取统计信息"""
        avg_time = self.total_time / self.request_count if self.request_count > 0 else 0
        return {
            "service_id": self.service_id,
            "total_requests": self.request_count,
            "average_time": f"{avg_time:.3f}s",
            "total_time": f"{self.total_time:.3f}s"
        }
 
def stress_test(num_services=1, requests_per_service=10):
    """压力测试"""
    ray.init()
    
    print(f"=== 并发压力测试 ===")
    print(f"服务数: {num_services}, 每服务请求数: {requests_per_service}")
    
    # 创建服务
    services = []
    for i in range(num_services):
        service = ConcurrentLLMService.remote(service_id=i)
        services.append(service)
    
    # 生成请求
    prompts = [
        "什么是AI？",
        "机器学习是什么？",
        "深度学习应用？",
        "自然语言处理？",
        "计算机视觉？"
    ]
    
    # 并发执行
    start_time = time.time()
    futures = []
    
    for service_idx, service in enumerate(services):
        for req_idx in range(requests_per_service):
            prompt = prompts[req_idx % len(prompts)]
            future = service.concurrent_inference.remote(prompt, f"req_{service_idx}_{req_idx}")
            futures.append(future)
    
    # 等待所有请求完成
    results = ray.get(futures)
    total_time = time.time() - start_time
    
    print(f"\n=== 压力测试结果 ===")
    print(f"总请求数: {len(results)}")
    print(f"总处理时间: {total_time:.3f}秒")
    print(f"平均QPS: {len(results) / total_time:.1f} 请求/秒")
    
    # 服务统计
    for service in services:
        stats = ray.get(service.get_stats.remote())
        print(f"服务 {stats['service_id']}: {stats['total_requests']} 请求, 平均 {stats['average_time']}")
    
    ray.shutdown()
 
if __name__ == "__main__":
    stress_test(num_services=1, requests_per_service=5)
EOF

=== 压力测试结果 ===
总请求数: 5
总处理时间: 13.191秒
平均QPS: 0.4 请求/秒
服务 0: 5 请求, 平均 0.413s