import ray
import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
import time
 
@ray.remote(num_gpus=1)
class RealLLMService:
    def __init__(self, service_id, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        self.service_id = service_id
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.load_model()
        
    def load_model(self):
        print(f"服务 {self.service_id} 正在加载模型: {self.model_name}")
        start_time = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        load_time = time.time() - start_time
        print(f"服务 {self.service_id} 模型加载完成，耗时: {load_time:.2f}秒")
        
    def real_inference(self, prompt, max_new_tokens=50):
        """真实LLM推理"""
        start_time = time.time()
        
        # 确保输入在正确设备
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 只取新生成的部分
        input_length = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        inference_time = time.time() - start_time
        
        return {
            "service_id": self.service_id,
            "prompt": prompt,
            "response": response,
            "inference_time": f"{inference_time:.3f}s",
            "input_tokens": input_length,
            "output_tokens": len(new_tokens),
            "gpu_memory": f"{torch.cuda.memory_allocated() / 1024**2:.1f}MB",
            "model": self.model_name
        }
 
@ray.remote
class DistributedLLMCoordinator:
    def __init__(self):
        self.services = []
        
    def register_service(self, service):
        """注册LLM服务"""
        self.services.append(service)
        
    def parallel_inference(self, prompts, max_new_tokens=50):
        """并行推理"""
        results = []
        
        for i, prompt in enumerate(prompts):
            # 轮询分配服务
            service = self.services[i % len(self.services)]
            
            # 执行真实推理
            result = ray.get(service.real_inference.remote(prompt, max_new_tokens))
            results.append(result)
            
        return results
    
    def batch_inference(self, prompts, max_new_tokens=50):
        """批量推理（所有服务并行）"""
        if len(prompts) <= len(self.services):
            # 每个服务处理一个请求
            futures = []
            for i, prompt in enumerate(prompts):
                future = self.services[i].real_inference.remote(prompt, max_new_tokens)
                futures.append(future)
            
            results = ray.get(futures)
        else:
            # 超过服务数量，使用轮询
            results = self.parallel_inference(prompts, max_new_tokens)
            
        return results
 
if __name__ == "__main__":
    ray.init()
    
    print("=== 真实LLM分布式推理测试 ===")
    
    # 创建协调器
    coordinator = DistributedLLMCoordinator.remote()
    
    # 创建LLM服务（使用1.5B模型节省内存）
    num_services = int(ray.cluster_resources().get('GPU', 0))
    services = []
    
    for i in range(num_services):
        # 使用较小的模型避免内存不足
        service = RealLLMService.remote(service_id=i, model_name="Qwen/Qwen2.5-1.5B-Instruct")
        services.append(service)
        ray.get(coordinator.register_service.remote(service))
    
    print(f"创建了 {num_services} 个真实LLM服务")
    
    # 测试提示
    test_prompts = [
        "什么是人工智能？",
        "机器学习的原理是什么？",
        "深度学习有什么应用？",
        "自然语言处理技术有哪些？"
    ]
    
    print(f"\n=== 并行推理测试 ({len(test_prompts)} 个提示) ===")
    
    # 并行推理
    start_time = time.time()
    parallel_results = ray.get(coordinator.parallel_inference.remote(test_prompts))
    parallel_time = time.time() - start_time
    
    print(f"并行推理总时间: {parallel_time:.3f}秒")
    
    # 显示结果
    print(f"\n=== 真实推理结果 ===")
    total_input_tokens = 0
    total_output_tokens = 0
    total_inference_time = 0
    
    for result in parallel_results:
        print(f"\n服务 {result['service_id']}:")
        print(f"  提示: {result['prompt']}")
        print(f"  回答: {result['response']}")
        print(f"  推理时间: {result['inference_time']}")
        print(f"  输入token: {result['input_tokens']}, 输出token: {result['output_tokens']}")
        print(f"  GPU内存: {result['gpu_memory']}")
        print(f"  模型: {result['model']}")
        
        total_input_tokens += result['input_tokens']
        total_output_tokens += result['output_tokens']
        total_inference_time += float(result['inference_time'].replace('s', ''))
    
    print(f"\n=== 性能统计 ===")
    print(f"总处理时间: {parallel_time:.3f}秒")
    print(f"总处理token: {total_input_tokens + total_output_tokens}")
    print(f"平均推理时间: {total_inference_time / len(test_prompts):.3f}秒/请求")
    print(f"吞吐量: {(total_input_tokens + total_output_tokens) / parallel_time:.1f} tokens/秒")
    
    # 单服务对比
    if len(services) > 0:
        print(f"\n=== 单服务对比测试 ===")
        single_service = services[0]
        
        start_time = time.time()
        single_result = ray.get(single_service.real_inference.remote("请简单介绍一下AI技术。", 50))
        single_time = time.time() - start_time
        
        print(f"单服务推理时间: {single_time:.3f}秒")
        print(f"并行vs单服务加速比: {single_time / (parallel_time / len(test_prompts)):.2f}x")
    
    ray.shutdown()
EOF

并行推理总时间: 8.221秒

=== 真实推理结果 ===

服务 0:
  提示: 什么是人工智能？
  回答: 它有什么特点和应用领域？

人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能才能完成的任务的机器或软件。这些任务包括学习、推理、问题解决、感知以及
  推理时间: 1.348s
  输入token: 3, 输出token: 50
  GPU内存: 2952.5MB
  模型: Qwen/Qwen2.5-1.5B-Instruct

服务 0:
  提示: 机器学习的原理是什么？
  回答:  1. 数据：数据是机器学习的基础，它包括输入和输出两个部分。输入数据可以来自各种来源，如图像、文本、音频等；而输出则是根据输入数据进行预测或分类的结果。

2. 模
  推理时间: 1.027s
  输入token: 6, 输出token: 50
  GPU内存: 2952.5MB
  模型: Qwen/Qwen2.5-1.5B-Instruct

服务 0:
  提示: 深度学习有什么应用？
  回答:  1. 图像识别：图像识别是深度学习的一个重要领域，可以用于人脸识别、物体检测、医学影像分析等领域。例如，Google的DeepMind团队开发了AlphaGo，它使用的是深度神经网络来玩围棋。


  推理时间: 1.025s
  输入token: 5, 输出token: 50
  GPU内存: 2952.5MB
  模型: Qwen/Qwen2.5-1.5B-Instruct

服务 0:
  提示: 自然语言处理技术有哪些？
  回答:  NLP（自然语言处理）是人工智能领域中一个重要的分支，它旨在使计算机能够理解和生成人类的自然语言。NLP技术广泛应用于许多方面，包括但不限于：

1. 机器翻译：将一种语言自动转换为
  推理时间: 1.018s
  输入token: 6, 输出token: 50
  GPU内存: 2952.5MB
  模型: Qwen/Qwen2.5-1.5B-Instruct

=== 性能统计 ===
总处理时间: 8.221秒
总处理token: 220
平均推理时间: 1.105秒/请求
吞吐量: 26.8 tokens/秒

=== 单服务对比测试 ===
单服务推理时间: 1.019秒
并行vs单服务加速比: 0.50x