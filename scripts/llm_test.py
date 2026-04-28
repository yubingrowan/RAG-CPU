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