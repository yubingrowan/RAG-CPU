#!/usr/bin/env python3
"""
LLM Factory - 工厂模式创建不同LLM实例
支持配置化切换不同LLM提供商
"""

import os
from enum import Enum
from typing import Dict, Any

from .base_llm import BaseLLM
from .openai_llm import OpenAILLM
# 后续可以导入其他LLM实现
# from .llama_cpp_llm import LlamaCppLLM


class LLMProvider(Enum):
    """LLM提供商枚举"""
    OPENAI = "openai"
    LLAMA_CPP = "llama_cpp"
    # 可以继续添加其他提供商
    # AZURE_OPENAI = "azure_openai"
    # HUGGINGFACE = "huggingface"


class LLMFactory:
    """LLM工厂类"""
    
    @staticmethod
    def create_llm(provider: LLMProvider, **kwargs) -> BaseLLM:
        """
        创建LLM实例
        
        Args:
            provider: LLM提供商
            **kwargs: 提供商特定的配置参数
            
        Returns:
            LLM实例
        """
        if provider == LLMProvider.OPENAI:
            return OpenAILLM(
                api_key=kwargs.get("api_key"),
                base_url=kwargs.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                model=kwargs.get("model", "qwen-plus"),
                max_retries=kwargs.get("max_retries", 3),
                timeout=kwargs.get("timeout", 30)
            )
        
        elif provider == LLMProvider.LLAMA_CPP:
            # 这里可以添加llama.cpp的实现
            # from .llama_cpp_llm import LlamaCppLLM
            # return LlamaCppLLM(
            #     model_path=kwargs.get("model_path"),
            #     context_size=kwargs.get("context_size", 1024)
            # )
            raise NotImplementedError("llama.cpp实现待添加")
        
        else:
            raise ValueError(f"不支持的LLM提供商: {provider}")
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> BaseLLM:
        """
        从配置字典创建LLM实例
        
        Args:
            config: 配置字典
            
        Returns:
            LLM实例
        """
        provider = LLMProvider(config.get("provider", "openai"))
        return LLMFactory.create_llm(provider, **config.get("config", {}))
    
    @staticmethod
    def get_available_providers() -> list:
        """
        获取可用的LLM提供商列表
        
        Returns:
            提供商列表
        """
        return [provider.value for provider in LLMProvider]


# 全局LLM实例缓存（可选）
_llm_cache = {}


def get_cached_llm(provider: LLMProvider, **kwargs) -> BaseLLM:
    """
    获取缓存的LLM实例
    
    Args:
        provider: LLM提供商
        **kwargs: 配置参数
        
    Returns:
        LLM实例（缓存或新建）
    """
    cache_key = f"{provider.value}_{hash(str(kwargs))}"
    
    if cache_key not in _llm_cache:
        _llm_cache[cache_key] = LLMFactory.create_llm(provider, **kwargs)
    
    return _llm_cache[cache_key]


def clear_llm_cache():
    """清空LLM缓存"""
    global _llm_cache
    _llm_cache.clear()


if __name__ == "__main__":
    # 测试工厂模式
    import asyncio
    
    async def test_factory():
        # 测试OpenAI配置
        config = {
            "provider": "openai",
            "config": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": "qwen-plus"
            }
        }
        
        llm = LLMFactory.create_from_config(config)
        info = llm.get_model_info()
        print(f"模型信息: {info}")
        
        # 测试基本对话
        response = await llm.predict_with_messages(
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=50
        )
        print(f"响应: {response}")
    
    asyncio.run(test_factory())
