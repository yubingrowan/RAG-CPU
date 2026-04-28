#!/usr/bin/env python3
"""
Base LLM Interface - 抽象化不同LLM提供商
支持llama.cpp和OpenAI API的无缝切换
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseLLM(ABC):
    """LLM基础接口类"""
    
    @abstractmethod
    async def predict_with_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        发送聊天消息并获取响应
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "..."}]
            max_tokens: 最大生成token数
            temperature: 温度参数，控制随机性
            top_p: 核采样参数
            
        Returns:
            模型生成的响应文本
        """
        pass
    
    @abstractmethod
    async def predict_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        支持工具调用的聊天请求
        
        Args:
            messages: 消息列表
            tools: 工具定义列表
            max_tokens: 最大生成token数
            temperature: 温度参数
            
        Returns:
            包含工具调用或普通响应的字典
        """
        pass
    
    @abstractmethod
    async def predict_json(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        max_tokens: int = 512,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        生成JSON格式响应
        
        Args:
            prompt: 输入提示
            schema: JSON schema（可选）
            max_tokens: 最大生成token数
            temperature: 温度参数，JSON生成建议用较低值
            
        Returns:
            解析后的JSON对象
        """
        pass
    
    @abstractmethod
    async def embed(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        生成文本嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型配置信息
        """
        pass
