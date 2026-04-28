#!/usr/bin/env python3
"""
OpenAI API Client - 兼容阿里云DashScope
支持标准OpenAI API格式，可无缝替换llama.cpp
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

logger = logging.getLogger("OpenAILLM")


class OpenAILLM:
    """OpenAI兼容的LLM客户端"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen-plus",
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        初始化OpenAI兼容客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL，默认为阿里云DashScope
            model: 模型名称，默认为qwen-plus
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
        """
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )
        self.model = model
        self.max_retries = max_retries
        
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
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            content = response.choices[0].message.content
            logger.debug(f"OpenAI API响应: {content[:100]}...")
            return content
            
        except Exception as e:
            logger.error(f"OpenAI API调用失败: {e}")
            raise
    
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
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            message = response.choices[0].message
            result = {
                "content": message.content or "",
                "tool_calls": []
            }
            
            # 解析工具调用
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    result["tool_calls"].append({
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    })
            
            logger.debug(f"工具调用结果: {result}")
            return result
            
        except Exception as e:
            logger.error(f"工具调用失败: {e}")
            raise
    
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
        try:
            messages = [
                {"role": "system", "content": "请返回有效的JSON格式响应。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            import json
            result = json.loads(content)
            
            logger.debug(f"JSON响应: {result}")
            return result
            
        except Exception as e:
            logger.error(f"JSON生成失败: {e}")
            raise
    
    async def embed(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        生成文本嵌入向量（如果模型支持）
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        # 注意：qwen-plus可能不支持嵌入，需要使用专门的嵌入模型
        logger.warning("qwen-plus不支持嵌入功能，请使用专门的嵌入模型")
        raise NotImplementedError("当前模型不支持嵌入功能")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型配置信息
        """
        return {
            "model": self.model,
            "base_url": self.client.base_url,
            "supports_tools": True,
            "supports_json": True,
            "supports_embedding": False,
            "max_retries": self.max_retries
        }


# 便捷函数，用于快速测试
async def test_openai_connection(api_key: str, base_url: str = None) -> bool:
    """
    测试OpenAI API连接
    
    Args:
        api_key: API密钥
        base_url: 自定义base_url（可选）
        
    Returns:
        连接是否成功
    """
    try:
        client = OpenAILLM(
            api_key=api_key,
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        response = await client.predict_with_messages(
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=10
        )
        
        logger.info("OpenAI API连接测试成功")
        return True
        
    except Exception as e:
        logger.error(f"OpenAI API连接测试失败: {e}")
        return False


if __name__ == "__main__":
    # 简单测试
    import asyncio
    
    async def main():
        # 从环境变量获取API密钥
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("请设置OPENAI_API_KEY环境变量")
            return
        
        client = OpenAILLM(api_key=api_key)
        
        # 测试基本对话
        response = await client.predict_with_messages(
            messages=[{"role": "user", "content": "你好，请介绍一下你自己"}]
        )
        print(f"响应: {response}")
        
        # 测试工具调用
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市名称"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        
        tool_response = await client.predict_with_tools(
            messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
            tools=tools
        )
        print(f"工具调用响应: {tool_response}")
    
    asyncio.run(main())
