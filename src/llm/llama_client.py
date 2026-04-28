#!/usr/bin/env python3
"""
llama.cpp HTTP Client with automatic prompt truncation
"""

import logging
import time
import requests
import json
import asyncio
from typing import Optional, List, Dict, Generator

logger = logging.getLogger("LlamaClient")


class LlamaClient:
    """Client for llama.cpp HTTP server with auto-truncation"""

    def __init__(self, host: str = "localhost", port: int = 8080):
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()

    def health_check(self) -> bool:
        """Check if server is running"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            healthy = response.status_code == 200
            logger.info("Health check: status=%d healthy=%s", response.status_code, healthy)
            return healthy
        except requests.RequestException as exc:
            logger.warning("Health check failed: %s", exc)
            return False

    async def truncate_prompt(
        self,
        prompt: str,
        max_tokens: int,
        context_limit: int = 1024
    ) -> str:
        """
        如果 prompt 太长，自动截断至适合上下文长度
        
        Args:
            prompt: 原始 prompt
            max_tokens: 期望生成的最大 token 数
            context_limit: 模型的上下文长度（默认 1024）
        
        Returns:
            截断后的 prompt（如果未超限则返回原 prompt）
        """
        # 获取 prompt 的 token 序列
        prompt_tokens = await asyncio.to_thread(self.tokenize, prompt)
        prompt_len = len(prompt_tokens)
        available_tokens = context_limit - max_tokens
        
        if available_tokens <= 0:
            raise ValueError(
                f"max_tokens ({max_tokens}) 已经达到或超过 context_limit ({context_limit})，"
                "无法生成任何内容。请减小 max_tokens 或增大 context_limit。"
            )
        
        if prompt_len <= available_tokens:
            logger.info(
                "Prompt 长度合适: %d tokens ≤ %d available",
                prompt_len, available_tokens
            )
            return prompt
        
        # 需要截断：保留最后 available_tokens 个 token（最新内容）
        truncated_tokens = prompt_tokens[-available_tokens:]
        truncated_prompt = await asyncio.to_thread(self.detokenize, truncated_tokens)
        
        logger.warning(
            "Prompt 被截断: 原始 %d tokens → %d tokens (可用空间 %d)",
            prompt_len, len(truncated_tokens), available_tokens
        )
        return truncated_prompt

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        context_limit: int = 1024,
        auto_truncate: bool = True
    ) -> str:
        """
        生成文本补全（异步），支持自动截断 prompt
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成 token 数
            temperature: 采样温度 (0.0 - 1.0)
            top_p: 核采样参数
            stream: 是否流式响应（注意：自动截断时 stream 仍正常工作）
            context_limit: 模型上下文长度（用于自动截断）
            auto_truncate: 是否自动截断超长 prompt
        
        Returns:
            生成的文本
        """
        # 自动截断（如果需要）
        if auto_truncate:
            prompt = await self.truncate_prompt(prompt, max_tokens, context_limit)
        
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream
        }

        # 打印请求详情（调试用）
        logger.info(
            "LLM complete: prompt_len=%d chars, max_tokens=%d, temp=%.2f, top_p=%.2f",
            len(prompt), max_tokens, temperature, top_p
        )
        
        start_time = time.perf_counter()
        response = await asyncio.to_thread(
            self.session.post,
            f"{self.base_url}/completion",
            json=payload,
            timeout=300
        )

        elapsed = time.perf_counter() - start_time
        logger.info("LLM complete finished: elapsed=%.3fs status=%d", elapsed, response.status_code)
        logger.info(
            "LLM completion payload: %s",
            json.dumps(payload, ensure_ascii=False)
        )
        logger.info(
            "LLM /completion response: status=%d headers=%s text=%s",
            response.status_code,
            dict(response.headers),
            response.text[:4000]
        )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            error_details = {
                "status_code": response.status_code,
                "response_text": response.text,
                "payload": payload,
                "url": f"{self.base_url}/completion"
            }
            logger.error(
                "LLM /completion failed: %s",
                json.dumps(error_details, ensure_ascii=False, indent=2)
            )
            raise requests.HTTPError(f"LLM request failed: {exc}") from exc
        except Exception as exc:
            logger.error(
                "LLM /completion unexpected error: %s payload=%s",
                str(exc),
                json.dumps(payload, ensure_ascii=False)
            )
            raise

        if stream:
            return self._stream_response(response)
        else:
            data = response.json()
            logger.info("LLM complete response content length=%d", len(data.get("content", "")))
            logger.info("LLM complete response body: %s", data)
            return data["content"]
    
    def _stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        """处理流式响应"""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "content" in data:
                    yield data["content"]
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        context_limit: int = 1024,
        auto_truncate: bool = True
    ) -> str:
        """
        聊天补全，自动构建 prompt 并支持截断
        """
        prompt = self._build_chat_prompt(messages)
        logger.info(
            "Chat request: messages=%d prompt_len=%d chars",
            len(messages), len(prompt)
        )
        return await self.complete(
            prompt, max_tokens, temperature, top_p, stream,
            context_limit=context_limit, auto_truncate=auto_truncate
        )
    
    def _build_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """从消息列表构建聊天 prompt（使用 llama.cpp 默认的 chat 模板）"""
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt
    
    def tokenize(self, text: str) -> List[int]:
        """
        将文本转换为 token ID 列表（同步方法）
        """
        payload = {"content": text}
        response = self.session.post(f"{self.base_url}/tokenize", json=payload)
        response.raise_for_status()
        return response.json()["tokens"]
    
    def detokenize(self, tokens: List[int]) -> str:
        """
        将 token ID 列表转换回文本（同步方法）
        """
        payload = {"tokens": tokens}
        response = self.session.post(f"{self.base_url}/detokenize", json=payload)
        response.raise_for_status()
        return response.json()["content"]


if __name__ == "__main__":
    # 测试客户端
    import asyncio
    
    async def test():
        client = LlamaClient()
        
        if client.health_check():
            print("Server is healthy")
            
            # 构造一个超长的 prompt（例如 1500 tokens）
            long_prompt = "Hello, how are you? " * 500
            
            try:
                # 自动截断将生效
                response = await client.complete(
                    long_prompt,
                    max_tokens=50,
                    context_limit=2048,
                    auto_truncate=True
                )
                print(f"Response: {response}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Server is not responding")
    
    asyncio.run(test())