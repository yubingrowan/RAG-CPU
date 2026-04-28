#!/usr/bin/env python3
"""
RAG Pipeline
Complete retrieval-augmented generation pipeline with hybrid search, rerank, and context management
"""

from typing import List, Dict, Any, Optional
import logging
import os
import time
from llm import LLMFactory, LLMProvider
from retrieval.hybrid_retrieval import HybridRetriever
from retrieval.reranker import Reranker
from cache.conversation_cache import ConversationCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


class RAGPipeline:
    """Complete RAG pipeline"""
    
    def __init__(
        self,
        llm_provider: str = "llama_cpp",
        llm_config: Dict[str, Any] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379
    ):
        """
        Initialize RAG pipeline
        
        Args:
            llm_provider: LLM provider ("openai" or "llama_cpp")
            llm_config: LLM configuration dictionary
            redis_host: Redis host
            redis_port: Redis port
        """
        # 创建LLM实例
        if llm_provider == "openai":
            provider = LLMProvider.OPENAI
            self.llm_client = LLMFactory.create_llm(provider, **llm_config)
        elif llm_provider == "llama_cpp":
            # 保持向后兼容
            from llm.llama_client import LlamaClient
            host = llm_config.get("host", "localhost")
            port = llm_config.get("port", 8080)
            self.llm_client = LlamaClient(host=host, port=port)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
            
        self.hybrid_retriever = HybridRetriever()
        self.reranker = Reranker()
        self.cache = ConversationCache(host=redis_host, port=redis_port)

        self.max_context_length = 4000  # Max characters for context
        self.max_history = 5  # Max conversation turns to include
        self.default_top_k = 3  # Default top_k value
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def query(
        self,
        query: str,
        session_id: str,
        use_rerank: bool = True,
        top_k: int = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Execute RAG query

        Args:
            query: User query
            session_id: Session identifier for conversation history
            use_rerank: Whether to use reranking
            top_k: Number of documents to retrieve
            temperature: LLM temperature
        
        Returns:
            Response with answer and metadata
        """
        if top_k is None:
            top_k = self.default_top_k

        start_time = time.perf_counter()
        self.logger.info(
            "RAG query start: session=%s top_k=%s use_rerank=%s",
            session_id,
            top_k,
            use_rerank
        )

        # Retrieve documents
        retrieve_start = time.perf_counter()
        results = await self.hybrid_retriever.search(query, top_k=top_k * 2)
        retrieve_time = time.perf_counter() - retrieve_start
        self.logger.info(
            "Retrieval complete: results=%d retrieve_time=%.3fs",
            len(results),
            retrieve_time
        )

        rerank_time = 0.0
        if use_rerank and results:
            rerank_start = time.perf_counter()
            results = self.reranker.rerank_with_metadata(query, results, top_k=top_k)
            rerank_time = time.perf_counter() - rerank_start
            self.logger.info(
                "Rerank complete: top_k=%d rerank_time=%.3fs",
                top_k,
                rerank_time
            )

        # Build context from retrieved documents
        context = self._build_context(results[:top_k])

        # Get conversation history
        history = self.cache.get_history(session_id, limit=self.max_history)

        # Build prompt
        prompt_build_start = time.perf_counter()
        prompt = self._build_prompt(query, context, history)
        prompt_build_time = time.perf_counter() - prompt_build_start
        self.logger.debug(
            "Prompt built: prompt_len=%d history_count=%d context_len=%d prompt_build_time=%.3fs",
            len(prompt),
            len(history),
            len(context),
            prompt_build_time
        )

        # Generate response
        llm_start = time.perf_counter()
        
        # 统一调用接口
        if hasattr(self.llm_client, 'predict_with_messages'):
            # OpenAI API接口
            response = await self.llm_client.predict_with_messages(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
        else:
            # llama.cpp接口（向后兼容）
            response = await self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                context_limit=1024
            )
        
        llm_time = time.perf_counter() - llm_start
        self.logger.info(
            "LLM generation complete: llm_time=%.3fs answer_len=%d",
            llm_time,
            len(response)
        )

        # Cache the conversation
        self.cache.add_message(session_id, "user", query)
        self.cache.add_message(session_id, "assistant", response)

        total_time = time.perf_counter() - start_time
        self.logger.info(
            "RAG query complete: session=%s total_time=%.3fs",
            session_id,
            total_time
        )

        return {
            "answer": response,
            "sources": results[:top_k],
            "context": context,
            "session_id": session_id,
            "timing": {
                "retrieval_time": retrieve_time,
                "rerank_time": rerank_time,
                "prompt_build_time": prompt_build_time,
                "llm_time": llm_time,
                "total_time": total_time
            }
        }
    
    def _build_context(self, results: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents"""
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Document {i}] {result['text']}"
            )
        return "\n\n".join(context_parts)
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        history: List[Dict[str, str]]
    ) -> str:
        """Build prompt with context and history"""
        prompt = """You are a knowledgeable assistant specialized in providing accurate, well-structured answers based on the given context.

CORE PRINCIPLES:
- Answer ONLY using information from the provided context
- If the context doesn't contain the answer, explicitly state this limitation
- Always cite which document(s) you're referencing
- Maintain consistency with previous conversation when relevant
- Be concise but thorough - don't be overly verbose

INSTRUCTIONS:

1. **Source Attribution**: Always cite your sources. When referencing information, specify which document it came from.
   - Good: "According to Document 1, RAG systems..."
   - Bad: "RAG systems..." (no citation)

2. **Handling Insufficient Information**: If the context doesn't contain enough information:
   - Clearly state: "The provided context does not contain information about [topic]."
   - If partial information exists, answer what you can and acknowledge gaps
   - Never make up or hallucinate information

3. **Answer Structure**:
   - Start with a direct answer to the question
   - Use bullet points for lists of related information
   - Use numbered lists for sequential information
   - Keep paragraphs focused and relatively short (2-3 sentences each)

4. **Technical Accuracy**:
   - Preserve technical terminology exactly as it appears in the context
   - Don't oversimplify complex concepts unless asked
   - If multiple perspectives exist in the context, present them clearly

5. **Context Integration**:
   - Synthesize information from multiple documents when relevant
   - Note contradictions between documents if they exist
   - Prioritize more recent or authoritative sources if indicated

6. **Conversation Consistency**:
   - Reference previous exchanges when relevant
   - Maintain a consistent tone and style
   - If the user asks follow-up questions, build on previous answers

EXAMPLES OF GOOD RESPONSES:
- "Based on Document 2, the RAG architecture consists of..."
- "According to Documents 1 and 3, the key components are..."
- "The context mentions X in Document 1, but doesn't provide details about Y."

EXAMPLES OF BAD RESPONSES:
- "RAG systems work by..." (no citation, no context reference)
- "I think..." (opinion not based on context)
- Making up facts not present in the context

"""
        
        if context:
            prompt += f"CONTEXT:\n{context}\n\n"
        
        if history:
            prompt += "CONVERSATION HISTORY:\n"
            for msg in history:
                prompt += f"{msg['role']}: {msg['content']}\n"
            prompt += "\n"
        
        prompt += f"USER QUESTION:\n{query}\n\n"
        prompt += "ANSWER:"
        
        # Truncate if too long
        if len(prompt) > self.max_context_length:
            prompt = prompt[-self.max_context_length:]
        
        return prompt
    
    def clear_session(self, session_id: str) -> None:
        """Clear conversation history for a session"""
        self.cache.clear_history(session_id)


if __name__ == "__main__":
    # Test RAG pipeline
    pipeline = RAGPipeline()
    
    # Test query (requires documents to be indexed first)
    session_id = "test_session"
    query = "What is machine learning?"
    
    try:
        result = pipeline.query(query, session_id, use_rerank=True)
        
        print("RAG Response:")
        print(f"Answer: {result['answer']}")
        print(f"\nSources: {len(result['sources'])}")
        for source in result['sources']:
            print(f"  - {source['text'][:100]}...")
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Make sure documents are indexed first using indexing_pipeline.py")
