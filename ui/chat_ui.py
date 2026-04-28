"""
RAG Chat UI
Streamlit-based chat interface for RAG system with query and index functionality
"""

import streamlit as st
import asyncio
import sys
import os
import datetime
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from workflows.query_graph import QueryWorkflow
from workflows.index_graph import IndexWorkflow
from kafka import KafkaProducer
from rag_pipeline import RAGPipeline
from llm import LLMProvider
import json


# Page configuration
st.set_page_config(
    page_title="RAG Chat System",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "mode" not in st.session_state:
    st.session_state.mode = "query"  # "query" or "index"

# LLM配置状态
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "llama_cpp"

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None


def get_kafka_producer():
    """Get or create Kafka producer"""
    if "kafka_producer" not in st.session_state:
        st.session_state.kafka_producer = KafkaProducer(
            bootstrap_servers='localhost:9092',
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    return st.session_state.kafka_producer


def send_index_task(documents):
    """Send indexing task to Kafka"""
    try:
        producer = get_kafka_producer()
        task = {
            "task_id": f"task_{len(st.session_state.messages)}",
            "documents": documents,
            "timestamp": str(datetime.datetime.now())
        }
        producer.send('index_tasks', value=task)
        producer.flush()
        return True, "Index task sent to Kafka"
    except Exception as e:
        return False, f"Error: {str(e)}"


async def query_mode():
    """Query mode - chat with RAG system"""
    st.header("🔍 Query Mode")
    st.write("Ask questions about your documents")
    
    # LLM配置面板
    with st.sidebar.expander("🔧 LLM配置", expanded=False):
        llm_provider = st.selectbox(
            "LLM提供商",
            options=["llama_cpp", "openai"],
            index=0 if st.session_state.get("llm_provider") == "llama_cpp" else 1,
            format_func=lambda x: "llama.cpp" if x == "llama_cpp" else "OpenAI API"
        )
        
        if llm_provider != st.session_state.get("llm_provider"):
            st.session_state.llm_provider = llm_provider
            st.session_state.rag_pipeline = None  # 重置pipeline以重新初始化
            st.success(f"✅ 已切换到 {llm_provider}")
        
        # OpenAI API配置
        if llm_provider == "openai":
            api_key = st.text_input(
                "API密钥",
                type="password",
                value=os.getenv("OPENAI_API_KEY", ""),
                help="输入OpenAI API密钥"
            )
            model = st.selectbox(
                "模型选择",
                options=["qwen-plus", "qwen-turbo"],
                index=0
            )
            st.session_state.llm_config = {
                "api_key": api_key,
                "model": model
            }
        else:
            st.session_state.llm_config = {
                "host": st.text_input("服务器地址", value="localhost"),
                "port": st.number_input("端口", value=8080)
            }
    
    # RAG参数配置
    with st.sidebar.expander("⚙️ RAG参数", expanded=False):
        st.session_state.use_rerank = st.checkbox(
            "启用重排序",
            value=st.session_state.get("use_rerank", True)
        )
        st.session_state.top_k = st.slider(
            "检索文档数量",
            min_value=1,
            max_value=10,
            value=st.session_state.get("top_k", 3)
        )
        st.session_state.temperature = st.slider(
            "温度参数",
            min_value=0.1,
            max_value=2.0,
            value=st.session_state.get("temperature", 0.7),
            step=0.1
        )
        st.session_state.max_tokens = st.slider(
            "最大Token数",
            min_value=50,
            max_value=2048,
            value=st.session_state.get("max_tokens", 512),
            step=50
        )
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 查看来源"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"**来源 {i}** (RRF分数: {source.get('rrf_score', 0):.4f})")
                        st.text(source['text'][:200] + "...")
                        st.caption(f"来源: {source.get('metadata', {}).get('source', 'Unknown')}")
    
    # Chat input - 确保在消息显示后
    st.markdown("---")  # 添加分隔线
    prompt = st.chat_input("请输入你的问题：", key="chat_input")
    
    # Add user message
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("🔄 正在搜索并生成答案..."):
                try:
                    # 初始化或重置RAG Pipeline
                    if (st.session_state.rag_pipeline is None or 
                        st.session_state.current_provider != st.session_state.get("llm_provider")):
                        
                        if st.session_state.get("llm_provider") == "openai":
                            st.session_state.rag_pipeline = RAGPipeline(
                                llm_provider="openai",
                                llm_config=st.session_state.get("llm_config", {})
                            )
                        else:
                            st.session_state.rag_pipeline = RAGPipeline(
                                llm_provider="llama_cpp",
                                llm_config=st.session_state.get("llm_config", {})
                            )
                        
                        st.session_state.current_provider = st.session_state.get("llm_provider")
                        st.success(f"✅ 已初始化 {st.session_state.get('llm_provider')} RAG Pipeline")
                    
                    # 使用RAG Pipeline查询 - 包装为同步调用
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            result = await st.session_state.rag_pipeline.query(
                                query=prompt,
                                session_id="streamlit_session",
                                top_k=st.session_state.get("top_k", 3),
                                use_rerank=st.session_state.get("use_rerank", True),
                                max_tokens=st.session_state.get("max_tokens", 512),
                                temperature=st.session_state.get("temperature", 0.3)
                            )
                        else:
                            # 如果没有运行的事件循环，创建新的
                            result = asyncio.run(
                                st.session_state.rag_pipeline.query(
                                    query=prompt,
                                    session_id="streamlit_session",
                                    top_k=st.session_state.get("top_k", 3),
                                    use_rerank=st.session_state.get("use_rerank", True),
                                    max_tokens=st.session_state.get("max_tokens", 512),
                                    temperature=st.session_state.get("temperature", 0.3)
                                )
                            )
                    except Exception as async_error:
                        # 如果异步调用失败，尝试同步调用
                        st.warning("⚠️ 异步调用失败，尝试同步方式")
                        try:
                            # 创建同步版本的查询方法
                            def sync_query():
                                return asyncio.run(st.session_state.rag_pipeline.query(
                                    query=prompt,
                                    session_id="streamlit_session",
                                    top_k=st.session_state.get("top_k", 3),
                                    use_rerank=st.session_state.get("use_rerank", True),
                                    max_tokens=st.session_state.get("max_tokens", 512),
                                    temperature=st.session_state.get("temperature", 0.3)
                                ))
                            
                            result = sync_query()
                        except Exception as sync_error:
                            st.error(f"❌ 同步调用也失败: {str(sync_error)}")
                            raise
                    
                    st.markdown(result["answer"])
                    
                    # Add assistant message with sources
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"]
                    })
                    
                except Exception as e:
                    st.error(f"❌ 查询失败: {str(e)}")
                    st.error("请检查配置或服务状态")
                
                with st.expander("View Sources"):
                    for i, source in enumerate(result["sources"], 1):
                        st.write(f"**Source {i}** (RRF Score: {source.get('rrf_score', 0):.4f})")
                        st.text(source['text'][:200] + "...")
                        st.caption(f"From: {source.get('metadata', {}).get('source', 'Unknown')}")


def index_mode():
    """Index mode - add documents to RAG system"""
    st.header("📄 Index Mode")
    st.write("Add documents to your RAG system")
    
    # Document input options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Text Input")
        text_input = st.text_area("Enter document text", height=200)
        doc_name = st.text_input("Document name (optional)")
    
    with col2:
        st.subheader("File Upload")
        uploaded_file = st.file_uploader("Upload a file (txt, md, pdf)", type=['txt', 'md', 'pdf'])
    
    # Index button
    if st.button("Index Document", type="primary"):
        documents = []
        
        # Process text input
        if text_input:
            documents.append({
                'text': text_input,
                'metadata': {
                    'source': doc_name if doc_name else 'manual_input',
                    'type': 'text'
                }
            })
        
        # Process uploaded file
        if uploaded_file:
            file_content = uploaded_file.read().decode('utf-8')
            documents.append({
                'text': file_content,
                'metadata': {
                    'source': uploaded_file.name,
                    'type': 'file_upload'
                }
            })
        
        if not documents:
            st.warning("Please enter text or upload a file")
            return
        
        # Send to Kafka for async indexing
        success, message = send_index_task(documents)
        
        if success:
            st.success(message)
            st.info("Document will be indexed asynchronously. You can query it in a few moments.")
        else:
            st.error(message)
    
    # Alternative: Direct indexing (synchronous)
    with st.expander("Direct Indexing (Synchronous)"):
        st.warning("This will block the UI until indexing completes")
        if st.button("Index Directly"):
            if documents:
                try:
                    workflow = IndexWorkflow()
                    result = asyncio.run(workflow.run(documents))
                    
                    if result["status"] == "completed":
                        st.success(f"Successfully indexed {result['indexed_count']} chunks")
                    else:
                        st.error(f"Indexing failed: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


def main():
    """Main application"""
    st.title("🤖 RAG Chat System")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Mode selection
        st.session_state.mode = st.radio(
            "Mode",
            ["Query", "Index"],
            horizontal=True,
            key="mode_selection"
        )
        
        st.divider()
        
        # Query settings
        st.subheader("Query Settings")
        st.session_state.use_rerank = st.checkbox("Use Reranking", value=True)
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1
        )
        
        st.divider()
        
        # System info
        st.subheader("System Info")
        st.info(f"Messages: {len(st.session_state.messages)}")
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Main content based on mode
    if st.session_state.mode == "Query":
        # 创建同步包装器
        def sync_query_mode():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(query_mode())
            finally:
                loop.close()
        sync_query_mode()
    else:
        index_mode()


if __name__ == "__main__":
    import pandas as pd
    import asyncio
    main()
