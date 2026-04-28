"""
LangGraph Query Workflow
RAG query workflow using LangGraph for orchestration
"""

from typing import TypedDict, Annotated, Sequence
from operator import add
from langgraph.graph import StateGraph, END
from rag_pipeline import RAGPipeline


class QueryState(TypedDict):
    """State for query workflow"""
    query: str
    session_id: str
    top_k: int
    use_rerank: bool
    temperature: float
    retrieved_docs: list
    answer: str
    sources: list
    context: str


class QueryWorkflow:
    """LangGraph workflow for RAG query"""
    
    def __init__(self):
        self.rag_pipeline = RAGPipeline()
        self.workflow = self._build_workflow()
    
    async def _retrieve(self, state: QueryState) -> QueryState:
        """Retrieve documents based on query"""
        result = await self.rag_pipeline.query(
            query=state["query"],
            session_id=state["session_id"],
            use_rerank=state["use_rerank"],
            top_k=state["top_k"],
            temperature=state["temperature"]
        )
        
        state["answer"] = result["answer"]
        state["sources"] = result["sources"]
        state["context"] = result["context"]
        state["retrieved_docs"] = result["sources"]
        
        return state
    
    def _generate(self, state: QueryState) -> QueryState:
        """Generate answer from retrieved documents"""
        # Answer is already generated in retrieve step
        return state
    
    def _build_workflow(self) -> StateGraph:
        """Build the query workflow graph"""
        workflow = StateGraph(QueryState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        # Add edges
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    async def run(
        self,
        query: str,
        session_id: str = "default",
        top_k: int = 5,
        use_rerank: bool = True,
        temperature: float = 0.3
    ) -> dict:
        """Run the query workflow"""
        initial_state = {
            "query": query,
            "session_id": session_id,
            "top_k": top_k,
            "use_rerank": use_rerank,
            "temperature": temperature,
            "retrieved_docs": [],
            "answer": "",
            "sources": [],
            "context": ""
        }
        
        result = await self.workflow.ainvoke(initial_state)
        
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "context": result["context"],
            "session_id": result["session_id"]
        }


if __name__ == "__main__":
    # Test the workflow
    workflow = QueryWorkflow()
    
    import asyncio
    result = asyncio.run(workflow.run(
        query="什么是RAG?",
        session_id="test"
    ))
    
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['sources'])}")
