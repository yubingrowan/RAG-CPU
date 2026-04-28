"""
LangGraph Index Workflow
Document indexing workflow using LangGraph for orchestration
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from indexing_pipeline import IndexingPipeline
import json
import asyncio


class IndexState(TypedDict):
    """State for index workflow"""
    documents: list
    indexed_count: int
    status: str
    error: str


class IndexWorkflow:
    """LangGraph workflow for document indexing"""
    
    def __init__(self):
        self.indexing_pipeline = IndexingPipeline()
        self.workflow = self._build_workflow()
    
    def _validate(self, state: IndexState) -> IndexState:
        """Validate input documents"""
        if not state["documents"]:
            state["status"] = "failed"
            state["error"] = "No documents provided"
            return state
        
        for doc in state["documents"]:
            if not isinstance(doc, dict) or "text" not in doc:
                state["status"] = "failed"
                state["error"] = "Invalid document format"
                return state
        
        state["status"] = "validated"
        return state
    
    async def _index(self, state: IndexState) -> IndexState:
        """Index documents to vector DB and BM25"""
        try:
            # Run async indexing
            result = await self.indexing_pipeline.index_documents(state["documents"])
            
            state["indexed_count"] = result["total_chunks"]
            state["status"] = "completed"
            
        except Exception as e:
            state["status"] = "failed"
            state["error"] = str(e)
        
        return state
    
    def _build_workflow(self) -> StateGraph:
        """Build the index workflow graph"""
        workflow = StateGraph(IndexState)
        
        # Add nodes
        workflow.add_node("validate", self._validate)
        workflow.add_node("index", self._index)
        
        # Set entry point
        workflow.set_entry_point("validate")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "validate",
            lambda x: "index" if x["status"] == "validated" else END,
            {
                "index": "index",
                END: END
            }
        )
        
        workflow.add_edge("index", END)
        
        return workflow.compile()
    
    async def run(self, documents: list) -> dict:
        """Run the index workflow"""
        initial_state = {
            "documents": documents,
            "indexed_count": 0,
            "status": "pending",
            "error": ""
        }
        
        result = await self.workflow.ainvoke(initial_state)
        
        return {
            "status": result["status"],
            "indexed_count": result["indexed_count"],
            "error": result.get("error", "")
        }


if __name__ == "__main__":
    # Test the workflow
    workflow = IndexWorkflow()
    
    test_docs = [
        {
            'text': 'This is a test document for LangGraph indexing workflow.',
            'metadata': {'source': 'test', 'type': 'test_doc'}
        }
    ]
    
    result = asyncio.run(workflow.run(test_docs))
    
    print(f"Status: {result['status']}")
    print(f"Indexed: {result['indexed_count']} chunks")
