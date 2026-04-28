#!/usr/bin/env python3
"""
Ray RAG Scheduler
Distributed RAG system with Ray for parallel processing and load balancing
"""

import os
import ray
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from rag_pipeline import RAGPipeline
from llm import LLMFactory, LLMProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGTask:
    """RAG task definition"""
    task_id: str
    query: str
    session_id: str
    llm_config: Dict[str, Any]
    rag_config: Dict[str, Any]
    priority: int = 1

@ray.remote
class RAGWorker:
    """Distributed RAG worker"""
    
    def __init__(self, worker_id: str, llm_provider: str, llm_config: Dict[str, Any]):
        import sys
        import os
        # Add project path for worker
        project_path = os.path.join(os.path.dirname(__file__), "..", "..")
        if project_path not in sys.path:
            sys.path.insert(0, project_path)
        
        self.worker_id = worker_id
        self.llm_provider = llm_provider
        self.llm_config = llm_config
        self.rag_pipeline = None
        self._init_pipeline()
    
    def _init_pipeline(self):
        """Initialize RAG pipeline"""
        try:
            from src.rag_pipeline import RAGPipeline
            self.rag_pipeline = RAGPipeline(
                llm_provider=self.llm_provider,
                llm_config=self.llm_config
            )
            logger.info(f"Worker {self.worker_id} initialized successfully")
        except Exception as e:
            logger.error(f"Worker {self.worker_id} initialization failed: {e}")
            raise
    
    async def process_task(self, task: RAGTask) -> Dict[str, Any]:
        """Process RAG task"""
        try:
            result = await self.rag_pipeline.query(
                query=task.query,
                session_id=task.session_id,
                **task.rag_config
            )
            
            return {
                "task_id": task.task_id,
                "worker_id": self.worker_id,
                "status": "success",
                "result": result,
                "timestamp": ray.get_runtime_context().current_time
            }
            
        except Exception as e:
            logger.error(f"Worker {self.worker_id} task {task.task_id} failed: {e}")
            return {
                "task_id": task.task_id,
                "worker_id": self.worker_id,
                "status": "error",
                "error": str(e),
                "timestamp": ray.get_runtime_context().current_time
            }
    
    def health_check(self) -> bool:
        """Health check"""
        try:
            return self.rag_pipeline is not None
        except:
            return False

class RayRAGScheduler:
    """Ray RAG scheduler for distributed processing"""
    
    def __init__(self, num_workers: int = 2, llm_provider: str = "openai"):
        self.num_workers = num_workers
        self.llm_provider = llm_provider
        self.workers = []
        self.task_queue = []
        self.results = {}
        self._init_cluster()
        self._init_workers()
    
    def _init_cluster(self):
        """Initialize Ray cluster"""
        try:
            if not ray.is_initialized():
                # Try to connect to existing cluster or start new one
                ray.init(
                    address="auto",
                    log_to_driver=True,
                    ignore_reinit_error=True
                )
            logger.info("Ray cluster initialized successfully")
        except Exception as e:
            logger.error(f"Ray cluster initialization failed: {e}")
            # Try to start a new cluster
            try:
                ray.init(
                    address=None,
                    log_to_driver=True,
                    ignore_reinit_error=True
                )
                logger.info("Ray cluster started successfully")
            except Exception as e2:
                logger.error(f"Ray cluster startup failed: {e2}")
                raise
    
    def _init_workers(self):
        """Initialize RAG workers"""
        try:
            # Default LLM config
            llm_config = {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": "qwen-plus"
            } if self.llm_provider == "openai" else {
                "host": "localhost",
                "port": 8080
            }
            
            # Create workers
            for i in range(self.num_workers):
                worker_id = f"rag-worker-{i+1}"
                worker = RAGWorker.remote(worker_id, self.llm_provider, llm_config)
                self.workers.append(worker)
            
            logger.info(f"Initialized {self.num_workers} RAG workers")
            
        except Exception as e:
            logger.error(f"Worker initialization failed: {e}")
            raise
    
    async def submit_task(self, task: RAGTask) -> str:
        """Submit RAG task"""
        try:
            # Select worker (round-robin)
            worker = self.workers[len(self.task_queue) % len(self.workers)]
            
            # Submit task
            future = worker.process_task.remote(task)
            self.task_queue.append(future)
            
            logger.info(f"Task {task.task_id} submitted to worker")
            return task.task_id
            
        except Exception as e:
            logger.error(f"Task submission failed: {e}")
            raise
    
    async def submit_batch(self, tasks: List[RAGTask]) -> List[str]:
        """Submit batch of RAG tasks"""
        task_ids = []
        
        # Submit all tasks
        for task in tasks:
            task_id = await self.submit_task(task)
            task_ids.append(task_id)
        
        logger.info(f"Submitted batch of {len(tasks)} tasks")
        return task_ids
    
    async def get_results(self, task_ids: List[str], timeout: int = 30) -> Dict[str, Any]:
        """Get task results"""
        try:
            # Wait for results using ray.get() in sync context
            results = []
            for future in self.task_queue:
                try:
                    result = ray.get(future, timeout=timeout)
                    results.append(result)
                except Exception as e:
                    results.append(e)
            
            # Process results
            task_results = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    task_results[task_ids[i]] = {
                        "status": "error",
                        "error": str(result)
                    }
                else:
                    task_results[task_ids[i]] = result
            
            return task_results
            
        except Exception as e:
            logger.error(f"Getting results failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, bool]:
        """Check worker health"""
        try:
            health_results = await asyncio.gather(
                *[worker.health_check.remote() for worker in self.workers],
                return_exceptions=True
            )
            
            worker_health = {}
            for i, health in enumerate(health_results):
                worker_id = f"rag-worker-{i+1}"
                if isinstance(health, Exception):
                    worker_health[worker_id] = False
                else:
                    worker_health[worker_id] = health
            
            return worker_health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {}
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster statistics"""
        try:
            return ray.nodes()
        except Exception as e:
            logger.error(f"Getting cluster stats failed: {e}")
            return {}
    
    def shutdown(self):
        """Shutdown Ray cluster"""
        try:
            ray.shutdown()
            logger.info("Ray cluster shutdown successfully")
        except Exception as e:
            logger.error(f"Ray cluster shutdown failed: {e}")
