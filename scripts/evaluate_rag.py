#!/usr/bin/env python3
"""
RAG Evaluation Script using RAGAS
Evaluates RAG system quality using RAGAS metrics
"""

import sys
import os
import asyncio
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from rag_pipeline import RAGPipeline
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_correctness
)


async def load_dataset(dataset_path: str = "data/eval_dataset.json"):
    """
    Load evaluation dataset from JSON file
    
    Args:
        dataset_path: Path to evaluation dataset JSON file
        
    Returns:
        Dictionary with questions and ground truth answers
    """
    full_path = project_root / dataset_path
    with open(full_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


async def collect_rag_results(questions: list, rag_pipeline: RAGPipeline):
    """
    Collect RAG results for evaluation
    
    Args:
        questions: List of questions to query
        rag_pipeline: RAGPipeline instance
        
    Returns:
        Dictionary with questions, contexts, answers, and ground truth
    """
    results = {
        "question": [],
        "contexts": [],
        "answer": [],
        "ground_truth": []
    }
    
    print(f"Processing {len(questions)} questions...")
    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Querying: {question}")
        
        # Query RAG system
        response = await rag_pipeline.query(
            query=question,
            session_id=f"eval_{i}",
            use_rerank=True,
            top_k=5,
            temperature=0.3
        )
        
        # Extract contexts from sources
        contexts = [source['text'] for source in response['sources']]
        
        # Store results
        results["question"].append(question)
        results["contexts"].append(contexts)
        results["answer"].append(response['answer'])
        
    return results


async def run_evaluation():
    """
    Run RAG evaluation using RAGAS
    """
    print("=== RAG Evaluation using RAGAS ===\n")
    
    # Load dataset
    print("Loading evaluation dataset...")
    dataset = await load_dataset()
    questions = dataset["questions"]
    ground_truth = dataset["ground_truth"]
    
    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    rag_pipeline = RAGPipeline()
    
    # Collect RAG results
    print("\nCollecting RAG results...")
    rag_results = await collect_rag_results(questions, rag_pipeline)
    rag_results["ground_truth"] = ground_truth
    
    # Create HuggingFace Dataset
    print("\nCreating evaluation dataset...")
    eval_dataset = Dataset.from_dict(rag_results)
    
    # Configure RAGAS metrics
    print("Configuring RAGAS metrics...")
    # Note: Some metrics require LLM API keys (OpenAI, etc.)
    # For now, we'll use metrics that don't require external APIs
    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_correctness
    ]
    
    # Run evaluation
    print("\nRunning RAGAS evaluation...")
    print("Note: Some metrics may require LLM API keys (OpenAI, etc.)")
    print("If evaluation fails, please set OPENAI_API_KEY environment variable\n")
    
    try:
        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics
        )
        
        # Print results
        print("\n=== Evaluation Results ===")
        print(result.to_pandas())
        
        # Save results
        output_path = project_root / "data" / "evaluation_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {output_path}")
        
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        print("\nPossible reasons:")
        print("1. Missing LLM API keys (OPENAI_API_KEY)")
        print("2. Network connectivity issues")
        print("3. RAGAS version compatibility")
        print("\nAlternative: Manual evaluation of collected results")
        print("Results collected in 'data/eval_results_raw.json'")
        
        # Save raw results for manual inspection
        raw_output_path = project_root / "data" / "eval_results_raw.json"
        with open(raw_output_path, 'w', encoding='utf-8') as f:
            json.dump(rag_results, f, indent=2, ensure_ascii=False)
        print(f"Raw results saved to {raw_output_path}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
