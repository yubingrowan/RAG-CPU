#!/usr/bin/env python3
"""
Simple RAG Evaluation Script (No external APIs required)
Evaluates RAG system using basic metrics
"""

import sys
import json
from pathlib import Path
from typing import List, Dict
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def calculate_overlap(text1: str, text2: str) -> float:
    """
    Calculate keyword overlap between two texts
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Overlap ratio (0-1)
    """
    # Tokenize and normalize
    tokens1 = set(re.findall(r'\w+', text1.lower()))
    tokens2 = set(re.findall(r'\w+', text2.lower()))
    
    if not tokens1 or not tokens2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    return intersection / union if union > 0 else 0.0


def calculate_context_relevance(question: str, contexts: List[str]) -> float:
    """
    Calculate context relevance by measuring overlap between question and contexts
    
    Args:
        question: User question
        contexts: List of retrieved contexts
        
    Returns:
        Average relevance score (0-1)
    """
    if not contexts:
        return 0.0
    
    relevance_scores = []
    for context in contexts:
        overlap = calculate_overlap(question, context)
        relevance_scores.append(overlap)
    
    return sum(relevance_scores) / len(relevance_scores)


def calculate_answer_relevance(answer: str, ground_truth: str) -> float:
    """
    Calculate answer relevance by measuring overlap with ground truth
    
    Args:
        answer: Generated answer
        ground_truth: Expected answer
        
    Returns:
        Relevance score (0-1)
    """
    return calculate_overlap(answer, ground_truth)


def calculate_context_coverage(ground_truth: str, contexts: List[str]) -> float:
    """
    Calculate how well contexts cover ground truth information
    
    Args:
        ground_truth: Expected answer
        contexts: List of retrieved contexts
        
    Returns:
        Coverage score (0-1)
    """
    if not contexts:
        return 0.0
    
    # Combine all contexts
    combined_context = " ".join(contexts)
    
    return calculate_overlap(ground_truth, combined_context)


def evaluate_results(results: Dict) -> Dict:
    """
    Evaluate RAG results using basic metrics
    
    Args:
        results: Dictionary with questions, contexts, answers, ground_truth
        
    Returns:
        Evaluation metrics
    """
    questions = results["question"]
    contexts_list = results["contexts"]
    answers = results["answer"]
    ground_truths = results["ground_truth"]
    
    metrics = {
        "context_relevance": [],
        "answer_relevance": [],
        "context_coverage": [],
        "num_contexts": []
    }
    
    for i in range(len(questions)):
        question = questions[i]
        contexts = contexts_list[i]
        answer = answers[i]
        ground_truth = ground_truths[i]
        
        # Calculate metrics
        context_rel = calculate_context_relevance(question, contexts)
        answer_rel = calculate_answer_relevance(answer, ground_truth)
        context_cov = calculate_context_coverage(ground_truth, contexts)
        
        metrics["context_relevance"].append(context_rel)
        metrics["answer_relevance"].append(answer_rel)
        metrics["context_coverage"].append(context_cov)
        metrics["num_contexts"].append(len(contexts))
    
    # Calculate averages
    summary = {
        "avg_context_relevance": sum(metrics["context_relevance"]) / len(metrics["context_relevance"]),
        "avg_answer_relevance": sum(metrics["answer_relevance"]) / len(metrics["answer_relevance"]),
        "avg_context_coverage": sum(metrics["context_coverage"]) / len(metrics["context_coverage"]),
        "avg_num_contexts": sum(metrics["num_contexts"]) / len(metrics["num_contexts"]),
        "num_questions": len(questions)
    }
    
    return {
        "summary": summary,
        "per_question": metrics,
        "details": [
            {
                "question": questions[i],
                "answer": answers[i],
                "ground_truth": ground_truths[i],
                "context_relevance": metrics["context_relevance"][i],
                "answer_relevance": metrics["answer_relevance"][i],
                "context_coverage": metrics["context_coverage"][i],
                "num_contexts": metrics["num_contexts"][i]
            }
            for i in range(len(questions))
        ]
    }


def main():
    """Main evaluation function"""
    print("=== Simple RAG Evaluation ===\n")
    
    # Load raw results
    results_path = project_root / "data" / "eval_results_raw.json"
    print(f"Loading results from {results_path}...")
    
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Evaluate
    print("Evaluating results...")
    evaluation = evaluate_results(results)
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    summary = evaluation["summary"]
    print(f"Number of questions: {summary['num_questions']}")
    print(f"Average context relevance: {summary['avg_context_relevance']:.4f}")
    print(f"Average answer relevance: {summary['avg_answer_relevance']:.4f}")
    print(f"Average context coverage: {summary['avg_context_coverage']:.4f}")
    print(f"Average number of contexts: {summary['avg_num_contexts']:.1f}")
    
    # Print per-question details
    print("\n=== Per-Question Details ===")
    for detail in evaluation["details"]:
        print(f"\nQuestion: {detail['question']}")
        print(f"Context relevance: {detail['context_relevance']:.4f}")
        print(f"Answer relevance: {detail['answer_relevance']:.4f}")
        print(f"Context coverage: {detail['context_coverage']:.4f}")
        print(f"Answer: {detail['answer'][:100]}...")
        print(f"Ground truth: {detail['ground_truth'][:100]}...")
    
    # Save evaluation results
    output_path = project_root / "data" / "evaluation_results_simple.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation results saved to {output_path}")


if __name__ == "__main__":
    main()
