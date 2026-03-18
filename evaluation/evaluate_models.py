"""
Evaluation Module

Provides utilities to evaluate RAG and fine-tuned models on benchmark tasks.
"""

import csv
from typing import Dict, List, Tuple
import numpy as np


class ModelEvaluator:
    """
    Evaluates model performance on QA tasks using common metrics.
    """
    
    def __init__(self, output_file: str = "evaluation/results.csv"):
        self.output_file = output_file
        self.results = []
    
    def compute_exact_match(self, prediction: str, ground_truth: str) -> bool:
        """
        Check if prediction exactly matches ground truth (after normalization).
        """
        pred_normalized = prediction.strip().lower()
        truth_normalized = ground_truth.strip().lower()
        return pred_normalized == truth_normalized
    
    def compute_f1_score(self, prediction: str, ground_truth: str) -> float:
        """
        Compute token-level F1 score between prediction and ground truth.
        """
        pred_tokens = set(prediction.lower().split())
        truth_tokens = set(ground_truth.lower().split())
        
        common = pred_tokens & truth_tokens
        
        if len(common) == 0:
            return 0.0
        
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(truth_tokens) if truth_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def evaluate_batch(
        self,
        predictions: List[str],
        ground_truths: List[str],
        model_name: str,
        task_name: str,
    ) -> Dict[str, float]:
        """
        Evaluate a batch of predictions.
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length.")
        
        em_scores = []
        f1_scores = []
        
        for pred, truth in zip(predictions, ground_truths):
            em = self.compute_exact_match(pred, truth)
            f1 = self.compute_f1_score(pred, truth)
            
            em_scores.append(float(em))
            f1_scores.append(f1)
        
        metrics = {
            "model": model_name,
            "task": task_name,
            "exact_match": np.mean(em_scores),
            "f1_score": np.mean(f1_scores),
            "num_samples": len(predictions),
        }
        
        self.results.append(metrics)
        return metrics
    
    def save_results(self):
        """
        Save evaluation results to CSV file.
        """
        if not self.results:
            print("[!] No results to save.")
            return
        
        keys = self.results[0].keys()
        
        with open(self.output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"[+] Results saved to {self.output_file}")
    
    def print_results(self):
        """
        Print evaluation results in a readable format.
        """
        if not self.results:
            print("[!] No results to display.")
            return
        
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        for result in self.results:
            print(f"\nModel: {result['model']} | Task: {result['task']}")
            print(f"  Exact Match (EM): {result['exact_match']:.4f}")
            print(f"  F1 Score:         {result['f1_score']:.4f}")
            print(f"  Samples:          {result['num_samples']}")
        
        print("\n" + "="*80 + "\n")


def evaluate_rag_vs_finetuned(
    rag_predictions: List[str],
    finetuned_predictions: List[str],
    ground_truths: List[str],
):
    """
    Compare RAG and fine-tuned model performance.
    """
    evaluator = ModelEvaluator()
    
    # Evaluate RAG
    rag_metrics = evaluator.evaluate_batch(
        rag_predictions,
        ground_truths,
        model_name="RAG",
        task_name="Question Answering"
    )
    
    # Evaluate Fine-tuned
    ft_metrics = evaluator.evaluate_batch(
        finetuned_predictions,
        ground_truths,
        model_name="Fine-Tuned",
        task_name="Question Answering"
    )
    
    # Print results
    evaluator.print_results()
    evaluator.save_results()
    
    # Print comparison
    print("COMPARISON SUMMARY")
    print("-" * 40)
    print(f"RAG EM:        {rag_metrics['exact_match']:.4f}")
    print(f"Fine-tuned EM: {ft_metrics['exact_match']:.4f}")
    print(f"Difference:    {abs(rag_metrics['exact_match'] - ft_metrics['exact_match']):.4f}")
    print()
    print(f"RAG F1:        {rag_metrics['f1_score']:.4f}")
    print(f"Fine-tuned F1: {ft_metrics['f1_score']:.4f}")
    print(f"Difference:    {abs(rag_metrics['f1_score'] - ft_metrics['f1_score']):.4f}")


if __name__ == "__main__":
    # Example usage
    rag_preds = ["The capital of France is Paris", "Machine learning is a subset of AI"]
    ft_preds = ["Paris is the capital of France", "Machine learning is part of AI"]
    truths = ["Paris is the capital of France", "Machine learning is a subset of AI"]
    
    evaluate_rag_vs_finetuned(rag_preds, ft_preds, truths)