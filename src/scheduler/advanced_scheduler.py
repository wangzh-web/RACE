"""
Advanced Scheduler: Online Calibration + Risk Budget + Semantic Entropy

1. Semantic Entropy (Nature 2024) - 2. Online Conformal Calibration - 3. Risk Budget Manager - """
from typing import Dict, Optional
from ..router.online_conformal import OnlineConformalPredictor
from ..router.semantic_entropy import SemanticEntropyRouter
from .risk_budget import RiskBudgetManager


class AdvancedTrustworthyScheduler:
    def __init__(self, edge_worker, cloud_worker, 
                 alpha: float = 0.05, 
                 risk_budget: int = 5,
                 use_semantic_entropy: bool = True):
        """
        edge_worker:         cloud_worker:         alpha:         risk_budget:         use_semantic_entropy: Semantic Entropy (Nature 2024)
        """
        self.edge_worker = edge_worker
        self.cloud_worker = cloud_worker
        self.use_semantic_entropy = use_semantic_entropy
        
        if use_semantic_entropy:
            self.se_router = SemanticEntropyRouter()
        else:
            self.se_router = None
        
        self.conformal = OnlineConformalPredictor(alpha=alpha)
        
        self.risk_manager = RiskBudgetManager(budget=risk_budget)
        
        self.stats = {
            "total": 0, 
            "edge": 0, 
            "cloud": 0, 
            "forced_cloud": 0,
            "cost": 0.0, 
            "correct": 0,
            "latency": 0.0
        }
    
    def schedule(self, question: str, choices: list, ground_truth: str) -> Dict:
        self.stats["total"] += 1
        
        if self.use_semantic_entropy:
            uncertainty, responses = self.se_router.predict_uncertainty(
                question, self.edge_worker, n_samples=3
            )
        else:
            uncertainty = 0.5
        
        if self.risk_manager.should_force_cloud():
            decision = "cloud"
            result = self.cloud_worker.answer_mcq(question, choices, ground_truth)
            self.stats["forced_cloud"] += 1
            self.stats["cloud"] += 1
        elif self.conformal.should_offload(uncertainty):
            decision = "cloud"
            result = self.cloud_worker.answer_mcq(question, choices, ground_truth)
            self.stats["cloud"] += 1
        else:
            decision = "edge"
            result = self.edge_worker.answer_mcq(question, choices)
            self.stats["edge"] += 1
        
        is_correct = result["parsed_answer"] == ground_truth
        if is_correct:
            self.stats["correct"] += 1
        
        self.conformal.update_and_calibrate(uncertainty, int(is_correct))
        
        self.risk_manager.allocate_risk(uncertainty, decision == "edge")
        
        self.stats["cost"] += result["cost"]
        self.stats["latency"] += result.get("latency", 0)
        
        return {
            "decision": decision,
            "answer": result["parsed_answer"],
            "is_correct": is_correct,
            "cost": result["cost"],
            "uncertainty": uncertainty,
            "threshold": self.conformal.threshold,
            "remaining_budget": self.risk_manager.get_remaining_budget()
        }
    
    def get_metrics(self) -> Dict:
        n = self.stats["total"]
        if n == 0:
            return {}
        return {
            "accuracy": self.stats["correct"] / n,
            "avg_cost": self.stats["cost"] / n,
            "total_cost": self.stats["cost"],
            "avg_latency": self.stats["latency"] / n,
            "edge_ratio": self.stats["edge"] / n,
            "cloud_ratio": self.stats["cloud"] / n,
            "forced_cloud_ratio": self.stats["forced_cloud"] / n,
            "adaptation_stats": self.conformal.get_adaptation_stats(),
            "budget_stats": self.risk_manager.get_stats()
        }
    
    def reset(self):
        self.stats = {
            "total": 0, "edge": 0, "cloud": 0, "forced_cloud": 0,
            "cost": 0.0, "correct": 0, "latency": 0.0
        }
        self.conformal.reset()
        self.risk_manager.reset()


if __name__ == "__main__":
    from ..workers.cloud_worker import CloudWorker
    
    class MockEdgeWorker:
        def generate(self, prompt):
            return {"response": "A"}
        def answer_mcq(self, q, c):
            return {"parsed_answer": "A", "cost": 0, "latency": 0.1}
    
    edge = MockEdgeWorker()
    cloud = CloudWorker()
    
    scheduler = AdvancedTrustworthyScheduler(
        edge_worker=edge,
        cloud_worker=cloud,
        alpha=0.05,
        risk_budget=5,
        use_semantic_entropy=False
    )
    
    result = scheduler.schedule(
        "What is 2+2?",
        ["3", "4", "5", "6"],
        "B"
    )
    print(f"Result: {result}")
    print(f"Metrics: {scheduler.get_metrics()}")
