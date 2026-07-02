"""
Conformal Prediction: 
    P(Y_true ∈ C(X)) ≥ 1 - α

α 0.05"""
import numpy as np
from typing import List, Tuple


class ConformalPredictor:
    def __init__(self, alpha: float = 0.05):
        """
        alpha: 0.05 = 5%        """
        self.alpha = alpha
        self.threshold = 0.5
        self.calibration_scores = []
        
    def calibrate(self, val_data: List[Tuple[float, int]]):
        """
                
        val_data: [(uncertainty_score, is_correct), ...]
        
        Split Conformal Prediction         """
        scores = []
        for uncertainty, is_correct in val_data:
            if is_correct:
                scores.append(uncertainty)
            else:
                scores.append(1 - uncertainty)
        
        scores.sort()
        n = len(scores)
        
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_index = int(q_level * n) - 1
        q_index = max(0, min(q_index, n - 1))
        
        self.threshold = scores[q_index]
        self.calibration_scores = scores
        
        print(f"Calibrated threshold: {self.threshold:.4f}")
        print(f"Expected coverage: {1 - self.alpha:.1%}")
        
        return self.threshold
    
    def should_offload(self, uncertainty: float) -> bool:
        return uncertainty > self.threshold
    
    def get_coverage_guarantee(self) -> str:
        return f"$\\Pr(Y_{{true}} \\in C(X)) \\geq {1 - self.alpha:.0%}$"


if __name__ == "__main__":
    predictor = ConformalPredictor(alpha=0.05)
    
    import random
    val_data = [(random.random(), random.choice([0, 1])) for _ in range(100)]
    
    predictor.calibrate(val_data)
    
    test_uncertainty = 0.3
    decision = "cloud" if predictor.should_offload(test_uncertainty) else "edge"
    print(f"Uncertainty {test_uncertainty:.2f} -> {decision}")
