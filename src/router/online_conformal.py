"""
Online Conformal Calibration: distribution shift 
Conformal Prediction N """
import numpy as np
from collections import deque
from typing import Tuple, List, Dict


class OnlineConformalPredictor:
    def __init__(self, alpha: float = 0.05, window_size: int = 100, 
                 recalibrate_interval: int = 20):
        """
        alpha:         window_size:         recalibrate_interval:         """
        self.alpha = alpha
        self.window_size = window_size
        self.recalibrate_interval = recalibrate_interval
        
        self.threshold = 0.5
        self.score_buffer = deque(maxlen=window_size)
        self.request_count = 0
        self.threshold_history = []
        
    def update_and_calibrate(self, uncertainty: float, is_correct: int) -> float:
        """
                
        Returns:
                    """
        score = uncertainty if is_correct else (1 - uncertainty)
        self.score_buffer.append(score)
        self.request_count += 1
        
        if self.request_count % self.recalibrate_interval == 0:
            self._recalibrate()
            
        return self.threshold
    
    def _recalibrate(self):
        if len(self.score_buffer) < 10:
            return
            
        scores = sorted(list(self.score_buffer))
        n = len(scores)
        
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_index = int(q_level * n) - 1
        q_index = max(0, min(q_index, n - 1))
        
        self.threshold = scores[q_index]
        self.threshold_history.append((self.request_count, self.threshold))
        
    def should_offload(self, uncertainty: float) -> bool:
        return uncertainty > self.threshold
    
    def get_adaptation_stats(self) -> Dict:
        return {
            "threshold_history": self.threshold_history,
            "current_threshold": self.threshold,
            "buffer_size": len(self.score_buffer),
            "total_recalibrations": len(self.threshold_history)
        }
    
    def reset(self):
        self.threshold = 0.5
        self.score_buffer.clear()
        self.request_count = 0
        self.threshold_history = []


if __name__ == "__main__":
    predictor = OnlineConformalPredictor(alpha=0.05)
    
    import random
    for i in range(100):
        uncertainty = random.random()
        is_correct = random.choice([0, 1])
        predictor.update_and_calibrate(uncertainty, is_correct)
    
    stats = predictor.get_adaptation_stats()
    print(f"Final threshold: {stats['current_threshold']:.4f}")
    print(f"Total recalibrations: {stats['total_recalibrations']}")
    print(f"Threshold history: {stats['threshold_history'][:5]}...")
