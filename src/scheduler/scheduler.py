"""
Scheduler: Conformal Prediction """
import numpy as np
from typing import Dict, List

# Base module for scheduler package
class TrustworthyScheduler:
    def __init__(self, *args, **kwargs): pass
    def schedule(self, *args, **kwargs): pass
    def get_metrics(self): return {}

class RECO:
    
    def __init__(self, alpha=0.1, window_size=100, eta_0=0.5):
        self.alpha = alpha
        self.window_size = window_size
        self.eta_0 = eta_0
        self.theta = 0.0
        self.risk_budget = 0.0
        self.t = 0
        self.history = []
    
    @property
    def threshold(self):
        """Current threshold (Sigmoid)"""
        return 1.0 / (1.0 + np.exp(-self.theta))
    
    def route(self, uncertainty):
        self.t += 1
        
        # Conservative Mode: if budget depleted
        if self.risk_budget >= self.t * self.alpha:
            return 1 # Cloud
        
        if uncertainty > self.threshold:
            return 1 # Cloud
        else:
            self.risk_budget += uncertainty # Consume budget
            return 0 # Edge
    
    def update(self, loss):
        """Update based on feedback"""
        self.history.append(loss)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # Online Mirror Descent update
        eta_t = self.eta_0 / np.sqrt(self.t)
        avg_loss = np.mean(self.history) if self.history else 0
        
        # Correct direction:
        # Loss > Alpha -> Need safer (Cloud) -> Lower Threshold -> Lower Theta
        self.theta += eta_t * (self.alpha - avg_loss)


class StaticCP:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    def route(self, uncertainty):
        return 1 if uncertainty > self.threshold else 0
    def update(self, loss): pass

class ACI:
    def __init__(self, alpha=0.1, gamma=0.05):
        self.alpha = alpha
        self.gamma = gamma
        self.threshold = alpha # Start with threshold = alpha
    
    def route(self, uncertainty):
        return 1 if uncertainty > self.threshold else 0
    
    def update(self, loss):
        self.threshold += self.gamma * (self.alpha - loss)
        self.threshold = np.clip(self.threshold, 0.01, 0.99)
