"""
Risk Budget Manager: 
Risk Budget100 5 0 1 conservative mode
"""
from typing import Tuple, Dict, List


class RiskBudgetManager:
    def __init__(self, budget: int = 5, period: int = 100):
        """
        budget:         period:         """
        self.budget = budget
        self.period = period
        
        self.current_risk = 0.0
        self.period_count = 0
        self.conservative_mode = False
        
        self.budget_history: List[Dict] = []
        self.mode_switches = 0
        
    def allocate_risk(self, uncertainty: float, is_edge_decision: bool) -> Tuple[float, str]:
        """
                
        Returns:
            (risk_cost, mode):         """
        self.period_count += 1
        
        if self.period_count > self.period:
            self._reset_period()
        
        if self.conservative_mode:
            return 0.0, "conservative"
        
        if is_edge_decision:
            risk_cost = uncertainty
            self.current_risk += risk_cost
            
            if self.current_risk >= self.budget:
                self.conservative_mode = True
                self.mode_switches += 1
                return risk_cost, "switched_to_conservative"
        else:
            risk_cost = 0.0
        
        self.budget_history.append({
            "period_count": self.period_count,
            "current_risk": self.current_risk,
            "remaining_budget": self.budget - self.current_risk
        })
        
        return risk_cost, "normal"
    
    def _reset_period(self):
        self.period_count = 0
        self.current_risk = 0.0
        self.conservative_mode = False
        
    def get_remaining_budget(self) -> float:
        return max(0, self.budget - self.current_risk)
    
    def should_force_cloud(self) -> bool:
        return self.conservative_mode
    
    def get_stats(self) -> Dict:
        return {
            "total_mode_switches": self.mode_switches,
            "budget_utilization": self.current_risk / self.budget if self.budget > 0 else 0,
            "periods_completed": self.period_count // self.period if self.period > 0 else 0,
            "current_risk": self.current_risk,
            "remaining_budget": self.get_remaining_budget()
        }
    
    def reset(self):
        self.current_risk = 0.0
        self.period_count = 0
        self.conservative_mode = False
        self.budget_history = []
        self.mode_switches = 0


if __name__ == "__main__":
    manager = RiskBudgetManager(budget=5, period=100)
    
    import random
    for i in range(150):
        uncertainty = random.random()
        is_edge = random.choice([True, False])
        cost, mode = manager.allocate_risk(uncertainty, is_edge)
        if mode == "switched_to_conservative":
            print(f"Switched to conservative mode at request {i}")
    
    stats = manager.get_stats()
    print(f"Final stats: {stats}")
