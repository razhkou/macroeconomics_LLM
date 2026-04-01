from typing import List, Any


class FinancialMarket:
    def __init__(self, interest_rate: float = 0.005):
        self.interest_rate = interest_rate

    def reset(self):
        pass

    def accrue_interest(self, agents: List[Any]):
        for agent in agents:
            if hasattr(agent, 'state') and hasattr(agent.state, 'savings'):
                agent.state.savings *= (1 + self.interest_rate)