import json
import logging
from typing import Any, Dict, List, Optional

import litellm
import pydantic
from shachi.agent import Agent

logger = logging.getLogger(__name__)


class GovernmentState(pydantic.BaseModel):
    budget: float
    tax_income_rate: float
    tax_profit_rate: float
    key_interest_rate: float
    transfers: Dict[str, float]


class GovernmentAgent(Agent):
    def __init__(self, config: dict, model: str, temperature: float = 0.0, api_base: Optional[str] = None):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.api_base = api_base
        self.state = GovernmentState(
            budget=config.get('initial_budget', 0),
            tax_income_rate=config.get('tax_income', 0.13),
            tax_profit_rate=config.get('tax_profit', 0.20),
            key_interest_rate=config.get('key_interest_rate', 0.01),
            transfers=config.get('transfers', {}),
        )

    async def decide_policy(self, observation: Dict[str, Any]) -> Dict[str, float]:
        macro = observation.get('macro', {})
        prompt = f"""
Правительство города. Текущие показатели:
Инфляция: {macro.get('inflation', 0):.2%}
Безработица: {macro.get('unemployment', 0):.2%}
ВВП: {macro.get('gdp', 0):.2f}
Средняя зарплата: {macro.get('avg_wage', 0):.2f}
Текущая ключевая ставка: {self.state.key_interest_rate:.2%}
Подоходный налог: {self.state.tax_income_rate:.2%}
Налог на прибыль: {self.state.tax_profit_rate:.2%}
Бюджет: {self.state.budget:.2f}

Примите решение о новой ключевой ставке (0..10%), подоходном налоге (0..50%), налоге на прибыль (0..50%).
Ответ JSON: {{"key_interest_rate": число, "tax_income_rate": число, "tax_profit_rate": число}}
"""
        try:
            kwargs = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": self.temperature}
            if self.api_base:
                kwargs["api_base"] = self.api_base
            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content
            data = json.loads(content)
            new_key = float(data.get("key_interest_rate", self.state.key_interest_rate))
            new_tax_inc = float(data.get("tax_income_rate", self.state.tax_income_rate))
            new_tax_prof = float(data.get("tax_profit_rate", self.state.tax_profit_rate))
            new_key = max(0.0, min(0.1, new_key))
            new_tax_inc = max(0.0, min(0.5, new_tax_inc))
            new_tax_prof = max(0.0, min(0.5, new_tax_prof))
        except Exception as e:
            logger.error(f"Error in decide_policy: {e}")
            new_key = self.state.key_interest_rate
            new_tax_inc = self.state.tax_income_rate
            new_tax_prof = self.state.tax_profit_rate
        return {"key_interest_rate": new_key, "tax_income_rate": new_tax_inc, "tax_profit_rate": new_tax_prof}

    async def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        policy = await self.decide_policy(observation)
        self.state.key_interest_rate = policy["key_interest_rate"]
        self.state.tax_income_rate = policy["tax_income_rate"]
        self.state.tax_profit_rate = policy["tax_profit_rate"]
        return {
            "agent_id": "government",
            "type": "government",
            "key_interest_rate": self.state.key_interest_rate,
            "tax_income_rate": self.state.tax_income_rate,
            "tax_profit_rate": self.state.tax_profit_rate,
        }

    def collect_taxes(self, households: List[Agent], firms: List[Agent]):
        for hh in households:
            tax = hh.state.income * self.state.tax_income_rate
            hh.state.income -= tax
            self.state.budget += tax
        for firm in firms:
            tax = firm.state.profit * self.state.tax_profit_rate
            firm.state.cash -= tax
            self.state.budget += tax

    def pay_transfers(self, households: List[Agent]):
        for hh in households:
            transfer = self.state.transfers.get(hh.config['name'], 0)
            if transfer > 0:
                hh.state.deposit += transfer
                self.state.budget -= transfer