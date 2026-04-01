import json
import logging
from typing import Any, Dict, List, Optional

import litellm
import pydantic
from shachi.agent import Agent
from shachi.tool import tool

logger = logging.getLogger(__name__)


class GovernmentState(pydantic.BaseModel):
    budget: float
    tax_income_rate: float          # подоходный налог (ставка)
    tax_profit_rate: float           # налог на прибыль
    key_interest_rate: float         # ключевая ставка (устанавливается правительством)
    transfers: Dict[str, float]      # пособия по архетипам


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
            key_interest_rate=config.get('key_interest_rate', 0.01),  # 1% месячная
            transfers=config.get('transfers', {})  # например, {"poor_single": 200, "poor_family": 400}
        )
        self.memory = []

    async def decide_policy(self, observation: Dict[str, Any]) -> Dict[str, float]:
        """LLM принимает решения о ключевой ставке и налоговых ставках."""
        prompt = self._build_policy_prompt(observation)
        try:
            kwargs = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": self.temperature}
            if self.api_base:
                kwargs["api_base"] = self.api_base
            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content
            data = json.loads(content)
            # Ожидаем поля key_interest_rate, tax_income_rate, tax_profit_rate
            new_key_rate = float(data.get("key_interest_rate", self.state.key_interest_rate))
            new_tax_income = float(data.get("tax_income_rate", self.state.tax_income_rate))
            new_tax_profit = float(data.get("tax_profit_rate", self.state.tax_profit_rate))
            # Ограничиваем разумными пределами
            new_key_rate = max(0.0, min(0.1, new_key_rate))
            new_tax_income = max(0.0, min(0.5, new_tax_income))
            new_tax_profit = max(0.0, min(0.5, new_tax_profit))
        except Exception as e:
            logger.error(f"Error in decide_policy: {e}")
            new_key_rate = self.state.key_interest_rate
            new_tax_income = self.state.tax_income_rate
            new_tax_profit = self.state.tax_profit_rate

        return {
            "key_interest_rate": new_key_rate,
            "tax_income_rate": new_tax_income,
            "tax_profit_rate": new_tax_profit,
        }

    def _build_policy_prompt(self, obs: Dict[str, Any]) -> str:
        macro = obs.get('macro', {})
        return f"""
Вы – правительство города. Ваша цель – поддерживать стабильную экономику (низкая инфляция, низкая безработица, устойчивый рост).
Текущие показатели:
- Инфляция: {macro.get('inflation', 0):.2%}
- Безработица: {macro.get('unemployment', 0):.2%}
- ВВП: {macro.get('gdp', 0):.2f}
- Средняя зарплата: {macro.get('avg_wage', 0):.2f}
- Текущая ключевая ставка: {self.state.key_interest_rate:.2%}
- Текущий подоходный налог: {self.state.tax_income_rate:.2%}
- Текущий налог на прибыль: {self.state.tax_profit_rate:.2%}
- Бюджет: {self.state.budget:.2f}

Примите решение:
1. Какую ключевую ставку установить (месячную, от 0 до 10%)?
2. Какую ставку подоходного налога установить (от 0 до 50%)?
3. Какую ставку налога на прибыль установить (от 0 до 50%)?

Ответ в формате JSON: {{"key_interest_rate": число, "tax_income_rate": число, "tax_profit_rate": число}}.
"""

    async def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        policy = await self.decide_policy(observation)
        # Обновляем ставки
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
        """Сбор налогов по текущим ставкам."""
        income_tax = 0
        for hh in households:
            tax = hh.state.income * self.state.tax_income_rate
            income_tax += tax
            hh.state.income -= tax   # доход после налога
            # Налог уже вычтен из дохода, который пойдёт на потребление
        profit_tax = 0
        for firm in firms:
            # Прибыль рассчитывается как cash - (начальный cash + займы)? Упростим: profit = revenue - costs
            # У нас нет явного revenue, но можно вычислить как выручку от продаж (сделки) - затраты на зарплату и материалы.
            # В текущей реализации прибыль не хранится, поэтому пока пропустим налог на прибыль.
            # Для простоты можно добавить в FirmState поле profit и обновлять его при каждой сделке.
            pass
        self.state.budget += income_tax + profit_tax

    def pay_transfers(self, households: List[Agent]):
        """Выплата трансфертов (пенсии, пособия) бедным группам."""
        for hh in households:
            archetype = hh.config['name']
            transfer = self.state.transfers.get(archetype, 0)
            if transfer > 0:
                hh.state.savings += transfer
                self.state.budget -= transfer