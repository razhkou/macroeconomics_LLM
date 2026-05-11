import json
import logging
from typing import Any, Dict, List, Optional

import litellm
import pydantic
from shachi.agent import Agent

logger = logging.getLogger(__name__)


class FirmState(pydantic.BaseModel):
    name: str
    industry: str
    cash: float
    inventory: Dict[str, float]
    employees: int
    wage_bill: float
    price: float
    production: float
    profit: float
    production_capacity: float
    max_capacity: float
    investment_efficiency: float


class FirmAgent(Agent):
    def __init__(self, agent_id: str, config: dict, model: str, temperature: float = 0.0, api_base: Optional[str] = None):
        super().__init__()
        self.id = agent_id
        self.config = config
        self.model = model
        self.temperature = temperature
        self.api_base = api_base
        self.state = FirmState(
            name=config['name'],
            industry=config['industry'],
            cash=config['initial_cash'],
            inventory={config['output_good']: 0, **{k: 0 for k in config.get('inputs', {}).keys()}},
            employees=0,
            wage_bill=0,
            price=config['initial_price'],
            production=0,
            profit=0,
            production_capacity=config.get('production_capacity', 1000),
            max_capacity=config.get('max_capacity', 2000),
            investment_efficiency=config.get('investment_efficiency', 0.5),
        )
        self.memory = []

    def reset(self):
        self.state.cash = self.config['initial_cash']
        self.state.employees = 0
        self.state.wage_bill = 0
        self.state.price = self.config['initial_price']
        self.state.inventory = {self.config['output_good']: 0, **{k: 0 for k in self.config.get('inputs', {}).keys()}}
        self.state.production = 0
        self.state.profit = 0
        self.state.production_capacity = self.config.get('production_capacity', 1000)

    async def make_decisions(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_unified_prompt(observation)
        try:
            kwargs = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": self.temperature}
            if self.api_base:
                kwargs["api_base"] = self.api_base
            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content
            data = json.loads(content)

            production = float(data.get("production", 0))
            price = float(data.get("price", self.state.price))
            vacancies = int(data.get("vacancies", 0))
            purchases = data.get("purchases", {})
            investment_ratio = float(data.get("investment_ratio", 0.0))
            investment_good = data.get("investment_good", "machinery")

            production = max(0, min(production, self.state.production_capacity))
            price = max(0.01, price)
            max_possible = self.config.get('max_employees', 500) - self.state.employees
            vacancies = max(0, min(vacancies, max_possible))
            investment_ratio = max(0.0, min(1.0, investment_ratio))

        except Exception as e:
            logger.error(f"Error in make_decisions for {self.id}: {e}")
            production = 0
            price = self.state.price
            vacancies = 0
            purchases = {}
            investment_ratio = 0.0
            investment_good = "machinery"

        return {
            "production": production,
            "price": price,
            "vacancies": vacancies,
            "purchases": purchases,
            "investment_ratio": investment_ratio,
            "investment_good": investment_good,
        }

    def _build_unified_prompt(self, obs: Dict[str, Any]) -> str:
        macro = obs.get('macro', {})
        inputs = self.config.get('inputs', {})
        return f"""
Вы – фирма "{self.state.name}" (отрасль {self.state.industry}).
Цена: {self.state.price:.2f}
Запасы: {self.state.inventory}
Денежные средства: {self.state.cash:.2f}
Сотрудников: {self.state.employees} (макс {self.config.get('max_employees', 500)})
Средняя зарплата: {self.state.wage_bill / max(1, self.state.employees):.2f}
Мощность: {self.state.production_capacity:.0f} / макс {self.state.max_capacity:.0f}
Инфляция: {macro.get('inflation', 0):.2%}
Безработица: {macro.get('unemployment', 0):.2%}
Средняя зарплата по городу: {macro.get('avg_wage', 0):.2f}
Цены товаров: {macro.get('prices', {})}
Необходимые ресурсы: {inputs}

Примите решения:
1. Объём производства (не более {self.state.production_capacity:.0f})
2. Цена за единицу
3. Количество новых сотрудников (целое)
4. Закупки материалов (формат {{"товар": количество}})
5. Доля прибыли на инвестиции (0..1) и какой товар покупать (обычно machinery или construction_services)

Ответ JSON:
{{"production": число, "price": число, "vacancies": число, "purchases": {{}}, "investment_ratio": число, "investment_good": "строка"}}
"""

    async def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        self._update_state_from_observation(observation)
        decisions = await self.make_decisions(observation)
        return {"agent_id": self.id, "type": "firm", **decisions}

    def _update_state_from_observation(self, obs: Dict[str, Any]):
        state = obs.get('state', {})
        if state:
            for k, v in state.items():
                if hasattr(self.state, k):
                    setattr(self.state, k, v)