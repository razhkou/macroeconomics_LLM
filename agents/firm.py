import json
import logging
from typing import Any, Dict, List, Optional

import litellm
import pydantic
from shachi.agent import Agent
from shachi.tool import tool

logger = logging.getLogger(__name__)


class FirmState(pydantic.BaseModel):
    name: str
    industry: str
    cash: float
    inventory: Dict[str, float]   # запасы готовой продукции и материалов (по типам)
    employees: int
    wage_bill: float
    price: float
    production: float
    profit: float


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
            inventory={config['output_good']: 0, **{k: 0 for k in config.get('inputs', {}).keys()}},  # запасы входов
            employees=0,
            wage_bill=0,
            price=config['initial_price'],
            production=0,
            profit=0,
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

    @tool
    async def decide_production_price(self, observation: Dict[str, Any]) -> Dict[str, float]:
        prompt = self._build_production_prompt(observation)
        try:
            kwargs = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": self.temperature}
            if self.api_base:
                kwargs["api_base"] = self.api_base
            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content
            data = json.loads(content)
            production = float(data.get("production", 0))
            price = float(data.get("price", self.state.price))
            # Ограничения
            production = max(0, min(production, self.config.get('production_capacity', 1000)))
            price = max(0.01, price)
        except Exception as e:
            logger.error(f"Error in decide_production_price for {self.id}: {e}")
            production = 0
            price = self.state.price
        return {"production": production, "price": price}

    @tool
    async def decide_hiring(self, observation: Dict[str, Any]) -> Dict[str, int]:
        prompt = self._build_hiring_prompt(observation)
        try:
            kwargs = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": self.temperature}
            if self.api_base:
                kwargs["api_base"] = self.api_base
            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content
            data = json.loads(content)
            vacancies = int(data.get("vacancies", 0))
            max_possible = self.config.get('max_employees', 500) - self.state.employees
            vacancies = max(0, min(vacancies, max_possible))
        except Exception as e:
            logger.error(f"Error in decide_hiring for {self.id}: {e}")
            vacancies = 0
        return {"vacancies": vacancies}

    @tool
    async def decide_purchases(self, observation: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Решает, какие входные товары купить и в каком количестве."""
        prompt = self._build_purchase_prompt(observation)
        try:
            kwargs = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": self.temperature}
            if self.api_base:
                kwargs["api_base"] = self.api_base
            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content
            data = json.loads(content)
            purchases = data.get("purchases", {})
            # Ограничиваем разумными количествами (например, не более cash / цена)
            return {"purchases": purchases}
        except Exception as e:
            logger.error(f"Error in decide_purchases for {self.id}: {e}")
            return {"purchases": {}}

    def _build_production_prompt(self, obs: Dict[str, Any]) -> str:
        macro = obs.get('macro', {})
        inventory_str = {k: f"{v:.2f}" for k, v in self.state.inventory.items()}
        return f"""
Вы – фирма "{self.state.name}" в отрасли {self.state.industry}.
Текущая цена: {self.state.price:.2f}
Запасы: {inventory_str}
Денежные средства: {self.state.cash:.2f}
Количество сотрудников: {self.state.employees}
Средняя зарплата: {self.state.wage_bill / max(1, self.state.employees):.2f}
Инфляция: {macro.get('inflation', 0):.2%}
Безработица: {macro.get('unemployment', 0):.2%}
Средняя зарплата по городу: {macro.get('avg_wage', 0):.2f}
Цены на товары: {macro.get('prices', {})}

Примите решение:
1. Какой объём производства (единиц товара) вы хотите произвести в этом месяце (не более {self.config.get('production_capacity', 1000)})?
2. Какую цену за единицу товара вы установите?

Ответ в формате JSON: {{"production": число, "price": число}}.
"""

    def _build_hiring_prompt(self, obs: Dict[str, Any]) -> str:
        macro = obs.get('macro', {})
        return f"""
Вы – фирма "{self.state.name}" в отрасли {self.state.industry}.
Текущая цена: {self.state.price:.2f}
Денежные средства: {self.state.cash:.2f}
Количество сотрудников: {self.state.employees} (максимум {self.config.get('max_employees', 500)})
Средняя зарплата: {self.state.wage_bill / max(1, self.state.employees):.2f}
Инфляция: {macro.get('inflation', 0):.2%}
Безработица: {macro.get('unemployment', 0):.2%}
Средняя зарплата по городу: {macro.get('avg_wage', 0):.2f}

Сколько новых сотрудников вы хотите нанять в этом месяце (целое число, не более {self.config.get('max_employees', 500) - self.state.employees})?

Ответ в формате JSON с полем: vacancies.
"""

    def _build_purchase_prompt(self, obs: Dict[str, Any]) -> str:
        macro = obs.get('macro', {})
        inputs = self.config.get('inputs', {})
        if not inputs:
            return "{}"  # нет входов
        prompt = f"""
Вы – фирма "{self.state.name}" в отрасли {self.state.industry}.
Для производства вам нужны следующие входные товары: {inputs}
Текущие запасы: {self.state.inventory}
Денежные средства: {self.state.cash:.2f}
Цены на товары: {macro.get('prices', {})}

Сколько единиц каждого товара вы хотите закупить (целые числа, не более чем позволяет cash и разумные потребности)?
Ответ в формате JSON: {{"purchases": {{"товар1": количество, "товар2": количество}} }}.
"""
        return prompt

    async def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        self._update_state_from_observation(observation)
        prod_price = await self.decide_production_price(observation)
        hiring = await self.decide_hiring(observation)
        purchase_decision = await self.decide_purchases(observation)
        return {
            "agent_id": self.id,
            "type": "firm",
            "production": prod_price["production"],
            "price": prod_price["price"],
            "vacancies": hiring["vacancies"],
            "purchases": purchase_decision["purchases"],
        }

    def _update_state_from_observation(self, obs: Dict[str, Any]):
        state = obs.get('state', {})
        if state:
            for k, v in state.items():
                if hasattr(self.state, k):
                    setattr(self.state, k, v)