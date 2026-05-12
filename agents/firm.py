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
Ты — менеджер фирмы в отрасли «{self.industry}». Твоя цель — максимизировать долгосрочную прибыль компании.

Каждый месяц ты определяешь следующие параметры:
- production_qty (объём производства) — целое неотрицательное число, не превышающее производственную мощность.
- price (цена единицы продукции) — положительное число.
- vacancies (изменение количества сотрудников): положительное — нанять, отрицательное — уволить, 0 — без изменений.
- purchase_plan (доля от потребности в сырье для производства, которую ты хочешь закупить) — число от 0.0 до 1.0.
- invest_share (доля прибыли, направляемая на расширение мощности) — число от 0.0 до 1.0.

Твоё текущее состояние:
- Наличность: {self.cash:.0f}.
- Запасы готовой продукции: {self.inventory:.0f}.
- Количество сотрудников: {len(self.employees)} (максимально возможное — {self.max_workers}).
- Производственная мощность: {self.max_capacity} единиц в месяц.
- Текущая цена: {self.price:.0f}.

Макроэкономические показатели:
- Инфляция: {env.get('inflation', 0)*100:.1f}%.
- Уровень безработицы: {env.get('unemployment', 0)*100:.0f}%.
- Средняя зарплата в экономике: {env.get('avg_wage', 0):.0f} (это ориентир для оплаты твоих будущих сотрудников).
- Налог на прибыль: {env.get('profit_tax', 0.20)*100:.0f}%.
- Кредитная ставка: {env.get('loan_rate', 0.1)*100:.1f}%.

Принимай решения, руководствуясь рыночной логикой:
- Производство: если запасы велики (превышают среднемесячные продажи), сокращай производство; если запасы тают, наращивай, но не превышай мощность.
- Цена: при высоких запасах — понижай цену для стимулирования сбыта; при дефиците — повышай. Учитывай инфляцию: в стабильной экономике цены могут индексироваться на её уровень.
- Найм/увольнения: нанимай, если мощности недозагружены и есть кадровый резерв (безработица не слишком низкая). Увольняй при устойчиво низком спросе (большие запасы, падение продаж).
- Закупки сырья: обеспечивай сырьём плановый объём производства; purchase_plan = 1.0 соответствует полной потребности, меньшее значение — риск нехватки в будущем.
- Инвестиции: направляй часть прибыли на расширение мощности, если ты работаешь на пределе возможностей (высокий спрос, нулевые запасы).

Ответь строго валидным JSON объектом с полями:
"production_qty", "price", "vacancies", "purchase_plan", "invest_share".
Никакого дополнительного текста, только JSON.
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
