import json
import logging
from typing import Any, Dict, List, Optional

import litellm
import pydantic
from shachi.agent import Agent

logger = logging.getLogger(__name__)


class HouseholdState(pydantic.BaseModel):
    archetype: str
    population: float
    skills: List[float]
    deposit: float
    consumption_basket: Dict[str, float]
    employed: float
    income: float
    wage: float
    saving_rate: float
    loan_amount: float = 0.0
    loan_monthly_payment: float = 0.0
    consumption_budget: float = 0.0  # добавлено


class HouseholdAgent(Agent):
    def __init__(self, agent_id: str, config: dict, model: str, temperature: float = 0.0, api_base: Optional[str] = None):
        super().__init__()
        self.id = agent_id
        self.config = config
        self.model = model
        self.temperature = temperature
        self.api_base = api_base
        self.state = HouseholdState(
            archetype=config['name'],
            population=config['population'],
            skills=config['skills'],
            deposit=config['initial_savings'],
            consumption_basket=config['consumption_basket'],
            employed=1.0,
            income=0.0,
            wage=0.0,
            saving_rate=config.get('saving_rate', 0.1),
        )
        self.memory = []

    async def make_decisions(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_unified_prompt(observation)
        try:
            kwargs = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": self.temperature}
            if self.api_base:
                kwargs["api_base"] = self.api_base
            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content
            data = json.loads(content)

            work_ratio = float(data.get("work_ratio", 0.5))
            desired_wage = float(data.get("desired_wage", self.state.wage or 1000))
            consumption_ratio = float(data.get("consumption_ratio", 0.5))
            need_loan = data.get("need_loan", False)
            loan_amount = float(data.get("loan_amount", 0))

            work_ratio = max(0.0, min(1.0, work_ratio))
            desired_wage = max(0.0, desired_wage)
            consumption_ratio = max(0.0, min(1.0, consumption_ratio))
            loan_amount = max(0, min(loan_amount, 50000))

        except Exception as e:
            logger.error(f"Error in make_decisions for {self.id}: {e}")
            work_ratio = 0.5
            desired_wage = self.state.wage or 1000
            consumption_ratio = 0.5
            need_loan = False
            loan_amount = 0

        return {
            "work_ratio": work_ratio,
            "desired_wage": desired_wage,
            "consumption_ratio": consumption_ratio,
            "loan_request": {"need_loan": need_loan, "amount": loan_amount},
        }

    def _build_unified_prompt(self, obs: Dict[str, Any]) -> str:
        macro = obs.get('macro', {})
        return f"""
Вы – домохозяйство "{self.state.archetype}" (группа из {self.state.population} чел.).
Депозит в банке: {self.state.deposit:.2f}
Доля работающих: {self.state.employed:.2f}
Средняя зарплата: {self.state.wage:.2f}
Доход группы: {self.state.income:.2f}
Норма сбережения: {self.state.saving_rate:.2%}
Инфляция: {macro.get('inflation', 0):.2%}
Безработица: {macro.get('unemployment', 0):.2%}
Средняя зарплата по городу: {macro.get('avg_wage', 0):.2f}
Ставка по кредитам: {macro.get('interest_rate_loan', 0.01):.2%}

Примите решения:
1. Доля работающих (0..1)
2. Желаемая зарплата
3. Доля дохода на потребление (0..1) – остальное автоматически сберегается
4. Нужен ли кредит (true/false) и сумма (до 50000)

Ответ JSON:
{{"work_ratio": число, "desired_wage": число, "consumption_ratio": число, "need_loan": bool, "loan_amount": число}}
"""

    async def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        self._update_state_from_observation(observation)
        decisions = await self.make_decisions(observation)
        return {"agent_id": self.id, "type": "household", **decisions}

    def _update_state_from_observation(self, obs: Dict[str, Any]):
        state = obs.get('state', {})
        if state:
            for k, v in state.items():
                if hasattr(self.state, k):
                    setattr(self.state, k, v)