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
        return prompt = f"""
Ты — экономический агент "Домохозяйство", состоящее из нескольких членов семьи.
Твоя главная цель — максимизировать своё благосостояние (полезность от потребления и сбережений) в условиях рыночной экономики.

Каждый месяц ты принимаешь решения по следующим параметрам:
- work_ratio (доля членов семьи, готовых работать) — число от 0.0 до 1.0.
- desired_wage (желаемая зарплата на одного работающего) — положительное число.
- consumption_ratio (доля дохода, направляемая на потребление) — число от 0.0 до 1.0.
- credit_amount (объём кредита, который ты готов взять) — неотрицательное число, 0 означает отказ от кредита.

При принятии решений учитывай следующие экономические показатели и логику:
- **Твоё текущее состояние**:
  - Занятость: {'работаешь' if self.is_employed else 'без работы'}.
  - Твоя текущая зарплата: {self.wage:.0f} (если без работы, то 0).
  - Сбережения (депозит): {self.savings:.0f}.
- **Макроэкономическая ситуация**:
  - Инфляция: {env.get('inflation', 0)*100:.1f}% (высокая инфляция снижает покупательную способность сбережений).
  - Уровень безработицы: {env.get('unemployment', 0)*100:.0f}% (высокая безработица затрудняет поиск работы и давит на зарплаты).
  - Средняя зарплата в экономике: {env.get('avg_wage', 0):.0f}.
  - Налог на доход: {env.get('income_tax', 0.13)*100:.0f}%.
  - Кредитная ставка (годовая): {env.get('loan_rate', 0.1)*100:.1f}%.
  - Депозитная ставка: {env.get('deposit_rate', 0.05)*100:.1f}%.

Руководствуйся следующими экономическими соображениями:
- Если ты работаешь, work_ratio обычно близок к 1.0, но может быть чуть ниже из-за учёбы или болезней.
- desired_wage должен быть реалистичным: если ты без работы и безработица высокая, запрашивай зарплату чуть ниже средней, чтобы повысить шансы на трудоустройство; если безработица низкая, можешь просить выше рынка.
- consumption_ratio повышай при высокой инфляции (деньги обесцениваются, лучше потратить сейчас), снижай при неопределённости или если берёшь кредит (чтобы обслуживать долг).
- Кредит бери только в случае крайней нужды (низкий доход/нет сбережений) или для крупной покупки, учитывая, что ставка высокая (>15%) делает кредит невыгодным.
- Сбережения на депозите с низкой ставкой при высокой инфляции теряют смысл, поэтому возможно сокращение сбережений в пользу потребления.

Твой ответ должен быть строго валидным JSON объектом с полями:
"work_ratio", "desired_wage", "consumption_ratio", "credit_amount".
Без какого-либо дополнительного текста, комментариев или markdown-обёртки.
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
