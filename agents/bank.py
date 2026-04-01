import logging
from typing import Any, Dict, List, Optional

import pydantic
from shachi.agent import Agent

logger = logging.getLogger(__name__)


class Loan(pydantic.BaseModel):
    borrower_id: str
    amount: float
    interest_rate: float          # месячная ставка
    remaining: float
    monthly_payment: float


class BankState(pydantic.BaseModel):
    cash: float                   # наличные средства (собственный капитал + депозиты)
    deposits: float               # сумма депозитов (обязательства)
    loans: List[Loan]             # выданные кредиты
    interest_rate_deposit: float  # ставка по депозитам (может быть привязана к ключевой)
    interest_rate_loan: float     # ставка по кредитам
    reserve_ratio: float


class BankAgent(Agent):
    def __init__(self, config: dict):
        super().__init__()
        self.state = BankState(
            cash=config.get('bank_cash', 100000),
            deposits=0,
            loans=[],
            interest_rate_deposit=config.get('interest_rate_deposit', 0.005),   # 0.5% в месяц
            interest_rate_loan=config.get('interest_rate_loan', 0.01),          # 1% в месяц
            reserve_ratio=config.get('reserve_ratio', 0.1),
        )

    def reset(self):
        self.state.cash = 100000
        self.state.deposits = 0
        self.state.loans = []

    async def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        # Банк может использовать ключевую ставку от правительства
        macro = observation.get('macro', {})
        key_rate = macro.get('key_interest_rate', 0.01)
        # Простая привязка: депозитная ставка = ключевая - 0.5%, кредитная = ключевая + 1%
        self.state.interest_rate_deposit = max(0.0, key_rate - 0.005)
        self.state.interest_rate_loan = key_rate + 0.01
        return {
            "agent_id": "bank",
            "type": "bank",
            "interest_rate_deposit": self.state.interest_rate_deposit,
            "interest_rate_loan": self.state.interest_rate_loan,
        }

    def accept_deposit(self, amount: float) -> bool:
        if amount <= 0:
            return False
        self.state.deposits += amount
        self.state.cash += amount
        return True

    def withdraw_deposit(self, amount: float) -> float:
        withdraw = min(amount, self.state.cash)
        self.state.cash -= withdraw
        self.state.deposits -= withdraw
        return withdraw

    def request_loan(self, borrower_id: str, amount: float, purpose: str) -> Optional[Loan]:
        # Проверка кредитоспособности: сумма не больше доступных средств
        max_loan = max(0, self.state.cash * 0.7)
        if amount <= 0 or amount > max_loan:
            return None

        monthly_rate = self.state.interest_rate_loan
        months = 12
        if monthly_rate > 0:
            monthly_payment = amount * (monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
        else:
            monthly_payment = amount / months

        loan = Loan(
            borrower_id=borrower_id,
            amount=amount,
            interest_rate=monthly_rate,
            remaining=amount,
            monthly_payment=monthly_payment,
        )
        self.state.loans.append(loan)
        self.state.cash -= amount
        return loan

    def repay_loan(self, borrower_id: str, amount: float) -> float:
        for loan in self.state.loans:
            if loan.borrower_id == borrower_id:
                payment = min(amount, loan.remaining)
                loan.remaining -= payment
                self.state.cash += payment
                if loan.remaining <= 0:
                    self.state.loans.remove(loan)
                return payment
        return 0.0

    def process_monthly_payments(self) -> Dict[str, float]:
        """Списывает ежемесячные платежи по кредитам. Возвращает словарь borrower_id -> сумма."""
        payments = {}
        for loan in self.state.loans[:]:
            payment = min(loan.monthly_payment, loan.remaining)
            loan.remaining -= payment
            self.state.cash += payment
            payments[loan.borrower_id] = payments.get(loan.borrower_id, 0) + payment
            if loan.remaining <= 0:
                self.state.loans.remove(loan)
        return payments