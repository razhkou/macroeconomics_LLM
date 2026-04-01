import asyncio
import logging
import numpy as np
from typing import Dict, List, Any

from shachi.environment import Environment

from agents.household import HouseholdAgent
from agents.firm import FirmAgent
from agents.bank import BankAgent
from agents.government import GovernmentAgent
from markets.goods_market import GoodsMarket
from markets.labor_market import LaborMarket

logger = logging.getLogger(__name__)


class CityEnvironment(Environment):
    def __init__(self, config: dict):
        self.config = config
        self.households: List[HouseholdAgent] = []
        self.firms: List[FirmAgent] = []
        self.bank: BankAgent = None
        self.government: GovernmentAgent = None
        self.time = 0

        self.labor_market = LaborMarket()
        self.goods_market = GoodsMarket()

        self._create_agents()

        self.logs = {
            'gdp': [],
            'unemployment': [],
            'inflation': [],
            'avg_wage': [],
            'prices': [],
            'savings_by_group': [],
            'loans': [],
            'key_interest_rate': [],
            'tax_income_rate': [],
            'tax_profit_rate': [],
            'employment_by_sector': [],
            'total_loans_amount': [],
        }

    def _create_agents(self):
        # Домохозяйства
        for arch in self.config['households']['archetypes']:
            agent = HouseholdAgent(
                agent_id=f"hh_{arch['name']}",
                config=arch,
                model=self.config['model'],
                temperature=self.config['temperature'],
                api_base=self.config.get('api_base')
            )
            self.households.append(agent)

        # Фирмы
        for firm_conf in self.config['firms']['firms']:
            agent = FirmAgent(
                agent_id=f"firm_{firm_conf['name']}",
                config=firm_conf,
                model=self.config['model'],
                temperature=self.config['temperature'],
                api_base=self.config.get('api_base')
            )
            self.firms.append(agent)

        # Банк
        self.bank = BankAgent(self.config['macros'])

        # Правительство (с LLM)
        self.government = GovernmentAgent(
            config=self.config['macros'],
            model=self.config['model'],
            temperature=self.config['temperature'],
            api_base=self.config.get('api_base')
        )

    async def reset(self) -> Dict[str, Any]:
        self.time = 0
        for hh in self.households:
            hh.state.savings = hh.config['initial_savings']
            hh.state.employed = 1.0
            hh.state.income = 0
            hh.state.wage = 0
            hh.state.loan_amount = 0
            hh.state.loan_monthly_payment = 0
        for firm in self.firms:
            firm.reset()
        self.bank.reset()
        self.government.state.budget = self.config['macros'].get('initial_budget', 0)
        self.labor_market.reset()
        self.goods_market.reset()
        self.logs = {k: [] for k in self.logs}
        return self._get_observations()

    def _get_observations(self) -> Dict[str, Any]:
        macro = {
            'inflation': self._calc_inflation(),
            'unemployment': self._calc_unemployment(),
            'avg_wage': self._calc_avg_wage(),
            'gdp': self._calc_gdp(),
            'prices': self.goods_market.get_current_prices(),
            'interest_rate_deposit': self.bank.state.interest_rate_deposit,
            'interest_rate_loan': self.bank.state.interest_rate_loan,
            'key_interest_rate': self.government.state.key_interest_rate,
        }
        obs = {}
        for hh in self.households:
            obs[hh.id] = {
                'agent_id': hh.id,
                'type': 'household',
                'state': hh.state.dict(),
                'macro': macro,
            }
        for firm in self.firms:
            obs[firm.id] = {
                'agent_id': firm.id,
                'type': 'firm',
                'state': firm.state.dict(),
                'macro': macro,
            }
        obs['bank'] = {
            'agent_id': 'bank',
            'type': 'bank',
            'state': self.bank.state.dict(),
            'macro': macro,
        }
        obs['government'] = {
            'agent_id': 'government',
            'type': 'government',
            'state': self.government.state.dict(),
            'macro': macro,
        }
        return obs

    async def step(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        # --- 1. Получаем наблюдения для правительства и банка ---
        observations = self._get_observations()
        # Правительство принимает решение о ставках
        gov_action = await self.government.step(observations['government'])
        # Банк корректирует свои ставки на основе ключевой ставки
        bank_action = await self.bank.step(observations['bank'])

        # --- 2. Финансовый рынок (новый порядок) ---
        # 2.1 Сначала платежи по кредитам (списание со счетов заёмщиков)
        payments = self.bank.process_monthly_payments()
        for borrower_id, payment in payments.items():
            for hh in self.households:
                if hh.id == borrower_id:
                    hh.state.savings -= payment
                    hh.state.loan_amount -= payment
                    break
            for firm in self.firms:
                if firm.id == borrower_id:
                    firm.state.cash -= payment
                    if hasattr(firm.state, 'loan_amount'):
                        firm.state.loan_amount -= payment
                    break

        # 2.2 Приём депозитов (агенты могут класть деньги)
        #    Упрощённо: все сбережения автоматически считаются депозитами.
        #    Начисление процентов
        deposit_rate = self.bank.state.interest_rate_deposit
        for hh in self.households:
            hh.state.savings *= (1 + deposit_rate)
        for firm in self.firms:
            firm.state.cash *= (1 + deposit_rate)

        # 2.3 Выдача новых кредитов (заявки уже в actions)
        for hh in self.households:
            action = actions.get(hh.id, {})
            loan_req = action.get('loan_request', {})
            if loan_req.get('need_loan'):
                amount = loan_req['amount']
                loan = self.bank.request_loan(hh.id, amount, "consumption")
                if loan:
                    hh.state.savings += loan.amount
                    hh.state.loan_amount = loan.remaining
                    hh.state.loan_monthly_payment = loan.monthly_payment

        # --- 3. Рынок труда ---
        self.labor_market.reset()
        for firm in self.firms:
            action = actions.get(firm.id, {})
            if action.get('vacancies', 0) > 0:
                wage = action.get('wage_offer', self._calc_avg_wage())
                self.labor_market.add_vacancy(firm.id, wage, firm.config['required_skills'], action['vacancies'])
        for hh in self.households:
            action = actions.get(hh.id, {})
            work_ratio = action.get('work_ratio', 0.5)
            desired_wage = action.get('desired_wage', self._calc_avg_wage())
            self.labor_market.add_application(hh.id, work_ratio, desired_wage, hh.state.skills)
        self.labor_market.match()

        for hh in self.households:
            wage = self.labor_market.get_wage(hh.id)
            if wage > 0:
                work_ratio = actions.get(hh.id, {}).get('work_ratio', 0.5)
                hh.state.income = hh.state.population * work_ratio * wage
                hh.state.wage = wage
                hh.state.employed = work_ratio
            else:
                hh.state.income = 0
                hh.state.wage = 0
                hh.state.employed = 0

        for firm in self.firms:
            hired = self.labor_market.hiring.get(firm.id, 0)
            firm.state.employees += hired
            firm.state.wage_bill = sum(self.labor_market.wages.get(hh.id, 0) for hh in self.households if hh.id in self.labor_market.matches.get(firm.id, []))

        # --- 4. Производство с учётом материалов ---
        # Сначала собираем B2B покупки (фирмы закупают материалы)
        self.goods_market.reset()
        # Фирмы выставляют заявки на покупку входов (из решения)
        for firm in self.firms:
            action = actions.get(firm.id, {})
            purchases = action.get('purchases', {})
            for good, quantity in purchases.items():
                if quantity <= 0:
                    continue
                price = self.goods_market.get_current_prices().get(good, 100)
                self.goods_market.place_bid(firm.id, good, price, quantity)

        # Фирмы выставляют предложения на продажу своей продукции
        for firm in self.firms:
            action = actions.get(firm.id, {})
            price = action.get('price', firm.state.price)
            firm.state.price = price
            good = firm.config['output_good']
            quantity = firm.state.inventory.get(good, 0)
            if quantity > 0:
                self.goods_market.place_ask(firm.id, good, price, quantity)

        # Домохозяйства выставляют заявки на покупку потребительских товаров
        for hh in self.households:
            action = actions.get(hh.id, {})
            consumption_ratio = action.get('consumption_ratio', 0.5)
            total_income = hh.state.income
            budget = consumption_ratio * total_income
            basket = hh.state.consumption_basket
            for good, share in basket.items():
                amount = budget * share
                if amount > 0:
                    price = self.goods_market.get_current_prices().get(good, 100)
                    quantity = amount / price
                    self.goods_market.place_bid(hh.id, good, price, quantity)

        # Клиринг рынка
        trades = self.goods_market.clear()

        # Обработка сделок
        for seller_id, buyer_id, good, price, quantity in trades:
            seller = next((f for f in self.firms if f.id == seller_id), None)
            buyer = next((h for h in self.households if h.id == buyer_id), None)
            if seller and buyer:
                seller.state.cash += price * quantity
                seller.state.inventory[good] -= quantity
                buyer.state.savings -= price * quantity
            else:
                buyer_firm = next((f for f in self.firms if f.id == buyer_id), None)
                if seller and buyer_firm:
                    seller.state.cash += price * quantity
                    seller.state.inventory[good] -= quantity
                    buyer_firm.state.cash -= price * quantity
                    if good in buyer_firm.state.inventory:
                        buyer_firm.state.inventory[good] += quantity
                    else:
                        buyer_firm.state.inventory[good] = quantity

        # Теперь производство с учётом доступных материалов
        for firm in self.firms:
            action = actions.get(firm.id, {})
            planned = action.get('production', 0)
            if planned <= 0:
                continue
            labor_hours = firm.state.employees * firm.config['labor_hours_per_worker']
            max_by_labor = labor_hours * firm.config['productivity']
            inputs = firm.config.get('inputs', {})
            max_by_inputs = float('inf')
            for input_good, needed_per_unit in inputs.items():
                stock = firm.state.inventory.get(input_good, 0)
                if needed_per_unit > 0:
                    max_by_inputs = min(max_by_inputs, stock / needed_per_unit)
            max_by_capacity = firm.config.get('production_capacity', float('inf'))
            production = min(planned, max_by_labor, max_by_inputs, max_by_capacity)
            # Потребляем материалы
            for input_good, needed_per_unit in inputs.items():
                used = production * needed_per_unit
                firm.state.inventory[input_good] -= used
            firm.state.inventory[firm.config['output_good']] += production
            firm.state.production = production

        # --- 5. Налоги и трансферты ---
        self.government.collect_taxes(self.households, self.firms)
        self.government.pay_transfers(self.households)

        # --- 6. Обновление макропоказателей ---
        self._update_macro()
        self._log()

        self.time += 1
        return self._get_observations()

    def _calc_gdp(self) -> float:
        total_income = sum(hh.state.income for hh in self.households)
        total_profit = sum(f.state.profit for f in self.firms)
        return total_income + total_profit

    def _calc_unemployment(self) -> float:
        total_pop = sum(hh.state.population for hh in self.households)
        employed_pop = sum(hh.state.employed * hh.state.population for hh in self.households)
        return 1 - (employed_pop / total_pop) if total_pop > 0 else 0

    def _calc_avg_wage(self) -> float:
        total_wage = sum(hh.state.wage * hh.state.employed * hh.state.population for hh in self.households)
        total_employed = sum(hh.state.employed * hh.state.population for hh in self.households)
        return total_wage / total_employed if total_employed > 0 else 0

    def _calc_inflation(self) -> float:
        prices = self.goods_market.get_current_prices()
        if not prices:
            return 0
        if not self.logs['prices']:
            return 0
        prev_prices = self.logs['prices'][-1]
        if not prev_prices:
            return 0
        inflation = 0
        n = 0
        for good, price in prices.items():
            if good in prev_prices and prev_prices[good] > 0:
                inflation += (price - prev_prices[good]) / prev_prices[good]
                n += 1
        return inflation / n if n > 0 else 0

    def _update_macro(self):
        gdp = self._calc_gdp()
        self.logs['gdp'].append(gdp)
        unemp = self._calc_unemployment()
        self.logs['unemployment'].append(unemp)
        inf = self._calc_inflation()
        self.logs['inflation'].append(inf)
        avg_wage = self._calc_avg_wage()
        self.logs['avg_wage'].append(avg_wage)
        self.logs['prices'].append(self.goods_market.get_current_prices())
        savings_by_group = {hh.id: hh.state.savings for hh in self.households}
        self.logs['savings_by_group'].append(savings_by_group)
        self.logs['loans'].append([l.dict() for l in self.bank.state.loans])
        self.logs['key_interest_rate'].append(self.government.state.key_interest_rate)
        self.logs['tax_income_rate'].append(self.government.state.tax_income_rate)
        self.logs['tax_profit_rate'].append(self.government.state.tax_profit_rate)

        # Занятость по отраслям
        employment_by_sector = {}
        for firm in self.firms:
            sector = firm.config['industry']
            employment_by_sector[sector] = employment_by_sector.get(sector, 0) + firm.state.employees
        self.logs['employment_by_sector'].append(employment_by_sector)

        # Общая сумма выданных кредитов
        total_loans = sum(loan.amount for loan in self.bank.state.loans)
        self.logs['total_loans_amount'].append(total_loans)

    def _log(self):
        if self.time % self.config['log_freq'] == 0:
            gdp = self.logs['gdp'][-1]
            unemp = self.logs['unemployment'][-1]
            inf = self.logs['inflation'][-1]
            wage = self.logs['avg_wage'][-1]
            logger.info(f"Step {self.time}: GDP={gdp:.2f}, Unemployment={unemp:.2%}, Inflation={inf:.2%}, AvgWage={wage:.2f}")

    def _save_logs_to_csv(self, step: int):
        """Сохраняет основные логи в CSV-файл."""
        save_dir = self.config.get('save_dir', './output')
        os.makedirs(save_dir, exist_ok=True)

        # Основные временные ряды
        df = pd.DataFrame({
            'step': range(len(self.logs['gdp'])),
            'gdp': self.logs['gdp'],
            'unemployment': self.logs['unemployment'],
            'inflation': self.logs['inflation'],
            'avg_wage': self.logs['avg_wage'],
            'key_interest_rate': self.logs['key_interest_rate'],
            'tax_income_rate': self.logs['tax_income_rate'],
            'tax_profit_rate': self.logs['tax_profit_rate'],
            'total_loans_amount': self.logs['total_loans_amount'],
        })
        df.to_csv(os.path.join(save_dir, f'logs_step_{step}.csv'), index=False)

        # Сбережения по группам
        savings_df = pd.DataFrame(self.logs['savings_by_group']).T
        savings_df.index.name = 'household_group'
        savings_df.to_csv(os.path.join(save_dir, f'savings_step_{step}.csv'))

        # Цены по товарам
        prices_df = pd.DataFrame(self.logs['prices']).T
        prices_df.index.name = 'good'
        prices_df.to_csv(os.path.join(save_dir, f'prices_step_{step}.csv'))

        # Занятость по отраслям
        employment_df = pd.DataFrame(self.logs['employment_by_sector']).T
        employment_df.index.name = 'sector'
        employment_df.to_csv(os.path.join(save_dir, f'employment_step_{step}.csv'))

        # Сохраняем последние версии для быстрого доступа
        df.to_csv(os.path.join(save_dir, 'logs_latest.csv'), index=False)
        savings_df.to_csv(os.path.join(save_dir, 'savings_latest.csv'))
        prices_df.to_csv(os.path.join(save_dir, 'prices_latest.csv'))
        employment_df.to_csv(os.path.join(save_dir, 'employment_latest.csv'))

        logger.info(f"Saved logs to {save_dir} (step {step})")