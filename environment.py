import asyncio
import logging
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Any

from shachi.environment import Environment

from agents.household import HouseholdAgent
from agents.firm import FirmAgent
from agents.bank import BankAgent
from agents.government import GovernmentAgent
from markets.labor_market import LaborMarket
from markets.goods_market import GoodsMarket

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
            'deposits_by_group': [],
            'loans': [],
            'key_interest_rate': [],
            'tax_income_rate': [],
            'tax_profit_rate': [],
        }

    def _create_agents(self):
        for arch in self.config['households']['archetypes']:
            agent = HouseholdAgent(
                agent_id=f"hh_{arch['name']}",
                config=arch,
                model=self.config['model'],
                temperature=self.config['temperature'],
                api_base=self.config.get('api_base')
            )
            self.households.append(agent)

        for firm_conf in self.config['firms']['firms']:
            agent = FirmAgent(
                agent_id=f"firm_{firm_conf['name']}",
                config=firm_conf,
                model=self.config['model'],
                temperature=self.config['temperature'],
                api_base=self.config.get('api_base')
            )
            self.firms.append(agent)

        self.bank = BankAgent(self.config['macros'])
        self.government = GovernmentAgent(
            config=self.config['macros'],
            model=self.config['model'],
            temperature=self.config['temperature'],
            api_base=self.config.get('api_base')
        )

    async def reset(self) -> Dict[str, Any]:
        self.time = 0
        for hh in self.households:
            hh.state.deposit = hh.config['initial_savings']
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
        # Начальные депозиты зачисляем в банк
        for hh in self.households:
            if hh.state.deposit > 0:
                self.bank.accept_deposit(hh.state.deposit)
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
            obs[hh.id] = {'agent_id': hh.id, 'type': 'household', 'state': hh.state.dict(), 'macro': macro}
        for firm in self.firms:
            obs[firm.id] = {'agent_id': firm.id, 'type': 'firm', 'state': firm.state.dict(), 'macro': macro}
        obs['bank'] = {'agent_id': 'bank', 'type': 'bank', 'state': self.bank.state.dict(), 'macro': macro}
        obs['government'] = {'agent_id': 'government', 'type': 'government', 'state': self.government.state.dict(), 'macro': macro}
        return obs

    async def step(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Правительство и банк обновляют ставки
        observations = self._get_observations()
        gov_action = await self.government.step(observations['government'])
        bank_action = await self.bank.step(observations['bank'])

        # 2. Кредитные платежи и проценты по депозитам
        payments = self.bank.process_monthly_payments()
        for borrower_id, payment in payments.items():
            for hh in self.households:
                if hh.id == borrower_id:
                    withdrawn = self.bank.withdraw_deposit(payment)
                    hh.state.deposit -= withdrawn
                    hh.state.loan_amount -= payment
                    break
            for firm in self.firms:
                if firm.id == borrower_id:
                    firm.state.cash -= payment
                    break

        deposit_rate = self.bank.state.interest_rate_deposit
        interest_earned = self.bank.state.deposits * deposit_rate
        self.bank.state.deposits += interest_earned
        self.bank.state.cash += interest_earned
        if self.bank.state.deposits > 0:
            for hh in self.households:
                hh.state.deposit *= (1 + deposit_rate)

        # 3. Рынок труда
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
            workers = hh.state.population * work_ratio
            self.labor_market.add_application(hh.id, workers, desired_wage, hh.state.skills)
        self.labor_market.match()

        for hh in self.households:
            if hh.id in self.labor_market.wages:
                total_income = 0.0
                total_workers = 0.0
                for wage, workers in self.labor_market.wages[hh.id]:
                    total_income += wage * workers
                    total_workers += workers
                hh.state.income = total_income
                hh.state.wage = total_income / total_workers if total_workers > 0 else 0
                hh.state.employed = total_workers / hh.state.population
            else:
                hh.state.income = 0
                hh.state.wage = 0
                hh.state.employed = 0

        for firm in self.firms:
            if firm.id in self.labor_market.matches:
                total_hired = sum(workers for _, workers, _ in self.labor_market.matches[firm.id])
                firm.state.employees += total_hired
                if self.labor_market.matches[firm.id]:
                    wage = self.labor_market.matches[firm.id][0][2]
                    firm.state.wage_bill = wage * total_hired
            else:
                firm.state.wage_bill = 0

        # 4. Сбережения и потребление
        for hh in self.households:
            saving = hh.state.income * hh.state.saving_rate
            if saving > 0:
                self.bank.accept_deposit(saving)
                hh.state.deposit += saving
            consumption_budget = hh.state.income - saving
            if consumption_budget < 0:
                consumption_budget = 0
            hh.state.consumption_budget = consumption_budget

        # 5. Кредитные заявки
        for hh in self.households:
            action = actions.get(hh.id, {})
            loan_req = action.get('loan_request', {})
            if loan_req.get('need_loan'):
                amount = loan_req['amount']
                loan = self.bank.request_loan(hh.id, amount, "consumption")
                if loan:
                    self.bank.accept_deposit(loan.amount)
                    hh.state.deposit += loan.amount
                    hh.state.loan_amount = loan.remaining
                    hh.state.loan_monthly_payment = loan.monthly_payment

        # 6. Рынок товаров
        self.goods_market.reset()
        # Предложения от фирм
        for firm in self.firms:
            action = actions.get(firm.id, {})
            price = action.get('price', firm.state.price)
            firm.state.price = price
            good = firm.config['output_good']
            quantity = firm.state.inventory.get(good, 0)
            if quantity > 0:
                self.goods_market.place_ask(firm.id, good, price, quantity)

        # Заявки на покупку от домохозяйств
        for hh in self.households:
            budget = getattr(hh.state, 'consumption_budget', 0)
            if budget <= 0:
                continue
            for good, share in hh.state.consumption_basket.items():
                amount = budget * share
                if amount > 0:
                    price = self.goods_market.get_current_prices().get(good, 100)
                    qty = amount / price
                    self.goods_market.place_bid(hh.id, good, price, qty)

        # Заявки на инвестиции и материалы от фирм
        for firm in self.firms:
            action = actions.get(firm.id, {})
            invest_ratio = action.get('investment_ratio', 0.0)
            if invest_ratio > 0 and firm.state.profit > 0:
                invest_budget = firm.state.profit * invest_ratio
                good = action.get('investment_good', 'machinery')
                price = self.goods_market.get_current_prices().get(good, 100)
                if price > 0:
                    quantity = invest_budget / price
                    self.goods_market.place_bid(firm.id, good, price, quantity)
            purchases = action.get('purchases', {})
            for good, qty in purchases.items():
                if qty > 0:
                    price = self.goods_market.get_current_prices().get(good, 100)
                    self.goods_market.place_bid(firm.id, good, price, qty)

        # Клиринг рынка товаров
        trades = self.goods_market.clear()
        for seller_id, buyer_id, good, price, quantity in trades:
            seller = next((f for f in self.firms if f.id == seller_id), None)
            buyer = next((h for h in self.households if h.id == buyer_id), None)
            if seller and buyer:
                seller.state.cash += price * quantity
                seller.state.inventory[good] -= quantity
                spent = price * quantity
                withdrawn = self.bank.withdraw_deposit(spent)
                buyer.state.deposit -= withdrawn
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

        # 7. Производство
        for firm in self.firms:
            action = actions.get(firm.id, {})
            planned = action.get('production', 0)
            if planned <= 0:
                continue
            labor_hours = firm.state.employees * firm.config['labor_hours_per_worker']
            max_by_labor = labor_hours * firm.config['productivity']
            inputs = firm.config.get('inputs', {})
            max_by_inputs = float('inf')
            for input_good, needed in inputs.items():
                stock = firm.state.inventory.get(input_good, 0)
                if needed > 0:
                    max_by_inputs = min(max_by_inputs, stock / needed)
            production = min(planned, max_by_labor, max_by_inputs, firm.state.production_capacity)
            for input_good, needed in inputs.items():
                firm.state.inventory[input_good] -= production * needed
            firm.state.inventory[firm.config['output_good']] += production
            firm.state.production = production

        # 8. Налоги и трансферты
        self.government.collect_taxes(self.households, self.firms)
        self.government.pay_transfers(self.households)
        for hh in self.households:
            transfer = self.government.state.transfers.get(hh.config['name'], 0)
            if transfer > 0:
                self.bank.accept_deposit(transfer)
                hh.state.deposit += transfer

        # 9. Обновление макропоказателей
        self._update_macro()
        self._log()
        self._save_logs_to_csv(self.time + 1)
        self._save_plot()

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
        if not prices or not self.logs['prices']:
            return 0
        prev = self.logs['prices'][-1]
        if not prev:
            return 0
        inf = 0
        n = 0
        for g, p in prices.items():
            if g in prev and prev[g] > 0:
                inf += (p - prev[g]) / prev[g]
                n += 1
        return inf / n if n > 0 else 0

    def _update_macro(self):
        self.logs['gdp'].append(self._calc_gdp())
        self.logs['unemployment'].append(self._calc_unemployment())
        self.logs['inflation'].append(self._calc_inflation())
        self.logs['avg_wage'].append(self._calc_avg_wage())
        self.logs['prices'].append(self.goods_market.get_current_prices())
        self.logs['deposits_by_group'].append({hh.id: hh.state.deposit for hh in self.households})
        self.logs['loans'].append([l.dict() for l in self.bank.state.loans])
        self.logs['key_interest_rate'].append(self.government.state.key_interest_rate)
        self.logs['tax_income_rate'].append(self.government.state.tax_income_rate)
        self.logs['tax_profit_rate'].append(self.government.state.tax_profit_rate)

    def _log(self):
        if self.time % self.config['log_freq'] == 0:
            logger.info(f"Step {self.time}: GDP={self.logs['gdp'][-1]:.2f}, Unemployment={self.logs['unemployment'][-1]:.2%}, Inflation={self.logs['inflation'][-1]:.2%}, AvgWage={self.logs['avg_wage'][-1]:.2f}")

    def _save_logs_to_csv(self, step: int):
        save_dir = self.config.get('save_dir', './output')
        os.makedirs(save_dir, exist_ok=True)
        df = pd.DataFrame({
            'step': range(len(self.logs['gdp'])),
            'gdp': self.logs['gdp'],
            'unemployment': self.logs['unemployment'],
            'inflation': self.logs['inflation'],
            'avg_wage': self.logs['avg_wage'],
        })
        if self.logs['key_interest_rate']:
            df['key_interest_rate'] = self.logs['key_interest_rate']
        if self.logs['tax_income_rate']:
            df['tax_income_rate'] = self.logs['tax_income_rate']
        if self.logs['tax_profit_rate']:
            df['tax_profit_rate'] = self.logs['tax_profit_rate']
        df.to_csv(os.path.join(save_dir, f'logs_step_{step}.csv'), index=False)
        df.to_csv(os.path.join(save_dir, 'logs_latest.csv'), index=False)

    def _save_plot(self):
        from utils.plotting import plot_results
        save_dir = self.config.get('save_dir', './output')
        plot_results(self.logs, save_dir)