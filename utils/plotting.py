import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd


def plot_results(logs, save_dir):
    """Строит сетку графиков и сохраняет в results.png"""
    os.makedirs(save_dir, exist_ok=True)

    # Определяем, какие данные доступны
    has_rates = 'key_interest_rate' in logs and logs['key_interest_rate']
    has_taxes = 'tax_income_rate' in logs and logs['tax_income_rate']
    has_savings = 'savings_by_group' in logs and logs['savings_by_group']
    has_loans = 'total_loans_amount' in logs and logs['total_loans_amount']
    has_prices = 'prices' in logs and logs['prices']
    has_employment = 'employment_by_sector' in logs and logs['employment_by_sector']

    # Определяем количество подграфиков
    n_plots = 4  # базовые: GDP, безработица, инфляция, средняя зарплата
    if has_rates:
        n_plots += 1
    if has_taxes:
        n_plots += 1
    if has_savings:
        n_plots += 1
    if has_loans:
        n_plots += 1
    if has_prices:
        n_plots += 1
    if has_employment:
        n_plots += 1

    # Вычисляем размеры сетки (3 колонки, строки динамически)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()  # чтобы удобно обращаться по индексу
    idx = 0

    # 1. GDP
    axes[idx].plot(logs['gdp'])
    axes[idx].set_title('GDP')
    axes[idx].grid(True)
    idx += 1

    # 2. Unemployment
    axes[idx].plot(logs['unemployment'])
    axes[idx].set_title('Unemployment')
    axes[idx].grid(True)
    idx += 1

    # 3. Inflation
    axes[idx].plot(logs['inflation'])
    axes[idx].set_title('Inflation')
    axes[idx].grid(True)
    idx += 1

    # 4. Average Wage
    axes[idx].plot(logs['avg_wage'])
    axes[idx].set_title('Average Wage')
    axes[idx].grid(True)
    idx += 1

    # 5. Key interest rate
    if has_rates:
        axes[idx].plot(logs['key_interest_rate'])
        axes[idx].set_title('Key Interest Rate')
        axes[idx].grid(True)
        idx += 1

    # 6. Tax rates
    if has_taxes:
        axes[idx].plot(logs['tax_income_rate'], label='Income tax')
        axes[idx].plot(logs['tax_profit_rate'], label='Profit tax')
        axes[idx].set_title('Tax Rates')
        axes[idx].legend()
        axes[idx].grid(True)
        idx += 1

    # 7. Savings by household group
    if has_savings:
        df_savings = pd.DataFrame(logs['savings_by_group'])
        for col in df_savings.columns:
            axes[idx].plot(df_savings[col], label=col)
        axes[idx].set_title('Savings by Household Group')
        axes[idx].legend(loc='best')
        axes[idx].grid(True)
        idx += 1

    # 8. Total loans amount
    if has_loans:
        axes[idx].plot(logs['total_loans_amount'])
        axes[idx].set_title('Total Loans Outstanding')
        axes[idx].grid(True)
        idx += 1

    # 9. Prices of selected goods (первые 5 ключевых товаров)
    if has_prices:
        df_prices = pd.DataFrame(logs['prices'])
        # Выбираем несколько товаров (например, до 5)
        goods_to_plot = list(df_prices.columns)[:5]
        for good in goods_to_plot:
            axes[idx].plot(df_prices[good], label=good)
        axes[idx].set_title('Prices of Key Goods')
        axes[idx].legend()
        axes[idx].grid(True)
        idx += 1

    # 10. Employment by sector
    if has_employment:
        df_empl = pd.DataFrame(logs['employment_by_sector'])
        for sector in df_empl.columns:
            axes[idx].plot(df_empl[sector], label=sector)
        axes[idx].set_title('Employment by Sector')
        axes[idx].legend(loc='best')
        axes[idx].grid(True)
        idx += 1

    # Скрыть неиспользованные подграфики
    for i in range(idx, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'results.png'))
    plt.close()