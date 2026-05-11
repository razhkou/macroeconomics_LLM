import matplotlib.pyplot as plt
import os

def plot_results(logs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0,0].plot(logs['gdp'])
    axes[0,0].set_title('GDP')
    axes[0,1].plot(logs['unemployment'])
    axes[0,1].set_title('Unemployment')
    axes[1,0].plot(logs['inflation'])
    axes[1,0].set_title('Inflation')
    axes[1,1].plot(logs['avg_wage'])
    axes[1,1].set_title('Average Wage')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'results.png'))
    plt.close()