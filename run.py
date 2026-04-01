import asyncio
import logging
import yaml
import os
import signal
import sys
from environment import CityEnvironment
from utils.plotting import plot_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальная переменная для доступа к env в обработчике
env = None

def signal_handler(sig, frame):
    logger.info("Interrupt received, saving final logs and plot...")
    if env:
        env._save_logs_to_csv(env.time)
        # Сохраняем график на момент прерывания
        plot_results(env.logs, env.config.get('save_dir', './output'))
    sys.exit(0)

def load_config(config_dir="configs"):
    with open(os.path.join(config_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(config_dir, "households.yaml")) as f:
        cfg["households"] = yaml.safe_load(f)
    with open(os.path.join(config_dir, "firms.yaml")) as f:
        cfg["firms"] = yaml.safe_load(f)
    with open(os.path.join(config_dir, "macros.yaml")) as f:
        cfg["macros"] = yaml.safe_load(f)
    return cfg

async def run_simulation(env, cfg):
    await env.reset()
    save_dir = cfg.get('save_dir', './output')
    os.makedirs(save_dir, exist_ok=True)

    for step in range(cfg["episode_length"]):
        observations = env._get_observations()
        actions = {}
        for agent in env.households + env.firms:
            obs = observations.get(agent.id, {})
            action = await agent.step(obs)
            actions[agent.id] = action
        bank_obs = observations.get('bank', {})
        gov_obs = observations.get('government', {})
        actions['bank'] = await env.bank.step(bank_obs)
        actions['government'] = await env.government.step(gov_obs)
        await env.step(actions)

        # После каждого шага сохраняем CSV и обновляем график
        env._save_logs_to_csv(step + 1)          # сохраняем CSV с номером шага
        plot_results(env.logs, save_dir)         # перезаписываем results.png
        logger.info(f"Step {step+1}/{cfg['episode_length']} completed, saved logs and plot")

    logger.info("Simulation finished.")

def main():
    global env
    cfg = load_config()
    env = CityEnvironment(cfg)
    # Устанавливаем обработчик сигнала для Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    try:
        asyncio.run(run_simulation(env, cfg))
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        if env:
            env._save_logs_to_csv(env.time)
            plot_results(env.logs, env.config.get('save_dir', './output'))
        raise
    # Финальное сохранение (уже сделано в цикле, но на всякий случай)
    plot_results(env.logs, cfg.get('save_dir', './output'))
    logger.info(f"Final plots saved to {cfg['save_dir']}/results.png")

if __name__ == "__main__":
    main()