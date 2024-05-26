import gymnasium as gym
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from SHARED.params import *
from SHARED.aux_functions import *


def evaluate(model: BaseAlgorithm, num_episodes: int = 100, deterministic: bool = True,
) -> float:
    """
    Evaluate an RL agent for `num_episodes`.

    :param model: the RL Agent
    :param env: the gym Environment
    :param num_episodes: number of episodes to evaluate it
    :param deterministic: Whether to use deterministic or stochastic actions
    :return: Mean reward for the last `num_episodes`
    """
    # This function will only work for a single environment
    vec_env = model.get_env()
    obs = vec_env.reset()
    all_episode_rewards = []
    for _ in range(num_episodes):
        episode_rewards = []
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _info = vec_env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print(f"Mean reward: {mean_episode_reward:.2f} - Num episodes: {num_episodes}")

    return mean_episode_reward

def sample_state():
    #Sample K
    # k = np.random.randint(0,max_steps//4)
    # k = int(np.random.exponential(scale=1/0.01))
    k = np.random.randint(0, max_steps)

    growth_space_ub = 0.0035 + k*(0.0001)
    growth_space_lb = 0.0035
    
    #Sample Dry matter
    x1 = np.random.uniform(growth_space_lb,growth_space_ub)
    
    #Sample Indoor Temp
    x3 = np.random.uniform(10,20)
    
    #Sample Indoor C02
    x2 = np.random.uniform(C02_MIN_CONSTRAIN_MPC,C02_MAX_CONSTRAIN_MPC) #sample PPM
    x2 = co2ppm2dens(temp = x3,ppm = x2) #convert to dens
    
    #Sample Hum
    x4 = np.random.uniform(50,90)
    x4 = rh2vaporDens(temp = x3, rh = x4)
    
    
    return np.array([x1,x2,x3,x4]),k

