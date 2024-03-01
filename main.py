import gymnasium as gym
import PyFlyt.gym_envs
from PyFlyt.gym_envs import FlattenWaypointEnv
import GPUtil
import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)
context = ray.init(num_cpus=6, num_gpus=1)
print("this the URL dashboard", context.dashboard_url)

ppo_config = PPOConfig()
ppo_config = ppo_config.framework("torch")
ppo_config = ppo_config.training(model={"fcnet_hiddens": [64, 64]},
                                 lr=0.0001, kl_coeff=0.3, clip_param=0.2, gamma=0.99)
ppo_config = ppo_config.environment(env="Pendulum-v1")

agent = ppo_config.build()
agent.restore("/home/basel/PycharmProjects/pythonProject/checkpoints/pendulum2")
print(PPO.default_resource_request(ppo_config))

### Training Loops
for iterat in range(10):
    # result = agent.train()
    # print(f"{iterat + 1}, episode reward mean: {result['episode_reward_mean']}")
    # print(f"mean episode reward: {result['episode_reward_mean']}")
    # print(f"average episode length: {result['episode_len_mean']}")
    # print(f"max episode reward: {result['episode_reward_max']}")
    # print(f"min episode reward: {result['episode_reward_min']}")
    # print(f"number of agent steps trained: {result['num_agent_steps_trained']}")
    if iterat % 1 == 0:
        # checkpoint_path = agent.save("/home/basel/PycharmProjects/pythonProject/checkpoints/pendulum2")
    ###### Evaluation through the environment
        env = gym.make("Pendulum-v1", render_mode="human")
        episode_reward = 0
        terminated = truncated = False
        obs, _ = env.reset()
        while not (terminated or truncated):
            action = agent.compute_single_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            print(reward)
        env.close()

ray.shutdown()
