import gymnasium as gym
import PyFlyt.gym_envs
from PyFlyt.gym_envs import FlattenWaypointEnv
import GPUtil
import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)

ray.init(num_cpus=6, num_gpus=1)

ppo_config = PPOConfig()
ppo_config = ppo_config.framework("torch")
ppo_config = ppo_config.training(model={"fcnet_hiddens": [256, 256]},
                                 lr=0.0003, kl_coeff=0.3, clip_param=0.2, lambda_=0.95,
                                 gamma=0.99, train_batch_size=5000)
ppo_config = ppo_config.environment(env="Acrobot-v1")
ppo_config = ppo_config.resources(num_gpus=1, num_cpus_for_local_worker=2,
                                  num_gpus_per_worker=0.5, num_cpus_per_worker=2)
ppo_config = ppo_config.rollouts(num_rollout_workers=2)

agent = ppo_config.build()
agent.restore("/home/basel/PycharmProjects/pythonProject/checkpoints/checkpoints6")
print(PPO.default_resource_request(ppo_config))




### Training Loops
for iterat in range(11):

    ###### Evaluation through the environment
    env = gym.make("Acrobot-v1", render_mode="human")
    episode_reward = 0
    terminated = truncated = False
    obs, _ = env.reset()
    while not (terminated or truncated):
        action = agent.compute_single_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)

    # result = agent.train()
    # print(f"GPU system monitoring: {GPUtil.showUtilization()}")
    # print(f"{iterat + 1}, episode reward mean: {result['episode_reward_mean']}")
    # print(f"mean episode reward: {result['episode_reward_mean']}")
    # print(f"average episode length: {result['episode_len_mean']}")
    # print(f"max episode reward: {result['episode_reward_max']}")
    # print(f"min episode reward: {result['episode_reward_min']}")
    # print(f"number of agent steps trained: {result['num_agent_steps_trained']}")
    # print(f"GPU system monitoring: {GPUtil.showUtilization()}")
    # if iterat % 25 == 0:
    #     checkpoint_path = agent.save("/home/basel/PycharmProjects/pythonProject/checkpoints6")

ray.shutdown()
