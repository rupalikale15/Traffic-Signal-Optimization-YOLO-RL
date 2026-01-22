from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

NET_FILE = os.path.join("sumo_env", "cross_2lanes.net.xml")
ROUTE_FILE = os.path.join("sumo_env", "cross_2lanes.rou.xml")

def make_env():
    return SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=True,
        num_seconds=3600,
        delta_time=5,
        single_agent=True
    )

env = DummyVecEnv([make_env])

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="runs"
)

model.learn(total_timesteps=100_000)
model.save("models/ppo_cross")
