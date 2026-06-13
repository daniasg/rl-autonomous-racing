"""Train a PPO reinforcement-learning agent to drive in CarRacing."""
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

def main():
    env = make_vec_env("CarRacing-v3", n_envs=1)
    env = VecFrameStack(env, n_stack=4)       
    model = PPO("CnnPolicy", env, verbose=1)    
    model.learn(total_timesteps=100000)
    model.save("models/ppo_carracing")
    print("Done. Model saved to models/ppo_carracing.zip")

if __name__ == "__main__":
    main()