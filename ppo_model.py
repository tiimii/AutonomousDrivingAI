from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os
import gymnasium as gym


def initialize_model(model_path):
    # Initialize environment
    env = gym.make('CarRacing-v2')

    # Wrap environment
    env = DummyVecEnv([lambda: env])

    # Create Logs directory
    log_path = os.path.join("Training", "Logs")

    # Create model
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_path)

    # Save Model
    model.save(model_path)

    env.close()


def train(model_path, timesteps):
    # Initialize environment
    env = gym.make('CarRacing-v2')

    # Wrap environment
    env = DummyVecEnv([lambda: env])

    # Load model
    model = PPO.load(model_path, env=env)

    # Train model
    model.learn(total_timesteps=timesteps, progress_bar=True)

    # Save model
    model.save(model_path)

    # Close environment
    env.close()


def evaluate(model_path):
    # Initialize environment
    env = gym.make('CarRacing-v2')

    # Wrap environment
    env = DummyVecEnv([lambda: env])

    # Load model
    model = PPO.load(model_path)
    mean_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
    print(mean_reward)


def test(model_path):
    # Load model
    model = PPO.load(model_path)
    new_env = gym.make("CarRacing-v2", render_mode="human")

    episodes = 5
    for episode in range(1, episodes+1):
        obs, _ = new_env.reset()
        done = False
        score = 0

        while not done:
            new_env.render()
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = new_env.step(action)
            score += reward
        print(f"Episode {episode}:\tScore:{score}")
    new_env.close()
