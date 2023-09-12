from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os
import gymnasium as gym
import optuna


def initialize_model(model_path, params):
    # Initialize environment
    env = gym.make('CarRacing-v2')

    # Wrap environment
    env = DummyVecEnv([lambda: env])

    # Create Logs directory
    log_path = os.path.join("Training", "Logs")

    # Create model
    model = PPO("CnnPolicy", env, **params, verbose=1, tensorboard_log=log_path)

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


def objective(trial):
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
    ent_coef = trial.suggest_float("ent_coef", 1e-5, 0.1, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)

    # Initialize environment
    env = gym.make('CarRacing-v2')
    env = DummyVecEnv([lambda: env])

    # Create model with hyperparameters
    model = PPO("CnnPolicy", env, learning_rate=learning_rate, gamma=gamma, gae_lambda=gae_lambda,
                ent_coef=ent_coef, clip_range=clip_range, verbose=0)

    # Train model
    model.learn(total_timesteps=2000)  # Use a smaller timesteps for trials

    # Evaluate model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, render=False)

    return mean_reward


def hyperparameter_tuning():
    # Tuning process
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    best_params = study.best_params

    # Print best parameters
    print("Best hyperparameters:", best_params)

    # Create Model with best parameters
    model_path = os.path.join("Training", "Saved Models", "PPO_Driving_Model_TUNED")
    initialize_model(model_path, best_params)


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
