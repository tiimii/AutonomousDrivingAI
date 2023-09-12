import ppo_model
import gymnasium as gym
import os


def test_environment():
    env = gym.make('CarRacing-v2', render_mode="human")
    print(env.action_space)
    print(env.observation_space)

    episodes = 5
    for episode in range(1, episodes+1):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            env.render()
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            score += reward
        print(f'Episode {episode} Score: {score}')
    env.close()


def train_model(path, iterations):
    for iteration in range(iterations):
        ppo_model.train(path, 100000)
    ppo_model.evaluate(path)


if __name__ == '__main__':
    model_path = os.path.join("Training", "Saved Models", "PPO_Driving_Model")
    ppo_model.hyperparameter_tuning()
    # train_model(model_path, 5)
    # ppo_model.test(model_path)
