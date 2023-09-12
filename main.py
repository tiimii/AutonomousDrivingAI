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


def train_model(model_path):
    ppo_model.train(model_path, 200000)
    ppo_model.evaluate(model_path)


if __name__ == '__main__':
    model_path = os.path.join("Training", "Saved Models", "PPO_Driving_Model")
    train_model(model_path)
    # ppo_model.test(model_path)
