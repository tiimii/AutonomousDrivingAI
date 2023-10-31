# AutonomousDrivingAI

The AutonomousDrivingAI is a deep reinforcement learning model designed to navigate the CarRacing-v2 environment from the OpenAI Gym. The AI employs the Proximal Policy Optimization (PPO) algorithm to learn optimal driving policies for the car in various racing scenarios.

## Overview

- Environment: The AI is trained on the CarRacing-v2 environment from OpenAI Gym. This environment simulates a car racing game where the goal is to drive as fast as possible while avoiding obstacles and staying on the track.
- Model: The core of the AI is the PPO model, implemented using the stable_baselines3 library. PPO is an on-policy algorithm that optimizes the policy by making small updates, ensuring that the new policy is not too different from the old one.
- Training: The model undergoes iterative training sessions, each consisting of a specified number of timesteps. Training logs and model checkpoints are stored in the Training directory.
- Evaluation: After training, the model's performance can be evaluated using the evaluate function, which calculates the mean reward over a set number of episodes.
- Hyperparameter Tuning: The repository includes a hyperparameter tuning function that uses the Optuna library to search for the best hyperparameters for the PPO model.

## Files

- `main.py`: The main script that initializes the model, trains it, and tests its performance.
- `ppo_model.py`: Contains the implementation of the PPO model, training, evaluation, and hyperparameter tuning functions.

## Getting Started

- Clone the repository.
- Install the required libraries and dependencies.
- Run the main.py script to train and test the model.

## Training Process

The training process involves the following steps:

- **Initialization**: The PPO model is initialized with a specified architecture and hyperparameters. The model is saved in the Training/Saved Models directory.
- **Training Loop**: The model is trained iteratively over a specified number of timesteps. After each training session, the model is saved.
- **Evaluation**: After training, the model's performance is evaluated over a set number of episodes. The mean reward is calculated and printed.
- **Hyperparameter Tuning**: If desired, the hyperparameters of the model can be tuned using the Optuna library. The best hyperparameters are saved and can be used for subsequent training sessions.
