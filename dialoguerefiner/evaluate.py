# dialoguerefiner/evaluate.py

from stable_baselines3 import PPO
from .model import DialogueModel
from .dataset import DialogueDataset
from .environment import DialogueEnvironment
from .reward_model import RewardModel
import numpy as np

"""
RLHF Model Evaluation Module

This module provides functionality to evaluate a trained RLHF (Reinforcement Learning from Human Feedback) model.
It uses the trained model to generate responses in a simulated environment and calculates average rewards
to assess the model's performance.

Key components:
1. Trained PPO model: The RLHF-optimized language model
2. DialogueModel: The base language model
3. RewardModel: Used to calculate rewards for generated responses
4. DialogueEnvironment: Simulates interactions for evaluation
5. Dataset: Provides test data for evaluation

The evaluation process involves:
- Loading the trained model and necessary components
- Running the model through multiple episodes in the environment
- Calculating average rewards and other potential metrics

This implementation demonstrates how to assess the performance of an RLHF-trained model using open-source tools.
"""

def evaluate_model(model_path, test_dataset_path, num_episodes=100):
    """
    Evaluate the trained RLHF model on a test dataset.

    Args:
    model_path (str): Path to the saved PPO model
    test_dataset_path (str): Path to the test dataset
    num_episodes (int): Number of evaluation episodes to run

    Returns:
    tuple: Average reward and standard deviation of rewards
    """
    # Load the trained model
    trained_model = PPO.load(model_path)
    
    # Initialize necessary components
    dialogue_model = DialogueModel()
    reward_model = RewardModel(input_size=768)  # Make sure this matches your actual input size
    test_dataset = DialogueDataset(test_dataset_path)
    
    # Create an environment for evaluation
    env = DialogueEnvironment(dialogue_model, reward_model, test_dataset)
    
    total_rewards = []
    
    # Run evaluation episodes
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        # Single episode loop
        while not done:
            action, _ = trained_model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    # Calculate average reward and standard deviation
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f} (±{std_reward:.2f})")
    
    # Additional metrics could be added here, such as:
    # - Comparing model responses to 'chosen' responses in the dataset
    # - Calculating the percentage of times the model's response was preferred over the 'rejected' response
    # - Evaluating coherence, relevance, or other qualitative aspects of the responses

    return avg_reward, std_reward

if __name__ == "__main__":
    model_path = "dialoguerefiner_rlhf"
    test_dataset_path = "hh-helpful/test.jsonl"
    evaluate_model(model_path, test_dataset_path)