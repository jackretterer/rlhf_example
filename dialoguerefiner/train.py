import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from .model import DialogueModel
from .reward_model import RewardModel
from .dataset import DialogueDataset
from .environment import DialogueEnvironment

"""
RLHF Training Module

This module implements the training process for Reinforcement Learning from Human Feedback (RLHF).
It combines a dialogue model, a reward model, and a training environment to fine-tune the dialogue model
using the PPO (Proximal Policy Optimization) algorithm.

Key components:
1. DialogueModel: The language model being fine-tuned
2. RewardModel: Predicts rewards based on human preferences
3. DialogueEnvironment: Simulates interactions for RL training
4. PPO: The reinforcement learning algorithm used for optimization

The training process involves:
- Initializing or loading pre-trained models
- Setting up the training environment
- Running the PPO algorithm to optimize the dialogue model
- Saving checkpoints and the final model

This implementation showcases how to set up and run an RLHF training pipeline using open-source tools.
"""

class ProgressCallback(BaseCallback):
    """
    Custom callback to track and print training progress.
    """
    def __init__(self, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.progress = 0

    def _on_step(self) -> bool:
        self.progress += 1
        if self.progress % 10 == 0:  # Print every 10 steps for 1000 total steps
            print(f"Progress: {self.progress} / {self.total_timesteps} steps ({self.progress / self.total_timesteps * 100:.2f}%)")
        return True

def train_rlhf():
    """
    Main function to run the RLHF training process.
    """
    # Check for GPU availability
    print("Checking GPU availability...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize models and dataset
    print("Initializing models and dataset...")
    dialogue_model = DialogueModel()
    train_dataset = DialogueDataset('hh-helpful/train.jsonl')

    # Load or train reward model
    reward_model_path = "reward_model"
    if os.path.exists(reward_model_path):
        print("Loading existing reward model...")
        reward_model = RewardModel.load_model(reward_model_path)
    else:
        print("Training new reward model...")
        reward_model = RewardModel()
        reward_model.train_model(train_dataset, epochs=1)  # Reduced epochs for testing

    # Set up the RL environment
    print("Setting up the environment...")
    env = DialogueEnvironment(dialogue_model, reward_model, train_dataset)

    # Initialize the PPO model
    print("Initializing PPO model...")
    model = PPO("MlpPolicy", env, verbose=1, device=device)

    # Prepare for training
    print("Starting training...")
    total_timesteps = 1000  # Define total timesteps
    checkpoint_callback = CheckpointCallback(save_freq=100, save_path="./checkpoints/")
    progress_callback = ProgressCallback(total_timesteps)
    
    # Run the training loop
    try:
        model.learn(total_timesteps=1000, callback=[checkpoint_callback, progress_callback])  # Reduced timesteps for testing
        print("Training completed successfully.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return

    # Save the final model
    print("Saving final model...")
    model.save("dialoguerefiner_rlhf")
    print("Model saved successfully.")

if __name__ == "__main__":
    train_rlhf()