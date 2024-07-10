from transformers import GPT2Model, GPT2Tokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os

"""
RewardModel Class

This class implements a reward model for RLHF (Reinforcement Learning from Human Feedback).
It uses a pre-trained GPT-2 model to learn to predict human preferences between pairs of responses.

Key functionalities:
1. Initializes a GPT-2 based model for reward prediction.
2. Provides methods for forward pass, reward calculation, and model training.
3. Implements save and load functionality for the trained model.

The reward model is crucial in RLHF as it guides the policy model towards generating
responses that align with human preferences.

Usage example:
    reward_model = RewardModel()
    reward_model.train_model(dataset)
    reward = reward_model.get_reward("Hello", "Hi there!")
"""

class RewardModel(nn.Module):
    def __init__(self, model_name="gpt2"):
        super(RewardModel, self).__init__()
        # Initialize GPT-2 model and tokenizer
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # Add a linear layer to produce a scalar reward
        self.score = nn.Linear(self.gpt2.config.n_embd, 1)

        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set up GPU if available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, input_ids, attention_mask):
        # Process input through GPT-2 and score layer
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        return self.score(last_hidden_states.mean(dim=1)).squeeze(-1)

    def get_reward(self, prompt, response):
        # Calculate reward for a given prompt-response pair
        inputs = self.tokenizer(prompt + response, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            return self.forward(inputs['input_ids'], inputs['attention_mask']).item()

    def train_model(self, dataset, batch_size=8, epochs=3, learning_rate=1e-5):
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        # Prepare the dataset
        chosen_inputs = [self.tokenizer(sample['chosen'], return_tensors="pt", truncation=True, max_length=512) for sample in dataset.data]
        rejected_inputs = [self.tokenizer(sample['rejected'], return_tensors="pt", truncation=True, max_length=512) for sample in dataset.data]

        chosen_ids = [inputs['input_ids'].squeeze(0) for inputs in chosen_inputs]
        chosen_mask = [inputs['attention_mask'].squeeze(0) for inputs in chosen_inputs]
        rejected_ids = [inputs['input_ids'].squeeze(0) for inputs in rejected_inputs]
        rejected_mask = [inputs['attention_mask'].squeeze(0) for inputs in rejected_inputs]

        # Pad sequences
        chosen_ids = pad_sequence(chosen_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        chosen_mask = pad_sequence(chosen_mask, batch_first=True, padding_value=0)
        rejected_ids = pad_sequence(rejected_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        rejected_mask = pad_sequence(rejected_mask, batch_first=True, padding_value=0)

        train_dataset = TensorDataset(chosen_ids, chosen_mask, rejected_ids, rejected_mask)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                chosen_ids, chosen_mask, rejected_ids, rejected_mask = [b.to(self.device) for b in batch]

                chosen_reward = self.forward(chosen_ids, chosen_mask)
                rejected_reward = self.forward(rejected_ids, rejected_mask)

                loss = criterion(chosen_reward - rejected_reward, torch.ones_like(chosen_reward))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")

        print("Reward model training completed.")

    def save_model(self, path):
        # Save the trained model and tokenizer
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "reward_model.pth"))
        self.tokenizer.save_pretrained(path)
        print(f"Reward model saved to {path}")

    @classmethod
    def load_model(cls, path):
        # Load a saved model and tokenizer
        model = cls()
        model.load_state_dict(torch.load(os.path.join(path, "reward_model.pth")))
        model.tokenizer = GPT2Tokenizer.from_pretrained(path)
        print(f"Reward model loaded from {path}")
        return model