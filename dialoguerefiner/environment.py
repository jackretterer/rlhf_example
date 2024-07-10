import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np

"""
DialogueEnvironment Class

This class implements a custom Gym environment for RLHF (Reinforcement Learning from Human Feedback) 
in the context of dialogue generation. It simulates interactions between a dialogue model and a 
reward model, allowing for reinforcement learning to optimize the dialogue model's responses.

Key components:
1. dialogue_model: The language model generating responses
2. reward_model: Provides rewards based on the quality of generated responses
3. dataset: Source of prompts/contexts for dialogue generation

The environment follows the OpenAI Gym interface, providing methods for resetting the environment,
taking steps (generating responses), and rendering the current state.
"""

class DialogueEnvironment(gym.Env):
    def __init__(self, dialogue_model, reward_model, dataset):
        super(DialogueEnvironment, self).__init__()
        self.dialogue_model = dialogue_model
        self.reward_model = reward_model
        self.dataset = dataset

        # Define action and observation spaces
        # Action space: Single continuous value influencing response generation (e.g., temperature)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Observation space: Embedding space of the dialogue model
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.dialogue_model.model.config.n_embd,), dtype=np.float32)

        self.current_sample = None
        self.current_prompt = None

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to a new initial state.
        Returns the initial observation.
        """
        super().reset(seed=seed)
        self.current_sample = self.dataset.get_sample()
        self.current_prompt = self.dataset.get_prompt(self.current_sample)
        
        # Generate initial observation (embedding of the current prompt)
        with torch.no_grad():
            inputs = self.dialogue_model.tokenizer(self.current_prompt, return_tensors="pt", padding=True, truncation=True).to(self.dialogue_model.device)
            hidden_states = self.dialogue_model.model.transformer(inputs.input_ids, attention_mask=inputs.attention_mask).last_hidden_state
            initial_observation = hidden_states.mean(dim=1).cpu().numpy().flatten()
        
        return initial_observation, {}  # Return observation and an empty info dict

    def step(self, action):
        """
        Take a step in the environment: generate a response and get its reward.
        
        Args:
        action (float): Value influencing response generation (e.g., temperature)

        Returns:
        observation: The new state (embedding of the generated response)
        reward: The reward for the generated response
        done: Whether the episode has ended (always False in this implementation)
        truncated: Whether the episode was truncated (always False here)
        info: Additional information (contains the generated response)
        """
        # Map action to temperature for response generation
        temperature = np.clip(action[0], 0.1, 2.0)
        
        # Generate response
        with torch.no_grad():
            inputs = self.dialogue_model.tokenizer(self.current_prompt, return_tensors="pt", padding=True, truncation=True).to(self.dialogue_model.device)
            outputs = self.dialogue_model.model.generate(
                inputs.input_ids, 
                attention_mask=inputs.attention_mask,
                max_length=100, 
                num_return_sequences=1, 
                no_repeat_ngram_size=2,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.dialogue_model.tokenizer.pad_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

        # Decode the generated response
        response = self.dialogue_model.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Get reward for the generated response
        reward = self.reward_model.get_reward(self.current_prompt, response)

        # Prepare the next observation (embedding of the generated response)
        next_obs = outputs.hidden_states[-1][-1].mean(dim=1).cpu().numpy().flatten()

        done = False
        info = {"response": response}

        return next_obs, reward, done, False, info

    def render(self):
        """
        Render the current state of the environment (print prompt and generated response).
        """
        print(f"Prompt: {self.current_prompt}")
        print(f"Response: {self.dialogue_model.generate_response(self.current_prompt)}")