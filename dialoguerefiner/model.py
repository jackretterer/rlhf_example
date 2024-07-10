from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

"""
DialogueModel Class

This class encapsulates a GPT-2 based language model for generating dialogue responses.
It provides methods for initializing the model, generating responses, and updating the model parameters.

Key functionalities:
1. Initializes a pre-trained GPT-2 model and tokenizer.
2. Handles device management (CPU/GPU).
3. Generates responses to given prompts.
4. Allows for model parameter updates (useful in RL training scenarios).

Usage example:
    model = DialogueModel()
    response = model.generate_response("Hello, how are you?")
    print(response)

Note: This class is designed to be used as part of a larger RLHF (Reinforcement Learning from Human Feedback) system,
where the model can be continuously updated based on feedback.
"""

class DialogueModel:
    def __init__(self, model_name="gpt2"):
        # Initialize the tokenizer and model from pretrained GPT-2
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Set up GPU if available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Ensure there's a padding token (important for batching)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_response(self, prompt, max_length=50):
        # Tokenize the input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Generate a response using the model
        outputs = self.model.generate(
            **inputs, 
            max_length=max_length,  # Set maximum length of the generated response
            num_return_sequences=1,  # Generate only one response
            pad_token_id=self.tokenizer.pad_token_id  # Use the defined padding token
        )
        
        # Decode the generated token IDs back to text
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def update_model(self, new_params):
        # Update the model's parameters (used in RL training)
        self.model.load_state_dict(new_params)