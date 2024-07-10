import json
import random

"""
DialogueDataset Class

This class manages the dataset used for RLHF (Reinforcement Learning from Human Feedback) training.
It loads and provides access to dialogue samples, including chosen (preferred) and rejected responses.

The dataset is expected to be in JSONL format, where each line is a JSON object containing
'chosen' and 'rejected' fields representing human-preferred and non-preferred responses respectively.

Key functionalities:
1. Load the dataset from a file
2. Provide random samples
3. Extract chosen and rejected responses
4. Extract prompts from dialogue samples
5. Iterate through all samples

This class is crucial for providing training data to both the reward model and the RLHF training process.
"""

class DialogueDataset:
    def __init__(self, data_path):
        """
        Initialize the dataset by loading data from the specified file.

        Args:
        data_path (str): Path to the JSONL file containing the dialogue dataset
        """
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def get_sample(self):
        """
        Return a random sample from the dataset.

        Returns:
        dict: A randomly selected dialogue sample
        """
        return random.choice(self.data)

    def get_chosen_response(self, sample):
        """
        Extract the chosen (human-preferred) response from a sample.

        Args:
        sample (dict): A dialogue sample

        Returns:
        str: The chosen response
        """
        return sample['chosen']

    def get_rejected_response(self, sample):
        """
        Extract the rejected (less preferred) response from a sample.

        Args:
        sample (dict): A dialogue sample

        Returns:
        str: The rejected response
        """
        return sample['rejected']

    def get_prompt(self, sample):
        """
        Extract the prompt (human query) from a dialogue sample.
        
        This method assumes the prompt is the first part of the dialogue,
        preceding the assistant's response.

        Args:
        sample (dict): A dialogue sample

        Returns:
        str: The extracted prompt, or None if no prompt is found
        """
        dialogue = sample['chosen'].split('\n\nHuman: ', 1)
        if len(dialogue) > 1:
            return 'Human: ' + dialogue[1].split('\n\nAssistant:', 1)[0]
        return None

    def iterate_samples(self):
        """
        Provide an iterator over all samples in the dataset.

        Yields:
        dict: Each dialogue sample in the dataset
        """
        for sample in self.data:
            yield sample