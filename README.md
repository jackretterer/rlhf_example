# Dialogue Refiner: An Introduction to RLHF

## Overview

Dialogue Refiner is a GitHub repository designed to introduce Machine Learning Engineers (MLEs) and data scientists to Reinforcement Learning from Human Feedback (RLHF). This project aims to demonstrate how to take existing Large Language Models (LLMs) and fine-tune them to be more helpful and harmless using RLHF techniques.

This repository leverages Anthropic's public [hh-rlhf repo](https://github.com/anthropics/hh-rlhf/tree/master) and provides a basic, beginner-friendly implementation of RLHF.

## Features

- Basic implementation of RLHF using open-source tools
- Integration with pre-trained language models
- Custom reward model for learning human preferences
- Simulated environment for RLHF training
- Example scripts for training and evaluation

## Prerequisites

- Python 3.8+
- pip
- Virtual environment tool (e.g., venv)
- I'd recommend a GPU :)

## Installation

1. Clone the repository:

```git clone https://github.com/your-username/dialogue-refiner.git```

2. Create a virtual environment:

```python -m venv venv```

3. Activate the virtual environment

```source venv/bin/activate```

4. Ensure your IDE is using the virtual environment:
- For Visual Studio Code: Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux), type "Python: Select Interpreter", and choose the interpreter in the `venv` folder.

5. Install the necessary packages:

``` pip install -r requirements.txt```

## Usage

1. Prepare your dataset:
- Place your training data in the `hh-helpful` directory
- Ensure you have both `train.jsonl` and `test.jsonl` files

2. Train the RLHF model:

```python -m dialoguerefiner.train```

3. Evaluate the trained model:

```python -m dialoguerefiner.evaluate```