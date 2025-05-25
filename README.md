# MiniReason

This repository serves as a lightweight playground for exploring the reasoning abilities of LLMs. It allows you to inspect model behavior, run benchmarks, and experiment with prompts across a variety of tasks, including:

- GSM8K (Grade School Math)
- MATH-500 (Competition Math Problems)
- AIME 2024 (American Invitational Mathematics Exam)
- Game of 24
- 5×5 Crossword
- GPQA Diamond (Graduate-Level Physics QA):
- CommonsenseQA

Currently under development — full setup and features coming soon.

If you are looking for a complete evaluation framework, refer to https://github.com/EleutherAI/lm-evaluation-harness.

## Overview

To evaluate performance, we experiment with a range of Large Language Models (LLMs), including:

- GPT-2
- Pythia 70M and Pythia 2.8B
- Gemma2 2B and Gemma2 9B
- DeepSeek-R1-Distill-Llama-8B
- Llama 3-8b-instruct and Llama 3.2-3b

## Architecture

### Evaluation Framework

The project includes a modular evaluation system for measuring model performance on mathematical reasoning tasks:

Parameters:

- `model`: Initialized HFModel instance
- `split`: Dataset split (test/train)
- `max_samples`: Quick-test sample limit
- `max_new_tokens`: Answer length limit

### Quick Start

Custom model evaluation with sample limit

```bash
python main.py \
  --models google/gemma-2-2b-it gpt2 \
  --max_gsm8k_samples 50
```

Customized wandb project and run name

```bash
python main.py --wandb --wandb_project <PROJECT_NAME> --wandb_name <RUN_NAME>
```

Logged metrics include:

- Direct/CoT/ToT accuracy per benchmark
- pass@k metrics
- Hyperparameter values (temperature, top_p, etc.)

### Testing

```bash
# Run all tests
python -m pytest tests/test_models.py -v

# Run a specific test group
python -m pytest tests/test_models.py::test_cot_reasoning -v
```

