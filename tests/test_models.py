from models import HFModel
import pytest

# List of model names to test sequentially
MODEL_NAMES = [
    ("google/gemma-2-2b-it", "2B"),
    ("google/gemma-2-9b-it", "9B"),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "DeepSeek-8B"),
    ("gpt2", "GPT-2"),
]

# Unified prompts for all tests
BASIC_PROMPT = "What are the main applications of sparse autoencoders and how do they relate to interpretability in large language models?"
COT_PROMPT = (
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
    "How much does the ball cost? Explain your reasoning step by step."
)
# Create a more complex prompt for ToT that benefits from exploring multiple reasoning paths
TOT_PROMPT = (
    "You have 8 balls that all weigh the same, except for one ball which is slightly heavier. "
    "Using a balance scale, what is the minimum number of weighings needed to find the heavier ball? "
    "Explain your reasoning."
)

MCTS_PROMPT = TOT_PROMPT  # reuse same complex prompt

@pytest.mark.parametrize("model_name, label", MODEL_NAMES)
def test_basic_generation(model_name, label):
    """Test basic text generation for each Gemma model with a unified prompt"""
    model = HFModel(model_name=model_name)

    print(f"Prompt ({label}): {BASIC_PROMPT}")
    print("-" * 50)

    response = model.generate(BASIC_PROMPT)
    print(f"Response ({label}):\n{response}")
    print("=" * 80)
    # Free GPU memory
    model.free_memory()


@pytest.mark.parametrize("model_name, label", MODEL_NAMES)
def test_cot_reasoning(model_name, label):
    """Test chain-of-thought reasoning for each Gemma model with a unified prompt"""
    model = HFModel(model_name=model_name)

    print(f"Prompt ({label}): {COT_PROMPT}")
    print("-" * 50)

    response = model.generate_cot(COT_PROMPT)
    print(f"CoT Response ({label}):\n{response}")
    print("=" * 80)
    # Free GPU memory
    model.free_memory()


@pytest.mark.parametrize("model_name, label", MODEL_NAMES)
def test_tot_reasoning(model_name, label):
    """Test tree-of-thought reasoning for each Gemma model with a unified prompt"""
    model = HFModel(model_name=model_name)

    print(f"Prompt ({label}): {TOT_PROMPT}")
    print("-" * 50)

    # Configure ToT parameters for testing - smaller values for faster tests
    answer, reasoning = model.generate_tot(
        TOT_PROMPT,
        num_thoughts=2,  # Reduced branching factor for testing
        max_depth=2,  # Reduced depth for testing
        k_select=2,  # Keep top-2 states at each level
        max_new_tokens=128,
    )

    print(f"ToT Reasoning ({label}):\n{reasoning}")
    print(f"ToT Answer ({label}):\n{answer}")
    print("=" * 80)
    # Free GPU memory
    model.free_memory()


# Testing different hyperparameter configurations
TOT_CONFIGS = [
    # num_thoughts, max_depth, k_select, max_tokens
    (2, 2, 2, 128),  # Minimal config
    (3, 3, 2, 128),  # Medium config
]


@pytest.mark.parametrize(
    "model_name, label", [MODEL_NAMES[0]]
)  # Only test with first model
@pytest.mark.parametrize("num_thoughts, max_depth, k_select, max_tokens", TOT_CONFIGS)
def test_tot_hyperparameters(
    model_name, label, num_thoughts, max_depth, k_select, max_tokens
):
    """Test different Tree of Thought hyperparameter configurations"""
    model = HFModel(model_name=model_name)

    print(f"ToT Config: b={num_thoughts}, T={max_depth}, k={k_select}")
    print(f"Prompt ({label}): {TOT_PROMPT}")
    print("-" * 50)

    answer, reasoning = model.generate_tot(
        TOT_PROMPT,
        num_thoughts=num_thoughts,
        max_depth=max_depth,
        k_select=k_select,
        max_new_tokens=max_tokens,
    )

    print(f"ToT Reasoning ({label}):\n{reasoning}")
    print(f"ToT Answer ({label}):\n{answer}")
    print("=" * 80)
    # Free GPU memory
    model.free_memory()


@pytest.mark.parametrize("model_name, label", MODEL_NAMES)
def test_mcts_reasoning(model_name, label):
    """Test MCTS reasoning for each model"""
    model = HFModel(model_name=model_name)

    print(f"Prompt ({label}): {MCTS_PROMPT}")
    print("-" * 50)

    answer, reasoning = model.generate_mcts(
        MCTS_PROMPT,
        num_simulations=200,
        max_depth=3,
        expansion_thoughts=3,
        evaluation_temperature=0.0,
        max_new_tokens=128,
    )

    print(f"MCTS Reasoning ({label}):\n{reasoning}")
    print(f"MCTS Answer ({label}):\n{answer}")
    print("=" * 80)
    model.free_memory()


# Hyperparameter grid for MCTS
MCTS_CONFIGS = [
#    (10, 2, 2, 128),
    (15, 10, 2, 128),
]

@pytest.mark.parametrize("model_name, label", [MODEL_NAMES[0]])
@pytest.mark.parametrize("num_sim, max_depth, expansion_thoughts, max_tokens", MCTS_CONFIGS)
def test_mcts_hyperparameters(model_name, label, num_sim, max_depth, expansion_thoughts, max_tokens):
    """Test different MCTS hyperparameter settings"""
    model = HFModel(model_name=model_name)

    print(f"MCTS Config: sims={num_sim}, depth={max_depth}, thoughts={expansion_thoughts}")
    answer, reasoning = model.generate_mcts(
        MCTS_PROMPT,
        num_simulations=num_sim,
        max_depth=max_depth,
        expansion_thoughts=expansion_thoughts,
        max_new_tokens=max_tokens,
    )
    print(f"MCTS Reasoning ({label}):\n{reasoning}")
    print(f"MCTS Answer ({label}):\n{answer}")
    print("=" * 80)
    model.free_memory()


if __name__ == "__main__":
    print("Testing Gemma Model Integration")
    print("=" * 80)

    for model_name, label in MODEL_NAMES:

        print("\n--- Basic Generation Test ---")
        temp_model_basic = HFModel(model_name=model_name)
        print(f"Prompt ({label}): {BASIC_PROMPT}")
        print("-" * 50)
        response_basic = temp_model_basic.generate(BASIC_PROMPT)
        print(f"Response ({label}):\n{response_basic}")
        print("=" * 80)
        # Free GPU memory
        temp_model_basic.free_memory()

        print("\n--- CoT Reasoning Test ---")
        temp_model_cot = HFModel(model_name=model_name)
        print(f"Prompt ({label}): {COT_PROMPT}")
        print("-" * 50)
        response_cot = temp_model_cot.generate_cot(COT_PROMPT)
        print(f"CoT Response ({label}):\n{response_cot}")
        print("=" * 80)
        # Free GPU memory
        temp_model_cot.free_memory()

        print("\n--- Tree of Thought Reasoning Test ---")
        temp_model_tot = HFModel(model_name=model_name)
        print(f"Prompt ({label}): {TOT_PROMPT}")
        print("-" * 50)
        # Configure ToT parameters for testing - smaller values for faster tests
        answer_tot, reasoning_tot = temp_model_tot.generate_tot(
            TOT_PROMPT,
            num_thoughts=2,  # Reduced branching factor for testing
            max_depth=2,  # Reduced depth for testing
            k_select=2,  # Keep top-2 states at each level
            max_new_tokens=128,
        )
        print(f"ToT Reasoning ({label}):\n{reasoning_tot}")
        print(f"ToT Answer ({label}):\n{answer_tot}")
        print("=" * 80)
        # Free GPU memory
        temp_model_tot.free_memory()

        print("\n--- MCTS Reasoning Test ---")
        temp_model_mcts = HFModel(model_name=model_name)
        print(f"Prompt ({label}): {MCTS_PROMPT}")
        print("-" * 50)
        answer_mcts, reasoning_mcts = temp_model_mcts.generate_mcts(
            MCTS_PROMPT,
            num_simulations=200,
            max_depth=3,
            expansion_thoughts=3,
            evaluation_temperature=0.2,
            max_new_tokens=128,
        )
        print(f"MCTS Reasoning ({label}):\n{reasoning_mcts}")
        print(f"MCTS Answer ({label}):\n{answer_mcts}")
        print("=" * 80)
        # Free GPU memory
        temp_model_mcts.free_memory()
