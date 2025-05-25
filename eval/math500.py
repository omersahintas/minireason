import re
from datasets import load_dataset, Dataset, DatasetDict
from typing import Callable, List

from models import HFModel
from utils.memory import get_recommended_batch_size

# --------------------------------------------------------
# Helper functions
# --------------------------------------------------------

def _extract_boxed_answer(text: str) -> str | None:
    """Extract the answer inside a \boxed{ ... } pattern if present.

    Falls back to the **last** integer/float in the string otherwise.
    Returns `None` if nothing can be parsed.
    """
    # 1) Try MATH dataset's standard \boxed{...} markup
    boxed = re.search(r"\\boxed\s*{([^}]+)}", text)
    if boxed:
        return boxed.group(1).strip()

    # 2) Fallback – last number (int/float) in text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1]
    return None


# --------------------------------------------------------
# Shared evaluation helper
# --------------------------------------------------------

def _evaluate_common(
    model: HFModel,
    dataset_list: list,
    batch_size: int,
    prompt_fn: Callable[[dict], str],
    response_fn: Callable[[List[str]], List[str]],
    desc: str,
) -> tuple[float, int, int]:
    """Evaluation loop identical to GSM8K variant (batched)."""
    from tqdm import tqdm  # local import to avoid dependency if not used

    total = correct = 0
    pbar = tqdm(total=len(dataset_list), desc=desc)

    for start in range(0, len(dataset_list), batch_size):
        batch = dataset_list[start : start + batch_size]
        prompts = [prompt_fn(item) for item in batch]
        responses = response_fn(prompts)

        for item, response in zip(batch, responses):
            gold_raw = item["solution"].strip()
            gold = _extract_boxed_answer(gold_raw)
            pred = _extract_boxed_answer(response)

            if gold is not None and pred is not None and str(gold).strip() == str(pred).strip():
                correct += 1
            total += 1
        pbar.update(len(batch))

    pbar.close()
    accuracy = correct / total if total else 0.0
    return accuracy, total, correct


# --------------------------------------------------------
# Public API
# --------------------------------------------------------

def evaluate_math500(
    model: HFModel,
    split: str = "test",
    max_samples: int | None = 500,
    batch_size: int | None = None,
    max_new_tokens: int = 64,
) -> tuple[float, int, int]:
    """Evaluate on the first *500* math competition problems (MATH-500).

    By default uses the first 500 problems of Hendrycks et al. *MATH* dataset.
    Accuracy is computed via exact string match of `\boxed{...}` answer.
    """

    # Load the dataset (this returns a DatasetDict)
    dataset_dict = load_dataset("nlile/hendrycks-MATH-benchmark")
    dataset = dataset_dict[split]
    
    # Now dataset is a Dataset object, not a DatasetDict
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    if batch_size is None:
        batch_size = get_recommended_batch_size()

    dataset_list = list(dataset)

    # --- prompt & generation callbacks ---
    prompt_fn = lambda item: item["problem"].strip() + "\nAnswer:"

    def response_fn(prompts: List[str]) -> List[str]:
        return model.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            do_sample=False,
        )

    return _evaluate_common(
        model=model,
        dataset_list=dataset_list,
        batch_size=batch_size,
        prompt_fn=prompt_fn,
        response_fn=response_fn,
        desc=f"Evaluating MATH-500 ({split})",
    )

# ---------------------------------
# Chain-of-thought evaluation
# ---------------------------------

def evaluate_math500_cot(
    model: HFModel,
    split: str = "test",
    max_samples: int | None = 500,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
) -> tuple[float, int, int]:
    """Evaluate MATH-500 using `generate_cot` reasoning prompts."""

    # Skip CoT evaluation for GPT-2 which has too small context window
    if "gpt2" in model.model_name.lower():
        print(f"Skipping MATH-500 CoT for {model.model_name} - context window too small")
        # Return placeholder results to avoid breaking the evaluation flow
        return 0.0, 0, 0
        
    # Load the dataset (this returns a DatasetDict)
    dataset_dict = load_dataset("nlile/hendrycks-MATH-benchmark")
    dataset = dataset_dict[split]

    dataset_list = list(dataset)
    batch_size = 1  # CoT one by one

    prompt_fn = lambda item: item["problem"].strip()

    def response_fn(prompts: List[str]) -> List[str]:
        return [
            model.generate_cot(
                prompts[0],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        ]

    return _evaluate_common(
        model=model,
        dataset_list=dataset_list,
        batch_size=batch_size,
        prompt_fn=prompt_fn,
        response_fn=response_fn,
        desc=f"Evaluating MATH-500 CoT ({split})",
    )

# ---------------------------------
# Tree-of-Thought evaluation
# ---------------------------------

def evaluate_math500_tot(
    model: HFModel,
    split: str = "test",
    max_samples: int | None = 500,
    num_thoughts: int = 3,
    max_depth: int = 3,
    k_select: int = 2,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
) -> tuple[float, int, int]:
    """Evaluate MATH-500 with Tree-of-Thought reasoning (single sample per problem)."""
    # Skip ToT for GPT-2 due to context window
    if "gpt2" in model.model_name.lower():
        print(f"Skipping MATH-500 ToT for {model.model_name} - context window too small")
        return 0.0, 0, 0

    dataset_dict = load_dataset("nlile/hendrycks-MATH-benchmark")
    dataset = dataset_dict[split]
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    dataset_list = list(dataset)
    batch_size = 1  # ToT is iterative

    prompt_fn = lambda item: item["problem"].strip()

    def response_fn(prompts: List[str]) -> List[str]:
        answer, _reason = model.generate_tot(
            prompts[0],
            num_thoughts=num_thoughts,
            max_depth=max_depth,
            k_select=k_select,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return [answer]

    return _evaluate_common(
        model=model,
        dataset_list=dataset_list,
        batch_size=batch_size,
        prompt_fn=prompt_fn,
        response_fn=response_fn,
        desc=f"Evaluating MATH-500 ToT ({split})",
    ) 